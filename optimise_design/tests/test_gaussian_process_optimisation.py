from __future__ import annotations

import numpy as np
import pytest

from optimise_design.gaussian_process_optimisation import (
    GaussianProcessOptimiser,
    rbf_kernel,
)


def test_rbf_kernel_computes_expected_value() -> None:
    """RBF kernel should match the exp(-0.5 * ||x-y||^2 / ℓ^2) formula."""

    kernel = rbf_kernel(lengthscale=2.0)
    x = np.array([1.0, 2.0])
    y = np.array([4.0, 6.0])
    diff = x - y
    sqdist = np.dot(diff, diff)
    expected = float(np.exp(-0.5 * sqdist / (2.0**2)))

    assert abs(kernel(x, y) - expected) < 1e-12


def test_log_marginal_likelihood_runs() -> None:
    """log_marginal_likelihood should match a hand-computed value.

    No hyperparameter tuning; keeping fixed hyperparameters makes the expected
    value deterministic for the manual calculation.
    """

    X = np.array([[0.0], [1.0]])
    y = np.array([0.0, 1.0])
    gp = GaussianProcessOptimiser(
        kernel=None,  # use default RBF
        lengthscale=1.0,
        signal_variance=1.0,
        noise_variance=1e-2,
        ucb_beta=1.0,
    )
    gp._X = X
    gp._y = y
    val = gp.log_marginal_likelihood()

    # Hand-compute the same quantity: K = k(X,X) + σ_n² I, then use the standard
    # GP log marginal likelihood formula.
    def k(x: np.ndarray, z: np.ndarray) -> float:
        diff = x - z
        return float(np.exp(-0.5 * np.dot(diff, diff)))

    K = np.array(
        [
            [k(X[0], X[0]), k(X[0], X[1])],
            [k(X[1], X[0]), k(X[1], X[1])],
        ],
        dtype=float,
    )
    K += 1e-2 * np.eye(2)
    L = np.linalg.cholesky(K + 1e-8 * np.eye(2))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    expected = -0.5 * float(y.T @ alpha)
    expected -= float(np.sum(np.log(np.diag(L))))
    expected -= 0.5 * X.shape[0] * np.log(2 * np.pi)

    assert np.isfinite(val)
    assert abs(val - expected) < 1e-6


def test_tune_hyperparameters_improves_lml() -> None:
    """Hyperparameter tuning should increase the log marginal likelihood."""

    X = np.array([[0.0], [0.5], [1.0]])
    y = np.array([0.0, 0.6, 1.0])
    gp = GaussianProcessOptimiser(
        kernel=None,  # default RBF
        lengthscale=2.0,
        signal_variance=0.5,
        noise_variance=0.5,
        ucb_beta=1.0,
    )
    gp._X = X
    gp._y = y

    initial_lml = gp.log_marginal_likelihood()
    gp.tune_hyperparameters(
        initial=(2.0, 0.5, 0.5),
        bounds=((0.1, 5.0), (0.1, 5.0), (1e-4, 1.0)),
    )
    tuned_lml = gp.log_marginal_likelihood()

    assert tuned_lml >= initial_lml


def test_tune_hyperparameters_across_many_seeds() -> None:
    """Hyperparameter tuning should behave across many random initialisations.
    On every random tiny dataset, tuning should be greater."""

    rng = np.random.default_rng(42)
    for _ in range(100):
        # Random tiny dataset.
        X = rng.uniform(-1.0, 1.0, size=(3, 1))
        y = rng.normal(size=3)
        gp = GaussianProcessOptimiser(
            kernel=None,
            lengthscale=rng.uniform(0.5, 2.0),
            signal_variance=rng.uniform(0.5, 2.0),
            noise_variance=rng.uniform(1e-3, 0.5),
            ucb_beta=1.0,
        )
        gp._X = X
        gp._y = y
        initial = gp.log_marginal_likelihood()
        gp.tune_hyperparameters(
            initial=(gp.lengthscale, gp.signal_variance, gp.noise_variance),
            bounds=((0.1, 3.0), (0.1, 3.0), (1e-4, 1.0)),
            optimizer_options={"maxiter": 5},
        )
        tuned = gp.log_marginal_likelihood()
        assert tuned > initial


def test_set_data_triggers_tuning_on_first_ingest() -> None:
    """set_data should tune on first ingestion so log marginal likelihood improves."""

    rng = np.random.default_rng(123)
    for _ in range(10):
        # Use at least 20 points to avoid the small-data warning.
        X = rng.uniform(-1.0, 1.0, size=(20, 1))
        y = rng.normal(size=20)

        # Baseline GP with deliberately non-ideal hyperparameters.
        gp_baseline = GaussianProcessOptimiser(
            kernel=None,
            lengthscale=3.0,
            signal_variance=0.2,
            noise_variance=0.8,
            ucb_beta=1.0,
        )
        gp_baseline._X = X
        gp_baseline._y = y
        baseline_lml = gp_baseline.log_marginal_likelihood()

        # Same starting point, but use set_data (which tunes on first ingest).
        gp = GaussianProcessOptimiser(
            kernel=None,
            lengthscale=3.0,
            signal_variance=0.2,
            noise_variance=0.8,
            ucb_beta=1.0,
        )
        gp.set_data(
            X,
            y,
            tune_kwargs={
                "bounds": ((0.1, 5.0), (0.1, 5.0), (1e-4, 2.0)),
                "optimizer_options": {"maxiter": 20},
            },
        )
        tuned_lml = gp.log_marginal_likelihood()

        assert tuned_lml >= baseline_lml


def test_training_covariance_matches_kernel() -> None:
    """Covariance utility should include kernel plus noise on the diagonal.

    Hyperparameters are fixed (no tuning) to keep the expected matrix
    deterministic for manual comparison.
    """

    X = np.array([[0.0], [1.0]])
    gp = GaussianProcessOptimiser(
        kernel=None,
        lengthscale=1.0,
        signal_variance=1.0,
        noise_variance=0.1,
        ucb_beta=1.0,
    )
    gp._X = X
    gp._y = np.array([0.0, 1.0])

    cov = gp.training_covariance()
    # Manually compute expected.
    def k(x: np.ndarray, z: np.ndarray) -> float:
        diff = x - z
        return float(np.exp(-0.5 * np.dot(diff, diff)))

    expected = np.array(
        [
            [k(X[0], X[0]), k(X[0], X[1])],
            [k(X[1], X[0]), k(X[1], X[1])],
        ]
    )
    expected += 0.1 * np.eye(2)
    assert np.allclose(cov, expected)


def test_joint_prior_covariances_blocks() -> None:
    """joint_prior_covariances should return all four blocks explicitly.

    Hyperparameters are fixed (no tuning) so the expected blocks remain
    deterministic for manual comparison.
    """

    X_train = np.array([[0.0], [1.0]])
    X_test = np.array([[0.5], [1.5]])
    gp = GaussianProcessOptimiser(
        kernel=None,
        lengthscale=1.0,
        signal_variance=1.0,
        noise_variance=0.2,
        ucb_beta=1.0,
    )
    gp._X = X_train
    gp._y = np.array([0.0, 1.0])

    K_tt, K_tT, K_Tt, K_TT = gp.joint_prior_covariances(X_test)

    # Manually compute expected blocks.
    def k(x: np.ndarray, z: np.ndarray) -> float:
        diff = x - z
        return float(np.exp(-0.5 * np.dot(diff, diff)))

    expected_tt = np.array(
        [
            [k(X_train[0], X_train[0]), k(X_train[0], X_train[1])],
            [k(X_train[1], X_train[0]), k(X_train[1], X_train[1])],
        ]
    )
    expected_tt += 0.2 * np.eye(2)

    expected_tT = np.array(
        [
            [k(X_train[0], X_test[0]), k(X_train[0], X_test[1])],
            [k(X_train[1], X_test[0]), k(X_train[1], X_test[1])],
        ]
    )
    expected_Tt = expected_tT.T
    expected_TT = np.array(
        [
            [k(X_test[0], X_test[0]), k(X_test[0], X_test[1])],
            [k(X_test[1], X_test[0]), k(X_test[1], X_test[1])],
        ]
    )

    assert np.allclose(K_tt, expected_tt)
    assert np.allclose(K_tT, expected_tT)
    assert np.allclose(K_Tt, expected_Tt)
    assert np.allclose(K_TT, expected_TT)


def test_compute_posterior() -> None:
    """Posterior mean/cov should match manual GP conditioning.

    Hyperparameters are fixed (no tuning) to keep the manual conditioning
    calculation deterministic.
    """

    X_train = np.array([[0.0], [1.0]])
    y_train = np.array([0.0, 1.0])
    X_test = np.array([[0.5]])
    gp = GaussianProcessOptimiser(
        kernel=None,
        lengthscale=1.0,
        signal_variance=1.0,
        noise_variance=0.1,
        ucb_beta=1.0,
    )
    gp._X = X_train
    gp._y = y_train

    mean, cov = gp.compute_posterior(X_test)

    # Manual conditioning.
    def k(x: np.ndarray, z: np.ndarray) -> float:
        diff = x - z
        return float(np.exp(-0.5 * np.dot(diff, diff)))

    K = np.array(
        [
            [k(X_train[0], X_train[0]), k(X_train[0], X_train[1])],
            [k(X_train[1], X_train[0]), k(X_train[1], X_train[1])],
        ]
    )
    K += 0.1 * np.eye(2)
    k_star = np.array([[k(X_train[0], X_test[0])], [k(X_train[1], X_test[0])]])
    k_star_star = np.array([[k(X_test[0], X_test[0])]])

    K_inv = np.linalg.inv(K)
    expected_mean = k_star.T @ K_inv @ y_train
    expected_cov = k_star_star - k_star.T @ K_inv @ k_star

    assert np.allclose(mean, expected_mean)
    assert np.allclose(cov, expected_cov)


def test_predict_with_and_without_noise() -> None:
    """predict should match manual posterior mean/variance, with optional noise.

    Hyperparameters are fixed (no tuning) so the manual posterior calculation
    stays deterministic.
    """

    X_train = np.array([[0.0], [1.0]])
    y_train = np.array([0.0, 1.0])
    X_test = np.array([[0.25], [0.75]])
    gp = GaussianProcessOptimiser(
        kernel=None,
        lengthscale=1.0,
        signal_variance=1.0,
        noise_variance=0.2,
        ucb_beta=1.0,
    )
    gp._X = X_train
    gp._y = y_train

    mean, cov = gp.compute_posterior(X_test)
    mean_pred, var_pred = gp.predict(X_test, noisy=False)
    mean_pred_noisy, var_pred_noisy = gp.predict(X_test, noisy=True)

    expected_var = np.diag(cov)
    expected_var_noisy = expected_var + 0.2

    assert np.allclose(mean_pred, mean.ravel())
    assert np.allclose(var_pred, expected_var)
    assert np.allclose(var_pred_noisy, expected_var_noisy)


def test_acquire_next_defaults_to_ucb_and_accepts_custom_acquisition() -> None:
    """acquire_next should pick the best candidate under UCB and allow custom scores."""

    # Training data near 0 and 1.
    X_train = np.array([[0.0], [1.0]])
    y_train = np.array([0.0, 1.0])
    gp = GaussianProcessOptimiser(
        kernel=None,
        lengthscale=1.0,
        signal_variance=1.0,
        noise_variance=0.05,
        ucb_beta=4.0,  # encourage exploration
    )
    gp._X = X_train
    gp._y = y_train

    # Candidate set: one close to data (low variance), one far (higher variance).
    X_candidates = np.array([[0.5], [2.0]])
    mean, var = gp.predict(X_candidates, noisy=False)

    # Under UCB with larger beta, the far point should score higher due to variance.
    best_x, scores = gp.acquire_next(X_candidates)
    assert np.allclose(best_x, X_candidates[1])
    assert scores.shape == (2,)

    # Custom acquisition: pure exploitation of mean only; should pick the higher mean.
    def exploit_only(mu: np.ndarray, _: np.ndarray) -> np.ndarray:
        return mu

    best_x_exploit, scores_exploit = gp.acquire_next(X_candidates, acquisition=exploit_only)
    # assert best_x_exploit is the candidate with highest mean (the score is just the mean, so argmax of scores)
    assert np.allclose(best_x_exploit, X_candidates[np.argmax(scores_exploit)]) 

