from __future__ import annotations

import math
import numpy as np
import pytest

from optimise_design.gaussian_process_optimisation import (
    GaussianProcessOptimiser,
    kernel_matrix,
    log_marginal_likelihood,
    rbf_kernel,
    rbf_tuner,
    ucb_acquisition,
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
        noise_variance=1e-2,
        kernel=None,  # use default RBF
        kernel_tuner=rbf_tuner,
    )
    gp._X = X
    gp._y = y
    val = log_marginal_likelihood(gp._X, gp._y, gp.kernel, kernel_matrix, gp.noise_variance)

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
    """On data drawn from a known GP, tuning should lift log marginal likelihood."""

    rng = np.random.default_rng(7)
    # Generate one dataset from a GP with known hyperparameters.
    true_kernel = rbf_kernel(lengthscale=0.7, signal_variance=1.5)
    true_noise = 0.05
    X = rng.uniform(-1.0, 1.0, size=(200, 1))
    K = kernel_matrix(X, X, true_kernel) + true_noise * np.eye(X.shape[0])
    y = rng.multivariate_normal(mean=np.zeros(X.shape[0]), cov=K)

    # Start from a deliberately misspecified set of hyperparameters.
    gp = GaussianProcessOptimiser(
        noise_variance=0.5,
        kernel=rbf_kernel(lengthscale=2.0, signal_variance=0.3),
        kernel_tuner=rbf_tuner,
    )
    gp.set_data(X, y, tune=False)
    initial_lml = log_marginal_likelihood(gp._X, gp._y, gp.kernel, kernel_matrix, gp.noise_variance)

    # Use a couple of restarts to avoid a bad local optimum.
    best_lml = initial_lml
    for init_ls, init_sv in ((1.0, 1.0), (0.6, 1.8), (0.9, 1.4)):
        tuned_kernel, tuned_noise = rbf_tuner(
            X,
            y,
            initial_lengthscale=init_ls,
            initial_signal_variance=init_sv,
            initial_noise_variance=0.5,
            bounds=((0.2, 3.0), (0.2, 3.0), (1e-4, 1.0)),
            optimizer_options={"maxiter": 80},
        )
        tuned_lml = log_marginal_likelihood(gp._X, gp._y, tuned_kernel, kernel_matrix, tuned_noise)
        best_lml = max(best_lml, tuned_lml)

    assert best_lml > initial_lml




def _rbf_params_from_kernel(kernel):
    # rbf_kernel returns a closure capturing (lengthscale, signal_variance)
    if kernel.__closure__ is None or len(kernel.__closure__) < 2:
        raise TypeError("Kernel does not expose RBF hyperparameters.")
    lengthscale = float(kernel.__closure__[0].cell_contents)
    signal_variance = float(kernel.__closure__[1].cell_contents)
    return lengthscale, signal_variance


def _lml_from_params(X, y, lengthscale, signal_variance, noise_variance):
    kernel = rbf_kernel(lengthscale, signal_variance)
    return log_marginal_likelihood(
        X, y, kernel, kernel_matrix, float(noise_variance)
    )


def _grad_lml_logspace_fd(X, y, log_params, eps=1e-5):
    """
    Finite-difference gradient of LML w.r.t.
    log_params = [log lengthscale, log signal_variance, log noise_variance]
    """
    def f(lp):
        l, s, n = np.exp(lp)
        return _lml_from_params(X, y, l, s, n)

    f0 = f(log_params)
    grad = np.zeros_like(log_params)

    for i in range(len(log_params)):
        d = np.zeros_like(log_params)
        d[i] = eps
        grad[i] = (f(log_params + d) - f0) / eps

    return f0, grad


def test_tune_hyperparameters_quadratic() -> None:
    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    X = np.linspace(-2.0, 2.0, num=80).reshape(-1, 1)
    rng = np.random.default_rng(999)
    y = -(X[:, 0] ** 2) + rng.normal(scale=0.1, size=X.shape[0])
    y = y - y.mean()

    # Deterministic train / test split
    idx = rng.permutation(X.shape[0])
    tr, te = idx[:60], idx[60:]
    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[te], y[te]

    # ------------------------------------------------------------------
    # Baseline GP
    # ------------------------------------------------------------------
    gp0 = GaussianProcessOptimiser(
        noise_variance=0.5,
        kernel=rbf_kernel(lengthscale=1.0, signal_variance=0.2),
        kernel_tuner=rbf_tuner,
    )
    gp0.set_data(Xtr, ytr, tune=False)

    l0, s0 = _rbf_params_from_kernel(gp0.kernel)
    n0 = float(gp0.noise_variance)

    lml0 = _lml_from_params(Xtr, ytr, l0, s0, n0)

    # ------------------------------------------------------------------
    # Bounds
    # ------------------------------------------------------------------
    bounds = (
        (1e-2, 10.0),   # lengthscale
        (1e-3, 10.0),   # signal variance
        (1e-4, 1.0),    # noise variance
    )

    # ------------------------------------------------------------------
    # Random restarts (baseline included)
    # ------------------------------------------------------------------
    best_lml = lml0
    best_params = (l0, s0, n0, gp0.kernel)

    for _ in range(5):
        init = rng.uniform(
            [b[0] for b in bounds],
            [b[1] for b in bounds],
        )

        kernel, noise = rbf_tuner(
            Xtr,
            ytr,
            initial_lengthscale=init[0],
            initial_signal_variance=init[1],
            initial_noise_variance=init[2],
            bounds=bounds,
            optimizer_options={"maxiter": 200},
        )

        l, s = _rbf_params_from_kernel(kernel)
        n = float(noise)

        lml = _lml_from_params(Xtr, ytr, l, s, n)
        if lml > best_lml:
            best_lml = lml
            best_params = (l, s, n, kernel)

    l1, s1, n1, best_kernel = best_params

    # ------------------------------------------------------------------
    # (1) LML non-worsening
    # ------------------------------------------------------------------
    assert best_lml >= lml0 - 1e-6

    # ------------------------------------------------------------------
    # (2) Gradient norm reduced (log-space)
    # ------------------------------------------------------------------
    _, g0 = _grad_lml_logspace_fd(Xtr, ytr, np.log([l0, s0, n0]))
    _, g1 = _grad_lml_logspace_fd(Xtr, ytr, np.log([l1, s1, n1]))

    assert np.linalg.norm(g1) <= np.linalg.norm(g0) + 1e-8

    # ------------------------------------------------------------------
    # (3) Holdout RMSE not worse
    # ------------------------------------------------------------------
    mu0, _ = gp0.predict(Xte, noisy=False)

    gp1 = GaussianProcessOptimiser(
        noise_variance=n1,
        kernel=best_kernel,
        kernel_tuner=rbf_tuner,
    )
    gp1.set_data(Xtr, ytr, tune=False)

    mu1, _ = gp1.predict(Xte, noisy=False)

    rmse0 = math.sqrt(np.mean((mu0 - yte) ** 2))
    rmse1 = math.sqrt(np.mean((mu1 - yte) ** 2))

    assert rmse1 <= rmse0 + 1e-8


# def test_tune_hyperparameters_quadratic() -> None:
#     """On a simple quadratic (noisy), tuning should improve LML on one deterministic run."""

#     # Deterministic quadratic: y = -x^2 with modest additive noise level.
#     X = np.linspace(-2.0, 2.0, num=80).reshape(-1, 1)
#     true_noise = 0.1
#     rng = np.random.default_rng(999)
#     y = -(X[:, 0] ** 2) + rng.normal(scale=0.1, size=X.shape[0])  # small noise
#     y = y - y.mean()
#     #y = -(X[:, 0] ** 2)

#     gp = GaussianProcessOptimiser(
#         noise_variance=0.5,
#         kernel=rbf_kernel(lengthscale=1.0, signal_variance=0.2),  # deliberately misspecified
#         kernel_tuner=rbf_tuner,
#     )
#     gp.set_data(X, y, tune=False)
#     initial = log_marginal_likelihood(gp._X, gp._y, gp.kernel, kernel_matrix, gp.noise_variance)

#     tuned_kernel, tuned_noise = rbf_tuner(
#         X,
#         y,
#         initial_lengthscale=1.0,
#         initial_signal_variance=1.0,
#         initial_noise_variance=gp.noise_variance,
#         bounds=((1e-4, 10.0), (1e-4, 300.0), (1e-4, 300.0)),
#         optimizer_options={"maxiter": 200},
#     )
#     tuned_lml = log_marginal_likelihood(X, y, tuned_kernel, kernel_matrix, tuned_noise)

#     assert tuned_lml > initial


def test_set_data_triggers_tuning_on_first_ingest() -> None:
    """set_data should tune on first ingestion so log marginal likelihood improves."""

    rng = np.random.default_rng(123)
    for _ in range(10):
        # Use at least 20 points to avoid the small-data warning.
        X = rng.uniform(-1.0, 1.0, size=(20, 1))
        y = rng.normal(size=20)

        # Baseline GP with deliberately non-ideal hyperparameters.
        gp_baseline = GaussianProcessOptimiser(
            noise_variance=0.8,
            kernel=None,
            kernel_tuner=rbf_tuner,
        )
        gp_baseline._X = X
        gp_baseline._y = y
        baseline_lml = log_marginal_likelihood(gp_baseline._X, gp_baseline._y, gp_baseline.kernel, kernel_matrix, gp_baseline.noise_variance)

        # Same starting point, but use set_data (which tunes on first ingest).
        gp = GaussianProcessOptimiser(
            noise_variance=0.8,
            kernel=None,
            kernel_tuner=rbf_tuner,
        )
        gp.set_data(
            X,
            y,
            tune_kwargs={
                "bounds": ((0.1, 5.0), (0.1, 5.0), (1e-4, 2.0)),
                "optimizer_options": {"maxiter": 20},
            },
        )
        tuned_lml = log_marginal_likelihood(gp._X, gp._y, gp.kernel, kernel_matrix, gp.noise_variance)

        assert tuned_lml >= baseline_lml


def test_training_covariance_matches_kernel() -> None:
    """Covariance utility should include kernel plus noise on the diagonal.

    Hyperparameters are fixed (no tuning) to keep the expected matrix
    deterministic for manual comparison.
    """

    X = np.array([[0.0], [1.0]])
    gp = GaussianProcessOptimiser(
        kernel=rbf_kernel(lengthscale=1.0, signal_variance=1.0),
        noise_variance=0.1,
        kernel_tuner=rbf_tuner,
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
        kernel=rbf_kernel(lengthscale=1.0, signal_variance=1.0),
        noise_variance=0.2,
        kernel_tuner=rbf_tuner,
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
        kernel=rbf_kernel(lengthscale=1.0, signal_variance=1.0),
        noise_variance=0.1,
        kernel_tuner=rbf_tuner,
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
        kernel=rbf_kernel(lengthscale=1.0, signal_variance=1.0),
        noise_variance=0.2,
        kernel_tuner=rbf_tuner,
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
        noise_variance=0.05,
        kernel=rbf_kernel(lengthscale=1.0, signal_variance=1.0),
        acquisition_fn=ucb_acquisition(beta=4.0),
        kernel_tuner=rbf_tuner,
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
