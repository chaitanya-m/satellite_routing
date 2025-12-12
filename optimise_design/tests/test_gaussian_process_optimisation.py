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
    """log_marginal_likelihood should match a hand-computed value."""

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
