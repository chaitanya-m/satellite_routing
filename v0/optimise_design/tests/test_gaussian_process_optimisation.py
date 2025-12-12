from __future__ import annotations

import math
import numpy as np
import pytest

from optimise_design.gaussian_process_optimisation import (
    GaussianProcessOptimiser,
    RBFKernel,
    log_marginal_likelihood,
    rbf_tuner,
)


# ---------------------------------------------------------------------
# Kernel tests
# ---------------------------------------------------------------------

def test_rbf_kernel_computes_expected_value() -> None:
    """RBF kernel should match exp(-0.5 * ||x-y||^2 / ℓ^2)."""

    kernel = RBFKernel(lengthscale=2.0, signal_variance=1.0)
    x = np.array([1.0, 2.0])
    y = np.array([4.0, 6.0])

    diff = x - y
    sqdist = float(diff @ diff)
    expected = math.exp(-0.5 * sqdist / (2.0 ** 2))

    assert abs(kernel(x, y)[0, 0] - expected) < 1e-12


# ---------------------------------------------------------------------
# Marginal likelihood
# ---------------------------------------------------------------------

def test_log_marginal_likelihood_matches_manual_small_case() -> None:
    """LML should match a hand-computed value on a tiny deterministic example."""

    X = np.array([[0.0], [1.0]])
    y = np.array([0.0, 1.0])

    kernel = RBFKernel(lengthscale=1.0, signal_variance=1.0)
    noise = 1e-2

    val = log_marginal_likelihood(X, y, kernel, noise)

    # Manual computation
    def k(x, z):
        return math.exp(-0.5 * (x - z) ** 2)

    K = np.array(
        [
            [k(0.0, 0.0), k(0.0, 1.0)],
            [k(1.0, 0.0), k(1.0, 1.0)],
        ]
    )
    K += noise * np.eye(2) + 1e-8 * np.eye(2)

    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))

    expected = (
        -0.5 * y @ alpha
        - np.sum(np.log(np.diag(L)))
        - 0.5 * 2 * math.log(2 * math.pi)
    )

    assert abs(val - expected) < 1e-6


# ---------------------------------------------------------------------
# GP prediction
# ---------------------------------------------------------------------

def test_predict_shapes_and_nonnegative_variance() -> None:
    """predict returns correct shapes and non-negative variances."""

    X = np.linspace(-1.0, 1.0, 10).reshape(-1, 1)
    y = np.sin(X[:, 0])

    gp = GaussianProcessOptimiser(
        kernel=RBFKernel(1.0, 1.0),
        noise_variance=0.1,
    )
    gp.set_data(X, y, tune=False)

    X_star = np.array([[0.0], [0.5]])
    mean, var = gp.predict(X_star, noisy=False)

    assert mean.shape == (2,)
    assert var.shape == (2,)
    assert np.all(var >= 0.0)


def test_predict_noisy_adds_noise_variance() -> None:
    """noisy=True should add noise variance to predictive variance."""

    X = np.array([[0.0], [1.0]])
    y = np.array([0.0, 1.0])

    gp = GaussianProcessOptimiser(
        kernel=RBFKernel(1.0, 1.0),
        noise_variance=0.2,
    )
    gp.set_data(X, y, tune=False)

    X_star = np.array([[0.5]])
    _, var_clean = gp.predict(X_star, noisy=False)
    _, var_noisy = gp.predict(X_star, noisy=True)

    assert abs(var_noisy[0] - (var_clean[0] + 0.2)) < 1e-8


# ---------------------------------------------------------------------
# Hyperparameter tuning (functional, not brittle)
# ---------------------------------------------------------------------

def test_tuning_does_not_degrade_holdout_rmse() -> None:
    """Hyperparameter tuning should not significantly worsen holdout RMSE."""

    rng = np.random.default_rng(123)

    X = np.linspace(-2.0, 2.0, 80).reshape(-1, 1)
    y = -(X[:, 0] ** 2) + rng.normal(scale=0.1, size=X.shape[0])
    y = y - y.mean()

    idx = rng.permutation(len(X))
    tr, te = idx[:60], idx[60:]

    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[te], y[te]

    # Baseline GP
    gp0 = GaussianProcessOptimiser(
        kernel=RBFKernel(1.0, 0.2),
        noise_variance=0.5,
    )
    gp0.set_data(Xtr, ytr, tune=False)
    mu0, _ = gp0.predict(Xte, noisy=False)
    rmse0 = math.sqrt(np.mean((mu0 - yte) ** 2))

    # Tuned GP
    gp1 = GaussianProcessOptimiser(
        kernel=RBFKernel(1.0, 0.2),
        noise_variance=0.5,
        kernel_tuner=rbf_tuner,
    )
    gp1.set_data(
        Xtr,
        ytr,
        tune=True,
        tuner_options={
            "bounds": ((1e-2, 10.0), (1e-3, 10.0), (1e-4, 1.0)),
            "optimizer_options": {"maxiter": 100},
        },
    )
    mu1, _ = gp1.predict(Xte, noisy=False)
    rmse1 = math.sqrt(np.mean((mu1 - yte) ** 2))

    assert rmse1 <= rmse0 + 1e-6
