from __future__ import annotations

import numpy as np

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
