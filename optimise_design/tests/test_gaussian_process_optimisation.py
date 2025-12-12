from __future__ import annotations

import numpy as np

from optimise_design.gaussian_process_optimisation import rbf_kernel


def test_rbf_kernel_computes_expected_value() -> None:
    """RBF kernel should match the exp(-0.5 * ||x-y||^2 / ℓ^2) formula."""

    kernel = rbf_kernel(lengthscale=2.0)
    x = np.array([1.0, 2.0])
    y = np.array([4.0, 6.0])
    diff = x - y
    sqdist = np.dot(diff, diff)
    expected = float(np.exp(-0.5 * sqdist / (2.0**2)))

    assert abs(kernel(x, y) - expected) < 1e-12
