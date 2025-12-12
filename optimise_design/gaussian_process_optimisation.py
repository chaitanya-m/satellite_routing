"""Gaussian Process Optimisation (readable overview)

Gaussian Process regression (GPR)
---------------------------------
- A Gaussian Process (GP) is a distribution over functions, defined by a mean
  (often 0) and a kernel with hyperparameters (e.g., RBF with lengthscale `ℓ`,
  signal variance `σ_f^2`, noise variance `σ_n^2`). Small `ℓ` favours wigglier
  functions; large `σ_f^2` allows larger amplitude. Noise variance `σ_n^2` is
  the observation noise parameter.
- Prior: f ~ GP(0, k_θ), where θ = (ℓ, σ_f^2, σ_n^2).

Kernel hyperparameters and marginal likelihood
----------------------------------------------
- Tune θ by maximising the GP marginal likelihood (closed-form Gaussian
  integral):
    log p(y | X, θ) = -0.5 yᵀ (K_θ + σ_n² I)⁻¹ y
                      -0.5 log det(K_θ + σ_n² I)
                      -0.5 n log 2π
- This step happens before posterior prediction and trades off fit vs.
  complexity; very small lengthscales or low noise are penalised.

GP posterior
------------
- Kernel matrix: K_θ(X, X) = [k_θ(x_i, x_j)].
- Joint prior over train (X) and test (X*):
    [f(X); f(X*)] ~ N(0, [[K_θ + σ_n² I, K_θ(X, X*)],
                          [K_θ(X*, X),   K_θ(X*, X*)]]).
- Conditioning yields posterior at X*:
    μ*(X*) = K_θ(X*, X) (K_θ + σ_n² I)⁻¹ y
    Σ*(X*) = K_θ(X*, X*) - K_θ(X*, X) (K_θ + σ_n² I)⁻¹ K_θ(X, X*)
  Posterior mean is the best estimate of the latent function; posterior
  covariance is remaining uncertainty.
- Prediction at a single x*:
    f(x*) | D ~ N(μ*(x*), σ*²(x*))
    y* | x*, D ~ N(μ*(x*), σ*²(x*) + σ_n²)

Using the posterior for optimisation (Bayesian optimisation)
------------------------------------------------------------
- Goal: pick x to maximise/minimise an expensive/noisy function.
- The posterior mean μ_t(x) suggests good regions; posterior variance σ_t²(x)
  shows uncertainty.
- An acquisition function a_t(x) trades off exploration/exploitation; choose
  next point x_{t+1} = arg max_x a_t(x). Update with new observation and repeat.

Common acquisition functions
----------------------------
- Upper Confidence Bound (UCB):
    a_UCB(x) = μ_t(x) + sqrt(β_t) * σ_t(x)
    β_t controls exploration (larger → more exploration).
- Expected Improvement (EI):
    a_EI(x) = E[max(0, f(x) - f(x⁺)) | D_t], where x⁺ is best so far. Balances
    high mean and high variance.
- Thompson Sampling (TS):
    Sample a function f_t from the posterior GP; pick x_{t+1} = arg max_x f_t(x).

UCB intuition
-------------
- μ_t(x) term = exploitation; sqrt(β_t)*σ_t(x) term = exploration.
- Picks points near the current upper confidence bound; β_t tunes the balance.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

import numpy as np


def rbf_kernel(lengthscale: float) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return an RBF kernel function with the given lengthscale."""

    def _k(x: np.ndarray, y: np.ndarray) -> float:
        diff = x - y
        sqdist = float(np.dot(diff, diff))
        return math.exp(-0.5 * sqdist / (lengthscale**2))

    return _k


class GaussianProcessOptimiser:
    """Minimal GP-based Bayesian optimiser with a UCB acquisition.

    Parameters
    ----------
    noise_variance:
        Observation noise variance σ_n² used in the GP likelihood.
    ucb_beta:
        UCB exploration weight β; larger values emphasise exploration.
    kernel:
        Positive-definite kernel function k(x, x') (e.g., RBF). Defaults to an
        RBF with the provided lengthscale if not supplied.
    lengthscale:
        Kernel lengthscale ℓ (used by many stationary kernels).
    signal_variance:
        Kernel signal variance σ_f².

    Before observing any data, we assume f ∼ GP(0, k_θ )
    That is, we assume the latent function is drawn from a Gaussian Process with mean 0 and kernel k_θ.
    θ = (lengthscale, signal_variance, noise_variance).    
    """

    def __init__(
        self,
        noise_variance: float,
        ucb_beta: float,
        kernel: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        lengthscale: float = 1.0,
        signal_variance: float = 1.0,
    ) -> None:

        self.noise_variance = noise_variance
        self.ucb_beta = ucb_beta
        self.kernel = kernel or rbf_kernel(lengthscale)
        self.lengthscale = lengthscale
        self.signal_variance = signal_variance

        # Lazy storage for training inputs/outputs; populated when observations are added.
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

    def _kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Build the kernel matrix K(X1, X2) using the provided kernel.

        Args:
            X1: Array of input points (n1, d); often training inputs.
            X2: Array of input points (n2, d); often training or test inputs.

        Typically:
        - For marginal likelihood/posterior: X1 = X2 = training inputs (_X).
        - For predictions: X1 = training inputs, X2 = test/candidate inputs.
        """
        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.empty((n1, n2), dtype=float)
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self.signal_variance * self.kernel(X1[i], X2[j])
        return K
