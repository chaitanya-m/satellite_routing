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
    p(y | X, θ) = ∫ p(y | f) p(f | X, θ) df
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

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple, Any

import numpy as np


# ---------------------------------------------------------------------
# Kernel abstraction
# ---------------------------------------------------------------------

class Kernel(Protocol):
    """
    Covariance kernel interface.
    """

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        ...

    def gram(self, X: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True)
class RBFKernel:
    """
    Squared exponential (RBF) kernel.

    __call__(X1, X2) for cross-covariance, gram(X) for same-set covariance:
    Used to build the kernel matrix K(X1, X2) using the provided kernel.

    Args:
        X1: Array of input points (n1, d); often training inputs.
        X2: Array of input points (n2, d); often training or test inputs.

    Typically:
    - For marginal likelihood/posterior: X1 = X2 = training inputs (_X).
    - For predictions: X1 = training inputs, X2 = test/candidate inputs.

    """
    lengthscale: float
    signal_variance: float

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)

        sqdist = (
            np.sum(X1 ** 2, axis=1)[:, None]
            + np.sum(X2 ** 2, axis=1)[None, :]
            - 2.0 * X1 @ X2.T
        )

        return self.signal_variance * np.exp(
            -0.5 * sqdist / (self.lengthscale ** 2)
        )

    def gram(self, X: np.ndarray) -> np.ndarray:
        return self(X, X)


# ---------------------------------------------------------------------
# Kernel tuner protocol
# ---------------------------------------------------------------------

class KernelTuner(Protocol):
    """
    Kernel tuner interface.

    A kernel tuner takes data and a kernel, and returns a new kernel
    instance together with a noise variance.

    """

    def __call__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        kernel: Kernel,
        noise_variance: float,
        **kwargs: Any,
    ) -> tuple[Kernel, float]:
        ...

# ---------------------------------------------------------------------
# Marginal likelihood
# ---------------------------------------------------------------------

def log_marginal_likelihood(
    X: np.ndarray,
    y: np.ndarray,
    kernel: Kernel,
    noise_variance: float,
    jitter: float = 1e-8,
) -> float:
    """
    Compute the Gaussian Process log marginal likelihood.

    Compute log p(y | X, θ) for the current data and hyperparameters.

    This is the likelihood of the observed data where model is the GP specified by a given kernel.
    
    The marginal likelihood is given by:
    p(y | X, θ) = ∫ p(y | f) p(f | X, θ) df
    
    But the latent function f is integrated out, and we are left with a simple closed-form expression:
    log p(y | X, θ) = -0.5 yᵀ (K_θ + σ_n² I)⁻¹  -0.5 log det(K_θ + σ_n² -0.5 n log 2π

    Returns:
        The log marginal likelihood value.

    No defaults as this is a pure function and not a default callable for a class.

    """
    n = X.shape[0]

    K = kernel.gram(X)
    K = K + (noise_variance + jitter) * np.eye(n)

    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))

    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * y @ alpha - 0.5 * log_det - 0.5 * n * np.log(2.0 * np.pi)



# ---------------------------------------------------------------------
# RBF kernel tuner
# ---------------------------------------------------------------------

def rbf_tuner(
    X: np.ndarray,
    y: np.ndarray,
    kernel: Kernel,
    noise_variance: float,
    *,
    bounds: Optional[Tuple[Tuple[float, float], ...]] = None,
    method: str = "L-BFGS-B",
    optimizer_options: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> tuple[Kernel, float]:
    """
    Tune RBF kernel hyperparameters via marginal likelihood maximisation.

    Tune (lengthscale, signal_variance, noise_variance) via maximising log marginal likelihood.

    The likelihood of the data given the model is low if the hyperparameters are poorly chosen.
    This method optimises the hyperparameters by tuning them to maximise the log marginal likelihood.

    This is one of the advantages of Gaussian Processes: we can learn the hyperparameters from data
    rather than having to set them manually. All covariance/prediction utilities then use the
    *current* hyperparameters; you can call this once and keep them fixed, or re-run it later if you
    believe the data distribution has drifted.

    Uses scipy.optimize.minimize with a default L-BFGS-B optimiser. Parameters
    are optimised in log-space to enforce positivity.

    """
    from scipy.optimize import minimize

    if not isinstance(kernel, RBFKernel):
        raise TypeError("rbf_tuner requires an RBFKernel.")

    x0 = np.log(
        np.array(
            [
                kernel.lengthscale,
                kernel.signal_variance,
                noise_variance,
            ],
            dtype=float,
        )
    )

    # --- validate bounds (once, before optimisation)
    if bounds is not None:
        for lo, hi in bounds:
            if lo <= 0.0 or hi <= 0.0 or lo >= hi:
                raise ValueError(
                    "Bounds must satisfy 0 < lower < upper for all hyperparameters."
                )
        log_bounds = [(np.log(lo), np.log(hi)) for lo, hi in bounds]
    else:
        log_bounds = None


    def objective(log_params: np.ndarray) -> float:
        l, s, n = np.exp(log_params)
        return -log_marginal_likelihood(X, y, RBFKernel(l, s), n)


    opt = minimize(
        objective,
        x0,
        method=method,
        bounds=log_bounds,
        options=optimizer_options,
    )

    # --- handle optimiser failure explicitly
    if not opt.success:
        # Conservative fallback: keep previous hyperparameters
        return kernel, float(noise_variance)

    l_opt, s_opt, n_opt = np.exp(opt.x)
    return RBFKernel(l_opt, s_opt), float(n_opt)




# ---------------------------------------------------------------------
# Gaussian Process regressor
# ---------------------------------------------------------------------

class GaussianProcessOptimiser:
    """
    Gaussian Process regressor with optional hyperparameter tuning.

    Minimal GP-based Bayesian optimiser with a UCB acquisition.

    Parameters
    ----------
    noise_variance:
        Observation noise variance σ_n² used in the GP likelihood.
    kernel:
        Positive-definite kernel function k(x, x') (e.g., RBF). If
        not provided, a default RBF kernel is constructed.
    acquisition_fn:
        Acquisition callable taking (mean, var) arrays and returning
        scores per candidate. Required if you do not want to use UCB default with its default parameters.

    Before observing any data, we assume f ∼ GP(0, k_θ )
    That is, we assume the latent function is drawn from a Gaussian Process with mean 0 and kernel k_θ.
    For the default RBF kernel θ = (lengthscale, signal_variance, noise_variance) where the first two belong to the 
    kernel and the last is the GP noise term. All three are tuned from data.

    """

    def __init__(
        self,
        kernel: Kernel,
        noise_variance: float,
        kernel_tuner: Optional[KernelTuner] = None,
    ):
        self.kernel = kernel
        self.noise_variance = noise_variance
        self.kernel_tuner = kernel_tuner

        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

    def set_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tune: bool = True,
        tuner_options: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Set training data and optionally tune hyperparameters.

        Parameters
        ----------
        X:
            Training inputs (n, d).
        y:
            Training targets (n,).
        tune:
            If True (default), tune kernel hyperparameters using the provided
            kernel tuner.

        tuner_options:
            Optional dictionary of keyword arguments to pass to the kernel
            tuner.

        """
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)

        self._X = X
        self._y = y

        if tune:
            if self.kernel_tuner is None:
                raise RuntimeError("No kernel tuner provided.")

            kernel, noise = self.kernel_tuner(
                X,
                y,
                self.kernel,
                self.noise_variance,
                **(tuner_options or {}),
            )
            self.kernel = kernel
            self.noise_variance = noise

    def predict(
        self,
        X_star: np.ndarray,
        noisy: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict posterior mean and marginal variance.
        """
        X_star = np.atleast_2d(X_star)

        if self._X is None or self._y is None:
            raise ValueError("No training data set.")

        X, y = self._X, self._y

        K = self.kernel.gram(X)
        K += (self.noise_variance + 1e-8) * np.eye(len(X)) # jitter for stability
        L = np.linalg.cholesky(K)

        K_star = self.kernel(X, X_star)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))

        mean = K_star.T @ alpha

        v = np.linalg.solve(L, K_star)
        var = self.kernel.gram(X_star) - v.T @ v # may produce small negative values on diagonal due to float precision issues

        if noisy:
            var += self.noise_variance * np.eye(len(X_star))

        diag = np.diag(var)
        diag = np.maximum(diag, 0.0) # ensure non-negative variance produced due to float precision issues zeroed
        return mean.reshape(-1), diag

