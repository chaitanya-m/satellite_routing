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

import math
import warnings
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

    def set_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tune: bool = True,
        tune_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Set training data and optionally tune hyperparameters.

        Parameters
        ----------
        X:
            Training inputs (n, d).
        y:
            Training targets (n,).
        tune:
            If True (default), call :meth:`tune_hyperparameters` after each 
            data ingestion so later covariance/posterior calls use the latest 
            tuned hyperparameters θ.

            Set to False if you wish to keep existing hyperparameters (e.g., you already
            tuned once and do not want to re-tune right now).

        tune_kwargs:
            Optional keyword arguments forwarded to :meth:`tune_hyperparameters`
            (e.g., bounds, optimizer options).

        Example
        -------
        To run the default L-BFGS-B tuner with bounds (including a noise floor)
        and custom optimiser options:

        >>> gp.set_data(
        ...     X, y,
        ...     tune_kwargs={
        ...         "bounds": (
        ...             (1e-2, 10.0),   # lengthscale ℓ
        ...             (1e-3, 10.0),   # signal variance σ_f²
        ...             (1e-4, 1.0),    # noise variance σ_n² (noise floor via lower bound)
        ...         ),
        ...         "optimizer_options": {"maxiter": 100, "gtol": 1e-6},
        ...     },
        ... )
        """
        first_ingest = self._X is None
        self._X = np.asarray(X, dtype=float)
        self._y = np.asarray(y, dtype=float)

        # Always tune on first ingestion to avoid accidentally running with
        # untuned hyperparameters. If the caller explicitly disables tuning on
        # the first ingestion, warn in case this was unintentional. On subsequent
        # ingestions, caller may skip tuning by setting tune=False.
        if first_ingest and not tune:
            warnings.warn(
                "First data ingestion did not tune GP kernel hyperparameters; proceeding with existing "
                "values without tuning. Set tune=True (default) to optimise them from data.",
                UserWarning,
            )
        if first_ingest and self._X.shape[0] < 20:
            warnings.warn(
                "First data ingestion has fewer than 20 points; hyperparameter tuning may be unstable. "
                "Consider supplying a larger initial batch or setting hyperparameter bounds and noise floor.",
                UserWarning,
            )
        if first_ingest or tune:
            self.tune_hyperparameters(**(tune_kwargs or {}))

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

    def log_marginal_likelihood(self) -> float:
        """Compute log p(y | X, θ) for the current data and hyperparameters.

        This is the likelihood of the observed data where model is the GP specified by a given kernel.
        
        The marginal likelihood is given by:
        p(y | X, θ) = ∫ p(y | f) p(f | X, θ) df
        
        But the latent function f is integrated out, and we are left with a simple closed-form expression:
        log p(y | X, θ) = -0.5 yᵀ (K_θ + σ_n² I)⁻¹  -0.5 log det(K_θ + σ_n² -0.5 n log 2π
 
        Returns:
            The log marginal likelihood value.

        """
        if self._X is None or self._y is None:
            raise ValueError("No observations to evaluate marginal likelihood.")
        K = self._kernel_matrix(self._X, self._X)
        K += self.noise_variance * np.eye(K.shape[0])
        L = np.linalg.cholesky(K + 1e-8 * np.eye(K.shape[0]))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self._y))
        lml = -0.5 * float(self._y.T @ alpha)
        lml -= float(np.sum(np.log(np.diag(L))))
        lml -= 0.5 * self._X.shape[0] * np.log(2 * np.pi)
        return lml

    def tune_hyperparameters(
        self,
        initial: Optional[tuple[float, float, float]] = None,
        bounds: Optional[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = None,
        method: str = "L-BFGS-B",
        optimizer_options: Optional[dict[str, Any]] = None,
    ) -> None:
        """Tune (lengthscale, signal_variance, noise_variance) via maximising log marginal likelihood.

        The likelihood of the data given the model is low if the hyperparameters are poorly chosen.
        This method optimises the hyperparameters by tuning them to maximise the log marginal likelihood.

        This is one of the advantages of Gaussian Processes: we can learn the hyperparameters from data
        rather than having to set them manually. All covariance/prediction utilities then use the
        *current* hyperparameters; you can call this once and keep them fixed, or re-run it later if you
        believe the data distribution has drifted.

        Uses scipy.optimize.minimize with a default L-BFGS-B optimiser. Parameters
        are optimised in log-space to enforce positivity. If scipy is unavailable,
        an ImportError is raised.
        """
        from scipy.optimize import minimize

        if self._X is None or self._y is None:
            raise ValueError("No observations to tune hyperparameters.")

        x0 = initial or (self.lengthscale, self.signal_variance, self.noise_variance)
        log_x0 = np.log(np.array(x0, dtype=float))

        def objective(log_params: np.ndarray) -> float:
            l, s, n = np.exp(log_params)
            self.kernel = rbf_kernel(l)
            self.lengthscale = l
            self.signal_variance = s
            self.noise_variance = n
            return -self.log_marginal_likelihood()

        opt = minimize(
            objective,
            log_x0,
            method=method,
            bounds=[(np.log(b[0]), np.log(b[1])) for b in bounds] if bounds else None,
            options=optimizer_options,
        )
        # Update with optimised values.
        l_opt, s_opt, n_opt = np.exp(opt.x)
        self.kernel = rbf_kernel(l_opt)
        self.lengthscale = float(l_opt)
        self.signal_variance = float(s_opt)
        self.noise_variance = float(n_opt)

    def training_covariance(self) -> np.ndarray:
        """Return K(X, X) + σ_n² I for the current data."""
        if self._X is None:
            raise ValueError("No training inputs available.")
        K = self._kernel_matrix(self._X, self._X)
        return K + self.noise_variance * np.eye(K.shape[0])

    def joint_prior_covariances(
        self, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (K_train_train, K_train_test, K_test_train, K_test_test)."""
        if self._X is None:
            raise ValueError("No training inputs available.")

        # Train/train block with observation noise on the diagonal.
        K_train_train = self._kernel_matrix(self._X, self._X)
        K_train_train = K_train_train + self.noise_variance * np.eye(K_train_train.shape[0])

        # Cross-covariance blocks and test/test block from the tuned kernel.
        K_train_test = self._kernel_matrix(self._X, X_test)
        K_test_train = self._kernel_matrix(X_test, self._X)
        K_test_test = self._kernel_matrix(X_test, X_test)
        return K_train_train, K_train_test, K_test_train, K_test_test

    def compute_posterior(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute GP posterior mean/covariance at X_test given current data.

        Uses the closed-form conditioning formulas with t=train, T=test:
            μ_T = K_Tt (K_tt)^-1 y
            Σ_T = K_TT - K_Tt (K_tt)^-1 K_tT
        where K_tt includes the observation noise term.
        """
        if self._X is None or self._y is None:
            raise ValueError("No observations to condition on.")

        K_tt, K_tT, K_Tt, K_TT = self.joint_prior_covariances(X_test)
        # Cholesky for numerical stability.
        L = np.linalg.cholesky(K_tt + 1e-8 * np.eye(K_tt.shape[0]))
        # α = (K_tt)^-1 y
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self._y))
        mean = K_Tt @ alpha # @ means matrix multiplication
        # v = L^-1 K_tT
        v = np.linalg.solve(L, K_tT)
        cov = K_TT - K_Tt @ np.linalg.solve(L.T, v)
        return mean, cov

    def predict(self, X_test: np.ndarray, noisy: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Predict posterior mean and variance at test inputs.

        Parameters
        ----------
        X_test:
            Array of test inputs (m, d).
        noisy:
            If True, include observation noise σ_n² on the returned variances
            (i.e., predictive variance for y*). If False, return the latent
            function variance only.

        Returns
        -------
        mean:
            Posterior mean vector shaped (m,).
        var:
            Posterior variance vector shaped (m,) corresponding to the diagonal of the
            posterior covariance; adds σ_n² if ``noisy`` is True.
        """
        mean, cov = self.compute_posterior(X_test)
        var = np.diag(cov)
        if noisy:
            var = var + self.noise_variance
        return mean.ravel(), var # ravel() flattens the array to 1D



    def acquire_next(
        self,
        X_candidates: np.ndarray,
        acquisition: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        ucb_beta: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Acquires the next point in the input / design space to evaluate.
        
        Score candidate designs and return the best according to an acquisition.

        
        Parameters
        ----------
        X_candidates:
            Candidate test inputs (m, d) to evaluate.
        acquisition:
            Optional callable taking ``(mean, var)`` arrays and returning an
            acquisition score per candidate. Defaults to UCB.
        ucb_beta:
            Optional override for the UCB exploration weight β; defaults to
            ``self.ucb_beta`` when ``acquisition`` is not provided.

        Returns
        -------
        best_x:
            The candidate (d,) with the highest acquisition score.
        scores:
            Acquisition scores for all candidates (m,).
        """
        mean, var = self.predict(X_candidates, noisy=False)
        if acquisition is None:
            beta = self.ucb_beta if ucb_beta is None else ucb_beta
            scores = mean + np.sqrt(beta) * np.sqrt(var)
        else:
            scores = acquisition(mean, var)
        best_idx = int(np.argmax(scores))
        return X_candidates[best_idx], scores
