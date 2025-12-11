"""Bayesian linear (contextual) optimiser for vector-valued designs.

This optimiser fits a linear reward model ``y = w^T x + ε`` over feature
vectors and maintains a single Gaussian posterior over the weights ``w``. It
uses Thompson sampling on that weight posterior to choose among candidate
designs: each iteration, it draws one sample ``w̃`` from the posterior and
selects the candidate with the highest sampled score ``w̃^T x``. Because it
shares one weight vector across all designs, it avoids the arm explosion from
one-Beta-per-combination when optimising multivariate configurations.

Objective: maximise the expected score produced by the linear model over
feature vectors. Scores can be any real scalar; the model assumes a Gaussian
noise likelihood around ``w^T x`` and updates a shared weight posterior.

Optimisation loop:

1) On the first iteration, there are no observations. The optimiser samples a
   batch of candidate designs via ``problem.sample_one_design()`` and picks one
   at random to evaluate (since no posterior exists yet).
2) After observations exist, each iteration:
   - draws a batch of candidates,
   - samples weights ``w̃`` from the current Gaussian posterior ``N(μ, Σ)``, which
     encodes all previous (x, y) observations and the prior; this is the
     Thompson-sampling step that turns posterior uncertainty into exploration by
     occasionally drawing weight vectors that emphasise different features when
     the posterior is still wide,
   - scores each candidate as ``w̃^T x`` using ``problem.encode_vector``,
   - picks the highest-scoring candidate.
3) ``record_result`` updates the shared weight posterior with the new (x, y)
   pair.
4) ``current_best`` returns the highest-scoring design seen so far.

This loop repeats until budget is exhausted, continually refining a single
shared model of the reward landscape instead of tracking separate arms.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from .interface import DesignOptimiser, DesignProblem


def _encode(problem: DesignProblem, design: Any) -> np.ndarray:
    """Encode a design into a numeric feature vector via ``encode_vector``."""
    if not hasattr(problem, "encode_vector"):
        raise TypeError("Problem must implement encode_vector(design) for BayesianLinearOptimiser.")
    feats = problem.encode_vector(design)  # type: ignore[attr-defined]
    arr = np.asarray(feats, dtype=float)
    if arr.ndim != 1:
        raise ValueError("encode_vector must return a 1D array-like.")
    return arr


class BayesianLinearOptimiser(DesignOptimiser):
    """Bayesian linear optimiser with Thompson sampling over weight vectors.

    Parameters
    ----------
    prior_precision:
        Scalar λ for the Gaussian weight prior precision. Larger values impose
        stronger shrinkage toward zero weights (more conservative, less risk of
        overfitting), while smaller values allow more flexibility but can
        overfit if data are scarce. Must be provided explicitly; there is no
        default.
    noise_variance:
        Observation noise variance σ² for the linear reward model. Smaller
        values make the optimiser trust observations more (faster learning but
        more sensitivity to noise); larger values slow down updates and yield
        smoother, more cautious learning. Must be set explicitly.
    sample_candidates:
        Number of candidate designs to draw and score per iteration. Higher
        values widen exploration per step but cost more evaluations; lower
        values are cheaper but may miss good regions. Must be provided
        explicitly.
    rng:
        Optional numpy Generator for reproducibility; defaults to
        ``np.random.default_rng()`` if not supplied.

    Note:
    
    The Gaussian posterior itself is learned state, the mean/covariance (μ, Σ), which are 
    inferred from data given the hyperparameter choices of λ (prior_precision) and σ² (noise_variance).
    """

    def __init__(
        self,
        *,
        prior_precision: float,
        noise_variance: float,
        sample_candidates: int,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.prior_precision = prior_precision
        self.noise_variance = noise_variance
        self.sample_candidates = sample_candidates
        self.rng = rng or np.random.default_rng()

        self._A: Optional[np.ndarray] = None  # precision matrix
        self._b: Optional[np.ndarray] = None  # precision-weighted targets
        self._best: Optional[Tuple[Any, float]] = None
        self._problem_ref: Optional[DesignProblem] = None

    def _init_matrices(self, dim: int) -> None:
        self._A = self.prior_precision * np.eye(dim)
        self._b = np.zeros(dim, dtype=float)

    def propose_candidate(self, problem: DesignProblem) -> Any:
        """Sample weights from the posterior and pick the best-scoring candidate."""
        # Keep a reference so record_result can re-encode designs.
        self._problem_ref = problem
        # Draw a small batch of candidates from the problem.
        candidates = [problem.sample_one_design() for _ in range(self.sample_candidates)]

        # If we have no observations yet, fall back to random choice.
        if self._A is None or self._b is None:
            return self.rng.choice(candidates)

        # Compute posterior mean and covariance, then sample weights.
        cov = self.noise_variance * np.linalg.inv(self._A)
        mean = np.linalg.solve(self._A, self._b)
        w_sample = self.rng.multivariate_normal(mean, cov)

        # Score candidates with the sampled weights.
        best_cand = None
        best_score = -np.inf
        for cand in candidates:
            x = _encode(problem, cand)
            score_est = float(np.dot(w_sample, x))
            if score_est > best_score:
                best_score = score_est
                best_cand = cand

        return best_cand

    def record_result(self, design: Any, score: float) -> None:
        """Update the Gaussian posterior with a single observation."""
        if self._problem_ref is None:
            raise RuntimeError("record_result called before any design was proposed.")
        x = _encode(self._problem_ref, design)

        if self._A is None or self._b is None:
            self._init_matrices(len(x))

        assert self._A is not None and self._b is not None
        # Precision update: A += (1/σ^2) x x^T
        outer = np.outer(x, x)
        self._A += outer / self.noise_variance
        # Target update: b += (1/σ^2) x y
        self._b += (x * score) / self.noise_variance

        if self._best is None or score > self._best[1]:
            self._best = (design, score)

    def current_best(self) -> Optional[Tuple[Any, float]]:
        return self._best

    def supports(self, problem: DesignProblem) -> bool:
        """Requires the problem to provide ``encode_vector``."""
        has_encoder = hasattr(problem, "encode_vector")
        if has_encoder:
            # Keep a reference so record_result can re-encode designs.
            self._problem_ref = problem  # type: ignore[attr-defined]
        return has_encoder

    def export_posterior_summary(self) -> Optional[dict]:
        """Export mean/covariance of the weight posterior if available."""
        if self._A is None or self._b is None:
            return None
        mean = np.linalg.solve(self._A, self._b)
        cov = self.noise_variance * np.linalg.inv(self._A)
        return {
            "summary_type": "gaussian_weight_posterior",
            "optimiser_type": "BayesianVectorOptimiser",
            "mean": mean.tolist(),
            "cov": cov.tolist(),
        }
