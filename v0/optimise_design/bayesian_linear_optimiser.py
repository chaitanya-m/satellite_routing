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
   batch of candidate designs (``sample_candidates`` many; e.g., 10) via
   ``problem.sample_one_design()`` and picks one at random to evaluate (since
   no posterior exists yet).
2) After observations exist, each iteration:
   - draws a batch of candidates (e.g., 10 designs),
   - samples weights ``w̃`` from the current Gaussian posterior ``N(μ, Σ)``, which
     encodes all previous (x, y) observations and the prior; this is the
     Thompson-sampling step that turns posterior uncertainty into exploration by
     occasionally drawing weight vectors that emphasise different features when
     the posterior is still wide,
   - scores each candidate as ``w̃^T x`` using ``problem.encode_vector``,
   - picks the highest-scoring candidate from that batch and **runs a real
     simulation** via ``evaluate`` only for that single design (the other batch
     candidates are just scored cheaply).
3) ``record_result`` updates the shared weight posterior with the new (x, y)
   pair.
4) ``current_best`` returns the highest-scoring design seen so far.

This loop repeats until budget is exhausted, continually refining a single
shared model of the reward landscape instead of updating separate arms as in the bayesian discrete bandit.
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
    
        Scalar λ for the Gaussian weight prior precision 
        
        In a Gaussian prior, the precision matrix is just the inverse of the covariance matrix.
        Using precision is standard in Bayesian linear regression because the math (posterior updates) 
        is cleaner in terms of precision. 

        The covariance matrix here is over the weights.
        
        Thus, it describes your uncertainty about the weights: its diagonal entries are the 
        variances of each weight (how far they’re expected to stray from the mean), and the off‑diagonal 
        entries are covariances (how weight deviations co‑move). 
        
        A “tight” prior (small variances - the diagonal terms) means weights are expected to stay near the mean; 
        a “loose” prior (large variances) means weights can wander more.

        A larger precision (i.e., smaller covariance) means you assume weights are tightly 
        concentrated near the prior mean (here a zero-mean prior).

        Our prior covariance is λ^{-1}I, an isotropic Gaussian where all weights are independent and
        identically distributed. That is, we assume we have no prior knowledge favouring any interactions 
        between features. This means off-diagonal entries in the covariance (which would encode correlations 
        between weights) are zero. The precision matrix, being the inverse of covariance, is thus λI.
        This matrix is diagonal with all diagonal entries equal to λ, reflecting the isotropic assumption.
        
        A larger prior precision λ means a tighter prior around zero weights. That is, we are more
        confident that weights should be close to zero before seeing any data, meaning we expect the features to 
        be equally uninformative. This is a form of regularization that prevents overfitting to small datasets.

        Conversely, a smaller prior precision λ (larger covariance) means a looser prior, allowing weights to 
        vary more freely based on observed data, potentially capturing more complex relationships but risking overfitting.

        The prior precision λ is a hyperparameter that reflects our belief about the relative informativeness of the features
        before seeing any data.

        For example, with λ=10 the posterior will move slowly—several consistent observations are needed to shift weights
        meaningfully—whereas with λ=0.1 the posterior will adapt quickly but can overreact to a few noisy points. 
        
        Must be provided explicitly; there is no default.

        Note: regardless of whether we use a full covariance matrix with non-zero off-diagonals... This optimiser is 
        linear because the model is restricted to functions of the form f(x) = w^T x (it’s only as rich as the 
        features supplied)

    noise_variance:
        Observation noise variance σ² for the linear reward model. 
        
        Smaller values make the optimiser trust observations more (faster learning but
        more sensitivity to noise); larger values slow down updates and yield
        smoother, more cautious learning. 
        
        σ² controls how much each new
        observation updates the posterior; low σ² can lead to overreacting to
        noise, high σ² makes the posterior move slowly. Must be set explicitly.

    sample_candidates:
        Number of candidate designs to draw and score per iteration. Higher
        values widen exploration per step but cost more evaluations; lower
        values are cheaper but may miss good regions. Must be provided
        explicitly. 
        
        With small batches, the posterior gets updated after seeing only a
        narrow, potentially unrepresentative subset of designs. That makes it less 
        likely to sample underexplored regions; the posterior “settles” too quickly.
        Because it is the posterior that encodes all previous observations and thus 
        the current understanding, this means the balance between exploration and 
        exploitation shifts towards exploitation.
        
        Larger batches let you score a wider variety of vector designs before each
        update, supporting better-informed exploration - promising designs may be found
        in unexplored regions of the design space before the posterior is updated.

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
