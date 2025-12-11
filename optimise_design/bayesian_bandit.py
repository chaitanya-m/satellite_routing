"""Thompson sampling bandit optimiser built on a Beta-Bernoulli model.

This optimiser treats each distinct design as an arm. Scores in
``record_result`` are assumed to lie in ``[0, 1]``; they are interpreted as
fractional successes for a Beta posterior (e.g., coverage, conversion rate,
success probability).

Objective: maximise the expected score in ``[0, 1]``. Each recorded score is
treated as a (possibly fractional) success for that arm's Beta posterior, and
the optimiser seeks the arm with the highest posterior mean.

Optimisation loop:

1) Call :func:`run_optimisation`; on the first iteration there are no arms, so
   the optimiser samples a design from ``problem.sample_one_design``, creates
   an arm for it, and records its score as a Beta update. If
   ``min_observations_before_exploit`` > 1, it may sample a few more designs
   before switching to pure Thompson sampling.
2) As soon as at least one arm exists, each iteration does Thompson sampling:
   - draw one sample from each arm's Beta posterior,
   - pick the arm with the highest sample (exploit high-mean arms, explore
     uncertain arms with wide posteriors),
   - occasionally explore a brand-new design if allowed.
3) ``record_result`` updates only the chosen arm's posterior (alpha/beta) with
   the new score.
4) ``current_best`` returns the arm with the highest posterior mean so far.

This loop repeats until your budget is exhausted: the bandit keeps a compact
posterior per arm, proposes a design each step, and tracks the best design it
has seen according to the accumulated Beta posteriors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import random

from optimise_design.interface import DesignOptimiser, DesignProblem


@dataclass
class ArmState:
    """Posterior state for a single arm."""

    alpha: float
    beta: float
    n: int = 0
    sum_scores: float = 0.0

    @property
    def posterior_mean(self) -> float:
        denom = self.alpha + self.beta
        if denom <= 0:
            return 0.0
        return self.alpha / denom


class BayesianBanditOptimiser(DesignOptimiser):
    """Beta-Bernoulli Thompson sampling optimiser."""

    def __init__(
        self,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        min_observations_before_exploit: int = 1,
        max_arms: Optional[int] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.min_observations_before_exploit = min_observations_before_exploit
        self.max_arms = max_arms
        self.rng = rng or random.Random()
        self._arms: Dict[Any, ArmState] = {}
        self._designs: Dict[Any, Any] = {}
        self._total_observations = 0

    def supports(self, problem: DesignProblem) -> bool:  # noqa: ARG002
        """Bandit works with any problem where designs are hashable or keyable."""

        return True

    def _design_key(self, problem: DesignProblem, design: Any) -> Any:
        """Return a stable key for a design, using problem.design_key if available."""
        key_fn = getattr(problem, "design_key", None)
        if callable(key_fn):
            return key_fn(design)
        return repr(design)

    def propose_candidate(self, problem: DesignProblem) -> Any:
        """Select the next design to evaluate via Thompson sampling."""
        # If we have no observations yet, or no arms, explore.
        if self._total_observations < self.min_observations_before_exploit or not self._arms:
            design = problem.sample_one_design()
            key = self._design_key(problem, design)
            self._designs[key] = design
            return design

        # If we are allowed more arms, occasionally explore a new design.
        if self.max_arms is None or len(self._arms) < self.max_arms:
            # 10% chance to explore a new arm even after some observations.
            if self.rng.random() < 0.1:
                design = problem.sample_one_design()
                key = self._design_key(problem, design)
                self._designs[key] = design
                return design

        # Thompson sampling over known arms.
        best_key = None
        best_sample = float("-inf")
        for key, state in self._arms.items():
            sample = self.rng.betavariate(state.alpha, state.beta)
            if sample > best_sample:
                best_sample = sample
                best_key = key

        if best_key is None:
            design = problem.sample_one_design()
            key = self._design_key(problem, design)
            self._designs[key] = design
            return design

        return self._designs[best_key]

    def record_result(self, design: Any, score: float) -> None:
        """Update the posterior for the arm corresponding to ``design``."""
        key = self._design_key_from_any(design)
        if key not in self._arms:
            self._arms[key] = ArmState(alpha=self.alpha0, beta=self.beta0)
            self._designs[key] = design

        state = self._arms[key]
        # Treat score in [0,1] as fractional success.
        state.alpha += score
        state.beta += max(0.0, 1.0 - score)
        state.n += 1
        state.sum_scores += score
        self._total_observations += 1

    def current_best(self) -> Optional[Tuple[Any, float]]:
        """Return the design with the highest posterior mean, if any."""
        if not self._arms:
            return None
        best_key = max(self._arms.items(), key=lambda item: item[1].posterior_mean)[0]
        state = self._arms[best_key]
        return self._designs[best_key], state.posterior_mean

    def export_posterior_summary(self) -> Dict[str, Any]:
        """Return a serialisable summary of the current arm posteriors."""
        return {
            # summary_type is a small, stable contract consumed by the meta layer.
            "summary_type": "beta_posterior",
            # optimiser_type is retained for human readability/debugging.
            "optimiser_type": "BayesianBanditOptimiser",
            "arms": [
                {
                    "design_key": key,
                    "alpha": state.alpha,
                    "beta": state.beta,
                    "n": state.n,
                    "sum_scores": state.sum_scores,
                }
                for key, state in self._arms.items()
            ],
        }

    def _design_key_from_any(self, design: Any) -> Any:
        """Infer a key for ``design`` by matching stored designs or falling back to repr."""
        # If design already corresponds to a known key, preserve it; else repr.
        for key, stored in self._designs.items():
            if design is stored or design == stored:
                return key
        return repr(design)
