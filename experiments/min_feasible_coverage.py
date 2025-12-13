# experimental/min_feasible_coverage.py

from __future__ import annotations
from typing import Any, Dict
import math


class MinLambdaForCoverage:
    """
    Find the minimum design value whose probability of achieving
    coverage >= target_coverage is at least (1 - delta),
    with confidence (1 - alpha).
    """

    def __init__(
        self,
        target_coverage: float,
        delta: float,
        alpha: float,
    ):
        self.target_coverage = target_coverage
        self.delta = delta
        self.alpha = alpha

        self._trials: Dict[Any, int] = {}
        self._successes: Dict[Any, int] = {}
        self._last_success_metrics: Dict[Any, dict[str, float]] = {}

    def objective(self, design: Any, metrics: dict[str, float]) -> float:
        # Smooth signal to guide optimiser
        return float(metrics["coverage"])

    def on_evaluation(self, design: Any, metrics: dict[str, float]) -> None:
        # Ignore trivial worlds
        if metrics["n_ground"] == 0:
            return

        self._trials[design] = self._trials.get(design, 0) + 1

        if metrics["coverage"] >= self.target_coverage:
            self._successes[design] = self._successes.get(design, 0) + 1
            self._last_success_metrics[design] = metrics

    def _lower_confidence_bound(self, successes: int, trials: int) -> float:
        """
        Clopper–Pearson lower bound.
        Only the S = n case is implemented explicitly.
        """
        if trials == 0:
            return 0.0

        if successes == trials:
            # exact closed form
            return self.alpha ** (1.0 / trials)

        # Conservative fallback: not certified
        return 0.0

    def is_feasible(self, design: Any) -> bool:
        trials = self._trials.get(design, 0)
        successes = self._successes.get(design, 0)

        lcb = self._lower_confidence_bound(successes, trials)
        return lcb >= 1.0 - self.delta

    def select_min(self) -> tuple[Any, dict[str, float]]:
        feasible = [
            d for d in self._trials
            if self.is_feasible(d)
        ]

        if not feasible:
            raise AssertionError("No design certified feasible")

        best = min(feasible)
        return best, self._last_success_metrics[best]
