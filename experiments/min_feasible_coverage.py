# experiments/min_feasible_coverage.py

from __future__ import annotations
from typing import Any, Dict


class MinLambdaForCoverage:
    """
    Find the minimum design value whose empirical success rate
    (coverage >= target_coverage, conditional on n_ground > 0)
    exceeds a required threshold.
    """

    def __init__(
        self,
        target_coverage: float,
        min_success_rate: float,
    ):
        """
        Parameters
        ----------
        target_coverage :
            Coverage threshold defining success for a single simulation.
        min_success_rate :
            Required empirical success probability (Level A criterion).
            Example: 0.9 means "at least 90% of non-trivial runs succeed".
        """
        self.target_coverage = target_coverage
        self.min_success_rate = min_success_rate

        # Per-design counters
        self._trials: Dict[Any, int] = {}
        self._successes: Dict[Any, int] = {}

        # Store last successful metrics for reporting / debugging
        self._last_success_metrics: Dict[Any, dict[str, float]] = {}

    def objective(self, design: Any, metrics: dict[str, float]) -> float:
        """
        Scalar objective used by the optimiser.
        We still guide search using coverage magnitude.
        """
        return float(metrics["coverage"])

    def on_evaluation(self, design: Any, metrics: dict[str, float]) -> None:
        """
        Record one stochastic evaluation of a design.
        Trivial realisations (n_ground == 0) are ignored.
        """
        if metrics["n_ground"] == 0:
            return

        self._trials[design] = self._trials.get(design, 0) + 1

        if metrics["coverage"] >= self.target_coverage:
            self._successes[design] = self._successes.get(design, 0) + 1
            self._last_success_metrics[design] = metrics

    def success_rate(self, design: Any) -> float:
        """
        Empirical success probability for a design.
        """
        trials = self._trials.get(design, 0)
        if trials == 0:
            return 0.0
        return self._successes.get(design, 0) / trials

    def is_feasible(self, design: Any) -> bool:
        """
        Level A feasibility criterion: empirical success rate.
        """
        return self.success_rate(design) >= self.min_success_rate

    def select_min(self) -> tuple[Any, dict[str, float]]:
        """
        Select the minimum design value that is empirically feasible.
        """
        feasible_designs = [
            d for d in self._trials
            if self.is_feasible(d)
        ]

        if not feasible_designs:
            raise AssertionError("No feasible design found")

        best = min(feasible_designs)
        return best, self._last_success_metrics[best]
