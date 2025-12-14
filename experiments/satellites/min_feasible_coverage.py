# experiments/satellites/min_feasible_coverage.py

from __future__ import annotations
from typing import Any

from experiments.single_objective_discrete import (
    BernoulliExperiment,
)
from experiments.certificates.base import FeasibilityCertificate


class MinLambdaForCoverage(BernoulliExperiment):
    """
    Domain-specific experiment:

    Find the minimum design value whose probability of achieving
    coverage >= target_coverage is at least (1 - delta),
    with confidence provided by the feasibility certificate.
    """

    def __init__(
        self,
        *,
        target_coverage: float,
        delta: float,
        certificate: FeasibilityCertificate,
    ):
        super().__init__(delta=delta, certificate=certificate)
        self.target_coverage = target_coverage

    # ------------------------------------------------------------------
    # Bernoulli semantics (canonical)
    # ------------------------------------------------------------------

    def is_valid_trial(self, metrics: dict[str, float]) -> bool:
        """
        Ignore trivial worlds (no ground points), preserving the original
        experiment semantics.
        """
        return metrics.get("n_ground", 1) != 0

    def accept(self, Z: dict[str, float]) -> bool:
        return float(Z["coverage"]) >= self.target_coverage

    def objective(self, design: Any, metrics: dict[str, float]) -> float:
        """
        Smooth optimisation signal (not the certificate).
        """
        return float(metrics["coverage"])
