# experiments/satellites/min_feasible_dimensioning.py

from __future__ import annotations
from typing import Any

from experiments.single_objective_discrete import (
    BernoulliExperiment,
)
from orchestrator.certificates.base import FeasibilityCertificate


class MinFeasibleDimensioning(BernoulliExperiment):
    """
    Dimensioning experiment with multiple stochastic constraints.

    The design parameter is a single scalar (e.g. lambda), and the goal
    is to find the minimum design value such that, with high probability,
    all dimensioning constraints are simultaneously satisfied.

    A trial is considered SUCCESSFUL iff:
      - the trial is valid, and
      - coverage >= min_coverage, and
      - signal_intensity >= min_signal_intensity.

    Feasibility (optional) means:
        P(success | design) >= 1 - delta
    with confidence provided by the injected certificate.

    Notes:
    - This is a single-objective optimisation problem (minimise design).
    - Multiple constraints are handled via a compound success predicate.
    - The optimisation objective is independent of the feasibility check.
    """

    def __init__(
        self,
        *,
        min_coverage: float,
        min_signal_intensity: float,
        delta: float,
        certificate: FeasibilityCertificate,
    ):
        super().__init__(delta=delta, certificate=certificate)
        self.min_coverage = min_coverage
        self.min_signal_intensity = min_signal_intensity

    # ------------------------------------------------------------------
    # Bernoulli semantics (canonical)
    # ------------------------------------------------------------------

    def is_valid_trial(self, metrics: dict[str, float]) -> bool:
        """
        Ignore trivial worlds (e.g. no ground demand).
        """
        return metrics.get("n_ground", 1) != 0

    def accept(self, Z: dict[str, float]) -> bool:
        return (
            Z["coverage"] >= self.min_coverage
            and Z["signal_intensity"] >= self.min_signal_intensity
        )

    def objective(self, design: Any, metrics: dict[str, float]) -> float:
        """
        Smooth optimisation signal for the optimiser.

        This does NOT encode feasibility; it merely guides search.
        """
        # Keep it simple and monotone in lambda
        return float(metrics["coverage"])
