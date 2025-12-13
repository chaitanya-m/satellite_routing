from __future__ import annotations
from typing import Any, Dict, Tuple

from experiments.certificates.base import FeasibilityCertificate


ObjectiveVector = Tuple[float, ...]


class MinFeasibleMultiObjective:
    """
    Domain- and simulation-agnostic experiment for multi-objective
    probabilistic feasibility.

    A design is considered SUCCESSFUL in a trial iff:
      - the trial is valid, and
      - all objective values meet their respective thresholds.

    Feasibility means:
        P(success | design) >= 1 - delta
    with confidence provided by the injected FeasibilityCertificate.

    Notes:
    - Objectives are vector-valued and never scalarised here.
    - The optimiser/orchestrator may consume the objective vector directly.
    - "Minimum" refers only to ordering over designs, not objectives.
    """

    def __init__(
        self,
        *,
        objective_keys: Tuple[str, ...],
        objective_thresholds: Tuple[float, ...],
        delta: float,
        certificate: FeasibilityCertificate,
    ):
        if len(objective_keys) != len(objective_thresholds):
            raise ValueError(
                "objective_keys and objective_thresholds must have the same length"
            )

        self.objective_keys = objective_keys
        self.objective_thresholds = objective_thresholds
        self.delta = delta
        self.certificate = certificate

        self._trials: Dict[Any, int] = {}
        self._successes: Dict[Any, int] = {}
        self._last_success_metrics: Dict[Any, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Hooks for domain-specific semantics
    # ------------------------------------------------------------------

    def is_valid_trial(self, metrics: dict[str, float]) -> bool:
        """
        Return False to ignore this evaluation entirely (it will not count
        toward trials or successes).

        Default behaviour: every evaluation is a valid trial.
        """
        return True

    # ------------------------------------------------------------------
    # Objective observation (vector-valued, no scalarisation)
    # ------------------------------------------------------------------

    def objective_vector(self, design: Any, metrics: dict[str, float]) -> ObjectiveVector:
        """
        Return the raw objective vector associated with a single evaluation.
        """
        return tuple(float(metrics[k]) for k in self.objective_keys)

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def on_evaluation(self, design: Any, metrics: dict[str, float]) -> None:
        """
        Record the outcome of a single simulation run.
        """
        if not self.is_valid_trial(metrics):
            return

        self._trials[design] = self._trials.get(design, 0) + 1

        success = True
        for key, threshold in zip(self.objective_keys, self.objective_thresholds):
            if float(metrics[key]) < threshold:
                success = False
                break

        if success:
            self._successes[design] = self._successes.get(design, 0) + 1
            self._last_success_metrics[design] = metrics

    # ------------------------------------------------------------------
    # Feasibility
    # ------------------------------------------------------------------

    def is_feasible(self, design: Any) -> bool:
        """
        Check whether a design is feasible under the chosen certificate.
        """
        trials = self._trials.get(design, 0)
        successes = self._successes.get(design, 0)

        lcb = self.certificate.lower_confidence_bound(successes, trials)
        return lcb >= 1.0 - self.delta

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_min(self) -> tuple[Any, dict[str, float]]:
        """
        Select the minimum feasible design according to the design ordering.
        """
        feasible = [d for d in self._trials if self.is_feasible(d)]
        if not feasible:
            raise AssertionError("No design certified feasible")

        best = min(feasible)
        return best, self._last_success_metrics[best]
