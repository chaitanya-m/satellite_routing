from __future__ import annotations
from typing import Any, Dict
from abc import ABC, abstractmethod

from experiments.certificates.base import FeasibilityCertificate


class SingleObjectiveDiscreteExperiment(ABC):
    """
    Domain-agnostic base class for single-objective, discrete experiments
    with probabilistic feasibility guarantees.

    Responsibilities:
    - Track trials and successes per design
    - Apply a statistical feasibility certificate
    - Select the minimum feasible design

    Subclasses define:
    - what constitutes a valid trial
    - what constitutes success
    - the optimisation objective signal
    """

    def __init__(
        self,
        *,
        delta: float,
        certificate: FeasibilityCertificate,
    ):
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

        Default behaviour: count every evaluation as a trial.
        """
        return True

    @abstractmethod
    def is_success(self, metrics: dict[str, float]) -> bool:
        """
        Return True iff a valid trial is considered successful.
        """
        raise NotImplementedError

    @abstractmethod
    def objective(self, design: Any, metrics: dict[str, float]) -> float:
        """
        Return a scalar optimisation signal for the optimiser.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Generic experiment mechanics
    # ------------------------------------------------------------------

    def on_evaluation(self, design: Any, metrics: dict[str, float]) -> None:
        """
        Record the outcome of a single simulation run.
        """
        if not self.is_valid_trial(metrics):
            return

        self._trials[design] = self._trials.get(design, 0) + 1

        if self.is_success(metrics):
            self._successes[design] = self._successes.get(design, 0) + 1
            self._last_success_metrics[design] = metrics

    def is_feasible(self, design: Any) -> bool:
        """
        Check whether a design is feasible under the chosen certificate.
        """
        trials = self._trials.get(design, 0)
        successes = self._successes.get(design, 0)

        lcb = self.certificate.lower_confidence_bound(successes, trials)
        return lcb >= 1.0 - self.delta

    def select_min(self) -> tuple[Any, dict[str, float]]:
        """
        Select the minimum feasible design according to the design ordering.
        """
        feasible = [d for d in self._trials if self.is_feasible(d)]
        if not feasible:
            raise AssertionError("No design certified feasible")

        best = min(feasible)
        return best, self._last_success_metrics[best]
