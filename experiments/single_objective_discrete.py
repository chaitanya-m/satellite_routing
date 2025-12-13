from __future__ import annotations
from typing import Any, Dict
from abc import ABC, abstractmethod

from experiments.certificates.base import FeasibilityCertificate


class SingleObjectiveDiscreteExperiment(ABC):
    """
    Abstract base class for single-objective optimisation over repeated
    stochastic evaluations.

    This class defines a generic experiment interface in which, for a
    fixed design, repeated evaluations induce a Bernoulli process over
    a user-defined success event. Concrete subclasses supply the domain
    semantics, while this base class provides shared experiment mechanics.

    Provided by this base class:
    - accounting of trials and successes per design,
    - optional feasibility checks via an injected statistical certificate,
    - selection of a minimum design under an external ordering.

    Required of subclasses:
    - definition of what constitutes a valid trial,
    - definition of what constitutes success for a single evaluation,
    - definition of a scalar optimisation objective signal.

    Notes:
    - Statistical guarantees are optional and only applied if feasibility
      checks are invoked.
    - No assumptions are made about the evaluation source (simulation,
      online system, or real-world trials), the optimiser, or the domain.
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
