# experiments/multi_objective.py

from __future__ import annotations
from typing import Any, Dict, Tuple
from abc import ABC, abstractmethod

from experiments.certificates.base import FeasibilityCertificate


ObjectiveVector = Tuple[float, ...]


class MultiObjectiveExperiment(ABC):
    """
    Abstract base class for multi-objective optimisation over repeated
    stochastic evaluations.

    For a fixed design, repeated evaluations induce a stochastic process
    over vector-valued objectives. Optionally, a user-defined success
    predicate induces a Bernoulli process that can be analysed with
    statistical certificates.

    This class provides shared experiment mechanics, while concrete
    subclasses define domain-specific semantics.

    Provided by this base class:
    - accounting of trials and successes per design,
    - optional feasibility checks via an injected certificate,
    - selection of a minimum design under an external ordering.

    Required of subclasses:
    - definition of what constitutes a valid trial,
    - definition of the objective vector for a single evaluation,
    - definition of what constitutes success (if used).

    Notes:
    - Objectives are vector-valued and never scalarised here.
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

        Default behaviour: every evaluation is a valid trial.
        """
        return True

    @abstractmethod
    def objective_vector(self, design: Any, metrics: dict[str, float]) -> ObjectiveVector:
        """
        Return the vector-valued objective observed in a single evaluation.
        """
        raise NotImplementedError

    @abstractmethod
    def is_success(self, metrics: dict[str, float]) -> bool:
        """
        Return True iff a valid evaluation is considered a success.

        This predicate is optional in the sense that feasibility checks
        need not be invoked in plain optimisation workflows.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Generic experiment mechanics
    # ------------------------------------------------------------------

    def on_evaluation(self, design: Any, metrics: dict[str, float]) -> None:
        """
        Record the outcome of a single evaluation.
        """
        if not self.is_valid_trial(metrics):
            return

        self._trials[design] = self._trials.get(design, 0) + 1

        if self.is_success(metrics):
            self._successes[design] = self._successes.get(design, 0) + 1
            self._last_success_metrics[design] = metrics

    # ------------------------------------------------------------------
    # Feasibility (optional)
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
    # Selection (optional)
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
