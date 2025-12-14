# experiments/single_objective_discrete.py

from __future__ import annotations
from typing import Any, Dict
from abc import ABC, abstractmethod

from experiments.base import Experiment


class BernoulliExperiment(Experiment, ABC):
    """
    Abstract base class for single-objective optimisation over repeated
    stochastic evaluations.

    This class defines a generic experiment interface in which, for a
    fixed design, repeated evaluations induce a Bernoulli process over
    a user-defined success event. Concrete subclasses supply the domain
    semantics, while this base class provides shared experiment mechanics.

    Provided by this base class:
    - accounting of trials and successes per design,

    Required of subclasses:
    - definition of what constitutes a valid trial,
    - definition of what constitutes success for a single evaluation (accept),
    - definition of a scalar optimisation objective signal.

    Notes:
    - Statistical guarantees are optional and only applied if feasibility
      checks are invoked.
    - No assumptions are made about the evaluation source (simulation,
      online system, or real-world trials), the optimiser, or the domain.
    """
    def __init__(
        self,
    ):
        self._trials: Dict[Any, int] = {}
        self._successes: Dict[Any, int] = {}
        self._last_success_metrics: Dict[Any, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Hooks for domain-specific semantics
    # ------------------------------------------------------------------

    def metric(self, design: Any, metrics: dict[str, float]) -> dict[str, float]:
        """
        Canonical per-trial object Z(d, ω).
        For Bernoulli dimensioning experiments, Z is just the raw metrics.
        """
        return metrics

    @abstractmethod
    def accept(self, Z: dict[str, float]) -> bool:
        """
        Per-trial success event. This is the Bernoulli predicate.
        """
        raise NotImplementedError

    def is_valid_trial(self, metrics: dict[str, float]) -> bool:
        """
        Return False to ignore this evaluation entirely (it will not count
        toward trials or successes).

        Default behaviour: count every evaluation as a trial.
        """
        return True

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

        Z = self.metric(design, metrics)
        if self.accept(Z):
            self._successes[design] = self._successes.get(design, 0) + 1
            self._last_success_metrics[design] = metrics
