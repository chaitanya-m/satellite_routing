# experiments/base.py

from __future__ import annotations

from typing import Any


class Experiment:
    """
    Defines how simulator metrics are interpreted for optimisation.
    """

    def objective(self, design: Any, metrics: dict[str, float]) -> float:
        """
        Map simulator metrics to a scalar objective for the optimiser.
        """
        raise NotImplementedError

    def on_evaluation(self, design: Any, metrics: dict[str, float]) -> None:
        """
        Optional hook for recording evaluations or feasibility events.
        """
        pass
