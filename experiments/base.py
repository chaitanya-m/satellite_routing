# experiments/base.py

from __future__ import annotations

from typing import Any


class Experiment:
    """
    Defines the semantics of a single evaluation.
    """

    def metric(self, design: Any, metrics: dict[str, Any]) -> Any:
        """
        Canonical per-trial object Z(d, ω).
        Default: raw metrics.
        """
        return metrics

    def accept(self, Z: Any) -> bool:
        """
        Per-trial success predicate.
        Default: always accept.
        """
        return True

    def is_valid_trial(self, metrics: dict[str, Any]) -> bool:
        """
        Trial validity (orthogonal to success).
        """
        return True

    def on_evaluation(self, design: Any, metrics: dict[str, Any]) -> None:
        """
        Optional bookkeeping hook.
        """
        pass
