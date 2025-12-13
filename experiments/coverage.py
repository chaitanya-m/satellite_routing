from __future__ import annotations

from typing import Any
from experiments.base import Experiment


class CoverageObjective(Experiment):
    """Optimise for coverage only."""

    def objective(self, design: Any, metrics: dict[str, float]) -> float:
        return float(metrics["coverage"])
