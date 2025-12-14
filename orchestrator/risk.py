from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from orchestrator.base import RiskAggregator


@dataclass
class EmpiricalSuccessRate(RiskAggregator):
    _trials: Dict[Any, int] = field(default_factory=dict)
    _successes: Dict[Any, int] = field(default_factory=dict)

    def update(self, design: Any, Z: Any, accepted: bool | None = None) -> None:
        if accepted is None:
            raise ValueError("EmpiricalSuccessRate requires accepted=bool")

        self._trials[design] = self._trials.get(design, 0) + 1
        if accepted:
            self._successes[design] = self._successes.get(design, 0) + 1

    def score(self, design: Any) -> float:
        trials = self._trials.get(design, 0)
        if trials == 0:
            return 0.0
        return self._successes.get(design, 0) / trials

