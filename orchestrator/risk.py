from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from orchestrator.base import RiskAggregator


@dataclass
class EmpiricalSuccessRate(RiskAggregator):
    """
    Bernoulli risk aggregator: empirical success rate per design.

    score(d) = successes[d] / trials[d]
    """

    _trials: Dict[Any, int] = field(default_factory=dict)
    _successes: Dict[Any, int] = field(default_factory=dict)

    def update(self, design: Any, Z: Any, accepted: bool | None = None) -> None:
        if accepted is None:
            raise ValueError("EmpiricalSuccessRate requires accepted=bool")

        self._trials[design] = self._trials.get(design, 0) + 1
        if accepted:
            self._successes[design] = self._successes.get(design, 0) + 1

    def trials(self, design: Any) -> int:
        return self._trials.get(design, 0)

    def successes(self, design: Any) -> int:
        return self._successes.get(design, 0)

    def score(self, design: Any) -> float:
        trials = self.trials(design)
        if trials == 0:
            return 0.0
        return self.successes(design) / trials

    def should_stop(self, design: Any, *, min_trials: int, target: float) -> bool:
        return self.trials(design) >= min_trials and self.score(design) >= target
