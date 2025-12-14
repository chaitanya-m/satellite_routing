from __future__ import annotations

from typing import Any


class RiskAggregator:
    def update(self, design: Any, Z: Any, accepted: bool | None = None) -> None:
        """
        Consume one trial outcome.
        accepted is optional and only used by Bernoulli-style aggregators.
        """
        raise NotImplementedError

    def score(self, design: Any) -> float:
        """
        Return scalar risk / reward for optimiser.
        """
        raise NotImplementedError

