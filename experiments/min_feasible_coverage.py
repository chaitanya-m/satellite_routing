from __future__ import annotations
from typing import Any


class MinLambdaForCoverage:
    """
    Optimise for the minimum design value that achieves target coverage.
    """

    def __init__(self, target_coverage: float):
        self.target_coverage = target_coverage
        self._feasible: list[tuple[Any, dict[str, float]]] = []

    def objective(self, design: Any, metrics: dict[str, float]) -> float:
        # Optimiser guidance only
        return float(metrics["coverage"])

    def on_evaluation(self, design: Any, metrics: dict[str, float]) -> None:
        if metrics["coverage"] >= self.target_coverage:
            self._feasible.append((design, metrics))

    def select_min(self) -> tuple[Any, dict[str, float]]:
        if not self._feasible:
            raise AssertionError("No feasible design found")
        return min(self._feasible, key=lambda r: r[0])
