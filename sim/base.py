from typing import Protocol, Any

class Simulator(Protocol):
    def evaluate(self, x: Any) -> float:
        """Return a scalar score for design x."""
        ...
