from typing import Protocol, Any

class Optimiser(Protocol):
    def ask(self) -> Any:
        """Propose a new design to evaluate."""
        ...

    def tell(self, x: Any, y: float) -> None:
        """Inform the optimiser of the score of a design."""
        ...
