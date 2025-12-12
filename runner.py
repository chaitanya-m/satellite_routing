
from typing import Any, Optional, Tuple


def run(optimiser, simulator, budget: int) -> Optional[Tuple[Any, float]]:
    best_x: Any = None
    best_y: Optional[float] = None

    for _ in range(budget):
        x = optimiser.ask() # choose design
        y = simulator.evaluate(x) # score design
        optimiser.tell(x, y) # update optimiser

        if best_y is None or y > best_y:
            best_x = x
            best_y = y

    if best_y is None:
        return None
    return best_x, best_y
