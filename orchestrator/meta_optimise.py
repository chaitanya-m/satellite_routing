from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol


class Optimiser(Protocol):
    def ask(self) -> float: ...

    def tell(self, x: float, y: float) -> None: ...


@dataclass
class OptimiserRun:
    asked: list[float] = field(default_factory=list)
    evaluated: set[float] = field(default_factory=set)


def run_with_coverage(
    *,
    optimiser: Optimiser,
    candidates: Iterable[float],
    is_affordable: Callable[[float], bool],
    utility: Callable[[float], float],
    should_stop: Callable[[OptimiserRun], bool],
    max_stagnant_steps: int = 100,
) -> OptimiserRun:
    """
    Meta-search loop:
    - Uses optimiser.ask() proposals by default.
    - Guarantees eventual coverage of affordable candidates by occasionally
      injecting an unseen affordable candidate when the optimiser is stagnant.
    - Terminates only via should_stop().
    """
    run = OptimiserRun()
    candidate_values = list(dict.fromkeys(float(x) for x in candidates))
    affordable = [x for x in candidate_values if is_affordable(x)]
    unseen_affordable = set(affordable)

    stagnant_steps = 0
    while not should_stop(run):
        x = float(optimiser.ask())
        run.asked.append(x)

        x_is_new_affordable = is_affordable(x) and x in unseen_affordable
        if x_is_new_affordable:
            stagnant_steps = 0
        else:
            stagnant_steps += 1

        # If the optimiser keeps proposing already-seen or unaffordable points,
        # inject a new affordable candidate to prevent infinite unproductive looping.
        if unseen_affordable and stagnant_steps >= max_stagnant_steps:
            x = min(unseen_affordable)
            stagnant_steps = 0

        y = utility(x)
        optimiser.tell(x, y)
        run.evaluated.add(x)
        unseen_affordable.discard(x)

    return run

