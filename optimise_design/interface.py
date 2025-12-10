"""Generic design optimisation interfaces.

This module defines a minimal, domain-agnostic interface for optimisation over
design spaces. Concrete problems (satellite dimensioning, Go policy search,
etc.) should live in separate modules and implement :class:`DesignProblem`.

A simple mental model is the 1D quadratic example used in the tests:

* The design space is all real numbers ``x`` in some interval.
* ``sample_design`` draws a candidate ``x`` (e.g. uniformly from [-10, 10]).
* ``evaluate(x)`` returns a score such as ``-(x-3)^2``, which is maximised at
  ``x = 3``.
* A :class:`DesignOptimiser` proposes candidate ``x`` values, observes their
  scores, and keeps track of the best one it has seen.
"""
from __future__ import annotations

from typing import Any, Optional, Protocol, Tuple


class DesignProblem(Protocol):
    """Abstract optimisation problem over a design space.

    A design is intentionally left as ``Any`` here; concrete problems should
    document and enforce the structure they expect.
    """

    def sample_one_design(self) -> Any:
        """Return a single random or heuristic design.

        This may be used directly by simple optimisers (e.g. random search) or
        serve as a building block for more sophisticated proposal strategies.
        """

        ...

    def evaluate(self, design: Any) -> float:
        """Return a scalar score for the given design.

        Higher scores are interpreted as "better" designs by optimisers. A
        common pattern is to use the negative of a loss or cost so that
        maximisation is natural.
        """

        ...


class DesignOptimiser(Protocol):
    """Strategy for exploring a design space."""

    def propose_candidate(self, problem: DesignProblem) -> Any:
        """Return a single candidate design that should be evaluated next.

        Implementations may use their internal state (including any previously
        recorded results) to decide which parts of the design space to explore
        or exploit. The ``problem`` argument is available so optimisers can
        access ``sample_design`` or other helper methods if needed.
        """

        ...

    def record_result(self, design: Any, score: float) -> None:
        """Record the score for a single evaluated design."""

        ...

    def current_best(self) -> Optional[Tuple[Any, float]]:
        """Return the best candidate design seen so far, if any.

        Returns:
            A tuple ``(design, score)`` for the best candidate encountered
            according to the optimiser's internal notion of quality, or
            ``None`` if no candidates have been evaluated yet.
        """

        ...


def run_optimisation(
    problem: DesignProblem,
    optimiser: DesignOptimiser,
    budget: int,
) -> Optional[Tuple[Any, float]]:
    """Simple optimisation loop over a design problem.

    The loop is budgeted purely in terms of the number of design evaluations:

    * ``budget`` is the total number of calls to :meth:`DesignProblem.evaluate`
      that will be performed.

    This helper does not implement any convergence logic itself or any
    parallelism; it simply drives the :class:`DesignOptimiser` for ``budget``
    single-candidate evaluations and then returns ``optimiser.current_best()``.
    """

    if budget <= 0:
        return optimiser.current_best()

    remaining = budget
    while remaining > 0:
        design = optimiser.propose_candidate(problem)
        score = problem.evaluate(design)
        optimiser.record_result(design, score)
        remaining -= 1

    return optimiser.current_best()
