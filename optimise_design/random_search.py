"""Toy optimiser: random search over a design problem.

Overview
--------
This optimiser performs pure Monte Carlo search with very simple behaviour:

* At each call to :meth:`propose_candidate`, it samples designs independently
  using :meth:`DesignProblem.sample_one_design`. The samples are i.i.d.; there
  is no adaptation based on previous results.
* It never builds a model of the design space; it simply keeps the best scored
  candidate seen so far via :meth:`candidate_set`.

Usage with ``run_optimisation``
-------------------------------
When used with :func:`optimise_design.interface.run_optimisation`, a typical
configuration is:

* ``budget`` = total number of designs to evaluate (e.g. 200).
* ``batch_size`` = how many designs to evaluate per iteration (e.g. 10).

The resulting search is equivalent to drawing ``budget`` independent samples
from :meth:`DesignProblem.sample_design` and returning the one with the
highest score.

Notes
-----
There is no convergence logic beyond the usual probabilistic effect that more
random samples increase the chance of seeing a good design.

Despite its simplicity, random search is a useful baseline and a convenient
way to exercise the optimisation interfaces before introducing more
sophisticated methods.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

from .interface import DesignOptimiser, DesignProblem


@dataclass
class RandomSearchOptimiser(DesignOptimiser):
    """Baseline optimiser that samples designs independently."""

    _best: Optional[Tuple[Any, float]] = field(default=None, init=False)

    def propose_candidate(self, problem: DesignProblem) -> Any:
        return problem.sample_one_design()

    def record_result(self, design: Any, score: float) -> None:
        if self._best is None or score > self._best[1]:
            self._best = (design, score)

    def current_best(self) -> Optional[Tuple[Any, float]]:
        return self._best

    def supports(self, problem: DesignProblem) -> bool:  # noqa: ARG002
        """Random search can work with any DesignProblem."""

        return True
