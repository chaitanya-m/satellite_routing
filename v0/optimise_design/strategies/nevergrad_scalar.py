"""Adapter for Nevergrad scalar optimisers.

This module provides a thin :class:`DesignOptimiser` implementation that wraps
Nevergrad's scalar optimisation for problems where designs can be represented
as a single floating point value. The dependency on Nevergrad is optional; if
it is not installed, this optimiser will report ``supports() == False`` for
all problems.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

from optimise_design.interface import DesignOptimiser, DesignProblem

try:  # pragma: no cover - import guard
    import nevergrad as ng  # type: ignore[import]
except Exception:  # pragma: no cover - import guard
    ng = None  # type: ignore[assignment]


@dataclass
class NevergradScalarOptimiser(DesignOptimiser):
    """Nevergrad-based optimiser for scalar design problems.

    This adapter assumes that the problem's designs can be coerced to and from
    a single float. It uses Nevergrad's Recommendation API to propose
    candidates and record their evaluation results.
    """

    lower_bound: float = -10.0
    upper_bound: float = 10.0

    _optimizer: Any = field(default=None, init=False)
    _best: Optional[Tuple[Any, float]] = field(default=None, init=False)

    def _ensure_optimizer(self) -> None:
        if self._optimizer is not None or ng is None:
            return
        instrumentation = ng.p.Scalar(lower=self.lower_bound, upper=self.upper_bound)
        self._optimizer = ng.optimizers.OnePlusOne(parametrization=instrumentation, budget=None)  # type: ignore[assignment]

    def propose_candidate(self, problem: DesignProblem) -> Any:
        self._ensure_optimizer()
        if self._optimizer is None:
            # Fallback if Nevergrad is not available.
            return problem.sample_one_design()
        recommendation = self._optimizer.ask()
        # recommendation.value is a float in [lower_bound, upper_bound].
        return float(recommendation.value)

    def record_result(self, design: Any, score: float) -> None:
        if self._optimizer is not None:
            try:
                # Nevergrad minimises by default, so use negative score.
                self._optimizer.tell(design, -score)
            except Exception:
                # If something goes wrong, fall back to tracking best only.
                pass

        if self._best is None or score > self._best[1]:
            self._best = (design, score)

    def current_best(self) -> Optional[Tuple[Any, float]]:
        return self._best

    def supports(self, problem: DesignProblem) -> bool:
        """Support scalar float designs when Nevergrad is installed."""

        if ng is None:
            return False
        # We assume problems that return a numeric scalar from sample_one_design
        # are acceptable. We do not attempt deep type checking here.
        try:
            d = problem.sample_one_design()
            float(d)
        except Exception:
            return False
        return True

