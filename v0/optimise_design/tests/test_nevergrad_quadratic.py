from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import random

import pytest

from optimise_design.interface import DesignProblem, compatible_optimisers, run_optimisation
from optimise_design.random_search import RandomSearchOptimiser
from optimise_design.strategies.nevergrad_scalar import NevergradScalarOptimiser, ng


@dataclass
class ScalarQuadraticProblem(DesignProblem):
    """Quadratic problem suitable for scalar Nevergrad optimiser."""

    rng: random.Random = field(default_factory=random.Random)

    def sample_one_design(self) -> float:
        return self.rng.uniform(-10.0, 10.0)

    def evaluate(self, design: Any) -> float:
        x = float(design)
        return -(x - 3.0) ** 2


@pytest.mark.skipif(ng is None, reason="nevergrad is not installed")
def test_nevergrad_scalar_compatible_and_solves_quadratic():
    problem = ScalarQuadraticProblem()

    candidates = (
        RandomSearchOptimiser(),
        NevergradScalarOptimiser(),
    )

    compatible = compatible_optimisers(problem, candidates)

    # Both random search and NevergradScalarOptimiser should be compatible with
    # this scalar quadratic problem when nevergrad is available.
    names = {type(opt).__name__ for opt in compatible}
    assert "RandomSearchOptimiser" in names
    assert "NevergradScalarOptimiser" in names

    for opt in compatible:
        best = run_optimisation(problem, opt, budget=100)
        assert best is not None
        design, _score = best
        assert abs(float(design) - 3.0) < 1.0
