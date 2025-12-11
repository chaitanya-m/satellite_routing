"""Top-level package for design-space optimisation utilities."""

from .interface import (
    DesignOptimiser,
    DesignProblem,
    compatible_optimisers,
    run_optimisation,
)
from .parallel import run_optimisation_parallel
from .random_search import RandomSearchOptimiser
from .bayesian_bandit import BayesianBanditOptimiser

__all__ = [
    "DesignProblem",
    "DesignOptimiser",
    "run_optimisation",
    "run_optimisation_parallel",
    "compatible_optimisers",
    "RandomSearchOptimiser",
    "BayesianBanditOptimiser",
]
