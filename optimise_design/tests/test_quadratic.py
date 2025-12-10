from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import os
import random
import time

from ..interface import DesignProblem, run_optimisation
from ..parallel import run_optimisation_parallel
from ..random_search import RandomSearchOptimiser


@dataclass
class QuadraticProblem(DesignProblem):
    """Minimal example of a DesignProblem for documentation purposes.

    The design space is a single real number ``x``. We define the score as
    ``-(x-3)^2``, which is maximised at ``x = 3``. The optimiser therefore
    tries to propose values of ``x`` that get closer to 3 over time, but in
    this test we use pure random search as a simple baseline.
    """

    rng: random.Random = field(default_factory=random.Random)

    def sample_one_design(self) -> float:
        # Propose a candidate x uniformly from [-10, 10].
        return self.rng.uniform(-10.0, 10.0)

    def evaluate(self, design: Any) -> float:
        # Simulate an expensive evaluation to make parallelism meaningful.
        time.sleep(0.01)
        # Convert whatever we receive into a float and score it.
        x = float(design)
        return -(x - 3.0) ** 2


def test_random_search_quadratic_converges():
    """Random search should find an x reasonably close to 3, sequentially and in parallel.

    We run random search for a finite budget of evaluations and then check that
    the best design (the x with the highest score) lies within a small window
    around the optimum. Because random search samples uniformly from [-10, 10],
    more evaluations increase the chance that we see a good candidate.
    """

    parallel_faster_count = 0
    runs = 10

    for run_idx in range(runs):
        # Use a deterministic seed per run for reproducibility.
        rng = random.Random(run_idx)
        problem = QuadraticProblem(rng=rng)
        # Sequential optimisation
        seq_optimiser = RandomSearchOptimiser()
        t0 = time.perf_counter()
        best_seq = run_optimisation(problem, seq_optimiser, budget=100)
        seq_duration = time.perf_counter() - t0

        # Ensure the sequential run returns a result.
        assert best_seq is not None
        design_seq, _ = best_seq
        # The best x from the sequential run should be reasonably close to 3.
        assert abs(float(design_seq) - 3.0) < 2.0

        # Parallel optimisation.
        par_optimiser = RandomSearchOptimiser()
        t1 = time.perf_counter()
        max_workers = os.cpu_count() or 4
        best_par = run_optimisation_parallel(problem, par_optimiser, budget=80, max_workers=max_workers)
        par_duration = time.perf_counter() - t1

        # Ensure the parallel run also returns a result.
        assert best_par is not None
        design_par, _ = best_par
        # The best x from the parallel run should also be close to 3.
        assert abs(float(design_par) - 3.0) < 2.0

        # Sanity check that both runs completed and durations are positive.
        # Both measurements must be positive to confirm that timing was recorded.
        assert seq_duration > 0.0
        assert par_duration > 0.0

        # Count how many times parallel is strictly faster than sequential.
        if par_duration < seq_duration:
            parallel_faster_count += 1

    # Require that parallel evaluation is faster in 9 out of 10 runs.
    assert parallel_faster_count >= 9
