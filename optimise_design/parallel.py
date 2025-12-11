"""Parallel utilities for design optimisation.

These utilities build on the core :mod:`optimise_design.interface` abstractions
without changing them. Parallelism is treated as an implementation detail:
multiple candidate designs are evaluated concurrently, but optimisers still see
one ``propose_candidate`` / ``record_result`` pair at a time.
"""
from __future__ import annotations

from concurrent.futures import Executor, Future, ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import Any, Dict, Optional, Tuple

from .interface import DesignOptimiser, DesignProblem


def run_optimisation_parallel(
    problem: DesignProblem,
    optimiser: DesignOptimiser,
    budget: int,
    max_workers: int = 4,
    executor: Optional[Executor] = None,
) -> Optional[Tuple[Any, float]]:
    """Parallel optimisation loop over a design problem.

    Parameters
    ----------
    problem:
        The design problem to optimise.
    optimiser:
        Strategy that proposes candidates and records results.
    budget:
        Total number of design evaluations (calls to ``problem.evaluate``)
        that will be performed across all workers.
    max_workers:
        Upper bound on the number of evaluations to run concurrently when an
        internal :class:`ThreadPoolExecutor` is used.
    executor:
        Optional external :class:`concurrent.futures.Executor`. When provided,
        ``max_workers`` is ignored and the caller is responsible for sizing the
        executor. When ``None``, a :class:`ThreadPoolExecutor` is created and
        managed for the duration of the call.

    Notes
    -----
    This utility does not implement any convergence logic itself; it simply
    drives the optimiser until the evaluation budget is exhausted while keeping
    up to ``max_workers`` evaluations in flight. The optimiser remains unaware
    of the parallelism and processes one ``(design, score)`` pair at a time via
    :meth:`DesignOptimiser.record_result`.
    """

    if budget <= 0:
        return optimiser.current_best()

    # Choose an executor context: use the provided one or create a local pool.
    if executor is None:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return _run_parallel_with_executor(problem, optimiser, budget, pool)
    return _run_parallel_with_executor(problem, optimiser, budget, executor)


def _run_parallel_with_executor(
    problem: DesignProblem,
    optimiser: DesignOptimiser,
    budget: int,
    executor: Executor,
) -> Optional[Tuple[Any, float]]:
    """Internal utility that assumes an executor is already available."""

    submitted = 0
    in_flight: Dict[Future, Any] = {}

    # Prime the pipeline: submit up to the executor capacity or budget.
    max_parallel = getattr(executor, "_max_workers", None) or budget
    while submitted < budget and len(in_flight) < max_parallel:
        design = optimiser.propose_candidate(problem)
        future = executor.submit(problem.evaluate, design)
        in_flight[future] = design
        submitted += 1

    # Process completions and keep submitting until budget is exhausted.
    while in_flight:
        done, _pending = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
        for future in done:
            design = in_flight.pop(future)
            score = future.result()
            optimiser.record_result(design, score)

            # Submit a replacement job if we still have budget.
            if submitted < budget:
                new_design = optimiser.propose_candidate(problem)
                new_future = executor.submit(problem.evaluate, new_design)
                in_flight[new_future] = new_design
                submitted += 1

    return optimiser.current_best()
