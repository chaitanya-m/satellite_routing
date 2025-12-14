# tests/test_risk_orchestration_bounded_exploration.py

import random
import warnings
from collections import Counter

import torch
from botorch.exceptions import InputDataWarning

from experiments.satellites.min_feasible_coverage import MinLambdaForCoverage
from orchestrator.certificates.base import FeasibilityCertificate
from orchestrator.risk import EmpiricalSuccessRate
from orchestrator.meta_optimise import OptimiserRun, run_with_coverage
from sim.dimensioning_2d import Dimensioning_2D
from optim.discrete_bandit import DiscreteBanditOptimiser


class NullCertificate(FeasibilityCertificate):
    def lower_confidence_bound(self, successes: int, trials: int) -> float:
        return 0.0


def test_risk_orchestration_budgeted_optimiser_prefers_cheapest():
    """
    Risk-based orchestration without certificates, with a compute budget.

    The optimiser proposes designs, but the orchestrator:
    - blocks unaffordable designs (no simulation if estimated cost exceeds budget),
    - aggregates empirical risk (success rate under accept(Z)),
    - reports a cost-aware scalar to encourage the cheapest feasible design.
    """

    coverage_distance = 0.102
    coverage_threshold = 1.0
    target_success_rate = 0.95
    min_valid_trials = 100

    # Budget is expressed in expected operations per trial. For this simulator,
    # expected work scales roughly with E[n_ground] * E[n_sats] ~= inner_lambda * lambda.
    max_ops = 5_000
    expected_ground = 5.0

    experiment = MinLambdaForCoverage(
        target_coverage=coverage_threshold,
        delta=0.05,
        certificate=NullCertificate(),
    )

    reward = EmpiricalSuccessRate()

    torch.manual_seed(0)
    candidates = torch.tensor(
        [
            [20.0],
            [10.0],
            [40.0],
            [200.0],
            [400.0],
            [50.0],
            [150.0],
            [600.0],
            [800.0],
            [1000.0],
            [2000.0],
            [10_000_000.0],
        ],
        dtype=torch.double,
    )
    optimiser = DiscreteBanditOptimiser(candidates=candidates)

    affordable_max_lambda = max_ops / expected_ground

    utilities: dict[float, float] = {}
    evaluated: set[float] = set()
    asked: list[float] = []
    invalid_trials: dict[float, int] = {}
    attempted_trials: dict[float, int] = {}

    def estimate_trial_ops(lambda_outer: float) -> int:
        return int(expected_ground * float(lambda_outer))

    def evaluate_to_min_trials(lambda_outer: float) -> None:
        d = float(lambda_outer)

        while reward.trials(d) < min_valid_trials:
            # Design-local RNG tape: the k-th attempted trial for a given design
            # always uses the same seed, regardless of how other designs are explored.
            k = attempted_trials.get(d, 0)
            attempted_trials[d] = k + 1

            # Avoid hashing for reproducibility; use an arithmetic seed schedule.
            # Assumes designs are effectively integral in this test.
            seed = int(d) * 1_000_000 + k
            rng = random.Random(seed)

            sim = Dimensioning_2D(
                inner_lambda=5.0,
                inner_radius=1.0,
                outer_radius=1.1,
                coverage_distance=coverage_distance,
                rng=rng,
            )
            metrics = sim.evaluate(d)
            if not experiment.is_valid_trial(metrics):
                invalid_trials[d] = invalid_trials.get(d, 0) + 1
                continue

            Z = experiment.metric(d, metrics)
            reward.update(d, Z, accepted=experiment.accept(Z))

    def utility(lambda_outer: float) -> float:
        d = float(lambda_outer)
        if d in utilities:
            return utilities[d]

        # Hard resource constraint: never simulate unaffordable designs.
        ops = estimate_trial_ops(d)
        if ops > max_ops:
            utilities[d] = -2.0
            return utilities[d]

        evaluate_to_min_trials(d)

        sr = reward.score(d)
        if sr >= target_success_rate:
            utilities[d] = 1.0 - (d / affordable_max_lambda)
        else:
            utilities[d] = sr - 1.0
        return utilities[d]

    candidate_values = [float(x.item()) for x in candidates.squeeze(1)]

    def is_affordable(d: float) -> bool:
        return estimate_trial_ops(d) <= max_ops

    def should_stop(trace: OptimiserRun) -> bool:
        feasible = [
            d
            for d in trace.evaluated
            if is_affordable(d)
            and reward.trials(d) >= min_valid_trials
            and reward.score(d) >= target_success_rate
        ]
        if not feasible:
            return False

        cheapest_feasible = min(feasible)
        cheaper_affordable = [d for d in candidate_values if is_affordable(d) and d < cheapest_feasible]
        cheaper_exhausted = all(
            reward.trials(d) >= min_valid_trials and reward.score(d) < target_success_rate
            for d in cheaper_affordable
        )
        return cheaper_exhausted

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InputDataWarning)

        trace = run_with_coverage(
            optimiser=optimiser,
            candidates=candidate_values,
            is_affordable=is_affordable,
            utility=utility,
            should_stop=should_stop,
            max_stagnant_steps=50,
        )
        asked.extend(trace.asked)
        evaluated |= trace.evaluated

    print("\n=== Risk-driven, budgeted optimisation ===")
    print(f"max_ops_per_trial={max_ops} | affordable_max_lambda={affordable_max_lambda:.1f}")
    print(f"coverage_distance={coverage_distance} | accept: coverage>={coverage_threshold}")
    print(f"target_success_rate={target_success_rate} | min_valid_trials={min_valid_trials}")
    print("=========================================\n")

    asked_counts = Counter(asked)
    n_blocked = sum(1 for x in asked if estimate_trial_ops(x) > max_ops)
    print(f"asked_total={len(asked)} | unique={len(asked_counts)} | blocked={n_blocked}")
    for x, n in asked_counts.most_common(10):
        status = "BLOCKED" if estimate_trial_ops(x) > max_ops else "OK"
        print(f"asked lambda={x:.1f} | n={n:3d} | {status}")

    print("\n=== Empirical risk per design (evaluated) ===")
    for d in (x for x in evaluated if estimate_trial_ops(x) <= max_ops):
        trials = reward.trials(d)
        invalid = invalid_trials.get(d, 0)
        attempted = trials + invalid
        sr = reward.score(d)
        invalid_rate = (invalid / attempted) if attempted else 0.0
        print(
            f"lambda={d:.1f} | n_valid={trials:4d} | n_invalid={invalid:4d} | "
            f"invalid_rate={invalid_rate:.3f} | success_rate={sr:.3f} | exp_ops={estimate_trial_ops(d)}"
        )
    print("===========================================\n")

    assert reward.trials(10_000_000.0) == 0
    assert n_blocked < len(asked)

    feasible = [
        d
        for d in evaluated
        if reward.trials(d) >= min_valid_trials and reward.score(d) >= target_success_rate
    ]
    assert feasible, "No affordable design met the target risk"

    cheapest_feasible = min(feasible)

    assert optimiser.X is not None and optimiser.Y is not None
    best_idx = int(optimiser.Y.argmax().item())
    best_design = float(optimiser.X[best_idx, 0].item())

    assert best_design == cheapest_feasible
