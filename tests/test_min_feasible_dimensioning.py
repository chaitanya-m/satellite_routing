# tests/test_min_feasible_dimensioning.py

import random
import torch
import warnings

from botorch.exceptions import InputDataWarning

from experiments.satellites.min_feasible_dimensioning import MinFeasibleDimensioning
from experiments.certificates.bernoulli import HoeffdingCertificate
from sim.dimensioning_2d import Dimensioning_2D  # assumed to exist
from optim.discrete_bandit import DiscreteBanditOptimiser


MIN_COVERAGE = 0.95
MIN_SIGNAL_INTENSITY = 0.001

DELTA = 0.05
ALPHA = 0.05


def test_min_feasible_dimensioning_with_signal_constraint():
    """
    Single-objective dimensioning test with multiple stochastic constraints.

    Objective:
      - minimise lambda (design parameter)

    Constraints (per trial):
      - coverage >= MIN_COVERAGE
      - signal_intensity >= MIN_SIGNAL_INTENSITY

    Feasibility is certified using a Hoeffding bound.

    NOTE:
      This test assumes the simulator provides a 'signal_intensity' metric.
      The simulator itself will be enriched in a follow-up commit.
    """

    torch.manual_seed(0)

    candidates = torch.tensor(
        [[0.0], [2.0], [10.0], [20.0], [30.0], [40.0], [50.0], [60.0], [70.0], [80.0], [90.0], [100.0]],
        dtype=torch.double,
    )

    num_designs = 100
    evals_per_design = 100
    seed = 0

    optimiser = DiscreteBanditOptimiser(candidates=candidates)

    certificate = HoeffdingCertificate(alpha=ALPHA)

    experiment = MinFeasibleDimensioning(
        min_coverage=MIN_COVERAGE,
        min_signal_intensity=MIN_SIGNAL_INTENSITY,
        delta=DELTA,
        certificate=certificate,
    )

    # Diagnostics: empirical minima observed per design (from raw simulator metrics)
    min_cov_seen: dict[float, float] = {}
    min_sig_seen: dict[float, float] = {}
    n_valid_seen: dict[float, int] = {}
    n_invalid_seen: dict[float, int] = {}

    # Optimiser reward: empirical success rate
    n_success: dict[float, int] = {}
    n_reward_trials: dict[float, int] = {}


    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InputDataWarning)

        for _ in range(num_designs):
            lambda_outer = optimiser.ask()
            coverages = []

            for _ in range(evals_per_design):
                rng = random.Random(seed)
                seed += 1

                sim = Dimensioning_2D(
                    inner_lambda=5.0,
                    inner_radius=1.0,
                    outer_radius=1.1,
                    coverage_distance=0.2,
                    rng=rng,
                )

                metrics = sim.evaluate(lambda_outer)

                # Assumed metric (not yet implemented)
                assert "signal_intensity" in metrics

                # Track empirical minima per design (using raw metrics)
                d = float(lambda_outer)
                if metrics["n_ground"] == 0.0:
                    n_invalid_seen[d] = n_invalid_seen.get(d, 0) + 1
                else:
                    min_cov_seen[d] = min(min_cov_seen.get(d, float("inf")), metrics["coverage"])
                    min_sig_seen[d] = min(min_sig_seen.get(d, float("inf")), metrics["signal_intensity"])
                    n_valid_seen[d] = n_valid_seen.get(d, 0) + 1

                coverages.append(metrics["coverage"])
                # print(
                #     f"lambda={lambda_outer:.1f} | "
                #     f"coverage={metrics['coverage']:.3f} | "
                #     f"p10_signal={metrics['signal_intensity']:.6f}"
                # )

                experiment.on_evaluation(lambda_outer, metrics)
                d = float(lambda_outer)

                # Count only valid trials for reward purposes
                if metrics["n_ground"] > 0.0:
                    n_reward_trials[d] = n_reward_trials.get(d, 0) + 1
                    Z = experiment.metric(lambda_outer, metrics)
                    if experiment.accept(Z):
                        n_success[d] = n_success.get(d, 0) + 1


            d = float(lambda_outer)

            if n_reward_trials.get(d, 0) > 0:
                success_rate = n_success.get(d, 0) / n_reward_trials[d]
            else:
                success_rate = 0.0

            optimiser.tell(lambda_outer, success_rate)


    min_lambda, metrics = experiment.select_min()

    print("\n=== Empirical diagnostics per design ===")
    for d in sorted(n_valid_seen):
        successes = n_success.get(d, 0)
        trials = n_reward_trials.get(d, 0)
        success_rate = successes / trials if trials > 0 else 0.0

        print(
            f"lambda={d:.1f} | "
            f"n_valid={n_valid_seen[d]:4d} | "
            f"n_invalid={n_invalid_seen.get(d, 0):4d} | "
            f"success_rate={success_rate:.3f} | "
            f"min_coverage={min_cov_seen[d]:.3f} (>= {MIN_COVERAGE}) | "
            f"min_signal={min_sig_seen[d]:.6f} (>= {MIN_SIGNAL_INTENSITY})"
        )
    print("=======================================\n")

    feasible = [
        d for d in experiment._trials
        if experiment.is_feasible(d)
    ]

    assert feasible, "No design certified feasible"
    assert experiment.is_feasible(min_lambda)
    assert min_lambda == min(feasible)
