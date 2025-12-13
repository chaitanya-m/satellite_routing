# tests/test_dimensioning_2d.py

import random
import torch
import warnings

from botorch.exceptions import InputDataWarning

from experiments.min_feasible_coverage import MinLambdaForCoverage
from sim.dimensioning_2d import Dimensioning_2D
from optim.discrete_bandit import DiscreteBanditOptimiser


TARGET_COVERAGE = 0.7   
DELTA = 0.05            # allow 5% failure probability
ALPHA = 0.05            # 95% confidence

def test_dimensioning_2d_prefers_higher_outer_intensity():
    """
    Research objective:
    For a fixed ground PPP intensity, find the *minimum* satellite PPP intensity
    that achieves (near-)full coverage with high probability.

    Optimisation structure:
    - one optimiser ask() proposes a satellite intensity (lambda_outer)
    - that design is evaluated multiple times to estimate its stochastic behaviour
    - the aggregated result is fed back via a single tell()
    """

    torch.manual_seed(0)

    # OUTER PPP intensities (satellites)
    candidates = torch.tensor(
        [[0.0], [2.0], [10.0], [20.0], [30.0], [40.0], [500.0]],
        dtype=torch.double,
    )

    num_designs = 100                 # how many designs the optimiser may propose
    evals_per_design = 100             # how many simulator runs per design

    seed = 0

    # Optimiser and experiment persist across the entire optimisation run
    optimiser = DiscreteBanditOptimiser(candidates=candidates)
    experiment = MinLambdaForCoverage(
        target_coverage=TARGET_COVERAGE,
        delta=DELTA,
        alpha=ALPHA,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InputDataWarning)

        for _ in range(num_designs):
            # ---- ask: optimiser proposes a design (satellite intensity) ----
            lambda_outer = optimiser.ask()

            coverages = []

            # ---- evaluate: multiple independent simulator runs for this design ----
            for _ in range(evals_per_design):
                rng = random.Random(seed)
                seed += 1
                sim = Dimensioning_2D(
                    inner_lambda=5.0,       # ground stations (fixed)
                    inner_radius=1.0,
                    outer_radius=1.1,
                    coverage_distance=0.2,  # > |1.1 - 1.0| = 0.1
                    rng=rng,
                )

                metrics = sim.evaluate(lambda_outer)
                coverages.append(metrics["coverage"])

                # Track feasibility information separately from optimisation
                experiment.on_evaluation(lambda_outer, metrics)

                if metrics["coverage"] > 0.0:
                    saw_nonzero_coverage = True
                print(
                    f"[eval] λ_sat={lambda_outer:5.1f} | "
                    f"n_ground={metrics['n_ground']:2.0f} | "
                    f"n_sats={metrics['n_sats']:3.0f} | "
                    f"coverage={metrics['coverage']:.3f}"
                )

            # ---- aggregate: produce a single scalar feedback for the optimiser ----
            # We use mean coverage as a noisy but unbiased objective signal.
            mean_coverage = sum(coverages) / len(coverages)


            print(
                f"[tell] λ_sat={lambda_outer:5.1f} | "
                f"mean_coverage={mean_coverage:.3f} | "
                f"evals={len(coverages)}"
            )

            # ---- tell: single feedback per ask(), as expected by the optimiser ----
            optimiser.tell(lambda_outer, mean_coverage)

    # ---- post-optimisation selection ----
    # The experiment decides what "best" means: minimum lambda achieving target coverage.
    min_lambda, metrics = experiment.select_min()

    print(
        f"min feasible outer λ={min_lambda:5.1f} | "
        f"ground PPP={metrics['n_ground']:2.0f} | "
        f"sat PPP={metrics['n_sats']:3.0f} | "
        f"coverage={metrics['coverage']:.3f}"
    )

    feasible = [
        d for d in experiment._trials
        if experiment.is_feasible(d)
    ]
    assert feasible, "No design certified feasible"

    assert experiment.is_feasible(min_lambda)

    assert min_lambda == min(feasible)

