import random
import torch
import warnings

from botorch.exceptions import InputDataWarning

from experiments.min_feasible_coverage import MinLambdaForCoverage
from sim.dimensioning_2d import Dimensioning_2D
from optim.discrete_bandit import DiscreteBanditOptimiser


TARGET_COVERAGE = 0.95


def test_dimensioning_2d_prefers_higher_outer_intensity():
    torch.manual_seed(0)

    # OUTER PPP intensities (satellites)
    candidates = torch.tensor(
        [[0.0], [2.0], [10.0], [20.0], [30.0], [40.0], [50.0]],
        dtype=torch.double,
    )

    chosen_lambdas: list[float] = []
    num_simulations = 100
    num_optimiser_steps = 100  # design evaluations per simulation

    saw_nonzero_coverage = False

    for seed in range(num_simulations):
        sim = Dimensioning_2D(
            inner_lambda=5.0,       # ground stations (fixed per simulation)
            inner_radius=1.0,
            outer_radius=1.1,
            coverage_distance=0.2,  # > |1.1 - 1.0| = 0.1
            rng=random.Random(seed),
        )

        optimiser = DiscreteBanditOptimiser(candidates=candidates)
        experiment = MinLambdaForCoverage(TARGET_COVERAGE)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InputDataWarning)

            for _ in range(num_optimiser_steps):
                lambda_outer = optimiser.ask()
                metrics = sim.evaluate(lambda_outer)

                objective = experiment.objective(lambda_outer, metrics)
                optimiser.tell(lambda_outer, objective)
                experiment.on_evaluation(lambda_outer, metrics)

                if metrics["coverage"] > 0.0:
                    saw_nonzero_coverage = True

        # Delegate feasibility and selection to the experiment
        min_lambda, metrics = experiment.select_min()
        chosen_lambdas.append(min_lambda)

        print(
            f"seed={seed:2d} | "
            f"min feasible outer λ={min_lambda:5.1f} | "
            f"ground PPP={metrics['n_ground']:2.0f} | "
            f"sat PPP={metrics['n_sats']:3.0f} | "
            f"coverage={metrics['coverage']:.3f}"
        )

    # Zero-triviality check: simulator must produce some non-zero coverage
    assert saw_nonzero_coverage, "Simulator never produced non-zero coverage"

    # Provisional behavioural check: optimiser should not always require max intensity
    assert any(lam < 50.0 for lam in chosen_lambdas), chosen_lambdas
