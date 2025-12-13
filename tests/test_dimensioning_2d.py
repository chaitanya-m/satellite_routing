import random
import torch
import warnings

from botorch.exceptions import InputDataWarning

from sim.dimensioning_2d import Dimensioning_2D
from optim.discrete_bandit import DiscreteBanditOptimiser
from runner import run


def test_dimensioning_2d_prefers_higher_outer_intensity():
    torch.manual_seed(0)

    # OUTER PPP intensities (satellites)
    candidates = torch.tensor([[0.0], [2.0], [10.0], [20.0], [50.0]], dtype=torch.double)

    all_scores: list[float] = []
    chosen_lambdas: list[float] = []
    num_simulations = 100
    num_optimiser_steps = 100 # number of design evaluations per simulation

    for seed in range(num_simulations):
        sim = Dimensioning_2D(
            inner_lambda=5.0,       # ground stations (fixed)
            inner_radius=1.0,
            outer_radius=1.1,
            coverage_distance=0.2,  # > |1.1 - 1.0| = 0.1
            alpha=0.01,              # cost penalty coefficient for outer lambda, i.e for number of satellites
            rng=random.Random(seed),
        )

        optimiser = DiscreteBanditOptimiser(candidates=candidates)

        best_lambda = None
        best_score = None

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InputDataWarning)

            for _ in range(num_optimiser_steps):
                lambda_outer = optimiser.ask()      # ask optimiser to choose design value (outer lambda)
                coverage = sim.evaluate(lambda_outer)  # evaluate with simulator
                score = coverage - sim.alpha * lambda_outer  # objective: coverage - cost_penalty
                optimiser.tell(lambda_outer, score) # tell optimiser the result

                all_scores.append(score)

                if best_score is None or score > best_score:
                    best_score = score
                    best_lambda = lambda_outer

        assert best_lambda is not None
        assert best_score is not None

        print(
            f"seed={seed:2d} | "
            f"chosen outer λ={best_lambda:5.1f} | "
            f"last ground PPP={sim.last_n_ground:2d} | "
            f"last satellite PPP={sim.last_n_sats:3d} | "
            f"best score={best_score:.3f}"
        )

        chosen_lambdas.append(best_lambda)

    # Sanity check: objective must not be identically zero
    assert any(score > 0.0 for score in all_scores), all_scores

    # Behavioural invariant: optimiser should avoid very low satellite intensities
    assert all(lam >= 10.0 for lam in chosen_lambdas), chosen_lambdas
