import math
import torch

from sim.circle import CircleAlignmentSim
from optim.discrete_bandit import DiscreteBanditOptimiser
from runner import run


def test_circle_botorch_finds_target_angle():
    
    import runner
    print("RUN FUNCTION FROM:", runner.run.__code__.co_filename)

    sim = CircleAlignmentSim(target_angle=0.5)

    # Discrete candidate set on the circle
    angles = torch.linspace(-math.pi, math.pi, 200).unsqueeze(-1)
    angles_normalised = (angles + math.pi) / (2 * math.pi)
    
    optimiser = DiscreteBanditOptimiser(candidates=angles_normalised)

    best = run(optimiser, sim, budget=40)
    assert best is not None

    x_best, y_best = best
    assert abs(x_best - 0.5) < 0.2
    assert y_best > 0.8
