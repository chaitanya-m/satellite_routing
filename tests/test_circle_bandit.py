import math
import torch

from sim.circle import CircleAlignmentSim
from optim.discrete_bandit import DiscreteBanditOptimiser


def test_circle_botorch_finds_target_angle():
    sim = CircleAlignmentSim(target_angle=0.5)

    # Discrete candidate set on the circle (normalised to [0, 1])
    angles = torch.linspace(-math.pi, math.pi, 200)
    angles_normalised = ((angles + math.pi) / (2 * math.pi)).unsqueeze(-1)

    optimiser = DiscreteBanditOptimiser(candidates=angles_normalised)

    best_x = None
    best_y = float("-inf")

    for _ in range(40):
        x_norm = optimiser.ask()                 # in [0, 1]
        angle = x_norm * 2 * math.pi - math.pi   # back to radians

        y = sim.evaluate(angle)
        optimiser.tell(x_norm, y)

        if y > best_y:
            best_x = angle
            best_y = y

    assert best_x is not None
    assert abs(best_x - 0.5) < 0.2
    assert best_y > 0.8
