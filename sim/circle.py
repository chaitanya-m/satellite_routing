import math
import random


class CircleAlignmentSim:
    """Score angles on the unit circle by proximity to a target angle."""

    def __init__(self, target_angle: float, rng: random.Random | None = None):
        self.target_angle = float(target_angle)
        self.rng = rng or random.Random()

    def evaluate(self, angle: float) -> float:
        angle = float(angle)
        diff = abs((angle - self.target_angle + math.pi) % (2 * math.pi) - math.pi)
        return max(0.0, 1.0 - diff / math.pi)

