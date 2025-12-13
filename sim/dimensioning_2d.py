# sim/dimensioning_2d.py

import math
import random
from sim.stochastic.poisson import sample_poisson


class Dimensioning_2D:
    """
    2D concentric-circle PPP coverage simulator.

    Ground stations:
        PPP with intensity `inner_lambda` on radius `inner_radius` (fixed)

    Satellites:
        PPP with intensity `lambda_outer` on radius `outer_radius` (design variable)

    A ground station is covered if at least one satellite is within
    `coverage_distance` (Euclidean distance).
    """

    def __init__(
        self,
        *,
        inner_lambda: float,
        inner_radius: float,
        outer_radius: float,
        coverage_distance: float,
        rng: random.Random | None = None,
    ):
        self.inner_lambda = float(inner_lambda)
        self.inner_radius = float(inner_radius)
        self.outer_radius = float(outer_radius)
        self.coverage_distance = float(coverage_distance)
        self.rng = rng or random.Random()

        # Last realised counts (for inspection / testing only)
        self.last_n_ground: int | None = None
        self.last_n_sats: int | None = None

    def evaluate(self, lambda_outer: float) -> dict[str, float]:
        # Ground stations (NOT optimised)
        n_ground = sample_poisson(self.inner_lambda, self.rng)

        # Satellites (THIS is what the optimiser chooses)
        n_sats = sample_poisson(float(lambda_outer), self.rng)

        self.last_n_ground = n_ground
        self.last_n_sats = n_sats

        if n_ground == 0:
            return {
                "coverage": 1.0,   # vacuously covered
                "n_ground": 0.0,
                "n_sats": float(n_sats),
            }

        if n_sats == 0:
            return {
                "coverage": 0.0,
                "n_ground": float(n_ground),
                "n_sats": 0.0,
            }

        ground_angles = [
            self.rng.uniform(-math.pi, math.pi) for _ in range(n_ground)
        ]
        sat_angles = [
            self.rng.uniform(-math.pi, math.pi) for _ in range(n_sats)
        ]

        covered = 0
        for theta in ground_angles:
            for phi in sat_angles:
                dist = math.sqrt(
                    self.inner_radius**2
                    + self.outer_radius**2
                    - 2 * self.inner_radius * self.outer_radius * math.cos(theta - phi)
                )
                if dist <= self.coverage_distance:
                    covered += 1
                    break


        coverage = covered / n_ground if n_ground > 0 else 1.0

        return {
            "coverage": coverage,
            "n_ground": float(n_ground),
            "n_sats": float(n_sats),
        }
