# sim/dimensioning_2d.py

import math
import random
from sim.stochastic.poisson import sample_poisson

class Dimensioning_2D:
    """
    2D concentric-circle PPP coverage simulator.

    Ground stations:
        PPP with rate `inner_lambda` on radius `inner_radius` (fixed)

    Satellites:
        PPP with rate `lambda_outer` on radius `outer_radius` (design variable)

    A ground station is covered if at least one satellite is within
    `coverage_distance` (Euclidean distance).

    Signal model:
        For a ground station g and satellite s at distance d,
        received signal strength is
            S(d) = 1 / d^2     if d <= coverage_distance
            S(d) = 0           otherwise

    Reported signal metric:
        p10 (10th percentile) of best-server signal strength
        across all ground stations in the trial.
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

    @staticmethod
    def _signal_strength(distance: float) -> float:
        # Monotone distance-based signal model (used only when link is feasible)
        return 1.0 / (distance * distance)

    @staticmethod
    def _p10(values: list[float]) -> float:
        if not values:
            return 0.0
        values = sorted(values)
        k = max(0, math.ceil(0.10 * len(values)) - 1)
        return values[k]

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
                "signal_intensity": 0.0,
                "n_ground": 0.0,
                "n_sats": float(n_sats),
            }

        if n_sats == 0:
            return {
                "coverage": 0.0,
                "signal_intensity": 0.0,
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
        best_server_signals: list[float] = []

        for theta in ground_angles:
            best_signal = 0.0
            is_covered = False

            for phi in sat_angles:
                dist = math.sqrt(
                    self.inner_radius**2
                    + self.outer_radius**2
                    - 2 * self.inner_radius * self.outer_radius * math.cos(theta - phi)
                )

                if dist <= self.coverage_distance:
                    is_covered = True
                    signal = self._signal_strength(dist)
                    if signal > best_signal:
                        best_signal = signal
                # else: signal contribution is 0 by definition (do nothing)

            if is_covered:
                covered += 1

            # If not covered, best_signal remains 0.0, as desired.
            best_server_signals.append(best_signal)

        coverage = covered / n_ground
        signal_p10 = self._p10(best_server_signals)

        return {
            "coverage": coverage,
            "signal_intensity": signal_p10,
            "n_ground": float(n_ground),
            "n_sats": float(n_sats),
        }
