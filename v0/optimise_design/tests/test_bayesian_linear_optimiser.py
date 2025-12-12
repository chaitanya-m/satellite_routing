from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Tuple

import numpy as np

from optimise_design import BayesianLinearOptimiser, run_optimisation
from optimise_design.interface import DesignProblem


@dataclass
class TwoDQuadraticProblem(DesignProblem):
    """Simple 2D quadratic optimum at (1.5, -0.5).

    The initial encoder returns only linear terms [x, y, 1], which is
    deliberately insufficient to capture curvature; a richer encoder with
    quadratic terms follows below.
    """

    low: float = -3.0
    high: float = 3.0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def sample_one_design(self) -> Tuple[float, float]:
        x = float(self.rng.uniform(self.low, self.high))
        y = float(self.rng.uniform(self.low, self.high))
        return (x, y)

    def evaluate(self, design: Any) -> float:
        x, y = design
        return -((x - 1.5) ** 2 + (y + 0.5) ** 2)

    def encode_vector(self, design: Any) -> np.ndarray:
        x, y = design
        return np.array([x, y, 1.0], dtype=float)


@dataclass
class TwoDQuadraticProblemWithQuadratics(TwoDQuadraticProblem):
    """Same quadratic, but with quadratic feature terms to model curvature."""

    def encode_vector(self, design: Any) -> np.ndarray:
        x, y = design
        return np.array([x, y, x * x, y * y, x * y, 1.0], dtype=float)


def test_bayesian_linear_without_quadratics_is_inadequate() -> None:
    """A purely linear encoder cannot reach the quadratic peak."""

    problem = TwoDQuadraticProblem(rng=np.random.default_rng(123))
    optimiser = BayesianLinearOptimiser(
        prior_precision=1.0,
        noise_variance=0.1,
        sample_candidates=64,
        rng=np.random.default_rng(999),
    )

    best = run_optimisation(problem, optimiser, budget=400)
    assert best is not None
    design, score = best
    x, y = design
    # The linear encoder cannot capture curvature; expect to remain far (>0.3) from optimum.
    dist = math.hypot(x - 1.5, y + 0.5)
    assert dist > 0.3
    # Score will be below the true peak (0.0).
    assert score < -0.05

    # Posterior summary should be available and typed as a Gaussian weight posterior.
    summary = optimiser.export_posterior_summary()
    assert summary is not None
    assert summary.get("summary_type") == "gaussian_weight_posterior"
    assert "mean" in summary and "cov" in summary


def test_bayesian_linear_with_quadratics_moves_toward_peak() -> None:
    """Including quadratic terms lets the optimiser approach the true optimum."""

    problem = TwoDQuadraticProblemWithQuadratics(rng=np.random.default_rng(321))
    optimiser = BayesianLinearOptimiser(
        prior_precision=1.0,
        noise_variance=0.1,
        sample_candidates=64,
        rng=np.random.default_rng(111),
    )

    best = run_optimisation(problem, optimiser, budget=500)
    assert best is not None
    design, score = best
    x, y = design
    dist = math.hypot(x - 1.5, y + 0.5)
    assert dist < 0.3
    assert score > -0.1


class DualOrbitCoverageProblem(DesignProblem):
    """Optimise two orbit inclinations to cover PPP ground stations on a sphere.

    - Ground stations: PPP on the unit sphere with mean ``ground_mean``.
    - Satellites: two circular orbits at radius 1.1, with inclinations (deg)
      given by the design tuple (inc1, inc2) relative to the equatorial plane.
      Each orbit has a fixed number of satellites (``sats_per_orbit``).
    - Coverage: a ground point is covered if any satellite lies within
      ``coverage_distance`` in Euclidean space. Return the fraction covered.

    The encoder maps (inc1, inc2) -> [inc1, inc2] so the optimiser learns a
    weight for each inclination dimension.
    """

    def __init__(
        self,
        inc_min: float,
        inc_max: float,
        ground_mean: float,
        sats_per_orbit: int,
        coverage_distance: float,
        rng: np.random.Generator,
    ) -> None:
        self.inc_min = inc_min
        self.inc_max = inc_max
        self.ground_mean = ground_mean
        self.sats_per_orbit = sats_per_orbit
        self.coverage_distance = coverage_distance
        self.rng = rng

    def sample_one_design(self) -> Tuple[float, float]:
        inc1 = float(self.rng.uniform(self.inc_min, self.inc_max))
        inc2 = float(self.rng.uniform(self.inc_min, self.inc_max))
        return inc1, inc2

    def _sample_ground(self, mean: float) -> np.ndarray:
        n = self.rng.poisson(mean)
        if n <= 0:
            return np.zeros((0, 3), dtype=float)
        vec = self.rng.normal(size=(n, 3))
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        return vec / norms

    def _orbit_points(self, n: int, inclination_deg: float, radius: float = 1.1) -> np.ndarray:
        if n <= 0:
            return np.zeros((0, 3), dtype=float)
        phi = self.rng.uniform(0.0, 2.0 * math.pi, size=n)
        inc = math.radians(inclination_deg)
        x = radius * np.cos(phi)
        y = radius * np.sin(phi) * math.cos(inc)
        z = radius * np.sin(phi) * math.sin(inc)
        return np.stack([x, y, z], axis=1)

    def evaluate(self, design: Any) -> float:
        inc1, inc2 = design
        ground = self._sample_ground(self.ground_mean)
        sats1 = self._orbit_points(self.sats_per_orbit, inc1)
        sats2 = self._orbit_points(self.sats_per_orbit, inc2)
        sats = np.vstack([sats1, sats2]) if sats1.size or sats2.size else np.zeros((0, 3))

        if ground.shape[0] == 0:
            return 1.0
        if sats.shape[0] == 0:
            return 0.0

        covered = 0
        for g in ground:
            dists = np.linalg.norm(sats - g, axis=1)
            if np.any(dists <= self.coverage_distance):
                covered += 1
        return covered / float(ground.shape[0])

    def encode_vector(self, design: Any) -> np.ndarray:
        inc1, inc2 = design
        return np.array([inc1, inc2], dtype=float)


def test_bayesian_linear_favours_well_separated_orbits() -> None:
    """Linear bandit should learn to separate orbit inclinations for better coverage.

    Setup:
    - Continuous design space: two inclinations (inc1, inc2) sampled uniformly
      from [0, 90] degrees each iteration.
    - Per-iteration search: the bandit samples ``sample_candidates`` (=10)
      candidate inclination pairs, draws one weight vector from its Gaussian
      posterior (Thompson sampling), scores the candidates as w^T x (x =
      [inc1, inc2]) using that single weight draw, and **simulates only the
      top-scoring pair**. The other 9 are scored cheaply but not simulated.
    - Evaluation: coverage is computed on a fresh PPP realisation of ground
      stations and satellites each time. Coverage = fraction of ground stations
      within ``coverage_distance`` of any satellite.
    - Budget: 1000 evaluations -> 1000 distinct ground/satellite realisations.

    Expectation: because two coplanar or near-coplanar orbits leave many points
    uncovered, the optimiser should prefer well-separated inclinations (>= 45
    degrees apart). This assertion is coarse but checks the trend.
    """

    problem = DualOrbitCoverageProblem(
        inc_min=0.0,
        inc_max=90.0,
        ground_mean=10.0,
        sats_per_orbit=10,
        coverage_distance=0.35,
        rng=np.random.default_rng(1),
    )
    optimiser = BayesianLinearOptimiser(
        prior_precision=1.0,
        noise_variance=0.5,
        sample_candidates=10,
        rng=np.random.default_rng(2),
    )

    best = run_optimisation(problem, optimiser, budget=1000)
    assert best is not None
    design, _score = best
    inc1, inc2 = design
    rel_inclination = abs(inc1 - inc2)

    # The optimiser should pick a design with substantial separation. Note we allow a max inclination of 90 deg.
    assert rel_inclination >= 45.0
