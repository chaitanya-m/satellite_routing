"""Concrete optimisation problem: satellites on a circular orbit.

This module provides a first implementation of :class:`optimisation.interfaces.DesignProblem`
for satellite dimensioning on a single circular orbit around the Earth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import tau
from typing import Any, Sequence

import random

from dimensioning.ppp import PPPParams, CircularOrbit, Point, sample_ppp
from .interfaces import DesignProblem


@dataclass
class OrbitDesign:
    """Design for a single-orbit constellation dimensioning problem."""

    lambda_mean: float
    orbit_radius_km: float
    satellites: Sequence[Point]
    ground_station_angles: Sequence[float] = field(default_factory=list)


@dataclass
class CoverageValue:
    """Simple coverage-based value function for orbit designs."""

    earth_radius_km: float = 6371.0
    footprint_fraction: float = 1.0 / 20.0

    def __call__(self, design: OrbitDesign) -> float:
        if not design.ground_station_angles:
            return 0.0

        footprint_radius_km = self.earth_radius_km * self.footprint_fraction
        coverage_angle = footprint_radius_km / design.orbit_radius_km

        sat_angles = [
            float(sat.location["angle_rad"])
            for sat in design.satellites
            if isinstance(sat.location, dict) and "angle_rad" in sat.location
        ]
        if not sat_angles:
            return 0.0

        def angular_distance(a: float, b: float) -> float:
            diff = abs(a - b) % tau
            return min(diff, tau - diff)

        total_visible = 0.0
        for ground_angle in design.ground_station_angles:
            visible_count = sum(
                1.0 for sat_angle in sat_angles if angular_distance(sat_angle, ground_angle) <= coverage_angle
            )
            total_visible += visible_count

        return total_visible / float(len(design.ground_station_angles))


@dataclass
class OrbitDimensioningProblem(DesignProblem):
    """Design problem for satellites on a circular orbit."""

    min_lambda: int = 1
    max_lambda: int = 10
    orbit_radius_km: float = 7000.0
    num_ground_stations: int = 10
    rng: random.Random = field(default_factory=random.Random)
    value_fn: CoverageValue = field(default_factory=CoverageValue)

    def __post_init__(self) -> None:
        self._ground_station_angles = self._generate_ground_stations(self.num_ground_stations)

    @staticmethod
    def _generate_ground_stations(num_stations: int) -> Sequence[float]:
        if num_stations <= 0:
            return []
        step = tau / float(num_stations)
        return [i * step for i in range(num_stations)]

    def _sample_orbit_design(self, lambda_mean: float) -> OrbitDesign:
        orbit = CircularOrbit(radius_km=self.orbit_radius_km)
        params = PPPParams(mean_intensity=lambda_mean)
        satellites = sample_ppp(params, orbit, rng=self.rng)
        return OrbitDesign(
            lambda_mean=lambda_mean,
            orbit_radius_km=self.orbit_radius_km,
            satellites=satellites,
            ground_station_angles=self._ground_station_angles,
        )

    def sample_design(self) -> OrbitDesign:
        lam = self.rng.randint(self.min_lambda, self.max_lambda)
        return self._sample_orbit_design(float(lam))

    def evaluate(self, design: Any) -> float:
        if not isinstance(design, OrbitDesign):
            raise TypeError("OrbitDimensioningProblem.evaluate expects an OrbitDesign instance.")
        return self.value_fn(design)

