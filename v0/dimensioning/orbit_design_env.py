"""Orbit design MDP setup using PPP-sampled satellites.

This module wires together:

* The generic design-space MDP in :mod:`dimensioning.design_mdp`.
* PPP sampling utilities from :mod:`dimensioning.ppp`.

The design space here consists of constellations on a single circular orbit
around the Earth. A design point contains:

* The PPP mean (intensity) used to sample satellite positions along the orbit.
* The resulting list of satellite positions on that orbit.

A simple value function estimates a coverage score for a fixed set of ground
stations on the equator, assuming each satellite has a footprint with radius
equal to approximately 1/20th of the Earth's radius. This creates a design
problem where larger PPP means (more satellites) tend to yield higher scores.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import tau
from typing import Any, Protocol, Sequence, Tuple

import random

from .design_mdp import Action, DesignEnvironment, DesignTransitionFn, RewardFn, State, TerminationFn
from .ppp import PPPParams, CircularOrbit, Point, sample_ppp


@dataclass
class OrbitDesign:
    """Design point for a single-orbit constellation."""

    lambda_mean: float
    orbit_radius_km: float
    satellites: Sequence[Point]
    ground_station_angles: Sequence[float] = field(default_factory=list)


class DesignValueFunction(Protocol):
    """Interface for value functions on orbit designs."""

    def evaluate(self, design: OrbitDesign) -> float:
        ...


@dataclass
class CoverageValueFunction:
    """Concrete value function using a simple visibility model.

    Ground stations are placed at fixed angles around the equator. A satellite
    is considered "in view" of a ground station if the angular separation
    between them along the orbit is less than a threshold determined by a
    footprint radius equal to approximately 1/20th of the Earth's radius.
    """

    earth_radius_km: float = 6371.0
    footprint_fraction: float = 1.0 / 20.0

    def evaluate(self, design: OrbitDesign) -> float:
        if not design.ground_station_angles:
            return 0.0

        footprint_radius_km = self.earth_radius_km * self.footprint_fraction
        coverage_angle = footprint_radius_km / design.orbit_radius_km

        # Extract satellite angles from PPP points.
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

        # Average number of satellites in view across ground stations.
        return total_visible / float(len(design.ground_station_angles))


def _generate_ground_stations(num_stations: int) -> Sequence[float]:
    """Return equally spaced ground station angles on [0, 2π)."""

    if num_stations <= 0:
        return []
    step = tau / float(num_stations)
    return [i * step for i in range(num_stations)]


def _sample_orbit_design(
    lambda_mean: float,
    orbit_radius_km: float,
    ground_station_angles: Sequence[float],
    rng: random.Random,
) -> OrbitDesign:
    orbit = CircularOrbit(radius_km=orbit_radius_km)
    params = PPPParams(mean_intensity=lambda_mean)
    satellites = sample_ppp(params, orbit, rng=rng)
    return OrbitDesign(
        lambda_mean=lambda_mean,
        orbit_radius_km=orbit_radius_km,
        satellites=satellites,
        ground_station_angles=list(ground_station_angles),
    )


def create_orbit_design_environment(
    *,
    min_lambda: int = 1,
    max_lambda: int = 10,
    orbit_radius_km: float = 7000.0,
    num_ground_stations: int = 10,
    max_steps: int = 5,
    rng: random.Random | None = None,
) -> Tuple[DesignEnvironment, State, DesignValueFunction]:
    """Create a design-space MDP for satellite constellations on a circular orbit.

    The agent explores PPP means in the integer range [min_lambda, max_lambda]
    by choosing actions that set a new mean. For each design state, satellites
    are sampled anew from the PPP and a coverage-based reward is computed.

    The caller is expected to plug this environment into an RL library or
    custom exploration loop.
    """

    if rng is None:
        rng = random.Random()
    value_fn: DesignValueFunction = CoverageValueFunction()
    ground_station_angles = _generate_ground_stations(num_ground_stations)

    def clamp_lambda(value: float) -> float:
        return float(max(min_lambda, min(max_lambda, int(round(value)))))

    def design_transition(design: OrbitDesign, action: Action) -> OrbitDesign:
        # Interpret action.data as a proposed lambda; clamp to [min_lambda, max_lambda].
        try:
            proposed = float(action.data)
        except (TypeError, ValueError):
            proposed = design.lambda_mean
        new_lambda = clamp_lambda(proposed)
        return _sample_orbit_design(
            lambda_mean=new_lambda,
            orbit_radius_km=design.orbit_radius_km,
            ground_station_angles=design.ground_station_angles,
            rng=rng,
        )

    def reward_fn(state: State, action: Action, next_state: State) -> float:
        return value_fn.evaluate(next_state.design)  # type: ignore[arg-type]

    def termination_fn(state: State) -> bool:
        return state.t >= max_steps

    initial_design = _sample_orbit_design(
        lambda_mean=float(min_lambda),
        orbit_radius_km=orbit_radius_km,
        ground_station_angles=ground_station_angles,
        rng=rng,
    )
    initial_state = State(t=0, design=initial_design)

    env = DesignEnvironment(
        design_transition=design_transition,  # type: ignore[arg-type]
        reward_fn=reward_fn,
        termination_fn=termination_fn,
    )
    return env, initial_state, value_fn

