"""Poisson point process helpers for design-space experiments.

This module contains small utilities for sampling (homogeneous) marked PPPs on
an abstract region. It is intentionally separate from the MDP harness in
``design_mdp.py`` so that the latter remains agnostic to how candidate designs
are generated.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt, tau
from typing import Any, Callable, List, Mapping, Optional, Protocol

import random


@dataclass(frozen=True)
class Point:
    """A single point in a PPP realisation."""

    location: Any
    mark: Optional[Mapping[str, Any]] = None


class Region(Protocol):
    """Subset of the underlying space on which a PPP is sampled."""

    def volume(self) -> float:
        """Return the measure of the region (e.g. area, length, or volume)."""

        ...

    def sample_location(self, rng: random.Random) -> Any:
        """Draw a single location uniformly from the region."""

        ...


@dataclass(frozen=True)
class PPPParams:
    """Parameters controlling a homogeneous marked PPP sampler."""

    mean_intensity: float
    sample_mark: Optional[Callable[[random.Random], Mapping[str, Any]]] = None


def _poisson_sample(lam: float, rng: random.Random) -> int:
    """Draw a single Poisson(lam) variate using a basic algorithm."""

    if lam <= 0.0:
        return 0

    # Simple approximation for very large lambda to avoid long loops.
    if lam > 700:
        value = int(round(rng.normalvariate(lam, sqrt(lam))))
        return max(0, value)

    l = exp(-lam)
    k = 0
    p = 1.0
    while p > l:
        k += 1
        p *= rng.random()
    return k - 1


def sample_ppp(params: PPPParams, region: Region, rng: Optional[random.Random] = None) -> List[Point]:
    """Sample a realisation of a homogeneous marked PPP on ``region``."""

    if rng is None:
        rng = random.Random()

    lam = params.mean_intensity * region.volume()
    count = _poisson_sample(lam, rng)
    points: List[Point] = []
    for _ in range(count):
        location = region.sample_location(rng)
        mark = params.sample_mark(rng) if params.sample_mark is not None else None
        points.append(Point(location=location, mark=mark))
    return points


@dataclass(frozen=True)
class CircularOrbit(Region):
    """One-dimensional circular region representing a single orbital ring.

    This is a simple helper for satellite dimensioning experiments where we
    want a homogeneous PPP along a circular orbit around the Earth.

    Attributes:
        radius_km:
            Radius of the orbit in kilometres. The PPP is defined over the
            circle at this radius; the linear measure is ``2π * radius_km``.
        origin:
            Optional identifier or metadata describing the orbital frame. This
            is not used in sampling but may be helpful for downstream code.
    """

    radius_km: float
    origin: Optional[str] = None

    def volume(self) -> float:
        """Return the circumference, used as the 1D 'volume' of the region."""

        return tau * self.radius_km

    def sample_location(self, rng: random.Random) -> Any:
        """Return a uniform point on the orbit as an angle in radians."""

        angle = rng.random() * tau
        return {"angle_rad": angle, "radius_km": self.radius_km, "origin": self.origin}
