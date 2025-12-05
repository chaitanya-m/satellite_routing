"""
Constellation and ground-station generation utilities.

Satellites are sampled from a Poisson point process on the sphere, with
band-weighted latitude ranges that reflect common deployment choices.
Ground stations use the same machinery but a different default weighting.
"""

from dataclasses import dataclass
from math import asin, exp, radians, sin
from typing import List, Optional, Sequence
import math
import random

from nodes import Node


@dataclass(frozen=True)
class LatBand:
    """
    Latitude band with a relative weight.

    weight is proportional to the expected number of points in the band.
    """

    min_lat: float  # degrees
    max_lat: float  # degrees
    weight: float


# Popular latitude ranges for current LEO constellations:
# - Near-equatorial shells (OneWeb, Amazon Kuiper)
# - Mid-latitude shells around 53 deg (Starlink)
# - Sun-synchronous/polar shells around ~97 deg
POPULAR_LAT_BANDS: Sequence[LatBand] = (
    LatBand(-10.0, 10.0, 0.6),
    LatBand(40.0, 60.0, 1.0),
    LatBand(70.0, 98.0, 0.8),
)

# Ground station latitudes skewed toward populated mid-latitudes.
GROUND_STATION_BANDS: Sequence[LatBand] = (
    LatBand(-5.0, 5.0, 0.6),    # equatorial belt
    LatBand(5.0, 20.0, 1.0),    # tropics north
    LatBand(-20.0, -5.0, 1.0),  # tropics south
    LatBand(20.0, 55.0, 1.2),   # populated mid-lat north
    LatBand(-55.0, -20.0, 1.0), # populated mid-lat south
)


@dataclass(frozen=True)
class SatelliteNode(Node):
    """
    Concrete satellite node with geographic metadata.
    """

    _id: str
    lat: float          # degrees
    lon: float          # degrees
    altitude_km: float

    @property
    def id(self) -> str:
        return self._id


@dataclass(frozen=True)
class GroundStationNode(Node):
    """
    Concrete ground station node.
    """

    _id: str
    lat: float
    lon: float

    @property
    def id(self) -> str:
        return self._id


def _poisson(rng: random.Random, lam: float) -> int:
    """
    Sample a Poisson-distributed count with rate lam using Knuth's method.
    Suitable for modest lam (tens to hundreds) used in these generators.
    """
    if lam <= 0:
        return 0

    threshold = exp(-lam)
    prod = 1.0
    k = 0

    while prod > threshold:
        k += 1
        prod *= rng.random()
    return k - 1


def _sample_lat_lon(rng: random.Random, min_lat: float, max_lat: float) -> tuple[float, float]:
    """
    Sample a point uniformly on the spherical patch bounded by [min_lat, max_lat].
    Uses a sin-lat transform for equal-area sampling.
    """
    sin_min = sin(radians(min_lat))
    sin_max = sin(radians(max_lat))
    s = rng.uniform(sin_min, sin_max)
    lat = math.degrees(asin(s))
    lon = rng.uniform(-180.0, 180.0)
    return lat, lon


def generate_poisson_constellation(
    expected_count: int,
    bands: Sequence[LatBand] = POPULAR_LAT_BANDS,
    altitude_km: float = 550.0,
    seed: Optional[int] = None,
) -> List[SatelliteNode]:
    """
    Generate a Poisson-distributed satellite constellation across latitude bands.

    How it works:
    - Take the global expected_count and split it across the provided bands
      proportional to each band's weight; this produces a per-band rate λ that
      reflects how popular that latitude shell is.
    - For each band, sample an actual population from a Poisson distribution
      with rate λ (Knuth sampler). Poisson variation keeps total counts reasonable
      without hard-coding a fixed number per band, which better mimics
      deployment noise and avoids perfect symmetry.
    - For every sampled satellite, draw (lat, lon) uniformly by area within
      the band's latitude bounds (sin-lat transform) and assign a constant
      altitude_km. Nodes are sequentially numbered SAT-0, SAT-1, ...
    - If seed is provided, rng draws are deterministic for reproducibility.

    Args:
        expected_count: Target mean constellation size across all bands.
        bands: Latitude bands with relative weights (defaults model popular LEO shells).
        altitude_km: Orbital altitude attached to every node.
        seed: Optional RNG seed to reproduce a particular sampled constellation.

    Returns:
        List of SatelliteNode instances positioned according to the sampled process.

    Raises:
        ValueError: if band weights do not sum to a positive value.
    """
    rng = random.Random(seed)
    total_weight = sum(b.weight for b in bands)
    if total_weight <= 0:
        raise ValueError("Latitude band weights must sum to a positive value.")
    satellites: List[SatelliteNode] = []

    for idx, band in enumerate(bands):
        lam = expected_count * (band.weight / total_weight)
        count = _poisson(rng, lam)
        for _ in range(count):
            lat, lon = _sample_lat_lon(rng, band.min_lat, band.max_lat)
            sat_id = f"SAT-{len(satellites)}"
            satellites.append(SatelliteNode(sat_id, lat, lon, altitude_km))

    return satellites


def generate_ground_stations(
    expected_count: int,
    bands: Sequence[LatBand] = GROUND_STATION_BANDS,
    seed: Optional[int] = None,
) -> List[GroundStationNode]:
    """
    Generate ground stations with Poisson counts per latitude band.
    """
    rng = random.Random(seed)
    total_weight = sum(b.weight for b in bands)
    if total_weight <= 0:
        raise ValueError("Latitude band weights must sum to a positive value.")
    stations: List[GroundStationNode] = []

    for band in bands:
        lam = expected_count * (band.weight / total_weight)
        count = _poisson(rng, lam)
        for _ in range(count):
            lat, lon = _sample_lat_lon(rng, band.min_lat, band.max_lat)
            gs_id = f"GS-{len(stations)}"
            stations.append(GroundStationNode(gs_id, lat, lon))

    return stations
