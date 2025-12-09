"""
Interfaces and data models for pluggable satellite communications simulators.

The interfaces defined in this module are intentionally decoupled from any
particular simulator implementation. They can be used to wrap external tools
such as the LTE-focused `ltesat` binary described in ``docs/ltesat_*`` as well
as lighter-weight in-memory simulators. The goal is to provide a stable API for
routing experiments (e.g., using Dijkstra) alongside planning and dimensioning
work where constellation placement and link quality matter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Iterable, List, Mapping, Optional, Sequence, Set


class EphemerisSourceType(Enum):
    """How a simulator should interpret satellite orbit definitions.

    This mirrors the variations outlined in the ltesat integration docs while
    remaining generic enough for other engines. Implementations can decide which
    subset they support at runtime via :py:meth:`AbstractSatelliteSimulator.supported_capabilities`.
    """

    TLE_FILE = auto()
    TLE_INLINE = auto()
    ORBITAL_PARAMETERS = auto()
    DEFAULT_ORBIT = auto()


@dataclass(frozen=True)
class EphemerisSource:
    """Defines where orbital position information should be retrieved from."""

    source_type: EphemerisSourceType
    tle_filename: Optional[str] = None
    tle_records: Optional[Sequence[str]] = None
    orbital_params: Optional[Mapping[str, float]] = None
    default_orbit_profile: Optional[str] = None

    def describe(self) -> str:
        """Return a human friendly description useful for logging or telemetry."""

        if self.source_type is EphemerisSourceType.TLE_FILE:
            return f"TLE file: {self.tle_filename or 'unspecified'}"
        if self.source_type is EphemerisSourceType.TLE_INLINE:
            count = len(self.tle_records or [])
            return f"Inline TLE records ({count} lines)"
        if self.source_type is EphemerisSourceType.ORBITAL_PARAMETERS:
            keys = ", ".join(sorted((self.orbital_params or {}).keys()))
            return f"Orbital parameters: {keys or 'empty'}"
        if self.source_type is EphemerisSourceType.DEFAULT_ORBIT:
            return f"Default orbit: {self.default_orbit_profile or 'unspecified'}"
        return "Unknown ephemeris source"


@dataclass(frozen=True)
class SatellitePlatform:
    """A uniquely identifiable satellite with an associated ephemeris."""

    id: str
    display_name: str
    ephemeris: EphemerisSource
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class GroundSite:
    """Represents a ground station, user cluster, or point of interest."""

    id: str
    latitude_deg: float
    longitude_deg: float
    altitude_km: float = 0.0
    coverage_radius_km: float = 0.0
    supported_constellations: Optional[Set[str]] = None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Constellation:
    """A collection of satellites that are simulated together."""

    id: str
    name: str
    satellites: Sequence[SatellitePlatform]
    description: str = ""


@dataclass(frozen=True)
class TrafficDemand:
    """Describes expected traffic between two points for routing studies."""

    source_site_id: str
    destination_site_id: str
    megabits_per_second: float


@dataclass(frozen=True)
class SimulationScenario:
    """High-level scenario shared by all simulator backends."""

    constellations: Sequence[Constellation]
    ground_sites: Sequence[GroundSite]
    start_time: datetime
    duration: timedelta
    time_step: timedelta
    traffic_matrix: Sequence[TrafficDemand] = field(default_factory=list)
    min_elevation_deg: float = 0.0

    def total_satellites(self) -> int:
        """Return the total number of satellites across all constellations."""

        return sum(len(constellation.satellites) for constellation in self.constellations)


class SimulatorCapability(Enum):
    """Capabilities that a simulator backend may advertise."""

    COVERAGE_ANALYSIS = auto()
    ROUTING_GRAPH = auto()
    LINK_BUDGET = auto()
    EPHEMERIS_EXPORT = auto()
    DOPPLER_REPORT = auto()
    TRAFFIC_SCHEDULING = auto()


@dataclass
class CoverageSnapshot:
    """Visibility and link-quality information for a specific instant."""

    timestamp: datetime
    visible_satellites: Mapping[str, List[str]]
    signal_to_noise: Mapping[str, float]


@dataclass
class RoutingGraph:
    """Lightweight graph representation produced by a simulator backend."""

    nodes: Set[str]
    edges: Mapping[str, Mapping[str, float]]

    def neighbors(self, node_id: str) -> Mapping[str, float]:
        """Return neighbor weights with graceful handling of missing nodes."""

        return self.edges.get(node_id, {})


class AbstractSatelliteSimulator(ABC):
    """Base class for integrating external simulators such as ltesat.

    Concrete implementations should translate :class:`SimulationScenario`
    objects into simulator-specific configuration files or API calls. The base
    class remains intentionally narrow; it focuses on the cross-cutting tasks
    needed by routing and planning experiments without exposing low-level
    simulator details. Implementations are encouraged to be stateless and
    reusable to keep coupling low.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human readable simulator name used for logging and provenance."""

    @abstractmethod
    def supported_capabilities(self) -> Set[SimulatorCapability]:
        """Return capabilities the backend can provide for a given scenario."""

    @abstractmethod
    def validate(self, scenario: SimulationScenario) -> None:
        """Raise a ``ValueError`` if the scenario is not supported.

        Validation should cover ephemeris options, time windows, and ground site
        geometry. Implementations may offer deeper checks (e.g., whether ltesat
        can parse all ephemerides) but should avoid mutating the input scenario.
        """

    @abstractmethod
    def plan_constellation(self, scenario: SimulationScenario) -> Sequence[SatellitePlatform]:
        """Return a refined set of satellites suitable for coverage planning."""

    @abstractmethod
    def compute_coverage(self, scenario: SimulationScenario) -> Iterable[CoverageSnapshot]:
        """Yield coverage snapshots across the requested time window."""

    @abstractmethod
    def build_routing_graph(self, scenario: SimulationScenario) -> RoutingGraph:
        """Produce a routing graph usable by path-finding algorithms.

        The resulting graph can be fed into the existing Dijkstra and distance
        vector engines in this repository by converting node identifiers to
        :class:`nodes.Node` objects elsewhere.
        """


class AbstractEphemerisExporter(ABC):
    """Interface for exporting ephemeris data from a simulator backend."""

    @abstractmethod
    def export_ephemeris(self, satellites: Sequence[SatellitePlatform], *, export_time: Optional[datetime] = None) -> str:
        """Return serialized ephemeris suitable for persistence or logging."""


class AbstractRoutingPlanner(ABC):
    """Planner interface that delegates graph construction to a simulator."""

    simulator: AbstractSatelliteSimulator

    @abstractmethod
    def build_graph(self, scenario: SimulationScenario) -> RoutingGraph:
        """Return the routing graph for downstream algorithms (e.g., Dijkstra)."""

    @abstractmethod
    def shortest_paths(self, graph: RoutingGraph, sources: Sequence[str]) -> Mapping[str, Mapping[str, float]]:
        """Return per-source shortest path costs keyed by destination node id."""
