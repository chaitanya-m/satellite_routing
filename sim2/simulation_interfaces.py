"""Abstract interfaces for satellite communications simulators.

This module defines reusable, well-documented interfaces that allow the
application to plug in different simulation backends (e.g., ltesat or other
mission-control and coverage simulators). The goal is to separate high-level
routing, planning, and dimensioning logic from the concrete simulator
implementation so researchers can swap engines without rewriting business
logic.

The interfaces here are informed by the public ltesat integration materials
under ``docs/`` while remaining backend-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Protocol, Sequence
from typing import Tuple


class EphemerisSourceType(Enum):
    """Supported sources for satellite ephemeris data.

    The options mirror the inputs commonly accepted by tools like ltesat while
    allowing other simulators to supply similar data in their native formats.
    """

    TLE_FILE = auto()
    TLE_INLINE = auto()
    ORBITAL_PARAMETERS = auto()
    DEFAULT_ORBIT = auto()


@dataclass(frozen=True)
class SatelliteEphemeris:
    """Description of how satellite position and motion are provided.

    Attributes
    ----------
    source_type:
        The origin of the orbital information (e.g., TLE file, inline TLE
        lines, direct orbital elements, or a built-in orbit template).
    tle_filename:
        Optional path to a TLE file when ``source_type`` is ``TLE_FILE``.
    tle_records:
        Optional list of TLE strings when ``source_type`` is ``TLE_INLINE``.
    orbital_parameters:
        Optional mapping of orbital elements (semi-major axis, inclination,
        eccentricity, RAAN, etc.) when ``source_type`` is
        ``ORBITAL_PARAMETERS``.
    default_orbit_profile:
        Optional identifier for a predefined orbit shape (e.g., ``"LEO_600KM"``)
        when ``source_type`` is ``DEFAULT_ORBIT``.
    """

    source_type: EphemerisSourceType
    tle_filename: Path | None = None
    tle_records: Sequence[str] | None = None
    orbital_parameters: Mapping[str, float] | None = None
    default_orbit_profile: str | None = None

    def validate(self) -> None:
        """Validate that the ephemeris description is internally consistent.

        Raises
        ------
        ValueError
            If required fields for the chosen ``source_type`` are missing or
            mutually exclusive fields are provided.
        """

        if self.source_type is EphemerisSourceType.TLE_FILE:
            if not self.tle_filename:
                raise ValueError("TLE file source requires 'tle_filename'.")
        elif self.source_type is EphemerisSourceType.TLE_INLINE:
            if not self.tle_records:
                raise ValueError("Inline TLE source requires 'tle_records'.")
        elif self.source_type is EphemerisSourceType.ORBITAL_PARAMETERS:
            if not self.orbital_parameters:
                raise ValueError(
                    "Orbital parameter source requires 'orbital_parameters'."
                )
        elif self.source_type is EphemerisSourceType.DEFAULT_ORBIT:
            if not self.default_orbit_profile:
                raise ValueError(
                    "Default orbit source requires 'default_orbit_profile'."
                )

        # Guard against conflicting multiple inputs.
        provided_fields = sum(
            bool(field)
            for field in [
                self.tle_filename,
                self.tle_records,
                self.orbital_parameters,
                self.default_orbit_profile,
            ]
        )
        if provided_fields > 1:
            raise ValueError(
                "Only one ephemeris detail should be provided for a satellite."
            )


@dataclass(frozen=True)
class SatelliteAsset:
    """Represents a single satellite that may participate in simulations."""

    identifier: str
    display_name: str
    ephemeris: SatelliteEphemeris
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class GroundSite:
    """Represents a ground station, user terminal, or area of interest."""

    identifier: str
    latitude_deg: float
    longitude_deg: float
    altitude_m: float = 0.0


@dataclass(frozen=True)
class SimulationWindow:
    """Time bounds for a simulation or coverage query."""

    start: datetime
    duration: timedelta

    @property
    def end(self) -> datetime:
        """Return the calculated end time for the window."""

        return self.start + self.duration


class RoutingAlgorithm(Enum):
    """Supported routing algorithms for simulator requests."""

    DIJKSTRA = auto()
    DISTANCE_VECTOR = auto()
    CUSTOM = auto()


@dataclass(frozen=True)
class RoutingRequest:
    """Parameters describing a routing study over a constellation network."""

    algorithm: RoutingAlgorithm
    source_node: str
    destination_node: str
    time_window: SimulationWindow | None = None
    constraints: Mapping[str, float] | None = None


@dataclass(frozen=True)
class RoutingResult:
    """Simplified routing result returned by a simulator."""

    path: Sequence[str]
    total_cost: float
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CoverageRequest:
    """Describe a coverage or visibility computation.

    This object is designed to be compatible with ltesat-like engines that
    accept ephemeris, ground sites, and time windows to compute visibility,
    link budgets, or pass tables.
    """

    satellites: Sequence[SatelliteAsset]
    ground_sites: Sequence[GroundSite]
    window: SimulationWindow
    elevation_mask_deg: float = 0.0


@dataclass(frozen=True)
class CoverageResult:
    """Summary of coverage metrics produced by a simulator."""

    visibility: Mapping[Tuple[str, str], float]
    best_links: Mapping[str, str]
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DimensioningRequest:
    """Parameters for planning/dimensioning of satellite placement or beams."""

    target_area: Sequence[GroundSite]
    candidate_satellites: Sequence[SatelliteAsset]
    optimization_goal: str
    constraints: Mapping[str, float] | None = None


@dataclass(frozen=True)
class DimensioningResult:
    """Result describing a recommended constellation or beam plan."""

    selected_satellites: Sequence[str]
    coverage_score: float
    signal_intensity_score: float
    metadata: Mapping[str, str] = field(default_factory=dict)


class SimulatorAdapter(Protocol):
    """Common protocol for all simulator plugins.

    Implementations translate the neutral request objects above into concrete
    simulator invocations (e.g., writing ltesat config files or calling another
    vendor's API) and return structured results.
    """

    name: str

    def healthcheck(self) -> bool:
        """Return ``True`` if the adapter is ready to run simulations."""

    def capabilities(self) -> Iterable[str]:
        """Return human-readable capability identifiers."""


class CoverageSimulator(SimulatorAdapter, Protocol):
    """Protocol for coverage/visibility simulators."""

    def compute_coverage(self, request: CoverageRequest) -> CoverageResult:
        """Compute visibility and link quality for the requested scenario."""


class RoutingSimulator(SimulatorAdapter, Protocol):
    """Protocol for routing simulators (e.g., Dijkstra over a dynamic graph)."""

    def compute_route(self, request: RoutingRequest) -> RoutingResult:
        """Compute a path between two nodes according to ``request``."""


class DimensioningSimulator(SimulatorAdapter, Protocol):
    """Protocol for dimensioning/planning simulators."""

    def optimize_constellation(
        self, request: DimensioningRequest
    ) -> DimensioningResult:
        """Return placement or selection recommendations for satellites."""


class SimulationPluginRegistry:
    """Registry that tracks available simulator adapters.

    The registry keeps simulators loosely coupled to the rest of the codebase;
    call-sites can request a capability (coverage, routing, dimensioning) and
    receive the first adapter that advertises support.
    """

    def __init__(self) -> None:
        self._adapters: MutableMapping[str, SimulatorAdapter] = {}

    def register(self, adapter: SimulatorAdapter) -> None:
        """Register a simulator adapter by its declared name."""

        if adapter.name in self._adapters:
            raise ValueError(f"Adapter with name '{adapter.name}' already registered.")
        self._adapters[adapter.name] = adapter

    def get(self, name: str) -> SimulatorAdapter:
        """Return an adapter by name, raising ``KeyError`` if missing."""

        return self._adapters[name]

    def find_by_capability(self, capability: str) -> List[SimulatorAdapter]:
        """Return all adapters that advertise the given capability."""

        return [
            adapter
            for adapter in self._adapters.values()
            if capability in set(adapter.capabilities())
        ]

    def __iter__(self):
        return iter(self._adapters.values())
