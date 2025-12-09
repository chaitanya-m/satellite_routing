"""Shared data structures for pluggable satellite simulation backends.

The interfaces here are deliberately backend-agnostic so that engines such as
``ltesat`` (see ``docs/ltesat_api_integration_spec.txt``) or future custom
propagation models can be integrated behind a stable set of types.  Each class
includes rich docstrings to ensure requirements are clear to adapter authors
and to keep experiment code uncluttered by simulator-specific details.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class Capability(Enum):
    """Features that a simulator backend may advertise.

    Backends should expose which capabilities they support so orchestration
    logic can route work to the best available engine (e.g., ``ltesat`` for
    orbital visibility, a custom RF tool for link budget refinement, or a
    lightweight graph-only simulator for unit tests).
    """

    VISIBILITY = auto()
    ROUTING = auto()
    COVERAGE = auto()
    LINK_BUDGET = auto()
    CONSTELLATION_PLANNING = auto()
    DIMENSIONING = auto()


class EphemerisSource(Enum):
    """Describes how satellite trajectories are obtained.

    Inspired by the ltesat integration spec to keep adapters compatible while
    allowing other engines to publish their own ephemeris sources.
    """

    TLE_FILE = auto()
    TLE_INLINE = auto()
    ORBITAL_PARAMETERS = auto()
    DEFAULT_ORBIT = auto()


@dataclass(frozen=True)
class SimulationTimeWindow:
    """Time interval for a simulation request.

    Attributes:
        start_seconds: Epoch seconds for the start of the analysis window.
        end_seconds: Epoch seconds for the end of the analysis window.
        step_seconds: Optional sampling interval for time series style outputs.
    """

    start_seconds: float
    end_seconds: float
    step_seconds: Optional[float] = None


@dataclass(frozen=True)
class SimulationEnvironment:
    """Global settings that influence all simulator runs.

    Examples include atmospheric models, propagation constants, or Earth
    orientation parameters. Backends should ignore fields they do not
    understand rather than failing, to preserve loose coupling.
    """

    parameters: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LinkIdentifier:
    """Uniquely identifies a communication link between two nodes."""

    source_id: str
    destination_id: str
    link_type: str = "SATELLITE"


@dataclass(frozen=True)
class LinkBudget:
    """Summarises RF performance for a single link sample."""

    link: LinkIdentifier
    received_power_dbw: float
    snr_db: float
    capacity_bps: Optional[float] = None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class VisibilitySample:
    """Snapshot of line-of-sight and geometry between a satellite and a site."""

    link: LinkIdentifier
    time_seconds: float
    elevation_deg: float
    azimuth_deg: float
    range_km: float
    is_visible: bool
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RoutingRequest:
    """Describes a routing problem to solve using simulator outputs."""

    source: str
    destination: str
    graph_edges: Sequence[Tuple[str, str, float]]
    algorithm: str = "dijkstra"
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RoutingResult:
    """Represents a routing path and metrics derived from a simulator."""

    path: List[str]
    total_cost: float
    hops: int
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ConstellationDefinition:
    """Minimal description of satellites and their ephemeris inputs."""

    name: str
    satellites: Sequence[Mapping[str, object]]
    ephemeris_source: Optional[EphemerisSource]
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class GroundNetworkDefinition:
    """Describes ground nodes or gateways participating in scenarios."""

    sites: Sequence[Mapping[str, object]]
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationScenario:
    """Scenario definition combining network and time constraints."""

    name: str
    time_window: SimulationTimeWindow
    goal: str
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CoverageSimulationRequest:
    """Request object for coverage and visibility studies."""

    constellation: ConstellationDefinition
    ground_network: GroundNetworkDefinition
    scenario: SimulationScenario
    environment: SimulationEnvironment
    samples_per_site: Optional[int] = None


@dataclass(frozen=True)
class CoverageSimulationResult:
    """Results of coverage or visibility studies."""

    visibility: Iterable[VisibilitySample]
    link_budgets: Iterable[LinkBudget]
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ConstellationPlanningRequest:
    """Request for planning or dimensioning exercises.

    This keeps planning inputs (candidate plane counts, altitude families, or
    optimisation weights) opaque so backends can offer their own semantics.
    """

    constraints: Mapping[str, object] = field(default_factory=dict)
    objective: str = "maximize_coverage"
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ConstellationPlanningResult:
    """Proposed constellation layout and expected KPIs."""

    constellation: ConstellationDefinition
    expected_coverage: Mapping[str, float]
    expected_capacity_bps: Optional[float] = None
    metadata: Mapping[str, str] = field(default_factory=dict)
