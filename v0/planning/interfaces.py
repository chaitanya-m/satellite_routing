"""Domain-agnostic interfaces for planning and routing problems.

Planning answers the question:

    *Given a set of deployed resources and a time-varying topology, how
    should we route or schedule demand to achieve our objectives?*

These types are neutral and designed to sit above simulator or telemetry
layers. Satellite networks, data centres, road systems, and other graphs
can all be mapped into :class:`TopologySnapshot` instances for analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterable, Mapping, MutableMapping, Protocol, Sequence


@dataclass(frozen=True)
class Node:
    """A node in a planning graph.

    Nodes represent endpoints that can originate, carry, or terminate demand.
    Concrete domains may interpret nodes as routers, satellites, servers,
    intersections, or any other logical entity.
    """

    id: str
    attributes: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    """Directed edge between two nodes with associated attributes."""

    source: str
    target: str
    weight: float
    attributes: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TopologySnapshot:
    """Snapshot of a graph at a given planning instant.

    Implementations may choose to carry an explicit timestamp in
    :pyattr:`metadata` (e.g., ``{"timestamp": "..."}``) when representing
    time-varying networks.
    """

    nodes: Sequence[Node]
    edges: Sequence[Edge]
    metadata: Mapping[str, object] = field(default_factory=dict)


class RoutingObjective(Enum):
    """Canonical routing objectives understood by common planners."""

    SHORTEST_PATH = auto()
    MINIMISE_LATENCY = auto()
    MAXIMISE_THROUGHPUT = auto()
    BALANCE_LOAD = auto()


@dataclass(frozen=True)
class RoutingDemand:
    """Describes demand between two endpoints."""

    source: str
    destination: str
    volume: float = 1.0
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanningRequest:
    """Request for planning or routing over a topology snapshot."""

    topology: TopologySnapshot
    demands: Sequence[RoutingDemand]
    objective: RoutingObjective = RoutingObjective.SHORTEST_PATH
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Path:
    """Represents a single path between two nodes."""

    nodes: Sequence[str]
    total_weight: float
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanningResult:
    """Aggregated result of a planning or routing computation."""

    paths: Mapping[tuple[str, str], Path]
    link_utilisation: Mapping[tuple[str, str], float] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)


class PlanningBackend(Protocol):
    """Protocol implemented by planning and routing engines."""

    def describe(self) -> Mapping[str, str]:
        """Return backend metadata such as name and supported objectives."""

        ...

    def plan(self, request: PlanningRequest) -> PlanningResult:
        """Compute paths and utilisation metrics for the given request."""

        ...


class PlanningRegistry:
    """Simple registry managing multiple planning backends."""

    def __init__(self) -> None:
        self._backends: MutableMapping[str, PlanningBackend] = {}

    def register(self, name: str, backend: PlanningBackend) -> None:
        """Register ``backend`` under ``name``."""

        if name in self._backends:
            raise ValueError(f"Planning backend '{name}' is already registered.")
        self._backends[name] = backend

    def get(self, name: str) -> PlanningBackend:
        """Return the backend registered under ``name``."""

        return self._backends[name]

    def all(self) -> Mapping[str, PlanningBackend]:
        """Return a read-only copy of all registered backends."""

        return dict(self._backends)
