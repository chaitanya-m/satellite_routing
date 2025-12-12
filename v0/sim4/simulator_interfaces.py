"""High-level interfaces for plug-and-play satellite simulators.

The module defines abstraction layers to decouple the routing/analysis stack
from any particular satellite simulator (e.g., ltesat or future engines). The
interfaces are intended to support multiple workflows:

* **Routing studies** driven by graph algorithms such as Dijkstra's algorithm.
* **Planning and dimensioning** tasks that search for constellation layouts
  that maximize coverage, visibility, and signal strength.

The design is intentionally lightweight so that production-grade adapters can
wrap external binaries, while in-memory/mock implementations can be used for
unit tests and deterministic experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Iterable, Iterator, Mapping, MutableMapping, Sequence
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class TimeWindow:
    """Defines when and how a simulation should run.

    Attributes
    ----------
    start : datetime
        Start time of the simulation window (inclusive).
    end : datetime
        End time of the simulation window (exclusive).
    step : timedelta
        Step used by the simulator to sample positions/links. The value can be
        larger than the routing cadence; adapters are expected to interpolate
        if needed.
    """

    start: datetime
    end: datetime
    step: timedelta = field(default=timedelta(seconds=60))

    def validate(self) -> None:
        """Validate monotonicity and positivity of the window.

        Raises
        ------
        ValueError
            If ``end`` is not strictly greater than ``start`` or if ``step`` is
            non-positive.
        """

        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        if self.step.total_seconds() <= 0:
            raise ValueError("step must be positive")


@dataclass(frozen=True)
class GroundPoint:
    """Represents a ground asset to observe or serve.

    Fields mirror the kinds of inputs described in the ltesat API specs while
    remaining generic enough for other engines.
    """

    id: str
    latitude_deg: float
    longitude_deg: float
    altitude_m: float = 0.0


@dataclass(frozen=True)
class SatelliteBody:
    """Describes a satellite used in the scenario.

    The ``ephemeris_hint`` is intentionally unstructured so that adapters can
    map it to simulator-specific formats (e.g., TLE content for ltesat or an
    orbital-parameter payload for other tools).
    """

    id: str
    display_name: str
    ephemeris_hint: Mapping[str, object]
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationScenario:
    """Static description of the constellation and service targets."""

    satellites: Sequence[SatelliteBody]
    ground_points: Sequence[GroundPoint]
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class LinkBudget:
    """Represents link-quality attributes used by routing and planning."""

    latency_ms: float
    capacity_mbps: float
    packet_error_rate: float = 0.0
    received_power_dbm: float | None = None


@dataclass(frozen=True)
class LinkSnapshot:
    """Connectivity edge between two nodes at a given time."""

    source: str
    target: str
    budget: LinkBudget


@dataclass(frozen=True)
class CoverageMetric:
    """Per-ground-point coverage output returned by simulators."""

    ground_point_id: str
    visible_satellites: Sequence[str]
    coverage_probability: float
    average_snr_db: float | None = None


@dataclass(frozen=True)
class SimulationSnapshot:
    """Time-aligned snapshot returned by a simulator plugin."""

    timestamp: datetime
    links: Sequence[LinkSnapshot]
    coverage: Sequence[CoverageMetric]


@dataclass(frozen=True)
class RoutingGraph:
    """Graph view derived from a snapshot.

    The structure is kept simple so that routing algorithms in this repository
    (e.g., Dijkstra) can consume it without depending on the underlying
    simulator. Implementations should provide symmetric edges where applicable
    and include self-loops only when meaningful for the routing policy.
    """

    adjacency: Mapping[str, Mapping[str, LinkBudget]]

    @classmethod
    def from_links(cls, links: Iterable[LinkSnapshot]) -> "RoutingGraph":
        """Create a graph from link snapshots."""

        adjacency: MutableMapping[str, Dict[str, LinkBudget]] = {}
        for link in links:
            adjacency.setdefault(link.source, {})[link.target] = link.budget
        return cls(adjacency=adjacency)


@runtime_checkable
class SimulationPlugin(Protocol):
    """Generic contract for satellite simulators.

    A plugin is responsible for translating :class:`SimulationScenario`
    objects into simulator-native configuration (e.g., ltesat JSON) and
    streaming :class:`SimulationSnapshot` instances for routing and planning
    routines.
    """

    def describe(self) -> Mapping[str, str]:
        """Return metadata about the plugin (name, version, supported features)."""

    def prepare(self, scenario: SimulationScenario) -> None:
        """Load or generate any simulator configuration required for execution."""

    def run(self, window: TimeWindow) -> Iterator[SimulationSnapshot]:
        """Execute the simulation and yield snapshots across the time window."""


@runtime_checkable
class RoutingGraphFactory(Protocol):
    """Converts raw simulation snapshots into routing-friendly graphs."""

    def build_graph(self, snapshot: SimulationSnapshot) -> RoutingGraph:
        """Construct a routing graph for algorithms such as Dijkstra."""


@runtime_checkable
class PlanningAdvisor(Protocol):
    """Provides hooks for planning and dimensioning workflows."""

    def suggest_positions(
        self, scenario: SimulationScenario, objective: str = "coverage"
    ) -> Sequence[SatelliteBody]:
        """Propose satellite placements tuned for coverage or signal intensity."""


class LteSatSimulationAdapter:
    """Example adapter skeleton for the ltesat toolchain.

    The implementation details are intentionally omitted; the adapter serves as
    an integration guide showing how the :class:`SimulationPlugin` contract can
    wrap an external binary while remaining compatible with the routing stack.
    """

    def __init__(self, ltesat_binary: str = "ltesat") -> None:
        self.ltesat_binary = ltesat_binary

    def describe(self) -> Mapping[str, str]:
        return {
            "name": "ltesat",
            "executable": self.ltesat_binary,
            "supports": "coverage, routing, pass prediction",
        }

    def prepare(self, scenario: SimulationScenario) -> None:  # pragma: no cover
        raise NotImplementedError("Adapter must author ltesat configs and assets")

    def run(self, window: TimeWindow) -> Iterator[SimulationSnapshot]:  # pragma: no cover
        raise NotImplementedError("Adapter must invoke ltesat and parse outputs")


class InMemorySimulator:
    """Minimal simulator useful for tests and examples.

    The simulator emits a fully connected mesh across all satellites and ground
    points at each timestep, with synthetic link budgets derived from the time
    offset. Coverage is also reported deterministically.
    """

    def __init__(self, scenario: SimulationScenario) -> None:
        self.scenario = scenario

    def describe(self) -> Mapping[str, str]:
        return {"name": "in-memory", "supports": "routing"}

    def prepare(self, scenario: SimulationScenario) -> None:
        self.scenario = scenario

    def run(self, window: TimeWindow) -> Iterator[SimulationSnapshot]:
        window.validate()
        current = window.start
        index = 0
        while current < window.end:
            links: list[LinkSnapshot] = []
            for sat in self.scenario.satellites:
                for ground in self.scenario.ground_points:
                    budget = LinkBudget(
                        latency_ms=10 + index,
                        capacity_mbps=100 - index,
                        packet_error_rate=0.01 * index,
                        received_power_dbm=-90 + index,
                    )
                    links.append(
                        LinkSnapshot(
                            source=sat.id,
                            target=ground.id,
                            budget=budget,
                        )
                    )
            coverage = [
                CoverageMetric(
                    ground_point_id=ground.id,
                    visible_satellites=[sat.id for sat in self.scenario.satellites],
                    coverage_probability=1.0,
                    average_snr_db=20.0,
                )
                for ground in self.scenario.ground_points
            ]
            yield SimulationSnapshot(timestamp=current, links=links, coverage=coverage)
            current += window.step
            index += 1


class DefaultRoutingGraphFactory:
    """Build routing graphs directly from :class:`LinkSnapshot` objects."""

    def build_graph(self, snapshot: SimulationSnapshot) -> RoutingGraph:
        return RoutingGraph.from_links(snapshot.links)
