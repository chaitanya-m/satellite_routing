"""Domain-agnostic interfaces for dimensioning problems.

Dimensioning answers the question:

    *Given a set of candidate resources and an expected demand profile,
    which resources should we deploy so that objectives and constraints
    are satisfied?*

These interfaces are neutral and can be used in any setting where resources must 
be selected or sized before operations and routing are considered 
(e.g. satellite constellations, data centre capacity planning, logistics networks).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Mapping, MutableMapping, Protocol, Sequence


class ObjectiveKind(Enum):
    """Canonical objectives a dimensioning backend may support.

    Backends are free to support additional objectives by declaring them in
    :class:`DimensioningRequest.metadata` or by using custom strings, but
    these values provide a small common vocabulary for orchestrators.
    """

    MINIMISE_COST = auto()
    MAXIMISE_COVERAGE = auto()
    MAXIMISE_CAPACITY = auto()
    BALANCE_COST_AND_PERFORMANCE = auto()


@dataclass(frozen=True)
class ResourceCandidate:
    """A resource that may be deployed as part of a solution.

    Examples of resources include satellites, ground stations, servers, links,
    charging points, windmills, vehicles. Domain specific code can interpret arbitrary 
    attributes via the ``attributes`` mapping without requiring this module to change.
    """

    id: str
    attributes: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DemandProfile:
    """Describes demand that the dimensioned system should be able to serve.

    The profile is abstract. In a satellite context it might
    represent geographic traffic distributions; in a compute context it could
    capture workload mixes. The exact schema is left to higher-level code.
    """

    description: str
    parameters: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DimensioningConstraints:
    """Constraints that candidate solutions must satisfy.

    Typical constraints include budget limits, maximum resource counts, or
    regulatory and policy restrictions. The ``limits`` mapping allows
    experiments to define arbitrary scalar or structured constraints.
    """

    limits: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DimensioningRequest:
    """Aggregates input data for a dimensioning run.

    Attributes:
        candidates:
            The universe of resources that could be deployed.
        demand:
            A description of demand the dimensioned system is expected to
            serve. This does not need to be exhaustive; backends may combine
            it with historical measurements or internal models to enrich the 
            demand picture before solving the problem.
        constraints:
            Hard constraints that a solution must satisfy.
        objective:
            High-level optimisation objective guiding the search.
        metadata:
            Free-form information that backends may use for logging, tracing,
            or experiment tracking.
    """

    candidates: Sequence[ResourceCandidate]
    demand: DemandProfile
    constraints: DimensioningConstraints = field(default_factory=DimensioningConstraints)
    objective: ObjectiveKind = ObjectiveKind.MAXIMISE_COVERAGE
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DimensioningKPI:
    """Key performance indicators associated with a dimensioning outcome."""

    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DimensioningResult:
    """Output of a dimensioning backend.

    Attributes:
        selected_resources:
            The subset of input candidates chosen for deployment. Backends may
            choose to annotate these with additional attributes using
            :class:`ResourceCandidate.attributes`.
        kpi:
            Quantitative assessment of the solution quality. The exact metrics
            are domain specific (e.g., coverage percentages, expected blocking
            (request rejection) probability, cost).
        metadata:
            Additional backend-specific details such as optimiser status or
            convergence diagnostics.
    """

    selected_resources: Sequence[ResourceCandidate]
    kpi: DimensioningKPI
    metadata: Mapping[str, str] = field(default_factory=dict)


class DimensioningBackend(Protocol):
    """Protocol implemented by all dimensioning engines.

    Concrete implementations are responsible for translating neutral requests
    into domain-specific optimisation problems. For example, a satellite
    backend may couple these types to constellation design variables, whereas
    a compute backend might interact with a capacity planner.
    """

    def describe(self) -> Mapping[str, str]:
        """Return human-readable backend metadata."""

        ...

    def optimise(self, request: DimensioningRequest) -> DimensioningResult:
        """Solve the dimensioning problem encoded in ``request``."""

        ...


class DimensioningRegistry:
    """Lightweight registry for available dimensioning backends.

    This helps orchestrators and experiments remain loosely coupled to concrete
    optimisation engines. Registries are intentionally simple and kept in this
    module to avoid pulling in heavier plugin frameworks.
    """

    def __init__(self) -> None:
        self._backends: MutableMapping[str, DimensioningBackend] = {}

    def register(self, name: str, backend: DimensioningBackend) -> None:
        """Register ``backend`` under ``name``.

        If a backend is already registered under the same name this method
        raises :class:`ValueError`.
        """

        if name in self._backends:
            raise ValueError(f"Dimensioning backend '{name}' is already registered.")
        self._backends[name] = backend

    def get(self, name: str) -> DimensioningBackend:
        """Return the backend registered under ``name``."""

        return self._backends[name]

    def all(self) -> Mapping[str, DimensioningBackend]:
        """Return a read-only view of all registered backends."""

        return dict(self._backends)
