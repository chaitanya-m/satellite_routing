"""Base interfaces for satellite simulator adapters.

The intent is to separate experiment logic from simulator-specific details. The
interfaces here are intentionally verbose and well documented so both humans
and automated agents can implement adapters for tools like ``ltesat`` or custom
mission control engines without coupling the rest of the codebase to any one
engine.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, Sequence

from .models import (
    Capability,
    ConstellationDefinition,
    ConstellationPlanningRequest,
    ConstellationPlanningResult,
    CoverageSimulationRequest,
    CoverageSimulationResult,
    GroundNetworkDefinition,
    RoutingRequest,
    RoutingResult,
    SimulationEnvironment,
    SimulationScenario,
)


@dataclass(frozen=True)
class BackendScenarioHandle:
    """Lightweight identifier used by backends to track prepared scenarios.

    Backends are encouraged to pre-process inputs (e.g., convert a constellation
    into ltesat configuration files) and return a handle that can later be used
    for repeated routing or coverage queries without reloading configuration
    files. The payload is intentionally opaque to keep orchestration decoupled.
    """

    backend_name: str
    handle: str
    metadata: Mapping[str, str]


class SatelliteSimulationBackend(ABC):
    """Abstract base class for all simulator adapters.

    Implementations should be side-effect free where possible and prefer pure
    data inputs. Methods are separated by concern so experiment runners can
    orchestrate visibility, routing, planning, and dimensioning workflows
    independently.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name (e.g., ``"ltesat"``)."""

    @property
    @abstractmethod
    def capabilities(self) -> Sequence[Capability]:
        """Advertise supported features so orchestrators can route requests."""

    @abstractmethod
    def configure_environment(self, environment: SimulationEnvironment) -> None:
        """Apply global settings for future runs.

        Backends should ignore unknown keys in ``environment`` to preserve
        forward compatibility. This call is expected to be idempotent.
        """

    @abstractmethod
    def prepare_scenario(
        self,
        constellation: ConstellationDefinition,
        ground_network: GroundNetworkDefinition,
        scenario: SimulationScenario,
    ) -> BackendScenarioHandle:
        """Pre-compute backend artefacts for the given scenario."""

    @abstractmethod
    def run_coverage(
        self, request: CoverageSimulationRequest
    ) -> CoverageSimulationResult:
        """Compute visibility/coverage metrics for the given request."""

    @abstractmethod
    def run_routing(self, request: RoutingRequest) -> RoutingResult:
        """Solve a routing query using the backend's graph or link metrics."""

    @abstractmethod
    def plan_constellation(
        self, request: ConstellationPlanningRequest
    ) -> ConstellationPlanningResult:
        """Return constellation design suggestions given high-level goals."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release any backend resources (files, processes, caches)."""


class SimulationOrchestrator:
    """Facade that coordinates multiple backends.

    The orchestrator allows experiments to pick the best backend for each
    activity (e.g., use ``ltesat`` for precise visibility while using a
    lightweight in-memory graph for routing stress tests). It also demonstrates
    how adapters are expected to be consumed by higher-level logic.
    """

    def __init__(self, backends: Mapping[str, SatelliteSimulationBackend]):
        self.backends = dict(backends)

    def get_backend(self, capability: Capability) -> SatelliteSimulationBackend:
        """Select a backend that declares the requested capability."""

        for backend in self.backends.values():
            if capability in backend.capabilities:
                return backend
        raise ValueError(f"No backend available for capability: {capability}")

    def configure_all(self, environment: SimulationEnvironment) -> None:
        """Apply environment configuration to all registered backends."""

        for backend in self.backends.values():
            backend.configure_environment(environment)

    def prepare_all(
        self,
        constellation: ConstellationDefinition,
        ground_network: GroundNetworkDefinition,
        scenario: SimulationScenario,
    ) -> Mapping[str, BackendScenarioHandle]:
        """Prepare every backend for repeated use in a scenario."""

        handles = {}
        for name, backend in self.backends.items():
            handles[name] = backend.prepare_scenario(
                constellation=constellation,
                ground_network=ground_network,
                scenario=scenario,
            )
        return handles

    def shutdown(self) -> None:
        """Shutdown all backends gracefully."""

        for backend in self.backends.values():
            backend.shutdown()
