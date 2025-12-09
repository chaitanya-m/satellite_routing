"""Placeholder test suite for simulator interface primitives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pytest

from simulators import (
    BackendScenarioHandle,
    Capability,
    ConstellationDefinition,
    ConstellationPlanningRequest,
    ConstellationPlanningResult,
    CoverageSimulationRequest,
    CoverageSimulationResult,
    GroundNetworkDefinition,
    RoutingRequest,
    RoutingResult,
    SatelliteSimulationBackend,
    SimulationEnvironment,
    SimulationOrchestrator,
    SimulationScenario,
    SimulationTimeWindow,
)


@dataclass(frozen=True)
class _NoOpBackend(SatelliteSimulationBackend):
    name: str = "noop"
    capabilities: Sequence[Capability] = (Capability.VISIBILITY, Capability.ROUTING)

    def configure_environment(self, environment: SimulationEnvironment) -> None:
        # This backend ignores configuration by design.
        return None

    def prepare_scenario(
        self,
        constellation: ConstellationDefinition,
        ground_network: GroundNetworkDefinition,
        scenario: SimulationScenario,
    ) -> BackendScenarioHandle:
        return BackendScenarioHandle(
            backend_name=self.name,
            handle=f"scenario:{scenario.name}",
            metadata={"constellation": constellation.name},
        )

    def run_coverage(
        self, request: CoverageSimulationRequest
    ) -> CoverageSimulationResult:
        return CoverageSimulationResult(visibility=(), link_budgets=(), metadata={})

    def run_routing(self, request: RoutingRequest) -> RoutingResult:
        return RoutingResult(path=[request.source, request.destination], total_cost=0.0, hops=1)

    def plan_constellation(
        self, request: ConstellationPlanningRequest
    ) -> ConstellationPlanningResult:
        return ConstellationPlanningResult(
            constellation=ConstellationDefinition(
                name="noop", satellites=(), ephemeris_source=request.constraints.get("source", None)
            ),
            expected_coverage={},
            expected_capacity_bps=None,
        )

    def shutdown(self) -> None:
        return None


def test_orchestrator_picks_backend_by_capability():
    orchestrator = SimulationOrchestrator({"noop": _NoOpBackend()})
    backend = orchestrator.get_backend(Capability.VISIBILITY)
    assert backend.name == "noop"


def test_prepare_all_returns_handles():
    orchestrator = SimulationOrchestrator({"noop": _NoOpBackend()})
    window = SimulationTimeWindow(start_seconds=0, end_seconds=1)
    scenario = SimulationScenario(name="demo", time_window=window, goal="coverage")
    handles = orchestrator.prepare_all(
        constellation=ConstellationDefinition(name="demo", satellites=(), ephemeris_source=None),
        ground_network=GroundNetworkDefinition(sites=()),
        scenario=scenario,
    )
    assert "noop" in handles
    assert handles["noop"].handle.startswith("scenario:demo")


def test_missing_backend_raises_value_error():
    orchestrator = SimulationOrchestrator({})
    with pytest.raises(ValueError):
        orchestrator.get_backend(Capability.ROUTING)
