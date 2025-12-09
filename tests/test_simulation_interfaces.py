"""Placeholder tests for the simulation interface definitions.

These tests exercise lightweight behaviors to make sure the abstract contracts
are usable by concrete simulator plugins without committing to a specific
backend implementation.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable

import pytest

from simulation_interfaces import (
    CoverageRequest,
    CoverageResult,
    CoverageSimulator,
    DimensioningRequest,
    DimensioningResult,
    DimensioningSimulator,
    EphemerisSourceType,
    GroundSite,
    RoutingAlgorithm,
    RoutingRequest,
    RoutingResult,
    RoutingSimulator,
    SatelliteAsset,
    SatelliteEphemeris,
    SimulationPluginRegistry,
    SimulationWindow,
)


class DummyCoverageSimulator:
    name = "dummy_coverage"

    def healthcheck(self) -> bool:
        return True

    def capabilities(self) -> Iterable[str]:
        return ["coverage", "visibility"]

    def compute_coverage(self, request: CoverageRequest) -> CoverageResult:
        visibility = {
            (request.satellites[0].identifier, request.ground_sites[0].identifier): 1.0
        }
        return CoverageResult(visibility=visibility, best_links={"best": "dummy"})


class DummyRoutingSimulator:
    name = "dummy_routing"

    def healthcheck(self) -> bool:
        return True

    def capabilities(self) -> Iterable[str]:
        return ["routing", "dijkstra"]

    def compute_route(self, request: RoutingRequest) -> RoutingResult:
        return RoutingResult(path=[request.source_node, request.destination_node], total_cost=1.0)


class DummyDimensioningSimulator:
    name = "dummy_dimensioning"

    def healthcheck(self) -> bool:
        return True

    def capabilities(self) -> Iterable[str]:
        return ["dimensioning"]

    def optimize_constellation(self, request: DimensioningRequest) -> DimensioningResult:
        return DimensioningResult(
            selected_satellites=[sat.identifier for sat in request.candidate_satellites],
            coverage_score=0.5,
            signal_intensity_score=0.5,
        )


def test_ephemeris_validation_inline_tle_passes():
    ephemeris = SatelliteEphemeris(
        source_type=EphemerisSourceType.TLE_INLINE,
        tle_records=["line1", "line2"],
    )
    ephemeris.validate()


def test_ephemeris_validation_conflict_raises():
    ephemeris = SatelliteEphemeris(
        source_type=EphemerisSourceType.TLE_FILE,
        tle_filename=None,
        tle_records=["line1", "line2"],
    )
    with pytest.raises(ValueError):
        ephemeris.validate()


def test_plugin_registry_registration_and_lookup():
    registry = SimulationPluginRegistry()
    coverage_adapter = DummyCoverageSimulator()
    registry.register(coverage_adapter)

    assert registry.get("dummy_coverage") is coverage_adapter
    assert registry.find_by_capability("coverage") == [coverage_adapter]


def test_dummy_routing_flow_matches_protocol():
    registry = SimulationPluginRegistry()
    routing_adapter: RoutingSimulator = DummyRoutingSimulator()
    registry.register(routing_adapter)

    request = RoutingRequest(
        algorithm=RoutingAlgorithm.DIJKSTRA,
        source_node="A",
        destination_node="B",
    )
    result = routing_adapter.compute_route(request)

    assert result.path == ["A", "B"]
    assert result.total_cost == 1.0


def test_dimensioning_simulator_placeholder():
    registry = SimulationPluginRegistry()
    dim_adapter: DimensioningSimulator = DummyDimensioningSimulator()
    registry.register(dim_adapter)

    request = DimensioningRequest(
        target_area=[GroundSite("g1", 0.0, 0.0)],
        candidate_satellites=[
            SatelliteAsset(
                identifier="sat-1",
                display_name="Sat 1",
                ephemeris=SatelliteEphemeris(
                    source_type=EphemerisSourceType.DEFAULT_ORBIT,
                    default_orbit_profile="LEO_600KM",
                ),
            )
        ],
        optimization_goal="maximize_coverage",
    )

    result = dim_adapter.optimize_constellation(request)
    assert result.selected_satellites == ["sat-1"]
    assert result.coverage_score == pytest.approx(0.5)


def test_coverage_simulator_placeholder_integration():
    registry = SimulationPluginRegistry()
    coverage_adapter: CoverageSimulator = DummyCoverageSimulator()
    registry.register(coverage_adapter)

    request = CoverageRequest(
        satellites=[
            SatelliteAsset(
                identifier="sat-1",
                display_name="Sat 1",
                ephemeris=SatelliteEphemeris(
                    source_type=EphemerisSourceType.DEFAULT_ORBIT,
                    default_orbit_profile="LEO_600KM",
                ),
            )
        ],
        ground_sites=[GroundSite("g1", 10.0, 20.0)],
        window=SimulationWindow(start=datetime.utcnow(), duration=timedelta(minutes=1)),
        elevation_mask_deg=10.0,
    )
    result = coverage_adapter.compute_coverage(request)

    assert ("sat-1", "g1") in result.visibility
