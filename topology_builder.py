"""
Utilities to generate and connect Poisson-sampled constellations and ground stations.
"""

from typing import Dict, Iterable, List, Tuple
import math

from adjacency_list_graph import AdjacencyListGraph
from constellation import (
    GroundStationNode,
    SatelliteNode,
    generate_ground_stations,
    generate_poisson_constellation,
)


EARTH_RADIUS_KM = 6371.0


def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points in km."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def build_constellation_graph(
    expected_sats: int,
    expected_ground: int,
    seed: int | None = None,
    altitude_km: float = 550.0,
    sat_degree: int = 4,
    ground_links: int = 4,
) -> tuple[AdjacencyListGraph, List[SatelliteNode], List[GroundStationNode]]:
    """
    Generate satellites/ground stations and connect them with simple nearest-neighbour links.

    Args:
        expected_sats: target mean satellite count for the Poisson process.
        expected_ground: target mean ground station count for the Poisson process.
        seed: RNG seed for reproducibility.
        altitude_km: altitude assigned to all satellites.
        sat_degree: number of nearest satellites to connect each satellite to (bidirectional).
        ground_links: number of nearest satellites to connect each ground station to (bidirectional).
    """
    satellites = generate_poisson_constellation(expected_sats, altitude_km=altitude_km, seed=seed)
    ground = generate_ground_stations(expected_ground, seed=seed)

    graph = AdjacencyListGraph()
    for node in [*satellites, *ground]:
        graph.add_node(node)

    _connect_satellites(graph, satellites, sat_degree)
    _connect_ground_to_sats(graph, ground, satellites, ground_links)
    return graph, satellites, ground


def _connect_satellites(graph: AdjacencyListGraph, satellites: List[SatelliteNode], degree: int) -> None:
    if degree <= 0:
        return

    for sat in satellites:
        distances: List[tuple[float, SatelliteNode]] = []
        for other in satellites:
            if other is sat:
                continue
            dist = great_circle_km(sat.lat, sat.lon, other.lat, other.lon)
            distances.append((dist, other))

        distances.sort(key=lambda pair: pair[0])
        for _, neighbor in distances[:degree]:
            weight = _link_cost_km(sat, neighbor)
            graph.add_edge(sat, neighbor, weight)
            graph.add_edge(neighbor, sat, weight)


def _connect_ground_to_sats(
    graph: AdjacencyListGraph,
    ground: List[GroundStationNode],
    satellites: List[SatelliteNode],
    degree: int,
) -> None:
    if degree <= 0 or not satellites:
        return

    for gs in ground:
        distances: List[tuple[float, SatelliteNode]] = []
        for sat in satellites:
            dist = great_circle_km(gs.lat, gs.lon, sat.lat, sat.lon)
            distances.append((dist, sat))
        distances.sort(key=lambda pair: pair[0])
        for _, sat in distances[: min(degree, len(distances))]:
            weight = _link_cost_km(gs, sat)
            graph.add_edge(gs, sat, weight)
            graph.add_edge(sat, gs, weight)


def _link_cost_km(a: SatelliteNode | GroundStationNode, b: SatelliteNode | GroundStationNode) -> float:
    """
    Simple link cost proxy using great-circle distance plus altitude term for satellites.
    """
    surface = great_circle_km(a.lat, a.lon, b.lat, b.lon)
    alt_a = a.altitude_km if isinstance(a, SatelliteNode) else 0.0
    alt_b = b.altitude_km if isinstance(b, SatelliteNode) else 0.0
    return surface + alt_a + alt_b
