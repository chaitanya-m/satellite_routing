from dataclasses import dataclass

from adjacency_list_graph import AdjacencyListGraph
from distance_vector_engine import SimpleDistanceVectorEngine
from dijkstra_engine import SimpleDijkstraEngine
import routers
from routers import (
    DijkstraRouter,
    GroundStationRouter,
    RouteSelectionPolicy,
    SatelliteRouter,
    set_route_selection_policy,
)
from routing import DVMessage
from simulation import run_dv_round
from nodes import Node


@dataclass(frozen=True)
class DummyNode(Node):
    _id: str

    @property
    def id(self) -> str:
        return self._id


def test_dv_only_matches_dijkstra_costs_on_simple_topology():
    """DV-only propagation should match Dijkstra costs on an acyclic small graph."""
    set_route_selection_policy(RouteSelectionPolicy.DV_ONLY)

    gs = DummyNode("GS")
    s1 = DummyNode("S1")
    s2 = DummyNode("S2")

    g = AdjacencyListGraph()
    for n in (gs, s1, s2):
        g.add_node(n)

    # GS <-> S1 (1), S1 <-> S2 (2)
    g.add_edge(gs, s1, 1.0)
    g.add_edge(s1, gs, 1.0)
    g.add_edge(s1, s2, 2.0)
    g.add_edge(s2, s1, 2.0)

    dv_engine = SimpleDistanceVectorEngine()
    dijkstra_engine = SimpleDijkstraEngine()

    gs_router = GroundStationRouter(gs, dv_engine, dijkstra_engine)
    s1_router = SatelliteRouter(s1, dv_engine)
    s2_router = SatelliteRouter(s2, dv_engine)
    dj_s2 = DijkstraRouter(s2, dijkstra_engine)

    routers_map = {gs: gs_router, s1: s1_router, s2: s2_router}

    for r in routers_map.values():
        r.recompute_on_topology(g, epoch=1)
    dj_s2.recompute_on_topology(g, epoch=1)

    # Two DV rounds suffice here
    run_dv_round(g, routers_map)
    run_dv_round(g, routers_map)

    dv_route = s2_router._routing_table[gs]  # type: ignore[attr-defined]
    dj_route = dj_s2.next_hop(gs)

    assert dv_route.next_hop == dj_route
    assert abs(dv_route.cost - 3.0) < 1e-6


def test_dijkstra_router_ignores_dv_messages():
    """Dijkstra-only router should ignore DV messages."""
    a = DummyNode("A")
    b = DummyNode("B")
    g = AdjacencyListGraph()
    for n in (a, b):
        g.add_node(n)
    g.add_edge(a, b, 1.0)
    g.add_edge(b, a, 100.0)

    dijkstra_engine = SimpleDijkstraEngine()
    dv_engine = SimpleDistanceVectorEngine()

    a_router = DijkstraRouter(a, dijkstra_engine)
    b_router = SatelliteRouter(b, dv_engine)

    a_router.recompute_on_topology(g, epoch=1)
    b_router.recompute_on_topology(g, epoch=1)

    # B sends a misleading advert claiming cost 0 to A (would make cost 100)
    a_router.handle_dv_message(b, DVMessage(dest=a, cost=0.0, origin_hc=None, epoch=1))
    # A ignores DV; route should remain via direct edge
    assert a_router.next_hop(b) == b


def test_run_dv_round_differs_by_policy():
    """Simulation helper should reflect policy: DV_ONLY vs PREFER_HC_EPOCH yield different next hops."""
    original_policy = routers.ROUTE_SELECTION_POLICY

    def run_policy(policy: RouteSelectionPolicy):
        set_route_selection_policy(policy)
        gs = DummyNode("GS")
        s1 = DummyNode("S1")
        s2 = DummyNode("S2")

        g = AdjacencyListGraph()
        for n in (gs, s1, s2):
            g.add_node(n)

        # S1 <-> S2 (1), GS <-> S2 (5); S1 never hears GS directly.
        g.add_edge(s1, s2, 1.0)
        g.add_edge(s2, s1, 1.0)
        g.add_edge(s2, gs, 5.0)
        g.add_edge(gs, s2, 5.0)

        dv_engine = SimpleDistanceVectorEngine()
        dijkstra_engine = SimpleDijkstraEngine()

        gs_router = GroundStationRouter(gs, dv_engine, dijkstra_engine)
        s1_router = SatelliteRouter(s1, dv_engine)
        s2_router = SatelliteRouter(s2, dv_engine)

        routers_map = {gs: gs_router, s1: s1_router, s2: s2_router}
        for r in routers_map.values():
            r.recompute_on_topology(g, epoch=2 if r is gs_router else 1)

        # Seed S1 with a cheap non-HC advert to GS (pretend S2 told it so)
        s1_router.handle_dv_message(s2, DVMessage(dest=gs, cost=0.0, origin_hc=None, epoch=1))

        # Two DV rounds: GS -> S1, then S1 -> S2 using current tables
        run_dv_round(g, routers_map)  # GS advert reaches S2; S1 cheap non-HC advert reaches S2
        run_dv_round(g, routers_map)  # second round stabilises tables

        route = s2_router._routing_table[gs]  # type: ignore[attr-defined]
        return s2_router.next_hop(gs), route.cost

    try:
        nh_dv_only, cost_dv_only = run_policy(RouteSelectionPolicy.DV_ONLY)
        nh_hc, cost_hc = run_policy(RouteSelectionPolicy.PREFER_HC_EPOCH)
    finally:
        set_route_selection_policy(original_policy)

    assert nh_dv_only != nh_hc
    assert nh_dv_only.id == "S1"
    assert nh_hc.id == "GS"
    assert abs(cost_dv_only - 2.0) < 1e-6
    assert abs(cost_hc - 5.0) < 1e-6
