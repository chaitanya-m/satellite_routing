from dataclasses import dataclass

from nodes import Node
from adjacency_list_graph import AdjacencyListGraph
from distance_vector_engine import SimpleDistanceVectorEngine
from dijkstra_engine import SimpleDijkstraEngine
import routers
from routers import (
    GroundStationRouter,
    RouteSelectionPolicy,
    SatelliteRouter,
    set_route_selection_policy,
)
from routing import DVMessage


@dataclass(frozen=True)
class DummyNode(Node):
    _id: str

    @property
    def id(self) -> str:
        return self._id


def test_satellite_learns_ground_route_via_dv():
    """Satellites should learn ground station paths via DV export from ground."""
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

    routers = {
        gs: gs_router,
        s1: s1_router,
        s2: s2_router,
    }

    gs_router.recompute_on_topology(g, epoch=1)
    s1_router.recompute_on_topology(g, epoch=1)
    s2_router.recompute_on_topology(g, epoch=1)

    # Ground station advertises to neighbours
    for msg in gs_router.outgoing_dv_messages()[s1]:
        s1_router.handle_dv_message(gs, msg)
    # S1 forwards to S2
    for msg in s1_router.outgoing_dv_messages()[s2]:
        s2_router.handle_dv_message(s1, msg)

    assert s2_router.next_hop(gs) == s1
    route = s2_router._routing_table[gs]  # type: ignore[attr-defined]
    assert abs(route.cost - 3.0) < 1e-6


def test_satellite_prefers_fresh_ground_epoch_over_cheaper_neighbor():
    """With ORACLE_DV, satellites prefer fresher HC adverts even if neighbor path looks cheaper."""
    original_policy = routers.ROUTE_SELECTION_POLICY
    set_route_selection_policy(RouteSelectionPolicy.ORACLE_DV)
    try:
        gs = DummyNode("GS")
        s1 = DummyNode("S1")
        s2 = DummyNode("S2")

        g = AdjacencyListGraph()
        for n in (gs, s1, s2):
            g.add_node(n)

        # Neighbor costs from S2: S1 (1), GS (5)
        g.add_edge(s2, s1, 1.0)
        g.add_edge(s2, gs, 5.0)
        g.add_edge(s1, s2, 1.0)
        g.add_edge(gs, s2, 5.0)

        dv_engine = SimpleDistanceVectorEngine()
        dijkstra_engine = SimpleDijkstraEngine()

        gs_router = GroundStationRouter(gs, dv_engine, dijkstra_engine)
        s1_router = SatelliteRouter(s1, dv_engine)
        s2_router = SatelliteRouter(s2, dv_engine)

        gs_router.recompute_on_topology(g, epoch=2)
        s1_router.recompute_on_topology(g, epoch=1)
        s2_router.recompute_on_topology(g, epoch=1)

        # S1 advertises a cheaper but stale path to GS (epoch 1, cost via S1: 1 + 1 = 2)
        msg_from_s1 = DVMessage(dest=gs, cost=1.0, origin_hc=None, epoch=1)
        s2_router.handle_dv_message(s1, msg_from_s1)

        # GS advertises with fresher epoch 2 (cost via GS: 5 + 0 = 5)
        msg_from_gs = DVMessage(dest=gs, cost=0.0, origin_hc=gs, epoch=2)
        s2_router.handle_dv_message(gs, msg_from_gs)

        assert s2_router.next_hop(gs) == gs  # prefers fresher HC advert despite higher cost
        route = s2_router._routing_table[gs]  # type: ignore[attr-defined]
        assert abs(route.cost - 5.0) < 1e-6
    finally:
        set_route_selection_policy(original_policy)


def test_satellite_cost_only_prefers_cheaper_neighbor_path():
    """DV_ONLY (cost-only) policy should pick the cheaper neighbor path even if HC exists."""
    # Ensure DV_ONLY policy is active
    set_route_selection_policy(RouteSelectionPolicy.DV_ONLY)

    gs = DummyNode("GS")
    s1 = DummyNode("S1")
    s2 = DummyNode("S2")

    g = AdjacencyListGraph()
    for n in (gs, s1, s2):
        g.add_node(n)

    # Neighbor costs from S2: S1 (1), GS (5)
    g.add_edge(s2, s1, 1.0)
    g.add_edge(s2, gs, 5.0)
    g.add_edge(s1, s2, 1.0)
    g.add_edge(gs, s2, 5.0)

    dv_engine = SimpleDistanceVectorEngine()
    dijkstra_engine = SimpleDijkstraEngine()

    gs_router = GroundStationRouter(gs, dv_engine, dijkstra_engine)
    s1_router = SatelliteRouter(s1, dv_engine)
    s2_router = SatelliteRouter(s2, dv_engine)

    gs_router.recompute_on_topology(g, epoch=2)
    s1_router.recompute_on_topology(g, epoch=1)
    s2_router.recompute_on_topology(g, epoch=1)

    # S1 advertises cost 1 (total 2 via S1)
    s2_router.handle_dv_message(s1, DVMessage(dest=gs, cost=1.0, origin_hc=None, epoch=1))
    # GS advertises fresher HC cost 0 (total 5 via GS)
    s2_router.handle_dv_message(gs, DVMessage(dest=gs, cost=0.0, origin_hc=gs, epoch=2))

    assert s2_router.next_hop(gs) == s1
    route = s2_router._routing_table[gs]  # type: ignore[attr-defined]
    assert abs(route.cost - 2.0) < 1e-6
