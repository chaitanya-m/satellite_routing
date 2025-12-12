"""
Router implementations for satsim.

Ground stations run Dijkstra as an oracle and export DV adverts.
Satellites run DV only and learn from ground-station adverts.
"""

from typing import Dict, List, Mapping, Optional, Tuple
from enum import Enum
import math

from algorithms import DijkstraEngine, DistanceVectorEngine
from graph import Graph
from nodes import Node
from routing import RouteEntry, Router


class RouteSelectionPolicy(Enum):
    """
    Global route-selection policy.

    DV_ONLY: prefer the cheapest DV cost (no HC/epoch bias).
    ORACLE_DV: prefer high-compute origins with fresher epochs, then cost.
    DIJKSTRA_ONLY: bypass DV entirely and rely on local Dijkstra at all nodes.
    """

    DV_ONLY = "dv_only"
    ORACLE_DV = "oracle_dv"
    DIJKSTRA_ONLY = "dijkstra_only"


# Global policy used by all routers; adjustable for experiments/comparisons.
ROUTE_SELECTION_POLICY: RouteSelectionPolicy = RouteSelectionPolicy.DV_ONLY


def set_route_selection_policy(policy: RouteSelectionPolicy) -> None:
    """Set the global route selection policy for all routers."""
    global ROUTE_SELECTION_POLICY
    ROUTE_SELECTION_POLICY = policy


class BaseDVRouter(Router):
    """
    Distance-vector router shared by satellites and ground stations.
    """

    def __init__(self, node: Node, dv_engine: DistanceVectorEngine) -> None:
        self._node = node
        self._dv_engine = dv_engine
        self._neighbor_costs: Dict[Node, float] = {}
        # Stored as neighbour -> (dest -> (cost, origin_hc, epoch))
        self._adverts: Dict[Node, Dict[Node, Tuple[float, Optional[Node], int]]] = {}
        self._routing_table: Dict[Node, RouteEntry] = {
            node: RouteEntry(node, node, 0.0, node, 0)
        }
        self._epoch = 0
        self._changed = False
        # Cache DV adverts so we don't rebuild the same messages every round.
        self._routing_dirty = True
        self._cached_adverts: Dict[Node, Dict[Node, Tuple[float, Optional[Node], int]]] = {}
        # Buffer inbound DV so we relax once per round.
        self._pending_dirty = False
        # Instrumentation: count DV relax work per router.
        self._ops_examined = 0
        self._ops_updates = 0

    # --- Router interface ---------------------------------------------------

    @property
    def node(self) -> Node:
        return self._node

    def recompute_on_topology(self, g: Graph, epoch: int) -> None:
        self._epoch = epoch
        self._neighbor_costs = dict(g.outgoing(self._node))
        # Drop adverts from nodes that are no longer neighbours
        self._adverts = {n: adv for n, adv in self._adverts.items() if n in self._neighbor_costs}
        self._routing_dirty = True
        self._pending_dirty = False
        self._ops_examined = 0
        self._ops_updates = 0
        self._relax()

    def next_hop(self, dest: Node) -> Optional[Node]:
        entry = self._routing_table.get(dest)
        return entry.next_hop if entry else None

    def outgoing_dv_messages(self) -> Mapping[Node, Mapping[Node, Tuple[float, Optional[Node], int]]]:
        """
        Build DV adverts to send to each neighbour using the current table.
        """
        if self._routing_dirty:
            self._cached_adverts = {}
            for neighbor in self._neighbor_costs:
                # Reuse the same advert maps per neighbour when routing is unchanged.
                advert_map: Dict[Node, Tuple[float, Optional[Node], int]] = {}
                for entry in self._routing_table.values():
                    advert_map[entry.dest] = (entry.cost, entry.origin_hc, entry.epoch)
                self._cached_adverts[neighbor] = advert_map
            self._routing_dirty = False
        return self._cached_adverts

    def handle_dv_message(self, src: Node, dest: Node, cost: float, origin_hc: Optional[Node], epoch: int) -> None:
        if src not in self._neighbor_costs:
            # Ignore messages from non-neighbours
            return

        self._adverts.setdefault(src, {})[dest] = (cost, origin_hc, epoch)
        self._pending_dirty = True

    def apply_pending_updates(self) -> None:
        """
        Apply any queued DV changes once per round to avoid repeated relax calls.
        """
        if self._pending_dirty:
            self._relax()
            self._pending_dirty = False

    # --- Internal helpers ---------------------------------------------------

    def reset_changed_flag(self) -> None:
        """Clear the per-round change marker used by convergence checks."""
        self._changed = False
        # Reset per-round instrumentation accumulators.
        self._ops_examined = 0
        self._ops_updates = 0

    def _relax(self) -> None:
        # Translate adverts into the form expected by the DV engine
        adverts_for_engine: Dict[Node, Dict[Node, float]] = {}
        for neighbor, dest_map in self._adverts.items():
            adverts_for_engine[neighbor] = {dest: payload[0] for dest, payload in dest_map.items()}

        current_costs = {dest: entry.cost for dest, entry in self._routing_table.items()}
        new_costs = self._dv_engine.relax(
            self_node=self._node,
            neighbor_costs=self._neighbor_costs,
            current_costs=current_costs,
            adverts=adverts_for_engine,
        )

        # Track best neighbour per destination
        best: Dict[Node, tuple[float, Node, Optional[Node], int]] = {}
        # Seed with existing routes to preserve paths when no better advert arrives
        for dest, entry in self._routing_table.items():
            best[dest] = (entry.cost, entry.next_hop, entry.origin_hc, entry.epoch)

        for neighbor, dest_map in self._adverts.items():
            if neighbor not in self._neighbor_costs:
                continue
            base = self._neighbor_costs[neighbor]
            for dest, payload in dest_map.items():
                msg_cost, msg_origin_hc, msg_epoch = payload
                candidate = base + msg_cost
                current = best.get(dest)
                candidate_entry = (candidate, neighbor, msg_origin_hc, msg_epoch)
                self._ops_examined += 1
                if current is None or self._is_better(candidate_entry, current):
                    best[dest] = candidate_entry
                    self._ops_updates += 1

        # Ensure self-route exists
        best[self._node] = (0.0, self._node, self._node, self._epoch)
        new_costs[self._node] = 0.0

        updated: Dict[Node, RouteEntry] = {}
        for dest, (cost, next_hop, origin_hc, epoch) in best.items():
            if dest != self._node and next_hop not in self._neighbor_costs:
                # Stale path; skip it.
                continue
            updated[dest] = RouteEntry(dest, next_hop, cost, origin_hc, epoch)

        # Mark if anything actually changed; avoid churn when stable.
        changed = updated != self._routing_table
        if changed:
            self._routing_table = updated
            self._routing_dirty = True
        self._changed = self._changed or changed

    def _is_better(
        self, candidate: tuple[float, Node, Optional[Node], int], current: tuple[float, Node, Optional[Node], int]
    ) -> bool:
        cand_cost, _, cand_origin, cand_epoch = candidate
        curr_cost, _, curr_origin, curr_epoch = current

        cand_hc = cand_origin is not None
        curr_hc = curr_origin is not None

        if ROUTE_SELECTION_POLICY == RouteSelectionPolicy.ORACLE_DV:
            # Prefer high-compute origin with fresher epoch.
            if cand_hc and curr_hc:
                if cand_epoch != curr_epoch:
                    return cand_epoch > curr_epoch
            elif cand_hc and not curr_hc:
                if cand_epoch >= curr_epoch:
                    return True
            elif not cand_hc and curr_hc:
                return False

        # Default and fallback: strictly better cost wins.
        return cand_cost < curr_cost


class SatelliteRouter(BaseDVRouter):
    """
    DV-only router for satellites.
    """

    pass


class DijkstraRouter(Router):
    """
    Router that relies solely on local Dijkstra computation (no DV).
    """

    def __init__(self, node: Node, dijkstra_engine: DijkstraEngine) -> None:
        self._node = node
        self._dijkstra = dijkstra_engine
        self._routing_table: Dict[Node, RouteEntry] = {
            node: RouteEntry(node, node, 0.0, node, 0)
        }
        self._epoch = 0

    @property
    def node(self) -> Node:
        return self._node

    def recompute_on_topology(self, g: Graph, epoch: int) -> None:
        self._epoch = epoch
        dist, parents = self._dijkstra.shortest_paths(g, self._node)
        routes: Dict[Node, RouteEntry] = {}
        for dest, cost in dist.items():
            if dest == self._node:
                routes[dest] = RouteEntry(dest, self._node, 0.0, self._node, epoch)
                continue
            next_hop = self._first_hop(dest, parents)
            if next_hop is None:
                continue
            routes[dest] = RouteEntry(dest, next_hop, cost, self._node, epoch)
        self._routing_table = routes
        # Record ops from the Dijkstra engine if available.
        if hasattr(self._dijkstra, "last_edges_examined"):
            self._ops_examined = getattr(self._dijkstra, "last_edges_examined", 0)
            self._ops_updates = getattr(self._dijkstra, "last_relaxed", 0)

    def next_hop(self, dest: Node) -> Optional[Node]:
        entry = self._routing_table.get(dest)
        return entry.next_hop if entry else None

    def outgoing_dv_messages(self) -> Mapping[Node, Mapping[Node, tuple[float, Optional[Node], int]]]:
        # Pure Dijkstra router does not emit DV.
        return {}

    def handle_dv_message(self, src: Node, dest: Node, cost: float, origin_hc: Optional[Node], epoch: int) -> None:
        # Ignores DV; relies on local computation.
        return

    def _first_hop(self, dest: Node, parents: Mapping[Node, Node]) -> Optional[Node]:
        step = dest
        while step in parents:
            parent = parents[step]
            if parent == self._node:
                return step
            step = parent
        return None


class GroundStationRouter(BaseDVRouter):
    """
    Router that runs Dijkstra locally and exports oracle DV adverts.
    """

    def __init__(
        self,
        node: Node,
        dv_engine: DistanceVectorEngine,
        dijkstra_engine: DijkstraEngine,
    ) -> None:
        super().__init__(node, dv_engine)
        self._dijkstra = dijkstra_engine

    def recompute_on_topology(self, g: Graph, epoch: int) -> None:
        self._epoch = epoch
        self._neighbor_costs = dict(g.outgoing(self._node))

        dist, parents = self._dijkstra.shortest_paths(g, self._node)
        routes: Dict[Node, RouteEntry] = {}
        for dest, cost in dist.items():
            if dest == self._node:
                routes[dest] = RouteEntry(dest, self._node, 0.0, self._node, epoch)
                continue

            next_hop = self._first_hop(dest, parents)
            if next_hop is None:
                continue
            routes[dest] = RouteEntry(dest, next_hop, cost, self._node, epoch)

        self._routing_table = routes
        # Ground stations ignore inbound DV—they are the oracle.
        self._adverts.clear()

    def handle_dv_message(self, src: Node, dest: Node, cost: float, origin_hc: Optional[Node], epoch: int) -> None:
        # Ground stations stick to their Dijkstra oracle and ignore DV updates.
        return

    def _first_hop(self, dest: Node, parents: Mapping[Node, Node]) -> Optional[Node]:
        step = dest
        while step in parents:
            parent = parents[step]
            if parent == self._node:
                return step
            step = parent
        return None
