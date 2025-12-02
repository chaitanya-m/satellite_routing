#!/usr/bin/env bash
set -e

# Extend README with routing/algorithm abstractions
cat >> README.md << 'EOF'

---

Routing abstractions introduced:

- `routing.Router`: per-node routing behaviour and DV handling.
- `routing.RouteEntry`, `routing.DVMessage`: basic routing data types.
- `algorithms.DijkstraEngine`, `algorithms.DistanceVectorEngine`: graph algorithm interfaces used by routers.
EOF

# routing interfaces
cat > routing.py << 'EOF'
"""
Routing abstractions for satsim.

Defines Router interface shared by both high-compute and low-compute nodes,
plus simple data structures for routes and DV messages.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from nodes import Node
from graph import Graph


@dataclass
class RouteEntry:
    """
    Single forwarding entry in a node's routing table.
    """
    dest: Node
    next_hop: Node
    cost: float
    origin_hc: Optional[Node]  # high-compute anchor, if any
    epoch: int                 # HC epoch for this entry (0 if none)


@dataclass(frozen=True)
class DVMessage:
    """
    Distance-vector advertisement sent between neighbours.

    For HC-originated routes, origin_hc and epoch identify the Dijkstra epoch
    that produced the advertised cost. For LC-only routes, origin_hc may be None.
    """
    dest: Node
    cost: float
    origin_hc: Optional[Node]
    epoch: int


class Router(ABC):
    """
    Common router API implemented by all nodes (HC and LC).

    High-compute routers may combine link-state + Dijkstra + DV export.
    Low-compute routers typically consume DV only and run local BF relaxations.
    """

    @property
    @abstractmethod
    def node(self) -> Node:
        """Node identity owned by this router."""
        raise NotImplementedError

    @abstractmethod
    def recompute_on_topology(self, g: Graph, epoch: int) -> None:
        """
        Called when a new topology snapshot (and epoch) is available.

        High-compute routers may run Dijkstra here and prepare DV adverts.
        Low-compute routers may only refresh local neighbour costs.
        """
        raise NotImplementedError

    @abstractmethod
    def next_hop(self, dest: Node) -> Optional[Node]:
        """
        Return the next hop toward dest under the current routing state.

        Returns None if dest is currently unreachable.
        """
        raise NotImplementedError

    @abstractmethod
    def handle_dv_message(self, src: Node, msg: DVMessage) -> None:
        """
        Process one incoming distance-vector message from neighbour src.
        """
        raise NotImplementedError
EOF

# algorithm interfaces
cat > algorithms.py << 'EOF'
"""
Algorithm interfaces for routing.

Keeps graph algorithms separate from router wiring and simulation details.
"""

from abc import ABC, abstractmethod
from typing import Dict, Mapping

from nodes import Node
from graph import Graph


class DijkstraEngine(ABC):
    """
    Interface for single-source shortest-path computation.
    """

    @abstractmethod
    def shortest_paths(self, graph: Graph, source: Node) -> Dict[Node, float]:
        """
        Compute shortest-path costs from source to all reachable nodes.

        Returns:
            Mapping dest_node -> path_cost(source -> dest_node).
        """
        raise NotImplementedError


class DistanceVectorEngine(ABC):
    """
    Interface for a Bellman–Ford-style distance-vector relaxation step.
    """

    @abstractmethod
    def relax(
        self,
        self_node: Node,
        neighbor_costs: Mapping[Node, float],
        current_costs: Mapping[Node, float],
        adverts: Mapping[Node, Mapping[Node, float]],
    ) -> Dict[Node, float]:
        """
        Perform one relaxation step for self_node.

        Args:
            self_node: node whose distances are updated.
            neighbor_costs: cost(self_node -> v) for each neighbour v.
            current_costs: current local estimate dest -> cost.
            adverts: for each neighbour v: advertised dest -> cost_v(dest).

        Returns:
            New dest -> cost map after one Bellman–Ford relaxation step.
        """
        raise NotImplementedError
EOF

echo "Commit 2 interfaces created: routing.py, algorithms.py, README updated."

