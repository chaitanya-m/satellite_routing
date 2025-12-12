"""
Concrete directed, weighted graph implementation for satsim.

Implements the Graph interface using a simple adjacency-list representation.
"""

from typing import Dict, Iterable, Mapping

from nodes import Node
from graph import Graph


class AdjacencyListGraph(Graph):
    """
    Directed, weighted graph backed by a node -> (neighbor -> weight) mapping.
    """

    def __init__(self) -> None:
        self._adj: Dict[Node, Dict[Node, float]] = {}

    # --- Mutation API (test/sim only, not part of Graph interface) -----------

    def add_node(self, node: Node) -> None:
        """Ensure node exists in the graph."""
        self._adj.setdefault(node, {})

    def add_edge(self, src: Node, dst: Node, weight: float) -> None:
        """
        Add or update a directed edge src -> dst with weight.
        Auto-adds nodes if they don't exist.
        """
        self.add_node(src)
        self.add_node(dst)
        self._adj[src][dst] = weight

    # --- Graph interface -----------------------------------------------------

    def nodes(self) -> Iterable[Node]:
        return self._adj.keys()

    def outgoing(self, node: Node) -> Mapping[Node, float]:
        return dict(self._adj.get(node, {}))  # defensive copy
