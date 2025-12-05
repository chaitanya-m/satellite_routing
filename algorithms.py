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
    def shortest_path_costs(self, graph: Graph, source: Node) -> Dict[Node, float]:
        """
        Compute shortest-path costs from source to all reachable nodes.

        Returns:
            Mapping dest_node -> path_cost(source -> dest_node).
        """
        raise NotImplementedError

    @abstractmethod
    def shortest_paths(
        self, graph: Graph, source: Node
    ) -> tuple[Dict[Node, float], Dict[Node, Node]]:
        """
        Compute shortest-path costs plus the predecessor chain for each dest.

        Returns:
            (dist, prev) where dist is the cost map and prev records parents.
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
