"""
Directed, weighted graph abstraction for satsim.

Nodes are Node instances.
Edges are directed: u -> v with float weight.
"""

from abc import ABC, abstractmethod
from typing import Iterable, Mapping

from nodes import Node


class Graph(ABC):
    """Directed, weighted graph over Node objects."""

    @abstractmethod
    def nodes(self) -> Iterable[Node]:
        """Return all nodes in the graph."""
        raise NotImplementedError

    @abstractmethod
    def outgoing(self, node: Node) -> Mapping[Node, float]:
        """
        Outgoing neighbors and edge weights for a given node.

        Returns: dict[Node, float]
        """
        raise NotImplementedError
