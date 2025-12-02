"""
Unit tests for SimpleDijkstraEngine using AdjacencyListGraph.
"""

from dataclasses import dataclass

from nodes import Node
from adjacency_list_graph import AdjacencyListGraph
from dijkstra_engine import SimpleDijkstraEngine


@dataclass(frozen=True)
class DummyNode(Node):
    """
    Minimal concrete Node implementation for Dijkstra tests.
    """
    _id: str

    @property
    def id(self) -> str:
        return self._id


def test_dijkstra_basic_paths():
    g = AdjacencyListGraph()
    a = DummyNode("A")
    b = DummyNode("B")
    c = DummyNode("C")

    # A -> B (1), A -> C (4), B -> C (2)
    g.add_edge(a, b, 1.0)
    g.add_edge(a, c, 4.0)
    g.add_edge(b, c, 2.0)

    engine = SimpleDijkstraEngine()
    dist = engine.shortest_paths(g, a)

    assert dist[a] == 0.0
    assert dist[b] == 1.0
    # Shortest A->C is A->B->C with cost 3.0
    assert dist[c] == 3.0


def test_dijkstra_unreachable_node_absent():
    g = AdjacencyListGraph()
    a = DummyNode("A")
    b = DummyNode("B")
    c = DummyNode("C")  # unreachable from A

    g.add_edge(a, b, 2.0)
    g.add_node(c)

    engine = SimpleDijkstraEngine()
    dist = engine.shortest_paths(g, a)

    assert dist[a] == 0.0
    assert dist[b] == 2.0
    # Unreachable node should not appear in the distance map
    assert c not in dist
