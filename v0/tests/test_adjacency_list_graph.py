"""
Unit tests for AdjacencyListGraph.
"""

from dataclasses import dataclass

from adjacency_list_graph import AdjacencyListGraph
from nodes import Node


@dataclass(frozen=True)
class DummyNode(Node):
    """
    Minimal concrete Node implementation for testing.
    """

    _id: str

    @property
    def id(self) -> str:
        return self._id


def test_add_nodes_and_edges():
    g = AdjacencyListGraph()

    a = DummyNode("A")
    b = DummyNode("B")
    c = DummyNode("C")

    g.add_edge(a, b, 1.0)
    g.add_edge(a, c, 2.0)
    g.add_edge(b, c, 3.0)

    assert set(g.nodes()) == {a, b, c}

    assert g.outgoing(a) == {b: 1.0, c: 2.0}
    assert g.outgoing(b) == {c: 3.0}
    assert g.outgoing(c) == {}


def test_outgoing_returns_copy():
    g = AdjacencyListGraph()
    a = DummyNode("A")
    b = DummyNode("B")

    g.add_edge(a, b, 1.0)

    out = g.outgoing(a)
    out.clear()

    # internal structure must remain intact
    assert g.outgoing(a) == {b: 1.0}
