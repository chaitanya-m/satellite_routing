#!/usr/bin/env bash
set -e

# Update README
cat >> README.md << 'EOF'

---

Graph Implementation:

- Added `AdjacencyListGraph` in `adjacency_list_graph.py`
- Directed adjacency-list graph implementing the `Graph` interface
- Included pytest tests under `tests/`
EOF

# Concrete directed graph implementation
cat > adjacency_list_graph.py << 'EOF'
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
EOF

# Tests package setup
mkdir -p tests

cat > tests/__init__.py << 'EOF'
# test package
EOF

cat > tests/test_adjacency_list_graph.py << 'EOF'
"""
Unit tests for AdjacencyListGraph.
"""

from dataclasses import dataclass

from adjacency_list_graph import AdjacencyListGraph
from nodes import Node


@dataclass(frozen=True)
class TestNode(Node):
    """
    Minimal concrete Node implementation for testing.
    """

    _id: str

    @property
    def id(self) -> str:
        return self._id


def test_add_nodes_and_edges():
    g = AdjacencyListGraph()

    a = TestNode("A")
    b = TestNode("B")
    c = TestNode("C")

    g.add_edge(a, b, 1.0)
    g.add_edge(a, c, 2.0)
    g.add_edge(b, c, 3.0)

    assert set(g.nodes()) == {a, b, c}

    assert g.outgoing(a) == {b: 1.0, c: 2.0}
    assert g.outgoing(b) == {c: 3.0}
    assert g.outgoing(c) == {}


def test_outgoing_returns_copy():
    g = AdjacencyListGraph()
    a = TestNode("A")
    b = TestNode("B")

    g.add_edge(a, b, 1.0)

    out = g.outgoing(a)
    out.clear()

    # internal structure must remain intact
    assert g.outgoing(a) == {b: 1.0}
EOF

echo "Created adjacency_list_graph.py and pytest tests. Run: pytest"


