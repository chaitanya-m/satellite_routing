from dataclasses import dataclass

from nodes import Node
from distance_vector_engine import SimpleDistanceVectorEngine


@dataclass(frozen=True)
class DummyNode(Node):
    _id: str

    @property
    def id(self) -> str:
        return self._id


def test_relax_updates_costs_from_adverts():
    """Single neighbour advert should update destination cost via link + advert."""
    dv = SimpleDistanceVectorEngine()
    me = DummyNode("me")
    n1 = DummyNode("n1")
    dest = DummyNode("dest")

    neighbor_costs: dict[Node, float] = {n1: 2.0}
    current_costs: dict[Node, float] = {me: 0.0}
    adverts: dict[Node, dict[Node, float]] = {n1: {dest: 3.0}}

    updated = dv.relax(me, neighbor_costs, current_costs, adverts)

    assert updated[me] == 0.0
    assert updated[dest] == 5.0  # 2 + 3


def test_relax_preserves_better_existing_costs():
    """Existing cheaper cost should not be overwritten by a worse advertised path."""
    dv = SimpleDistanceVectorEngine()
    me = DummyNode("me")
    n1 = DummyNode("n1")
    dest = DummyNode("dest")

    neighbor_costs: dict[Node, float] = {n1: 5.0}
    current_costs: dict[Node, float] = {me: 0.0, dest: 3.0}
    adverts: dict[Node, dict[Node, float]] = {n1: {dest: 3.0}}  # would be 8 via n1, worse than 3

    updated = dv.relax(me, neighbor_costs, current_costs, adverts)

    assert updated[dest] == 3.0


def test_relax_chooses_best_from_multiple_neighbours():
    """Choose the best neighbour per destination when multiple adverts compete."""
    dv = SimpleDistanceVectorEngine()
    me = DummyNode("me")
    n1 = DummyNode("n1")
    n2 = DummyNode("n2")
    d1 = DummyNode("d1")
    d2 = DummyNode("d2")

    neighbor_costs: dict[Node, float] = {n1: 1.0, n2: 5.0}
    adverts: dict[Node, dict[Node, float]] = {
        n1: {d1: 10.0, d2: 4.0},
        n2: {d1: 3.0, d2: 2.0},
    }
    current_costs: dict[Node, float] = {}

    updated = dv.relax(me, neighbor_costs, current_costs, adverts)

    # d1: min(1+10=11 via n1, 5+3=8 via n2)
    assert updated[d1] == 8.0
    # d2: min(1+4=5 via n1, 5+2=7 via n2)
    assert updated[d2] == 5.0
    assert updated[me] == 0.0


def test_relax_ignores_neighbours_without_adverts():
    """Neighbours lacking adverts should not affect the cost table."""
    dv = SimpleDistanceVectorEngine()
    me = DummyNode("me")
    n1 = DummyNode("n1")
    n2 = DummyNode("n2")
    dest = DummyNode("dest")

    neighbor_costs: dict[Node, float] = {n1: 1.0, n2: 2.0}
    adverts: dict[Node, dict[Node, float]] = {n1: {dest: 5.0}}  # n2 has no advert
    current_costs: dict[Node, float] = {}

    updated = dv.relax(me, neighbor_costs, current_costs, adverts)

    assert updated[dest] == 6.0  # via n1 only
    assert dest not in adverts.get(n2, {})
