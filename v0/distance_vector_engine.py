"""
Simple Bellman–Ford-style distance-vector engine.

Operates over neighbour adverts and returns an updated dest -> cost map.
"""

from typing import Dict, Mapping
import math

from nodes import Node
from algorithms import DistanceVectorEngine


class SimpleDistanceVectorEngine(DistanceVectorEngine):
    """
    One-step Bellman–Ford relaxation suitable for DV routers.
    """

    def relax(
        self,
        self_node: Node,
        neighbor_costs: Mapping[Node, float],
        current_costs: Mapping[Node, float],
        adverts: Mapping[Node, Mapping[Node, float]],
    ) -> Dict[Node, float]:
        """
        Perform one Bellman–Ford relaxation over neighbour adverts.

        We start from the current local costs, set our own node to 0,
        then for each neighbour combine the link cost to that neighbour
        with the neighbour's advertised cost to each destination. If the
        combined cost beats what we have locally, we update to the new cost.
        The advert map is a nested mapping: for each neighbour, we obtain the
        neighbour's advertised costs to every destination it knows about.
        """
        new_costs: Dict[Node, float] = dict(current_costs)
        new_costs[self_node] = 0.0

        for neighbor, neighbor_distance in neighbor_costs.items():
            advertised = adverts.get(neighbor)
            if not advertised:
                continue

            for dest, advert_cost in advertised.items():
                candidate = neighbor_distance + advert_cost
                if candidate < new_costs.get(dest, math.inf):
                    new_costs[dest] = candidate

        return new_costs
