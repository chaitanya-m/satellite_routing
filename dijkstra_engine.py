"""
Heap-based DijkstraEngine implementation for satsim.

Uses Python's heapq to compute single-source shortest paths over any Graph
implementation that satisfies the Graph interface.
"""

from typing import Dict
import heapq

from nodes import Node
from graph import Graph
from algorithms import DijkstraEngine


class SimpleDijkstraEngine(DijkstraEngine):
    """
    Single-source Dijkstra using a binary heap.

    Complexity:
        O(E log V) over the nodes reachable from the source.
    """

    def shortest_paths(self, graph: Graph, source: Node) -> Dict[Node, float]:
        # Distance map for reachable nodes
        dist: Dict[Node, float] = {source: 0.0}
        # Priority queue of (distance, node)
        pq = [(0.0, source)]

        while pq:
            d_u, u = heapq.heappop(pq)

            # Skip outdated entries
            if d_u != dist.get(u, float("inf")):
                continue

            for v, w in graph.outgoing(u).items():
                alt = d_u + w
                if alt < dist.get(v, float("inf")):
                    dist[v] = alt
                    heapq.heappush(pq, (alt, v))

        return dist
