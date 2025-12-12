"""
Heap-based DijkstraEngine implementation for satsim.

Uses Python's heapq to compute single-source shortest paths over any Graph
implementation that satisfies the Graph interface.
"""

from typing import Dict
import heapq
import math

from nodes import Node
from graph import Graph
from algorithms import DijkstraEngine


class SimpleDijkstraEngine(DijkstraEngine):
    """
    Single-source Dijkstra using a binary heap.

    Complexity:
        O(E log V) over the nodes reachable from the source.
    """

    def __init__(self) -> None:
        # Instrumentation counters per invocation.
        self.last_edges_examined = 0
        self.last_relaxed = 0
        self.last_heap_pops = 0
        self.last_heap_pushes = 0

    def shortest_path_costs(self, graph: Graph, source: Node) -> Dict[Node, float]:
        """
        Compute only the cost map for all reachable nodes from source.
        """
        self.last_edges_examined = 0
        self.last_relaxed = 0
        self.last_heap_pops = 0
        self.last_heap_pushes = 0

        dist: Dict[Node, float] = {source: 0.0}
        pq = [(0.0, source)]  # priority queue of (distance, node)

        while pq:
            d_u, u = heapq.heappop(pq)
            self.last_heap_pops += 1

            # Skip outdated entries
            if d_u != dist.get(u, float("inf")):
                continue

            for v, w in graph.outgoing(u).items():
                self.last_edges_examined += 1
                alt = d_u + w
                if alt < dist.get(v, float("inf")):
                    dist[v] = alt
                    heapq.heappush(pq, (alt, v))
                    self.last_heap_pushes += 1
                    self.last_relaxed += 1

        return dist

    def shortest_paths(
        self, graph: Graph, source: Node
    ) -> tuple[Dict[Node, float], Dict[Node, Node]]:
        """
        Dijkstra variant that also records predecessors for path reconstruction.

        It returns the distance map (dest -> cost from source) plus a
        predecessor map that lets you walk back from any reachable node to
        the source. Routers can use this to derive a concrete next hop for
        each destination by walking parents until the source is reached;
        that next hop is what gets installed in a forwarding table or shared
        in an advert. The predecessor map omits the source itself because it
        has no parent.
        """
        self.last_edges_examined = 0
        self.last_relaxed = 0
        self.last_heap_pops = 0
        self.last_heap_pushes = 0

        dist: Dict[Node, float] = {source: 0.0}
        prev: Dict[Node, Node] = {}
        pq = [(0.0, source)]

        while pq:
            d_u, u = heapq.heappop(pq)
            self.last_heap_pops += 1
            if d_u != dist.get(u, math.inf):
                continue

            for v, w in graph.outgoing(u).items():
                self.last_edges_examined += 1
                alt = d_u + w
                if alt < dist.get(v, math.inf):
                    dist[v] = alt
                    prev[v] = u
                    heapq.heappush(pq, (alt, v))
                    self.last_heap_pushes += 1
                    self.last_relaxed += 1

        return dist, prev
