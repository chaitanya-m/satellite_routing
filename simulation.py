"""
Simulation utilities for DV message exchange.
"""

from typing import Dict, List

from graph import Graph
from nodes import Node
from routing import DVMessage, Router


def run_dv_round(graph: Graph, routers: Dict[Node, Router]) -> None:
    """
    Perform one synchronous DV round across all routers using the graph topology.
    """
    deliveries: List[tuple[Node, Node, List[DVMessage]]] = []
    for router in routers.values():
        adverts = router.outgoing_dv_messages()
        for neighbor, msgs in adverts.items():
            if neighbor not in routers:
                continue
            if neighbor not in graph.outgoing(router.node):
                continue
            deliveries.append((neighbor, router.node, msgs))

    for dst, src, msgs in deliveries:
        dst_router = routers.get(dst)
        if not dst_router:
            continue
        for msg in msgs:
            dst_router.handle_dv_message(src, msg)
