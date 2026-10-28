"""Dijkstra shortest attack-path algorithm implementation using heapq."""

import heapq
import math
from typing import Any

import networkx as nx


def path_probability(total_cost: float) -> float:
    """Calculate cumulative traversal probability from total logarithmic cost.

    Since edge cost is defined as -ln(p), cumulative path probability is exp(-total_cost).

    Args:
        total_cost: Sum of logarithmic edge costs.

    Returns:
        float: Cumulative probability in [0.0, 1.0].
    """
    if math.isinf(total_cost) or total_cost < 0.0:
        return 0.0
    return math.exp(-total_cost)


def dijkstra_search(
    graph: nx.DiGraph,
    source: Any,
    target: Any,
    weight: str = "cost",
) -> tuple[list[Any] | None, float, int]:
    """Find the optimal attack path from source to target using Dijkstra's algorithm.

    Implemented using a min-heap (heapq). Explores nodes in increasing order of
    cumulative cost from the source node.

    Args:
        graph: Directed attack graph.
        source: Source node identifier.
        target: Target node identifier.
        weight: Edge attribute key representing traversal cost (default: 'cost').

    Returns:
        tuple containing:
            - list[Any] | None: Ordered node sequence along optimal path, or None if unreachable.
            - float: Total path cost (sum of edge weights), or float('inf') if unreachable.
            - int: Total number of nodes expanded during the search.
    """
    if source not in graph or target not in graph:
        return None, float("inf"), 0

    if source == target:
        return [source], 0.0, 0

    # Priority queue stores (cost, node)
    heap: list[tuple[float, Any]] = [(0.0, source)]
    distances: dict[Any, float] = {source: 0.0}
    parent: dict[Any, Any] = {}
    visited: set[Any] = set()
    nodes_expanded: int = 0

    while heap:
        cost, u = heapq.heappop(heap)

        if u in visited:
            continue
        visited.add(u)
        nodes_expanded += 1

        if u == target:
            # Reconstruct optimal path
            path: list[Any] = []
            curr: Any = target
            while curr is not None:
                path.append(curr)
                curr = parent.get(curr)
            path.reverse()
            return path, cost, nodes_expanded

        for v in graph.successors(u):
            edge_weight = float(graph[u][v].get(weight, 1.0))
            new_cost = cost + edge_weight

            if new_cost < distances.get(v, float("inf")):
                distances[v] = new_cost
                parent[v] = u
                heapq.heappush(heap, (new_cost, v))

    return None, float("inf"), nodes_expanded

