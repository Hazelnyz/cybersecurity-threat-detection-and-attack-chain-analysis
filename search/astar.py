"""A* attack-path search algorithm with an admissible and consistent heuristic."""

import heapq
from collections import deque
from typing import Any

import networkx as nx

from search.dijkstra import path_probability


def compute_reverse_bfs_hops(graph: nx.DiGraph, target: Any) -> dict[Any, int]:
    """Compute the minimum hop count from every graph node to the target using reverse BFS.

    Args:
        graph: Directed attack graph.
        target: Target destination node ID.

    Returns:
        dict[Any, int]: Mapping from reachable node ID to minimum hops to target.
    """
    if target not in graph:
        return {}

    hops: dict[Any, int] = {target: 0}
    queue: deque[tuple[Any, int]] = deque([(target, 0)])

    # Traverse incoming edges (predecessors)
    while queue:
        curr, dist = queue.popleft()
        for pred in graph.predecessors(curr):
            if pred not in hops:
                hops[pred] = dist + 1
                queue.append((pred, dist + 1))

    return hops


def astar_search(
    graph: nx.DiGraph,
    source: Any,
    target: Any,
    weight: str = "cost",
) -> tuple[list[Any] | None, float, int]:
    r"""Find the optimal attack path from source to target using the A* algorithm.

    Heuristic Definition:
        h(n) = min_edge_cost * hops_to_target(n)

    Proof of Admissibility and Consistency:
    ---------------------------------------
    1. Admissibility:
       Let h*(n) denote the true optimal cost from node n to target. Any directed
       path from n to target must traverse at least k = hops_to_target(n) edges,
       because reverse BFS computes the unweighted shortest path distance.
       Every edge e in the graph has a non-negative cost c(e) >= min_edge_cost.
       Thus, for any path consisting of m edges (where m >= k):
           cost(path) = \sum_{i=1}^m c(e_i) >= m * min_edge_cost >= k * min_edge_cost = h(n)
       Since cost(path) >= h(n) for all valid paths, the optimal cost h*(n) >= h(n).
       Therefore, h(n) never overestimates the true remaining cost (admissible).

    2. Consistency (Monotonicity):
       A heuristic is consistent if for every edge (u, v) with cost c(u, v):
           h(u) <= c(u, v) + h(v)
       By the triangle inequality on the unweighted reverse BFS graph:
           hops_to_target(u) <= 1 + hops_to_target(v)
       Multiplying both sides by min_edge_cost >= 0:
           h(u) = min_edge_cost * hops_to_target(u)
                <= min_edge_cost * (1 + hops_to_target(v))
                = min_edge_cost + h(v)
       Since min_edge_cost <= c(u, v) by definition of min_edge_cost:
           h(u) <= c(u, v) + h(v)
       Thus, the heuristic satisfies consistency. Because h is consistent, every
       node expanded by A* is visited with its optimal cost g(n), eliminating the
       need to re-open closed nodes.

    Args:
        graph: Directed attack graph with non-negative edge costs.
        source: Source node identifier.
        target: Target destination node identifier.
        weight: Edge attribute key for traversal cost (default: 'cost').

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

    # Determine minimum edge cost across all edges in the graph
    edge_costs = [
        float(data.get(weight, 1.0))
        for _, _, data in graph.edges(data=True)
        if weight in data
    ]
    min_edge_cost = min(edge_costs) if edge_costs else 0.0

    # Precompute unweighted hops to target via reverse BFS
    hops_map = compute_reverse_bfs_hops(graph, target)

    # If source cannot reach target topologically, abort early
    if source not in hops_map:
        return None, float("inf"), 0

    def heuristic(node: Any) -> float:
        hops = hops_map.get(node)
        if hops is None:
            return float("inf")
        return min_edge_cost * hops

    # Priority queue stores (f_score, g_score, node)
    initial_h = heuristic(source)
    heap: list[tuple[float, float, Any]] = [(initial_h, 0.0, source)]
    g_scores: dict[Any, float] = {source: 0.0}
    parent: dict[Any, Any] = {}
    visited: set[Any] = set()
    nodes_expanded: int = 0

    while heap:
        f_score, g_score, u = heapq.heappop(heap)

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
            return path, g_score, nodes_expanded

        for v in graph.successors(u):
            edge_weight = float(graph[u][v].get(weight, 1.0))
            tentative_g = g_score + edge_weight

            if tentative_g < g_scores.get(v, float("inf")):
                g_scores[v] = tentative_g
                parent[v] = u
                h_v = heuristic(v)
                f_v = tentative_g + h_v
                heapq.heappush(heap, (f_v, tentative_g, v))

    return None, float("inf"), nodes_expanded


__all__ = ["astar_search", "compute_reverse_bfs_hops", "path_probability"]
