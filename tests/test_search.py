"""Tests for attack-path search using Dijkstra and A* algorithms."""

import math

import networkx as nx
import numpy as np

from graph.topology import build_attack_graph, load_topology
from search.astar import astar_search, compute_reverse_bfs_hops
from search.dijkstra import dijkstra_search, path_probability


def test_dijkstra_and_astar_match_networkx_50_random_graphs() -> None:
    """Verify Dijkstra and A* match each other and NetworkX on 50 seeded random graphs."""
    for seed in range(50):
        # Generate random directed graph with 25 nodes and 20% edge probability
        g = nx.gnp_random_graph(n=25, p=0.20, directed=True, seed=seed)
        rng = np.random.default_rng(seed)

        # Assign strictly positive random edge costs
        for u, v in g.edges():
            g[u][v]["cost"] = float(rng.uniform(0.1, 5.0))

        src = 0
        target = 24

        d_path, d_cost, d_expanded = dijkstra_search(g, src, target, weight="cost")
        a_path, a_cost, a_expanded = astar_search(g, src, target, weight="cost")

        if nx.has_path(g, src, target):
            nx_cost = nx.shortest_path_length(g, src, target, weight="cost")

            assert d_path is not None, f"Seed {seed}: Dijkstra failed to find reachable path"
            assert a_path is not None, f"Seed {seed}: A* failed to find reachable path"
            assert math.isclose(d_cost, nx_cost, rel_tol=1e-6), (
                f"Seed {seed}: Dijkstra cost {d_cost} != NetworkX cost {nx_cost}"
            )
            assert math.isclose(a_cost, nx_cost, rel_tol=1e-6), (
                f"Seed {seed}: A* cost {a_cost} != NetworkX cost {nx_cost}"
            )
            assert math.isclose(d_cost, a_cost, rel_tol=1e-6), (
                f"Seed {seed}: Dijkstra cost {d_cost} != A* cost {a_cost}"
            )
            # A* with an admissible heuristic must never expand more nodes than Dijkstra + queue
            assert a_expanded <= d_expanded + len(g)
        else:
            assert d_path is None
            assert a_path is None
            assert math.isinf(d_cost)
            assert math.isinf(a_cost)


def test_heuristic_admissibility_never_exceeds_true_cost() -> None:
    """Verify that h(n) never exceeds the true optimal remaining cost h*(n) on all nodes."""
    for seed in range(10):
        g = nx.gnp_random_graph(n=20, p=0.30, directed=True, seed=seed)
        rng = np.random.default_rng(seed)

        for u, v in g.edges():
            g[u][v]["cost"] = float(rng.uniform(0.2, 4.0))

        target = 19
        hops_map = compute_reverse_bfs_hops(g, target)
        min_edge_cost = min((d["cost"] for _, _, d in g.edges(data=True)), default=0.0)

        for node in g.nodes():
            if nx.has_path(g, node, target):
                true_remaining_cost = nx.shortest_path_length(g, node, target, weight="cost")
                hops = hops_map[node]
                h = min_edge_cost * hops
                assert h <= true_remaining_cost + 1e-9, (
                    f"Heuristic violated at {node}: h={h} > true={true_remaining_cost}"
                )


def test_topology_attack_graph_3plus_hops_to_db01() -> None:
    """Verify topology attack graph has an optimal path from attacker-ext to db-01 with 3+ hops."""
    topology = load_topology()
    attack_graph = build_attack_graph(topology)

    source = "attacker-ext"
    target = "db-01"

    d_path, d_cost, d_expanded = dijkstra_search(attack_graph, source, target)
    a_path, a_cost, a_expanded = astar_search(attack_graph, source, target)

    assert d_path is not None, "Dijkstra must find path from attacker-ext to db-01"
    assert a_path is not None, "A* must find path from attacker-ext to db-01"

    # Number of hops = len(path) - 1
    d_hops = len(d_path) - 1
    a_hops = len(a_path) - 1

    assert d_hops >= 3, f"Expected 3+ hops to db-01, got {d_hops}: {d_path}"
    assert a_hops >= 3, f"Expected 3+ hops to db-01, got {a_hops}: {a_path}"

    assert math.isclose(d_cost, a_cost, rel_tol=1e-9)
    assert path_probability(d_cost) > 0.0
    assert path_probability(a_cost) > 0.0


def test_unreachable_target_returns_clear_result_no_crash() -> None:
    """Verify that search between disconnected nodes returns a clear result without crashing."""
    g = nx.DiGraph()
    g.add_node("nodeA")
    g.add_node("nodeB")
    # No edges between nodeA and nodeB

    d_path, d_cost, d_expanded = dijkstra_search(g, "nodeA", "nodeB")
    assert d_path is None
    assert math.isinf(d_cost)
    assert d_expanded >= 1
    assert path_probability(d_cost) == 0.0

    a_path, a_cost, a_expanded = astar_search(g, "nodeA", "nodeB")
    assert a_path is None
    assert math.isinf(a_cost)
    assert path_probability(a_cost) == 0.0

    # Test unknown node not in graph
    path, cost, expanded = dijkstra_search(g, "nonexistent", "nodeB")
    assert path is None
    assert math.isinf(cost)
    assert expanded == 0
