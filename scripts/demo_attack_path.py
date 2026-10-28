"""Demonstration script for attack-path search using Dijkstra and A* algorithms."""

import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.topology import build_attack_graph, load_topology
from search.astar import astar_search
from search.dijkstra import dijkstra_search, path_probability


def run_demo() -> None:
    """Run Dijkstra and A* attack path search from attacker-ext to all crown jewels."""
    topology = load_topology()
    attack_graph = build_attack_graph(topology)

    source = "attacker-ext"
    crown_jewels = topology.crown_jewels

    print("=" * 80)
    print("ThreatGraph Attack-Path Search Demonstration")
    print(
        f"Attack Graph: {attack_graph.number_of_nodes()} hosts, "
        f"{attack_graph.number_of_edges()} reachability edges"
    )
    print(f"Source Node:  {source}")
    print(f"Crown Jewels: {crown_jewels}")
    print("=" * 80)

    for target in crown_jewels:
        print(f"\n--- Searching Paths to Target: {target} ---")

        # 1. Dijkstra Search
        t0 = time.perf_counter()
        d_path, d_cost, d_expanded = dijkstra_search(attack_graph, source, target)
        d_time = (time.perf_counter() - t0) * 1000.0
        d_prob = path_probability(d_cost)

        print("\n[Dijkstra Algorithm]")
        print(f"  Path:            {' -> '.join(d_path) if d_path else 'None'}")
        print(f"  Hops:            {len(d_path) - 1 if d_path else 0}")
        print(f"  Total Cost:      {d_cost:.4f}")
        print(f"  Path Prob:       {d_prob:.6e} ({d_prob * 100:.4f}%)")
        print(f"  Nodes Expanded:  {d_expanded}")
        print(f"  Runtime:         {d_time:.3f} ms")

        # 2. A* Search
        t0 = time.perf_counter()
        a_path, a_cost, a_expanded = astar_search(attack_graph, source, target)
        a_time = (time.perf_counter() - t0) * 1000.0
        a_prob = path_probability(a_cost)

        print("\n[A* Algorithm (Admissible & Consistent Reverse BFS Heuristic)]")
        print(f"  Path:            {' -> '.join(a_path) if a_path else 'None'}")
        print(f"  Hops:            {len(a_path) - 1 if a_path else 0}")
        print(f"  Total Cost:      {a_cost:.4f}")
        print(f"  Path Prob:       {a_prob:.6e} ({a_prob * 100:.4f}%)")
        print(f"  Nodes Expanded:  {a_expanded}")
        print(f"  Runtime:         {a_time:.3f} ms")

        assert d_cost == a_cost or abs(d_cost - a_cost) < 1e-9, "Dijkstra and A* costs must match!"


if __name__ == "__main__":
    run_demo()
