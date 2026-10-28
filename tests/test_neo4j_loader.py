"""Tests for Neo4j persistence loader with mocked driver and optional live integration."""

import os
from unittest.mock import MagicMock, patch

import pytest

from data.synthetic import generate_synthetic_flows
from graph.builder import attach_topology_info, build_windows
from graph.neo4j_loader import Neo4jLoader
from graph.topology import load_topology


@pytest.fixture
def mock_loader() -> tuple[Neo4jLoader, MagicMock]:
    """Fixture providing a Neo4jLoader instance backed by a mocked driver."""
    with patch("neo4j.GraphDatabase.driver") as mock_driver_fn:
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver_fn.return_value = mock_driver

        loader = Neo4jLoader("bolt://mock:7687", "neo4j", "password")
        return loader, mock_session


def test_clear_all_refuses_without_confirm(mock_loader: tuple[Neo4jLoader, MagicMock]) -> None:
    """Ensure clear_all refuses to execute unless confirm=True is explicitly passed."""
    loader, session = mock_loader

    with pytest.raises(ValueError, match="confirm=True"):
        loader.clear_all(confirm=False)

    # Calling with confirm=True executes deletion
    loader.clear_all(confirm=True)
    session.run.assert_called_with("MATCH (n) DETACH DELETE n")


def test_parameterized_cypher_and_relationship_types(
    mock_loader: tuple[Neo4jLoader, MagicMock],
) -> None:
    """Ensure all Cypher statements are parameterized and use distinct relationship types."""
    loader, session = mock_loader

    topology = load_topology()
    flows = generate_synthetic_flows(seed=42, n_benign=30)
    window = build_windows(flows, window_seconds=600)[0]
    attach_topology_info(window, topology)

    # Load window and attack graph
    loader.load_window(window)
    loader.load_attack_graph(topology)

    observed_rel_types = set()
    simulated_rel_types = set()

    for call_args in session.run.call_args_list:
        query = call_args[0][0]
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("parameters", {})

        # Assert queries are parameterized with $batch
        assert "$batch" in query, f"Cypher query must be parameterized: {query}"
        assert "batch" in params, "Query parameters must supply 'batch'"

        # Verify no hardcoded string interpolation of flow IPs or values in query
        assert "203.0.113.50" not in query
        assert "10.0.1.10" not in query

        # Track relationship types used in observed vs simulated
        if "CONNECTS_TO" in query:
            observed_rel_types.add("CONNECTS_TO")
        if "HOSTED_ON" in query:
            observed_rel_types.add("HOSTED_ON")
        if "CAN_REACH" in query:
            simulated_rel_types.add("CAN_REACH")

    # Observed communication must use CONNECTS_TO / HOSTED_ON
    assert "CONNECTS_TO" in observed_rel_types
    assert "HOSTED_ON" in observed_rel_types

    # Simulated reachability must use CAN_REACH
    assert "CAN_REACH" in simulated_rel_types

    # Strict separation: observed and simulated relationship types must be disjoint
    assert observed_rel_types.isdisjoint(simulated_rel_types)


def test_batch_sizes_respected(mock_loader: tuple[Neo4jLoader, MagicMock]) -> None:
    """Ensure loaders break large records into chunks conforming to batch_size."""
    loader, session = mock_loader
    topology = load_topology()

    # Call load_attack_graph with small batch size of 5
    loader.load_attack_graph(topology, batch_size=5)

    for call_args in session.run.call_args_list:
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("parameters", {})
        batch = params.get("batch", [])
        assert len(batch) <= 5, f"Batch size exceeded: {len(batch)} > 5"


@pytest.mark.skipif(
    not os.getenv("NEO4J_URI"),
    reason="Integration test requires a live Neo4j instance via NEO4J_URI environment variable.",
)
def test_neo4j_live_integration() -> None:
    """Live integration test: loads a tiny window into Neo4j and verifies counts."""
    uri = os.environ["NEO4J_URI"]
    user = os.getenv("NEO4J_USER", "neo4j")
    pwd = os.getenv("NEO4J_PASSWORD", "change_me")

    with Neo4jLoader(uri, user, pwd) as loader:
        loader.clear_all(confirm=True)
        loader.create_constraints()

        topology = load_topology()
        loader.load_attack_graph(topology)

        flows = generate_synthetic_flows(seed=42, n_benign=10)
        window = build_windows(flows, window_seconds=600)[0]
        attach_topology_info(window, topology)
        loader.load_window(window)

        with loader._driver.session() as session:
            h_cnt = session.run("MATCH (h:Host) RETURN count(h) AS cnt").single()["cnt"]
            s_cnt = session.run("MATCH (s:Service) RETURN count(s) AS cnt").single()["cnt"]
            c_query = "MATCH ()-[r:CONNECTS_TO]->() RETURN count(r) AS cnt"
            c_cnt = session.run(c_query).single()["cnt"]
            r_query = "MATCH ()-[r:CAN_REACH]->() RETURN count(r) AS cnt"
            r_cnt = session.run(r_query).single()["cnt"]

            assert h_cnt > 0
            assert s_cnt > 0
            assert c_cnt > 0
            assert r_cnt > 0
