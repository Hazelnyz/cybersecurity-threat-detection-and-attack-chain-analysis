"""Neo4j graph persistence loader for observed windows and simulated reachability."""

from types import TracebackType
from typing import Any

import pandas as pd
from neo4j import Driver, GraphDatabase

from graph.builder import GraphWindow
from graph.topology import Topology, build_attack_graph

BATCH_SIZE: int = 1000


def _format_service_key(key: tuple[Any, ...]) -> str:
    """Format a service node tuple key into a unique string identifier.

    Args:
        key: Tuple of ('service', ip, port, proto).

    Returns:
        str: Canonical string key for Neo4j indexing.
    """
    _, ip, port, proto = key
    port_str = "null" if (port is None or pd.isna(port)) else str(int(port))
    return f"{ip}:{port_str}:{proto}"


class Neo4jLoader:
    """Manages connection and loading of ThreatGraph records into a Neo4j database."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        """Initialize Neo4j driver connection.

        Args:
            uri: Bolt or Neo4j URI (e.g. bolt://localhost:7687).
            user: Username (e.g. neo4j).
            password: Password.
        """
        self.uri = uri
        self.user = user
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        """Close active Neo4j driver connections."""
        if self._driver is not None:
            self._driver.close()

    def __enter__(self) -> "Neo4jLoader":
        """Support context-manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Support context-manager exit."""
        self.close()

    def create_constraints(self) -> None:
        """Create uniqueness constraints for Host and Service nodes."""
        queries = [
            "CREATE CONSTRAINT host_ip_unique IF NOT EXISTS FOR (h:Host) REQUIRE h.ip IS UNIQUE",
            (
                "CREATE CONSTRAINT service_key_unique IF NOT EXISTS "
                "FOR (s:Service) REQUIRE s.key IS UNIQUE"
            ),
        ]
        with self._driver.session() as session:
            for q in queries:
                session.run(q)

    def clear_all(self, confirm: bool = False) -> None:
        """Delete all nodes and relationships in the database.

        Args:
            confirm: Guard flag; must be True to execute deletion.

        Raises:
            ValueError: If confirm is False.
        """
        if not confirm:
            raise ValueError(
                "Refusing to clear database: clear_all requires confirm=True."
            )
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def load_window(self, window: GraphWindow, batch_size: int = BATCH_SIZE) -> None:
        """Load an observed GraphWindow into Neo4j using batched, parameterized Cypher.

        Creates Host and Service nodes and observed edges:
        - (:Host)-[:CONNECTS_TO {source: "observed", ...features}]->(:Service)
        - (:Service)-[:HOSTED_ON {source: "observed"}]->(:Host)

        Args:
            window: GraphWindow instance containing nodes and edges.
            batch_size: Maximum rows per batched UNWIND query (defaults to 1000).
        """
        # Separate host and service node records
        host_records: list[dict[str, Any]] = []
        service_records: list[dict[str, Any]] = []

        for _, row in window.nodes.iterrows():
            if row["node_type"] == "host":
                segment_val = (
                    str(row["segment"])
                    if "segment" in row and pd.notna(row["segment"])
                    else "unknown"
                )
                crit_val = (
                    int(row["asset_criticality"])
                    if "asset_criticality" in row and pd.notna(row["asset_criticality"])
                    else None
                )
                host_id_val = (
                    str(row["host_id"])
                    if "host_id" in row and pd.notna(row["host_id"])
                    else None
                )
                host_records.append({
                    "ip": str(row["ip"]),
                    "segment": segment_val,
                    "asset_criticality": crit_val,
                    "host_id": host_id_val,
                    "node_idx": int(row["node_idx"]),
                })
            elif row["node_type"] == "service":
                service_records.append({
                    "key": _format_service_key(row["node_key"]),
                    "ip": str(row["ip"]),
                    "port": int(row["port"]) if pd.notna(row["port"]) else None,
                    "proto": str(row["proto"]),
                    "node_idx": int(row["node_idx"]),
                })

        # Separate connects_to and hosted_on edge records
        connects_records: list[dict[str, Any]] = []
        hosted_records: list[dict[str, Any]] = []

        for _, row in window.edges.iterrows():
            if row["edge_type"] == "connects_to":
                atk_stage = (
                    str(row["attack_stage"]) if pd.notna(row["attack_stage"]) else None
                )
                connects_records.append({
                    "src_ip": str(row["src_key"][1]),
                    "dst_key": _format_service_key(row["dst_key"]),
                    "window_id": int(window.window_id),
                    "flow_count": int(row["flow_count"]),
                    "total_fwd_bytes": int(row["total_fwd_bytes"]),
                    "total_bwd_bytes": int(row["total_bwd_bytes"]),
                    "total_fwd_pkts": int(row["total_fwd_pkts"]),
                    "total_bwd_pkts": int(row["total_bwd_pkts"]),
                    "mean_duration_s": float(row["mean_duration_s"]),
                    "max_duration_s": float(row["max_duration_s"]),
                    "distinct_src_ports": int(row["distinct_src_ports"]),
                    "label_binary": int(row["label_binary"]),
                    "label_family": str(row["label_family"]),
                    "attack_stage": atk_stage,
                    "n_flows_attack": int(row["n_flows_attack"]),
                    "source": "observed",
                })
            elif row["edge_type"] == "hosted_on":
                hosted_records.append({
                    "service_key": _format_service_key(row["src_key"]),
                    "host_ip": str(row["dst_key"][1]),
                    "source": "observed",
                })

        cypher_merge_hosts = """
        UNWIND $batch AS row
        MERGE (h:Host {ip: row.ip})
        SET h.segment = row.segment,
            h.asset_criticality = row.asset_criticality,
            h.host_id = row.host_id,
            h.node_idx = row.node_idx
        """

        cypher_merge_services = """
        UNWIND $batch AS row
        MERGE (s:Service {key: row.key})
        SET s.ip = row.ip,
            s.port = row.port,
            s.proto = row.proto,
            s.node_idx = row.node_idx
        """

        cypher_create_connects = """
        UNWIND $batch AS row
        MATCH (h:Host {ip: row.src_ip})
        MATCH (s:Service {key: row.dst_key})
        CREATE (h)-[r:CONNECTS_TO {
            window_id: row.window_id,
            flow_count: row.flow_count,
            total_fwd_bytes: row.total_fwd_bytes,
            total_bwd_bytes: row.total_bwd_bytes,
            total_fwd_pkts: row.total_fwd_pkts,
            total_bwd_pkts: row.total_bwd_pkts,
            mean_duration_s: row.mean_duration_s,
            max_duration_s: row.max_duration_s,
            distinct_src_ports: row.distinct_src_ports,
            label_binary: row.label_binary,
            label_family: row.label_family,
            attack_stage: row.attack_stage,
            n_flows_attack: row.n_flows_attack,
            source: row.source
        }]->(s)
        """

        cypher_merge_hosted = """
        UNWIND $batch AS row
        MATCH (s:Service {key: row.service_key})
        MATCH (h:Host {ip: row.host_ip})
        MERGE (s)-[r:HOSTED_ON {source: row.source}]->(h)
        """

        with self._driver.session() as session:
            for i in range(0, len(host_records), batch_size):
                session.run(cypher_merge_hosts, {"batch": host_records[i : i + batch_size]})

            for i in range(0, len(service_records), batch_size):
                session.run(cypher_merge_services, {"batch": service_records[i : i + batch_size]})

            for i in range(0, len(hosted_records), batch_size):
                session.run(cypher_merge_hosted, {"batch": hosted_records[i : i + batch_size]})

            for i in range(0, len(connects_records), batch_size):
                session.run(
                    cypher_create_connects, {"batch": connects_records[i : i + batch_size]}
                )

    def load_attack_graph(
        self, topology: Topology, batch_size: int = BATCH_SIZE
    ) -> None:
        """Load simulated reachability edges into Neo4j using CAN_REACH relationships.

        Uses the relationship type CAN_REACH with source="simulated" to ensure
        complete structural separation from observed CONNECTS_TO relationships.

        Args:
            topology: Enterprise network topology.
            batch_size: Maximum batch size per parameterised query.
        """
        attack_graph = build_attack_graph(topology)

        # Merge host nodes
        host_records = [
            {
                "ip": host.ip,
                "segment": host.segment,
                "asset_criticality": host.asset_criticality,
                "host_id": host.id,
            }
            for host in topology.hosts
        ]

        cypher_merge_hosts = """
        UNWIND $batch AS row
        MERGE (h:Host {ip: row.ip})
        SET h.segment = row.segment,
            h.asset_criticality = row.asset_criticality,
            h.host_id = row.host_id
        """

        # Simulated reachability edges
        reach_records = []
        for u, v, data in attack_graph.edges(data=True):
            src_node = attack_graph.nodes[u]
            dst_node = attack_graph.nodes[v]
            reach_records.append({
                "src_ip": src_node["ip"],
                "dst_ip": dst_node["ip"],
                "p": float(data["p"]),
                "cost": float(data["cost"]),
                "source": "simulated",
            })

        cypher_merge_can_reach = """
        UNWIND $batch AS row
        MATCH (src:Host {ip: row.src_ip})
        MATCH (dst:Host {ip: row.dst_ip})
        MERGE (src)-[r:CAN_REACH {
            p: row.p,
            cost: row.cost,
            source: row.source
        }]->(dst)
        """

        with self._driver.session() as session:
            for i in range(0, len(host_records), batch_size):
                session.run(cypher_merge_hosts, {"batch": host_records[i : i + batch_size]})

            for i in range(0, len(reach_records), batch_size):
                session.run(cypher_merge_can_reach, {"batch": reach_records[i : i + batch_size]})
