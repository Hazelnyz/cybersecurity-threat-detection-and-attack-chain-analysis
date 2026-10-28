"""Demo script to generate synthetic flows, build graph windows, and load into Neo4j."""

import logging
import sys
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings
from data.synthetic import generate_synthetic_flows
from graph.builder import attach_topology_info, build_windows
from graph.neo4j_loader import Neo4jLoader
from graph.topology import load_topology

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("load_demo")


def run_demo() -> None:
    """Execute end-to-end demo loading synthetic graph windows into Neo4j."""
    settings = get_settings()

    logger.info("Loading enterprise topology...")
    topology = load_topology()

    logger.info("Generating synthetic network flows (seed=42)...")
    host_dicts = [{"id": h.id, "ip": h.ip, "segment": h.segment} for h in topology.hosts]
    flows = generate_synthetic_flows(seed=42, n_benign=1000, hosts=host_dicts)
    logger.info("Generated %d synthetic flows.", len(flows))

    logger.info("Building time-windowed subgraphs (window=300s)...")
    windows = build_windows(flows, window_seconds=300)
    logger.info("Constructed %d graph windows.", len(windows))

    logger.info("Enriching windows with topology metadata...")
    for w in windows:
        attach_topology_info(w, topology)

    logger.info("Connecting to Neo4j at %s...", settings.neo4j_uri)
    try:
        with Neo4jLoader(
            settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password
        ) as loader:
            logger.info("Resetting Neo4j database and applying uniqueness constraints...")
            loader.clear_all(confirm=True)
            loader.create_constraints()

            logger.info("Loading simulated attack graph (CAN_REACH edges)...")
            loader.load_attack_graph(topology)

            logger.info(
                "Loading %d observed graph windows (CONNECTS_TO / HOSTED_ON)...", len(windows)
            )
            for w in windows:
                loader.load_window(w)

            # Query and report counts
            with loader._driver.session() as session:
                n_hosts = session.run("MATCH (h:Host) RETURN count(h) AS cnt").single()["cnt"]
                n_services = session.run(
                    "MATCH (s:Service) RETURN count(s) AS cnt"
                ).single()["cnt"]
                n_connects = session.run(
                    "MATCH ()-[r:CONNECTS_TO]->() RETURN count(r) AS cnt"
                ).single()["cnt"]
                n_hosted = session.run(
                    "MATCH ()-[r:HOSTED_ON]->() RETURN count(r) AS cnt"
                ).single()["cnt"]
                n_can_reach = session.run(
                    "MATCH ()-[r:CAN_REACH]->() RETURN count(r) AS cnt"
                ).single()["cnt"]

            logger.info("=== Neo4j Loading Summary ===")
            logger.info("  Host nodes:           %d", n_hosts)
            logger.info("  Service nodes:        %d", n_services)
            logger.info("  CONNECTS_TO edges:    %d (observed)", n_connects)
            logger.info("  HOSTED_ON edges:      %d (observed structural)", n_hosted)
            logger.info("  CAN_REACH edges:      %d (simulated reachability)", n_can_reach)
            logger.info("=============================")

    except Exception as err:
        logger.warning(
            "Could not connect to Neo4j at %s (%s). "
            "To run with live Neo4j, start containers with 'make up' or 'docker compose up -d'.",
            settings.neo4j_uri,
            err,
        )


if __name__ == "__main__":
    run_demo()

