"""Unit tests for enterprise topology loading, validation, and attack graph modeling."""

import inspect

import networkx as nx
import pytest

import graph.topology as topo_module
from data.synthetic import generate_synthetic_flows
from graph.topology import (
    Topology,
    base_traversal_probability,
    build_attack_graph,
    ip_to_host,
    load_topology,
    reachable_services,
)


def _base_valid_dict() -> dict:
    """Helper returning a minimal valid topology dictionary."""
    return {
        "segments": ["lan", "wan"],
        "hosts": [
            {
                "id": "h1",
                "ip": "10.0.0.1",
                "segment": "lan",
                "os": "Linux",
                "asset_criticality": 3,
                "services": [
                    {
                        "name": "ssh",
                        "port": 22,
                        "proto": "tcp",
                        "vuln": {
                            "id": "V-1",
                            "name": "vuln1",
                            "exploitability": 0.5,
                        },
                    }
                ],
            },
            {
                "id": "h2",
                "ip": "10.0.0.2",
                "segment": "wan",
                "os": "Windows",
                "asset_criticality": 2,
                "services": [{"name": "http", "port": 80, "proto": "tcp"}],
            },
        ],
        "firewall_rules": [
            {
                "from_segment": "lan",
                "to_segment": "wan",
                "ports": [80],
                "proto": "tcp",
                "action": "allow",
            }
        ],
        "crown_jewels": ["h1"],
        "entry_points": ["h2"],
    }


def test_shipped_topology_loads_and_validates() -> None:
    """Ensure the shipped topology.yaml loads cleanly and satisfies all model constraints."""
    topology = load_topology()
    assert len(topology.hosts) >= 18
    assert "internet" in topology.segments
    assert "dmz" in topology.segments
    assert "dc-01" in topology.crown_jewels
    assert "db-01" in topology.crown_jewels
    assert "web-01" in topology.entry_points
    assert "mail-01" in topology.entry_points


def test_validation_duplicate_host_id() -> None:
    """Ensure duplicate host ID is rejected."""
    data = _base_valid_dict()
    data["hosts"][1]["id"] = "h1"
    with pytest.raises(ValueError, match="Duplicate host id"):
        Topology.model_validate(data)


def test_validation_duplicate_host_ip() -> None:
    """Ensure duplicate host IP is rejected."""
    data = _base_valid_dict()
    data["hosts"][1]["ip"] = "10.0.0.1"
    with pytest.raises(ValueError, match="Duplicate host IP"):
        Topology.model_validate(data)


def test_validation_invalid_ip_format() -> None:
    """Ensure malformed IP addresses are rejected."""
    data = _base_valid_dict()
    data["hosts"][0]["ip"] = "999.999.999.999"
    with pytest.raises(ValueError, match="Invalid IP address"):
        Topology.model_validate(data)


def test_validation_undefined_host_segment() -> None:
    """Ensure hosts cannot reference non-existent segments."""
    data = _base_valid_dict()
    data["hosts"][0]["segment"] = "unknown_zone"
    with pytest.raises(ValueError, match="references undefined segment"):
        Topology.model_validate(data)


def test_validation_undefined_firewall_rule_from_segment() -> None:
    """Ensure firewall rules reject unknown from_segment."""
    data = _base_valid_dict()
    data["firewall_rules"][0]["from_segment"] = "nonexistent"
    with pytest.raises(ValueError, match="references undefined from_segment"):
        Topology.model_validate(data)


def test_validation_undefined_firewall_rule_to_segment() -> None:
    """Ensure firewall rules reject unknown to_segment."""
    data = _base_valid_dict()
    data["firewall_rules"][0]["to_segment"] = "nonexistent"
    with pytest.raises(ValueError, match="references undefined to_segment"):
        Topology.model_validate(data)


def test_validation_invalid_asset_criticality() -> None:
    """Ensure asset criticality must be in 1..5."""
    data = _base_valid_dict()
    data["hosts"][0]["asset_criticality"] = 0
    with pytest.raises(ValueError, match="Asset criticality must be in 1..5"):
        Topology.model_validate(data)

    data["hosts"][0]["asset_criticality"] = 6
    with pytest.raises(ValueError, match="Asset criticality must be in 1..5"):
        Topology.model_validate(data)


def test_validation_invalid_exploitability() -> None:
    """Ensure vulnerability exploitability must be in (0, 1]."""
    data = _base_valid_dict()
    data["hosts"][0]["services"][0]["vuln"]["exploitability"] = 0.0
    with pytest.raises(ValueError, match="exploitability must be in \\(0, 1\\]"):
        Topology.model_validate(data)

    data["hosts"][0]["services"][0]["vuln"]["exploitability"] = 1.05
    with pytest.raises(ValueError, match="exploitability must be in \\(0, 1\\]"):
        Topology.model_validate(data)


def test_validation_invalid_port() -> None:
    """Ensure service and firewall ports must be in 0..65535."""
    data = _base_valid_dict()
    data["hosts"][0]["services"][0]["port"] = 70000
    with pytest.raises(ValueError, match="Service port must be in 0..65535"):
        Topology.model_validate(data)

    data_rule = _base_valid_dict()
    data_rule["firewall_rules"][0]["ports"] = [-5]
    with pytest.raises(ValueError, match="Firewall rule port must be in 0..65535"):
        Topology.model_validate(data_rule)


def test_validation_unknown_crown_jewel() -> None:
    """Ensure crown jewels must reference existing hosts."""
    data = _base_valid_dict()
    data["crown_jewels"] = ["ghost_host"]
    with pytest.raises(ValueError, match="Crown jewel references unknown host"):
        Topology.model_validate(data)


def test_validation_unknown_entry_point() -> None:
    """Ensure entry points must reference existing hosts."""
    data = _base_valid_dict()
    data["entry_points"] = ["ghost_host"]
    with pytest.raises(ValueError, match="Entry point references unknown host"):
        Topology.model_validate(data)


def test_segment_policy_default_deny_and_allow() -> None:
    """Verify default-allow within the same segment and default-deny across segments."""
    topology = load_topology()
    mapping = {h.id: h for h in topology.hosts}

    ws1 = mapping["ws-01"]
    ws2 = mapping["ws-02"]
    # Intra-segment (both user_lan): default-allow all services
    reachable_intra = reachable_services(ws1, ws2, topology)
    assert len(reachable_intra) == len(ws2.services)

    # Inter-segment: internet -> user_lan has no allow rule, so default-deny
    attacker = mapping["attacker-ext"]
    admin_ws = mapping["admin-ws-01"]
    reachable_inter_denied = reachable_services(attacker, admin_ws, topology)
    assert len(reachable_inter_denied) == 0

    p_denied = base_traversal_probability(attacker, admin_ws, topology)
    assert p_denied == 0.0


def test_internet_reachability_rules() -> None:
    """Ensure internet cannot reach user_lan but can reach allowed dmz web ports."""
    topology = load_topology()
    graph = build_attack_graph(topology)

    assert not graph.has_edge("attacker-ext", "admin-ws-01")
    assert not graph.has_edge("attacker-ext", "ws-01")

    # attacker-ext can reach web-01
    assert graph.has_edge("attacker-ext", "web-01")
    edge = graph.edges["attacker-ext", "web-01"]
    assert edge["edge_type"] == "simulated_reachability"
    assert edge["source"] == "simulated"
    assert edge["p"] > 0.0


def test_attack_graph_probabilities_and_costs() -> None:
    """Verify all attack-graph edges satisfy 0 < p <= 1 and cost >= 0."""
    topology = load_topology()
    graph = build_attack_graph(topology)

    assert graph.number_of_nodes() == len(topology.hosts)
    assert graph.number_of_edges() > 0

    for u, v, data in graph.edges(data=True):
        p = data["p"]
        cost = data["cost"]
        assert 0.0 < p <= 1.0, f"Edge ({u}, {v}) has invalid probability p={p}"
        assert cost >= 0.0, f"Edge ({u}, {v}) has negative cost={cost}"
        assert data["edge_type"] == "simulated_reachability"
        assert data["source"] == "simulated"


def test_observed_traffic_cannot_create_attack_edge() -> None:
    """Assert that topology functions do not take flow DataFrames as input."""
    for name, fn in inspect.getmembers(topo_module, inspect.isfunction):
        sig = inspect.signature(fn)
        for param_name, _param in sig.parameters.items():
            assert param_name not in ("flows", "flow_df", "df", "events"), (
                f"Function {name} must not accept observed flow parameters"
            )


def test_synthetic_flows_align_with_topology_and_lateral_movement_paths() -> None:
    """Ensure attack flows map to topology hosts and lateral movement flows have attack paths."""
    topology = load_topology()
    mapping = ip_to_host(topology)
    attack_graph = build_attack_graph(topology)

    host_dicts = [{"id": h.id, "ip": h.ip, "segment": h.segment} for h in topology.hosts]
    flows = generate_synthetic_flows(seed=42, n_benign=100, hosts=host_dicts)

    attack_flows = flows[flows["label_binary"] == 1]
    assert not attack_flows.empty

    # Assert every src_ip and dst_ip in attack flows exists in topology
    for _, row in attack_flows.iterrows():
        assert row["src_ip"] in mapping, f"src_ip {row['src_ip']} not in topology"
        assert row["dst_ip"] in mapping, f"dst_ip {row['dst_ip']} not in topology"

    # Assert each lateral-movement flow has a reachable path in the attack graph
    lateral_flows = flows[flows["attack_stage"] == "lateral_movement"]
    assert not lateral_flows.empty

    for _, row in lateral_flows.iterrows():
        src_id = mapping[row["src_ip"]].id
        dst_id = mapping[row["dst_ip"]].id
        has_path = nx.has_path(attack_graph, src_id, dst_id)
        assert has_path, f"No reachable attack-graph path between {src_id} and {dst_id}"
