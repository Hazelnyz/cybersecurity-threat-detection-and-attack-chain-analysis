"""Unit tests for time-windowed graph construction and feature extraction."""

import pandas as pd
import pytest

from data.schema import coerce_flows, make_dup_key, validate_flows
from data.synthetic import generate_synthetic_flows
from graph.builder import (
    FEATURE_COLUMNS_EDGE,
    FEATURE_COLUMNS_HOST,
    assert_no_leakage,
    attach_topology_info,
    build_windows,
    window_to_networkx,
)
from graph.topology import load_topology


def test_builder_determinism() -> None:
    """Ensure identical flow DataFrames produce identical GraphWindows."""
    flows = generate_synthetic_flows(seed=42, n_benign=100)
    windows1 = build_windows(flows, window_seconds=300)
    windows2 = build_windows(flows, window_seconds=300)

    assert len(windows1) == len(windows2)
    for w1, w2 in zip(windows1, windows2, strict=True):
        assert w1.window_id == w2.window_id
        assert w1.t_start == w2.t_start
        assert w1.t_end == w2.t_end
        pd.testing.assert_frame_equal(w1.nodes, w2.nodes)
        pd.testing.assert_frame_equal(w1.edges, w2.edges)


def test_windows_non_overlapping_and_coverage() -> None:
    """Verify windows are ordered, non-overlapping, and preserve all flow counts."""
    flows = generate_synthetic_flows(seed=42, n_benign=200)
    windows = build_windows(flows, window_seconds=300)

    assert len(windows) > 1

    total_flow_count_edges = 0
    total_window_flows = 0

    for i in range(len(windows)):
        w = windows[i]
        assert w.t_start < w.t_end
        total_flow_count_edges += int(w.edges["flow_count"].sum())
        total_window_flows += len(w.flows) if w.flows is not None else 0

        if i > 0:
            prev_w = windows[i - 1]
            assert prev_w.t_end <= w.t_start, "Windows must not overlap"

    # Every flow must land in exactly one window
    assert total_window_flows == len(flows)
    # Total flow_count across all edges equals input row count
    assert total_flow_count_edges == len(flows)


def test_assert_no_leakage() -> None:
    """Ensure assert_no_leakage passes on valid windows and raises on corruption."""
    flows = generate_synthetic_flows(seed=42, n_benign=50)
    windows = build_windows(flows, window_seconds=600)
    assert len(windows) > 0

    # Natural window passes
    w = windows[0]
    assert_no_leakage(w)

    # Corrupt by shifting window boundary early
    corrupted_w1 = build_windows(flows, window_seconds=600)[0]
    corrupted_w1.t_start = corrupted_w1.flows["ts_start"].max()  # flows now before t_start
    with pytest.raises(ValueError, match="Temporal leakage"):
        assert_no_leakage(corrupted_w1)

    # Corrupt by shifting window boundary backwards
    corrupted_w2 = build_windows(flows, window_seconds=600)[0]
    corrupted_w2.t_end = corrupted_w2.flows["ts_start"].min()  # flows now after t_end
    with pytest.raises(ValueError, match="Temporal leakage"):
        assert_no_leakage(corrupted_w2)


def test_hand_computed_node_features_on_tiny_example() -> None:
    """Verify host node features exactly match hand-computed values on a 6-flow scenario."""
    base_t = pd.Timestamp("2026-01-01 10:00:00", tz="UTC")

    records = [
        # Flow 1: A -> B on port 80 tcp (benign)
        {
            "ts_start": base_t + pd.Timedelta(seconds=1),
            "duration_s": 1.0,
            "src_ip": "10.0.0.1",
            "src_port": 50001,
            "dst_ip": "10.0.0.2",
            "dst_port": 80,
            "proto": "tcp",
            "fwd_pkts": 10,
            "bwd_pkts": 5,
            "fwd_bytes": 1000,
            "bwd_bytes": 500,
            "label_binary": 0,
            "label_family": "benign",
            "label_raw": "benign",
            "attack_stage": pd.NA,
        },
        # Flow 2: A -> B on port 443 tcp (benign)
        {
            "ts_start": base_t + pd.Timedelta(seconds=5),
            "duration_s": 2.0,
            "src_ip": "10.0.0.1",
            "src_port": 50002,
            "dst_ip": "10.0.0.2",
            "dst_port": 443,
            "proto": "tcp",
            "fwd_pkts": 20,
            "bwd_pkts": 10,
            "fwd_bytes": 2000,
            "bwd_bytes": 1000,
            "label_binary": 0,
            "label_family": "benign",
            "label_raw": "benign",
            "attack_stage": pd.NA,
        },
        # Flow 3: A -> C on port 80 tcp (attack)
        {
            "ts_start": base_t + pd.Timedelta(seconds=10),
            "duration_s": 3.0,
            "src_ip": "10.0.0.1",
            "src_port": 50003,
            "dst_ip": "10.0.0.3",
            "dst_port": 80,
            "proto": "tcp",
            "fwd_pkts": 30,
            "bwd_pkts": 15,
            "fwd_bytes": 3000,
            "bwd_bytes": 1500,
            "label_binary": 1,
            "label_family": "web_exploit",
            "label_raw": "exploit",
            "attack_stage": "initial_access",
        },
        # Flow 4: B -> C on port 22 tcp (benign)
        {
            "ts_start": base_t + pd.Timedelta(seconds=15),
            "duration_s": 4.0,
            "src_ip": "10.0.0.2",
            "src_port": 50004,
            "dst_ip": "10.0.0.3",
            "dst_port": 22,
            "proto": "tcp",
            "fwd_pkts": 40,
            "bwd_pkts": 20,
            "fwd_bytes": 4000,
            "bwd_bytes": 2000,
            "label_binary": 0,
            "label_family": "benign",
            "label_raw": "benign",
            "attack_stage": pd.NA,
        },
        # Flow 5: C -> B on port 80 tcp (benign)
        {
            "ts_start": base_t + pd.Timedelta(seconds=20),
            "duration_s": 5.0,
            "src_ip": "10.0.0.3",
            "src_port": 50005,
            "dst_ip": "10.0.0.2",
            "dst_port": 80,
            "proto": "tcp",
            "fwd_pkts": 50,
            "bwd_pkts": 25,
            "fwd_bytes": 5000,
            "bwd_bytes": 2500,
            "label_binary": 0,
            "label_family": "benign",
            "label_raw": "benign",
            "attack_stage": pd.NA,
        },
        # Flow 6: B -> A on port 53 udp (benign)
        {
            "ts_start": base_t + pd.Timedelta(seconds=25),
            "duration_s": 6.0,
            "src_ip": "10.0.0.2",
            "src_port": 50006,
            "dst_ip": "10.0.0.1",
            "dst_port": 53,
            "proto": "udp",
            "fwd_pkts": 60,
            "bwd_pkts": 30,
            "fwd_bytes": 6000,
            "bwd_bytes": 3000,
            "label_binary": 0,
            "label_family": "benign",
            "label_raw": "benign",
            "attack_stage": pd.NA,
        },
    ]

    df_flows = pd.DataFrame(records)
    df_flows["dataset"] = "synthetic"
    df_flows["flow_id"] = [f"f_{i}" for i in range(len(df_flows))]
    df_flows["dup_key"] = make_dup_key(df_flows)
    coerced = coerce_flows(df_flows)
    validate_flows(coerced)

    windows = build_windows(coerced, window_seconds=60)
    assert len(windows) == 1
    w = windows[0]

    nodes_df = w.nodes[w.nodes["node_type"] == "host"].set_index("ip")
    host_a = nodes_df.loc["10.0.0.1"]

    # Hand computed assertions for Host A:
    # Flows out: Flow 1, Flow 2, Flow 3 -> 3
    assert host_a["flows_out"] == 3
    # Flows in: Flow 6 -> 1
    assert host_a["flows_in"] == 1
    # Bytes out: 1000 + 2000 + 3000 = 6000
    assert host_a["bytes_out"] == 6000
    # Bytes in: 6000 (from Flow 6)
    assert host_a["bytes_in"] == 6000
    # Ratio out/in: 6000 / 6000 = 1.0
    assert host_a["ratio_out_in_bytes"] == pytest.approx(1.0)
    # Out degree services: (B, 80, tcp), (B, 443, tcp), (C, 80, tcp) -> 3
    assert host_a["out_degree_services"] == 3
    # In degree services: (53, udp) -> 1
    assert host_a["in_degree_services"] == 1
    # Distinct destination hosts out: B, C -> 2
    assert host_a["distinct_dst_hosts_out"] == 2
    # Distinct destination ports out: 80, 443 -> 2
    assert host_a["distinct_dst_ports_out"] == 2
    # Distinct source hosts in: B -> 1
    assert host_a["distinct_src_hosts_in"] == 1
    # Mean flow duration: (1.0 + 2.0 + 3.0) / 3 = 2.0 s
    assert host_a["mean_flow_duration"] == pytest.approx(2.0)
    # Host A originated Flow 3 (attack flow) -> label_binary must be 1
    assert host_a["label_binary"] == 1


def test_edge_label_aggregation_rules() -> None:
    """Verify edge labels: any attack -> 1, mixed/benign -> 0, all unlabeled -> -1."""
    base_t = pd.Timestamp("2026-01-01 10:00:00", tz="UTC")

    def _make_df(labels: list[int]) -> pd.DataFrame:
        rows = []
        for i, lbl in enumerate(labels):
            rows.append({
                "ts_start": base_t + pd.Timedelta(seconds=i),
                "duration_s": 0.5,
                "src_ip": "10.0.0.1",
                "src_port": 50000 + i,
                "dst_ip": "10.0.0.2",
                "dst_port": 80,
                "proto": "tcp",
                "fwd_pkts": 5,
                "bwd_pkts": 5,
                "fwd_bytes": 500,
                "bwd_bytes": 500,
                "label_binary": lbl,
                "label_family": "portscan" if lbl == 1 else "benign",
                "label_raw": "raw",
                "attack_stage": "recon" if lbl == 1 else pd.NA,
                "dataset": "synthetic",
                "flow_id": f"flow_{i}",
            })
        df = pd.DataFrame(rows)
        df["dup_key"] = make_dup_key(df)
        return coerce_flows(df)

    # 1. Any attack (mixed benign + attack) -> label_binary = 1
    w_attack = build_windows(_make_df([0, 1]), window_seconds=60)[0]
    conn_edge = w_attack.edges[w_attack.edges["edge_type"] == "connects_to"].iloc[0]
    assert conn_edge["label_binary"] == 1
    assert conn_edge["label_family"] == "portscan"

    # 2. Mixed benign and unlabeled -> label_binary = 0
    w_mixed = build_windows(_make_df([0, -1]), window_seconds=60)[0]
    conn_edge2 = w_mixed.edges[w_mixed.edges["edge_type"] == "connects_to"].iloc[0]
    assert conn_edge2["label_binary"] == 0
    assert conn_edge2["label_family"] == "benign"

    # 3. All unlabeled -> label_binary = -1
    w_unlabeled = build_windows(_make_df([-1, -1]), window_seconds=60)[0]
    conn_edge3 = w_unlabeled.edges[w_unlabeled.edges["edge_type"] == "connects_to"].iloc[0]
    assert conn_edge3["label_binary"] == -1


def test_feature_columns_contain_no_identifiers() -> None:
    """Ensure FEATURE_COLUMNS_* exclude all identifier, IP, timestamp, and label-derived columns."""
    forbidden_identifiers = [
        "ip",
        "node_idx",
        "window_id",
        "flow_id",
        "timestamp",
        "ts",
        "src_ip",
        "dst_ip",
    ]
    forbidden_labels = [
        "label_binary",
        "label_family",
        "attack_stage",
        "n_flows_attack",
    ]
    all_forbidden = set(forbidden_identifiers).union(set(forbidden_labels))

    for col in FEATURE_COLUMNS_HOST:
        assert col not in all_forbidden, f"Host feature '{col}' is forbidden (identifier or label)"
        assert not col.endswith("_ip"), f"Host feature '{col}' resembles an IP address"

    for col in FEATURE_COLUMNS_EDGE:
        assert col not in all_forbidden, f"Edge feature '{col}' is forbidden (identifier or label)"
        assert not col.endswith("_ip"), f"Edge feature '{col}' resembles an IP address"


def test_no_look_ahead_temporal_isolation() -> None:
    """Ensure adding flows after a window's t_end does not alter that window's outputs."""
    flows = generate_synthetic_flows(seed=42, n_benign=100)
    windows_full = build_windows(flows, window_seconds=300)

    # Cut flows to only those in the first window
    w0_end = windows_full[0].t_end
    flows_first_win = flows[flows["ts_start"] < w0_end].copy()
    windows_sub = build_windows(flows_first_win, window_seconds=300)

    # First window must be identical regardless of subsequent flows
    pd.testing.assert_frame_equal(windows_full[0].nodes, windows_sub[0].nodes)
    pd.testing.assert_frame_equal(windows_full[0].edges, windows_sub[0].edges)


def test_attach_topology_info_with_unknown_ips() -> None:
    """Ensure attach_topology_info correctly handles known and unknown IPs."""
    topology = load_topology()
    flows = generate_synthetic_flows(seed=42, n_benign=50)
    window = build_windows(flows, window_seconds=600)[0]

    attached = attach_topology_info(window, topology)
    assert "segment" in attached.nodes.columns
    assert "asset_criticality" in attached.nodes.columns
    assert "host_id" in attached.nodes.columns

    # Verify a known IP from topology
    web_node = attached.nodes[attached.nodes["ip"] == "10.0.1.10"]
    if not web_node.empty:
        assert (web_node["segment"] == "dmz").all()
        assert (web_node["asset_criticality"] == 3).all()
        assert (web_node["host_id"] == "web-01").all()

    # Verify an unknown IP gets segment="unknown"
    unknown_node = attached.nodes[attached.nodes["ip"] == "192.0.2.254"]
    if not unknown_node.empty:
        assert (unknown_node["segment"] == "unknown").all()
        assert unknown_node["asset_criticality"].isna().all()


def test_window_to_networkx() -> None:
    """Ensure window_to_networkx constructs a valid MultiDiGraph with nodes and edges."""
    flows = generate_synthetic_flows(seed=42, n_benign=30)
    window = build_windows(flows, window_seconds=600)[0]
    g = window_to_networkx(window)

    assert g.number_of_nodes() == len(window.nodes)
    assert g.number_of_edges() == len(window.edges)
