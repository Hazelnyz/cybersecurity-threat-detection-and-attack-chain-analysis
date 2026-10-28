"""Unit tests for synthetic network flow generation."""

import pandas as pd

from data.schema import validate_flows
from data.synthetic import generate_synthetic_flows


def test_synthetic_reproducibility() -> None:
    """Ensure same seed produces identical DataFrames and different seeds differ."""
    df1 = generate_synthetic_flows(seed=42, n_benign=100)
    df2 = generate_synthetic_flows(seed=42, n_benign=100)
    df3 = generate_synthetic_flows(seed=99, n_benign=100)

    pd.testing.assert_frame_equal(df1, df2)
    assert not df1.equals(df3)


def test_synthetic_attack_stages_temporal_order() -> None:
    """Ensure the four attack stages appear in strict chronological order with gaps."""
    df = generate_synthetic_flows(seed=42, n_benign=200)
    stages = ["recon", "initial_access", "lateral_movement", "exfiltration"]

    min_times = {}
    max_times = {}
    for stage in stages:
        stage_flows = df[df["attack_stage"] == stage]
        assert not stage_flows.empty, f"Stage {stage} should have flows"
        min_times[stage] = stage_flows["ts_start"].min()
        max_times[stage] = stage_flows["ts_start"].max()

    # Verify strict progression with temporal separation
    assert max_times["recon"] < min_times["initial_access"]
    assert max_times["initial_access"] < min_times["lateral_movement"]
    assert max_times["lateral_movement"] < min_times["exfiltration"]


def test_synthetic_labels_and_attack_metadata() -> None:
    """Ensure attack stages have label_binary == 1 and expected label_family."""
    df = generate_synthetic_flows(seed=42, n_benign=100)

    stage_family_map = {
        "recon": "portscan",
        "initial_access": "web_exploit",
        "lateral_movement": "lateral_movement",
        "exfiltration": "exfiltration",
    }

    for stage, expected_family in stage_family_map.items():
        sub = df[df["attack_stage"] == stage]
        assert (sub["label_binary"] == 1).all()
        assert (sub["label_family"] == expected_family).all()


def test_synthetic_hard_negatives() -> None:
    """Ensure hard negatives exist and have label_binary == 0."""
    df = generate_synthetic_flows(seed=42, n_benign=100)

    backup_flows = df[df["label_raw"] == "benign_backup"]
    admin_flows = df[df["label_raw"] == "benign_admin"]

    assert not backup_flows.empty, "Backup hard negatives must exist"
    assert not admin_flows.empty, "Admin hard negatives must exist"

    assert (backup_flows["label_binary"] == 0).all()
    assert (admin_flows["label_binary"] == 0).all()
    assert (backup_flows["label_family"] == "benign").all()
    assert (admin_flows["label_family"] == "benign").all()
    assert backup_flows["attack_stage"].isna().all()
    assert admin_flows["attack_stage"].isna().all()

    # Large transfer verification for backup
    assert (backup_flows["fwd_bytes"] > 5_000_000).all()


def test_synthetic_dup_key_stability() -> None:
    """Ensure dup_key values are stable across independent runs."""
    df1 = generate_synthetic_flows(seed=42, n_benign=50)
    df2 = generate_synthetic_flows(seed=42, n_benign=50)
    pd.testing.assert_series_equal(df1["dup_key"], df2["dup_key"])


def test_synthetic_with_custom_hosts() -> None:
    """Ensure generate_synthetic_flows accepts a custom host list."""
    custom_hosts = [
        {"id": "ext_attacker", "ip": "192.0.2.1", "segment": "external"},
        {"id": "ext_c2", "ip": "192.0.2.2", "segment": "external"},
        {"id": "dmz_web_custom", "ip": "172.16.1.10", "segment": "dmz"},
        {"id": "internal_db", "ip": "172.16.2.20", "segment": "internal"},
        {"id": "internal_ws", "ip": "172.16.3.30", "segment": "internal"},
    ]
    df = generate_synthetic_flows(seed=42, n_benign=50, hosts=custom_hosts)
    validate_flows(df)

    # Verify custom IPs are present in output
    all_ips = set(df["src_ip"]).union(set(df["dst_ip"]))
    assert "192.0.2.1" in all_ips
    assert "172.16.1.10" in all_ips

