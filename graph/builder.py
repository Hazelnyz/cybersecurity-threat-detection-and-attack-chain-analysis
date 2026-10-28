"""Time-windowed graph construction from network flows."""

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from config import WINDOW_SECONDS
from graph.topology import Topology, ip_to_host

# Numeric model feature column whitelist (strictly excludes identifiers and timestamps)
FEATURE_COLUMNS_HOST: list[str] = [
    "out_degree_services",
    "in_degree_services",
    "distinct_dst_hosts_out",
    "distinct_dst_ports_out",
    "distinct_src_hosts_in",
    "bytes_out",
    "bytes_in",
    "flows_out",
    "flows_in",
    "mean_flow_duration",
    "ratio_out_in_bytes",
]

FEATURE_COLUMNS_EDGE: list[str] = [
    "flow_count",
    "total_fwd_bytes",
    "total_bwd_bytes",
    "total_fwd_pkts",
    "total_bwd_pkts",
    "mean_duration_s",
    "max_duration_s",
    "distinct_src_ports",
    "n_flows_attack",
]


@dataclass
class GraphWindow:
    """A discrete, non-overlapping time-windowed sub-graph of observed communications."""

    window_id: int
    t_start: pd.Timestamp
    t_end: pd.Timestamp
    nodes: pd.DataFrame
    edges: pd.DataFrame
    node_index: dict[tuple[Any, ...], int]
    flows: pd.DataFrame | None = None


def assert_no_leakage(window: GraphWindow) -> None:
    """Verify that every flow in the window strictly falls inside [t_start, t_end).

    Args:
        window: The GraphWindow to verify.

    Raises:
        ValueError: If any flow timestamp lies outside [t_start, t_end).
    """
    if window.flows is None or window.flows.empty:
        return

    ts = window.flows["ts_start"]
    leaked_early = ts < window.t_start
    leaked_late = ts >= window.t_end

    if leaked_early.any() or leaked_late.any():
        n_early = int(leaked_early.sum())
        n_late = int(leaked_late.sum())
        raise ValueError(
            f"Temporal leakage in window {window.window_id} [{window.t_start}, {window.t_end}): "
            f"{n_early} flows before t_start, {n_late} flows at or after t_end."
        )


def _compute_edge_label_binary(labels: pd.Series) -> int:
    """Determine aggregate edge label_binary per priority rule.

    1 if any attack, 0 if all benign or mixed benign/unlabeled, -1 if all unlabeled.
    """
    if (labels == 1).any():
        return 1
    if (labels == 0).any():
        return 0
    return -1


def _compute_edge_label_family(group: pd.DataFrame) -> str:
    """Determine most frequent attack family if attack flows exist, else 'benign'."""
    attacks = group[group["label_binary"] == 1]
    if not attacks.empty:
        return str(attacks["label_family"].mode().iloc[0])
    return "benign"


def _compute_edge_attack_stage(group: pd.DataFrame) -> Any:
    """Select the most recent non-null attack_stage in the group, or pd.NA."""
    valid_stages = group.dropna(subset=["attack_stage"])
    if not valid_stages.empty:
        return valid_stages.sort_values(by="ts_start", ascending=False).iloc[0]["attack_stage"]
    return pd.NA


def build_windows(
    flows: pd.DataFrame,
    window_seconds: int = WINDOW_SECONDS,
) -> list[GraphWindow]:
    """Partition network flows into tumbling, non-overlapping graph windows.

    Windows are aligned to the floor of the earliest timestamp. Empty windows
    are skipped. For each window, extracts host and service nodes along with
    observed communication ('connects_to') and structural ('hosted_on') edges.

    Args:
        flows: Coerced and validated flow DataFrame.
        window_seconds: Duration of each tumbling window in seconds.

    Returns:
        list[GraphWindow]: Chronologically ordered list of GraphWindow instances.
    """
    if flows.empty:
        return []

    # Floor base timestamp to integer multiple of window_seconds
    min_ts = flows["ts_start"].min()
    base_epoch_sec = (int(min_ts.timestamp()) // window_seconds) * window_seconds
    t_base = pd.Timestamp(base_epoch_sec, unit="s", tz="UTC")

    # Vectorized computation of window index
    flow_sec = flows["ts_start"].astype("int64") // 10**9
    w_indices = (flow_sec - base_epoch_sec) // window_seconds

    windows: list[GraphWindow] = []

    # Process each non-empty tumbling window
    for window_id, (w_idx, w_flows) in enumerate(flows.groupby(w_indices, sort=True)):
        t_win_start = t_base + pd.Timedelta(seconds=int(w_idx) * window_seconds)
        t_win_end = t_win_start + pd.Timedelta(seconds=window_seconds)

        # -------------------------------------------------------------
        # 1. Observed Communication Edges ("connects_to")
        # -------------------------------------------------------------
        edge_groups = w_flows.groupby(
            ["src_ip", "dst_ip", "dst_port", "proto"], as_index=False, dropna=False
        )

        connects_records = []
        for (src_ip, dst_ip, dst_port, proto), grp in edge_groups:
            n_attack = int((grp["label_binary"] == 1).sum())
            lbl_binary = _compute_edge_label_binary(grp["label_binary"])
            lbl_family = _compute_edge_label_family(grp)
            atk_stage = _compute_edge_attack_stage(grp)

            p_val = None if pd.isna(dst_port) else int(dst_port)
            connects_records.append({
                "src_key": ("host", src_ip),
                "dst_key": ("service", dst_ip, p_val, proto),
                "edge_type": "connects_to",
                "source": "observed",
                "flow_count": int(len(grp)),
                "total_fwd_bytes": int(grp["fwd_bytes"].sum()),
                "total_bwd_bytes": int(grp["bwd_bytes"].sum()),
                "total_fwd_pkts": int(grp["fwd_pkts"].sum()),
                "total_bwd_pkts": int(grp["bwd_pkts"].sum()),
                "mean_duration_s": float(grp["duration_s"].mean()),
                "max_duration_s": float(grp["duration_s"].max()),
                "distinct_src_ports": int(grp["src_port"].nunique()),
                "label_binary": np.int8(lbl_binary),
                "label_family": str(lbl_family),
                "attack_stage": atk_stage,
                "n_flows_attack": n_attack,
            })

        df_connects = (
            pd.DataFrame(connects_records)
            if connects_records
            else pd.DataFrame(columns=[
                "src_key", "dst_key", "edge_type", "source", "flow_count",
                "total_fwd_bytes", "total_bwd_bytes", "total_fwd_pkts",
                "total_bwd_pkts", "mean_duration_s", "max_duration_s",
                "distinct_src_ports", "label_binary", "label_family",
                "attack_stage", "n_flows_attack",
            ])
        )

        # -------------------------------------------------------------
        # 2. Structural Edges ("hosted_on"): Service -> Host
        # -------------------------------------------------------------
        # Distinct services observed as destinations in this window
        distinct_services = (
            w_flows[["dst_ip", "dst_port", "proto"]]
            .drop_duplicates()
            .values
        )

        hosted_records = []
        for dst_ip, dst_port, proto in distinct_services:
            p_val = None if pd.isna(dst_port) else int(dst_port)
            hosted_records.append({
                "src_key": ("service", dst_ip, p_val, proto),
                "dst_key": ("host", dst_ip),
                "edge_type": "hosted_on",
                "source": "observed",
                "flow_count": 0,
                "total_fwd_bytes": 0,
                "total_bwd_bytes": 0,
                "total_fwd_pkts": 0,
                "total_bwd_pkts": 0,
                "mean_duration_s": 0.0,
                "max_duration_s": 0.0,
                "distinct_src_ports": 0,
                "label_binary": np.int8(0),
                "label_family": "hosted_on",
                "attack_stage": pd.NA,
                "n_flows_attack": 0,
            })

        df_hosted = (
            pd.DataFrame(hosted_records)
            if hosted_records
            else pd.DataFrame(columns=df_connects.columns)
        )

        all_edges = pd.concat([df_connects, df_hosted], ignore_index=True)

        # -------------------------------------------------------------
        # 3. Node Construction (Host Nodes & Service Nodes)
        # -------------------------------------------------------------
        all_hosts = sorted(set(w_flows["src_ip"]).union(set(w_flows["dst_ip"])))
        sorted_service_keys = sorted(
            [
                ("service", d_ip, None if pd.isna(d_p) else int(d_p), p)
                for d_ip, d_p, p in distinct_services
            ],
            key=lambda x: (x[0], x[1], -1 if x[2] is None else x[2], x[3]),
        )
        sorted_host_keys = [("host", h_ip) for h_ip in all_hosts]

        node_index: dict[tuple[Any, ...], int] = {}
        for idx, key in enumerate(sorted_host_keys + sorted_service_keys):
            node_index[key] = idx

        # Vectorized host feature extraction
        w_flows_copy = w_flows.copy()
        w_flows_copy["_svc"] = list(
            zip(w_flows["dst_ip"], w_flows["dst_port"], w_flows["proto"], strict=True)
        )
        w_flows_copy["_hosted_svc"] = list(
            zip(w_flows["dst_port"], w_flows["proto"], strict=True)
        )

        src_grp = w_flows_copy.groupby("src_ip")
        dst_grp = w_flows_copy.groupby("dst_ip")

        src_flows = src_grp["flow_id"].count().reindex(all_hosts, fill_value=0)
        dst_flows = dst_grp["flow_id"].count().reindex(all_hosts, fill_value=0)

        src_bytes = src_grp["fwd_bytes"].sum().reindex(all_hosts, fill_value=0)
        dst_bytes = dst_grp["fwd_bytes"].sum().reindex(all_hosts, fill_value=0)

        out_deg_svc = src_grp["_svc"].nunique().reindex(all_hosts, fill_value=0)
        in_deg_svc = dst_grp["_hosted_svc"].nunique().reindex(all_hosts, fill_value=0)

        dst_hosts_out = src_grp["dst_ip"].nunique().reindex(all_hosts, fill_value=0)
        dst_ports_out = src_grp["dst_port"].nunique().reindex(all_hosts, fill_value=0)
        src_hosts_in = dst_grp["src_ip"].nunique().reindex(all_hosts, fill_value=0)

        mean_dur = src_grp["duration_s"].mean().reindex(all_hosts, fill_value=0.0)

        host_attack = (
            w_flows[w_flows["label_binary"] == 1]
            .groupby("src_ip")["flow_id"]
            .count()
            .reindex(all_hosts, fill_value=0)
        )
        host_label_binary = (host_attack > 0).astype("int8")

        # Ratio out / in bytes (safe divide)
        ratio_bytes = np.where(
            dst_bytes > 0,
            src_bytes.astype(float) / dst_bytes.astype(float),
            np.where(src_bytes == 0, 0.0, src_bytes.astype(float)),
        )

        host_rows = []
        for i, h_ip in enumerate(all_hosts):
            host_rows.append({
                "node_key": ("host", h_ip),
                "node_idx": node_index[("host", h_ip)],
                "node_type": "host",
                "ip": h_ip,
                "port": pd.NA,
                "proto": pd.NA,
                "out_degree_services": int(out_deg_svc[h_ip]),
                "in_degree_services": int(in_deg_svc[h_ip]),
                "distinct_dst_hosts_out": int(dst_hosts_out[h_ip]),
                "distinct_dst_ports_out": int(dst_ports_out[h_ip]),
                "distinct_src_hosts_in": int(src_hosts_in[h_ip]),
                "bytes_out": int(src_bytes[h_ip]),
                "bytes_in": int(dst_bytes[h_ip]),
                "flows_out": int(src_flows[h_ip]),
                "flows_in": int(dst_flows[h_ip]),
                "mean_flow_duration": float(mean_dur[h_ip]),
                "ratio_out_in_bytes": float(ratio_bytes[i]),
                "label_binary": np.int8(host_label_binary[h_ip]),
            })

        service_rows = []
        for key in sorted_service_keys:
            _, s_ip, s_port, s_proto = key
            service_rows.append({
                "node_key": key,
                "node_idx": node_index[key],
                "node_type": "service",
                "ip": s_ip,
                "port": pd.NA if s_port is None else int(s_port),
                "proto": s_proto,
                "out_degree_services": 0,
                "in_degree_services": 0,
                "distinct_dst_hosts_out": 0,
                "distinct_dst_ports_out": 0,
                "distinct_src_hosts_in": 0,
                "bytes_out": 0,
                "bytes_in": 0,
                "flows_out": 0,
                "flows_in": 0,
                "mean_flow_duration": 0.0,
                "ratio_out_in_bytes": 0.0,
                "label_binary": np.int8(0),
            })

        all_nodes = pd.DataFrame(host_rows + service_rows)

        window = GraphWindow(
            window_id=window_id,
            t_start=t_win_start,
            t_end=t_win_end,
            nodes=all_nodes,
            edges=all_edges,
            node_index=node_index,
            flows=w_flows.copy(),
        )
        assert_no_leakage(window)
        windows.append(window)

    return windows


def window_to_networkx(window: GraphWindow) -> nx.MultiDiGraph:
    """Convert a GraphWindow into a NetworkX MultiDiGraph for graph inspection.

    Args:
        window: The GraphWindow instance.

    Returns:
        nx.MultiDiGraph: Graph containing host/service nodes and connects_to/hosted_on edges.
    """
    graph = nx.MultiDiGraph()

    for _, row in window.nodes.iterrows():
        key = row["node_key"]
        attrs = row.to_dict()
        graph.add_node(key, **attrs)

    for _, row in window.edges.iterrows():
        u = row["src_key"]
        v = row["dst_key"]
        attrs = row.to_dict()
        graph.add_edge(u, v, **attrs)

    return graph


def attach_topology_info(window: GraphWindow, topology: Topology) -> GraphWindow:
    """Attach enterprise topology metadata to host and service nodes.

    Adds 'segment', 'asset_criticality', and 'host_id' columns. IPs absent from
    the topology receive segment='unknown' and null criticality/host_id.
    These fields are designated METADATA for reasoning/search, NOT model features.

    Args:
        window: GraphWindow whose nodes are to be enriched.
        topology: Enterprise topology definition.

    Returns:
        GraphWindow: Enriched GraphWindow with metadata columns added to window.nodes.
    """
    mapping = ip_to_host(topology)

    segments: list[str] = []
    criticalities: list[Any] = []
    host_ids: list[Any] = []

    for _, row in window.nodes.iterrows():
        ip_val = row["ip"]
        if ip_val in mapping:
            host = mapping[ip_val]
            segments.append(host.segment)
            criticalities.append(host.asset_criticality)
            host_ids.append(host.id)
        else:
            segments.append("unknown")
            criticalities.append(pd.NA)
            host_ids.append(pd.NA)

    window.nodes["segment"] = pd.Series(segments, dtype="string")
    window.nodes["asset_criticality"] = pd.Series(criticalities, dtype="Int64")
    window.nodes["host_id"] = pd.Series(host_ids, dtype="string")

    return window
