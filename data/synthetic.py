"""Synthetic network flow generator with multi-stage attack and hard negatives."""

from typing import Any

import numpy as np
import pandas as pd

from data.schema import coerce_flows, make_dup_key, validate_flows


def _get_default_hosts() -> tuple[
    list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]
]:
    """Provide default set of ~18 enterprise hosts plus 2 external hosts.

    Returns:
        tuple containing (external_hosts, dmz_hosts, internal_hosts).
    """
    external_hosts = [
        {"id": "attacker-ext", "ip": "203.0.113.50", "segment": "internet"},
        {"id": "c2-ext", "ip": "198.51.100.25", "segment": "internet"},
    ]

    dmz_hosts = [
        {"id": "web-01", "ip": "10.0.1.10", "segment": "dmz"},
        {"id": "mail-01", "ip": "10.0.1.11", "segment": "dmz"},
        {"id": "dns-01", "ip": "10.0.1.53", "segment": "dmz"},
    ]

    internal_hosts = [
        # Server LAN
        {"id": "app-01", "ip": "10.0.2.10", "segment": "server_lan"},
        {"id": "file-01", "ip": "10.0.2.60", "segment": "server_lan"},
        {"id": "dc-01", "ip": "10.0.2.100", "segment": "server_lan"},
        # DB LAN
        {"id": "db-01", "ip": "10.0.2.50", "segment": "db_lan"},
        {"id": "db-02", "ip": "10.0.2.51", "segment": "db_lan"},
        {"id": "backup-01", "ip": "10.0.2.70", "segment": "db_lan"},
        # User LAN
        {"id": "admin-ws-01", "ip": "10.0.3.101", "segment": "user_lan"},
        {"id": "ws-01", "ip": "10.0.3.102", "segment": "user_lan"},
        {"id": "ws-02", "ip": "10.0.3.103", "segment": "user_lan"},
        {"id": "ws-03", "ip": "10.0.3.104", "segment": "user_lan"},
        {"id": "ws-04", "ip": "10.0.3.105", "segment": "user_lan"},
        {"id": "ws-05", "ip": "10.0.3.106", "segment": "user_lan"},
        {"id": "ws-06", "ip": "10.0.3.107", "segment": "user_lan"},
    ]
    return external_hosts, dmz_hosts, internal_hosts


def generate_synthetic_flows(
    seed: int,
    n_benign: int = 5000,
    hosts: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Generate deterministic synthetic network flows with attack stages and hard negatives.

    Produces a realistic 2-hour window of network communication containing:
    1. Benign background traffic (DNS, HTTP/S, DB, SSH, SMB, ICMP).
    2. Hard negative traffic (benign backup job with large bytes, admin SMB/RDP).
    3. Multi-stage attack progression (recon -> initial_access -> lateral_movement -> exfiltration).

    Args:
        seed: Random seed for deterministic flow generation.
        n_benign: Number of background benign flows to generate.
        hosts: Optional topology host records. If None, uses default enterprise network.

    Returns:
        pd.DataFrame: Validated, coerced, and sorted DataFrame of synthetic flows.
    """
    rng = np.random.default_rng(seed)

    if hosts is not None and len(hosts) > 0:
        ext = [
            h for h in hosts
            if h.get("segment") in ("internet", "external") or "ext" in h.get("id", "").lower()
        ]
        dmz = [
            h for h in hosts
            if h.get("segment") == "dmz" or "dmz" in h.get("id", "").lower()
        ]
        intl = [h for h in hosts if h not in ext and h not in dmz]

        # Fallbacks if segments are missing
        if not ext:
            ext = [
                {"id": "attacker-ext", "ip": "203.0.113.50", "segment": "internet"},
                {"id": "c2-ext", "ip": "198.51.100.25", "segment": "internet"},
            ]
        if not dmz:
            dmz = [intl.pop(0)] if intl else [
                {"id": "web-01", "ip": "10.0.1.10", "segment": "dmz"}
            ]
        if not intl:
            intl = dmz
    else:
        ext, dmz, intl = _get_default_hosts()

    base_time = pd.Timestamp("2026-01-15 08:00:00", tz="UTC")
    flows: list[dict[str, Any]] = []

    attacker_ip = ext[0]["ip"]
    c2_ip = ext[1]["ip"] if len(ext) > 1 else ext[0]["ip"]
    dmz_web_ip = next((h["ip"] for h in dmz if "web" in h.get("id", "").lower()), dmz[0]["ip"])

    # Target selections
    db_ip = next((h["ip"] for h in intl if "db-01" in h.get("id", "").lower()), intl[0]["ip"])
    file_ip = next(
        (h["ip"] for h in intl if "file" in h.get("id", "").lower()),
        intl[-1]["ip"],
    )
    admin_ws_ip = next(
        (h["ip"] for h in intl if "admin" in h.get("id", "").lower()),
        intl[0]["ip"],
    )
    backup_srv_ip = next(
        (h["ip"] for h in intl if "backup" in h.get("id", "").lower()),
        file_ip,
    )
    dns_ip = next((h["ip"] for h in dmz if "dns" in h.get("id", "").lower()), dmz[-1]["ip"])

    # Lateral movement targets: user_lan and server_lan hosts
    lateral_candidates = [
        h["ip"] for h in intl
        if h.get("segment") in ("server_lan", "user_lan")
        or any(k in h.get("id", "").lower() for k in ("app", "file", "dc", "admin", "ws"))
    ]
    if not lateral_candidates:
        lateral_candidates = [file_ip, admin_ws_ip]

    # Exfiltration sources: db-01 or file-01
    exfil_sources = [
        h["ip"] for h in intl
        if any(target in h.get("id", "").lower() for target in ("db-01", "file-01", "db", "file"))
    ]
    if not exfil_sources:
        exfil_sources = [db_ip, file_ip]

    # -------------------------------------------------------------
    # 1. Multi-Stage Attack Flows (Strict temporal order with realistic gaps)
    # Stage 1: Recon (port scan)       [600s .. 1200s]
    # Gap:     [1200s .. 1800s]
    # Stage 2: Initial Access (exploit)[1800s .. 2400s]
    # Gap:     [2400s .. 3600s]
    # Stage 3: Lateral Movement        [3600s .. 4500s]
    # Gap:     [4500s .. 5400s]
    # Stage 4: Exfiltration           [5400s .. 6300s]
    # -------------------------------------------------------------

    # Stage 1: Recon (Portscan against web-01)
    n_recon = 120
    recon_ports = [
        21, 22, 23, 25, 53, 80, 110, 135, 139, 143, 443, 445, 993, 995,
        1433, 1521, 3306, 3389, 5432, 5985, 8000, 8080, 8443, 8888, 9000, 9200,
    ]
    recon_times = np.sort(rng.uniform(600.0, 1200.0, size=n_recon))
    for t in recon_times:
        port = int(rng.choice(recon_ports))
        flows.append({
            "ts_start": base_time + pd.Timedelta(seconds=float(t)),
            "duration_s": round(float(rng.uniform(0.005, 0.05)), 6),
            "src_ip": attacker_ip,
            "src_port": int(rng.integers(40000, 65000)),
            "dst_ip": dmz_web_ip,
            "dst_port": port,
            "proto": "tcp",
            "fwd_pkts": int(rng.integers(1, 3)),
            "bwd_pkts": int(rng.integers(0, 2)),
            "fwd_bytes": int(rng.integers(40, 150)),
            "bwd_bytes": int(rng.integers(0, 100)),
            "label_binary": 1,
            "label_family": "portscan",
            "label_raw": "portscan",
            "attack_stage": "recon",
        })

    # Stage 2: Initial Access (Web Exploit against web-01)
    n_exploit = 35
    exploit_times = np.sort(rng.uniform(1800.0, 2400.0, size=n_exploit))
    for t in exploit_times:
        port = int(rng.choice([80, 443]))
        flows.append({
            "ts_start": base_time + pd.Timedelta(seconds=float(t)),
            "duration_s": round(float(rng.uniform(0.1, 1.5)), 6),
            "src_ip": attacker_ip,
            "src_port": int(rng.integers(40000, 65000)),
            "dst_ip": dmz_web_ip,
            "dst_port": port,
            "proto": "tcp",
            "fwd_pkts": int(rng.integers(12, 35)),
            "bwd_pkts": int(rng.integers(10, 30)),
            "fwd_bytes": int(rng.integers(2500, 12000)),
            "bwd_bytes": int(rng.integers(1500, 8000)),
            "label_binary": 1,
            "label_family": "web_exploit",
            "label_raw": "web_exploit",
            "attack_stage": "initial_access",
        })

    # Stage 3: Lateral Movement (from web-01 to user_lan / server_lan hosts)
    n_lateral = 30
    lateral_times = np.sort(rng.uniform(3600.0, 4500.0, size=n_lateral))
    for t in lateral_times:
        target_ip = str(rng.choice(lateral_candidates))
        port = int(rng.choice([445, 3389, 5985]))
        flows.append({
            "ts_start": base_time + pd.Timedelta(seconds=float(t)),
            "duration_s": round(float(rng.uniform(0.5, 4.0)), 6),
            "src_ip": dmz_web_ip,
            "src_port": int(rng.integers(40000, 65000)),
            "dst_ip": target_ip,
            "dst_port": port,
            "proto": "tcp",
            "fwd_pkts": int(rng.integers(15, 60)),
            "bwd_pkts": int(rng.integers(12, 50)),
            "fwd_bytes": int(rng.integers(1500, 9000)),
            "bwd_bytes": int(rng.integers(1200, 8000)),
            "label_binary": 1,
            "label_family": "lateral_movement",
            "label_raw": "lateral_movement",
            "attack_stage": "lateral_movement",
        })

    # Stage 4: Exfiltration (from db-01 / file-01 to c2-ext)
    n_exfil = 15
    exfil_times = np.sort(rng.uniform(5400.0, 6300.0, size=n_exfil))
    for t in exfil_times:
        port = int(rng.choice([443, 8443]))
        src_srv = str(rng.choice(exfil_sources))
        flows.append({
            "ts_start": base_time + pd.Timedelta(seconds=float(t)),
            "duration_s": round(float(rng.uniform(15.0, 80.0)), 6),
            "src_ip": src_srv,
            "src_port": int(rng.integers(40000, 65000)),
            "dst_ip": c2_ip,
            "dst_port": port,
            "proto": "tcp",
            "fwd_pkts": int(rng.integers(1500, 6000)),
            "bwd_pkts": int(rng.integers(400, 1500)),
            "fwd_bytes": int(rng.integers(5_000_000, 25_000_000)),
            "bwd_bytes": int(rng.integers(10_000, 60_000)),
            "label_binary": 1,
            "label_family": "exfiltration",
            "label_raw": "exfiltration",
            "attack_stage": "exfiltration",
        })

    # -------------------------------------------------------------
    # 2. Hard Negatives (benign high-volume / sensitive port flows)
    # -------------------------------------------------------------
    # Benign backup job with large outbound transfer
    n_backup = 8
    backup_times = rng.uniform(1000.0, 6500.0, size=n_backup)
    for t in backup_times:
        flows.append({
            "ts_start": base_time + pd.Timedelta(seconds=float(t)),
            "duration_s": round(float(rng.uniform(30.0, 150.0)), 6),
            "src_ip": db_ip,
            "src_port": int(rng.integers(40000, 65000)),
            "dst_ip": backup_srv_ip,
            "dst_port": 873,
            "proto": "tcp",
            "fwd_pkts": int(rng.integers(2000, 8000)),
            "bwd_pkts": int(rng.integers(600, 2500)),
            "fwd_bytes": int(rng.integers(8_000_000, 45_000_000)),
            "bwd_bytes": int(rng.integers(15_000, 80_000)),
            "label_binary": 0,
            "label_family": "benign",
            "label_raw": "benign_backup",
            "attack_stage": None,
        })

    # Benign admin doing SMB / RDP
    n_admin = 12
    admin_times = rng.uniform(500.0, 6800.0, size=n_admin)
    for t in admin_times:
        port = int(rng.choice([445, 3389]))
        target = str(rng.choice([dmz_web_ip, file_ip, db_ip]))
        flows.append({
            "ts_start": base_time + pd.Timedelta(seconds=float(t)),
            "duration_s": round(float(rng.uniform(5.0, 45.0)), 6),
            "src_ip": admin_ws_ip,
            "src_port": int(rng.integers(40000, 65000)),
            "dst_ip": target,
            "dst_port": port,
            "proto": "tcp",
            "fwd_pkts": int(rng.integers(25, 90)),
            "bwd_pkts": int(rng.integers(20, 80)),
            "fwd_bytes": int(rng.integers(3000, 18000)),
            "bwd_bytes": int(rng.integers(2500, 15000)),
            "label_binary": 0,
            "label_family": "benign",
            "label_raw": "benign_admin",
            "attack_stage": None,
        })

    # -------------------------------------------------------------
    # 3. Benign Background Traffic (n_benign flows across 2 hours)
    # -------------------------------------------------------------
    all_internal_ips = [h["ip"] for h in intl]
    workstation_ips = [
        h["ip"] for h in intl
        if h.get("segment") == "user_lan" or "ws" in h.get("id", "").lower()
    ]
    if not workstation_ips:
        workstation_ips = all_internal_ips

    dmz_web_ips = [h["ip"] for h in dmz if "web" in h.get("id", "").lower()] or [dmz_web_ip]
    app_hosts = [h["ip"] for h in intl if "app" in h.get("id", "").lower()] or [all_internal_ips[0]]
    db_hosts = [h["ip"] for h in intl if "db" in h.get("id", "").lower()] or [db_ip]
    dc_hosts = [h["ip"] for h in intl if "dc" in h.get("id", "").lower()] or [all_internal_ips[-1]]

    benign_times = rng.uniform(0.0, 7200.0, size=n_benign)
    for t in benign_times:
        pattern = int(rng.integers(0, 7))
        if pattern == 0:
            # DNS Query (Workstation -> DMZ DNS)
            src = str(rng.choice(workstation_ips))
            flows.append({
                "ts_start": base_time + pd.Timedelta(seconds=float(t)),
                "duration_s": round(float(rng.uniform(0.001, 0.04)), 6),
                "src_ip": src,
                "src_port": int(rng.integers(40000, 65000)),
                "dst_ip": dns_ip,
                "dst_port": 53,
                "proto": "udp",
                "fwd_pkts": int(rng.integers(1, 3)),
                "bwd_pkts": int(rng.integers(1, 3)),
                "fwd_bytes": int(rng.integers(50, 150)),
                "bwd_bytes": int(rng.integers(70, 350)),
                "label_binary": 0,
                "label_family": "benign",
                "label_raw": "benign",
                "attack_stage": None,
            })
        elif pattern == 1:
            # HTTPS Web traffic (Workstation -> DMZ Web)
            src = str(rng.choice(workstation_ips))
            dst = str(rng.choice(dmz_web_ips))
            flows.append({
                "ts_start": base_time + pd.Timedelta(seconds=float(t)),
                "duration_s": round(float(rng.uniform(0.05, 3.5)), 6),
                "src_ip": src,
                "src_port": int(rng.integers(40000, 65000)),
                "dst_ip": dst,
                "dst_port": 443,
                "proto": "tcp",
                "fwd_pkts": int(rng.integers(6, 40)),
                "bwd_pkts": int(rng.integers(8, 60)),
                "fwd_bytes": int(rng.integers(600, 5000)),
                "bwd_bytes": int(rng.integers(1200, 35000)),
                "label_binary": 0,
                "label_family": "benign",
                "label_raw": "benign",
                "attack_stage": None,
            })
        elif pattern == 2:
            # HTTP Web traffic (Workstation -> DMZ Web)
            src = str(rng.choice(workstation_ips))
            dst = str(rng.choice(dmz_web_ips))
            flows.append({
                "ts_start": base_time + pd.Timedelta(seconds=float(t)),
                "duration_s": round(float(rng.uniform(0.02, 2.0)), 6),
                "src_ip": src,
                "src_port": int(rng.integers(40000, 65000)),
                "dst_ip": dst,
                "dst_port": 80,
                "proto": "tcp",
                "fwd_pkts": int(rng.integers(4, 25)),
                "bwd_pkts": int(rng.integers(5, 35)),
                "fwd_bytes": int(rng.integers(300, 3000)),
                "bwd_bytes": int(rng.integers(600, 20000)),
                "label_binary": 0,
                "label_family": "benign",
                "label_raw": "benign",
                "attack_stage": None,
            })
        elif pattern == 3:
            # Internal App to Database
            src = str(rng.choice(app_hosts))
            dst = str(rng.choice(db_hosts))
            port = int(rng.choice([3306, 5432]))
            flows.append({
                "ts_start": base_time + pd.Timedelta(seconds=float(t)),
                "duration_s": round(float(rng.uniform(0.01, 0.8)), 6),
                "src_ip": src,
                "src_port": int(rng.integers(40000, 65000)),
                "dst_ip": dst,
                "dst_port": port,
                "proto": "tcp",
                "fwd_pkts": int(rng.integers(4, 20)),
                "bwd_pkts": int(rng.integers(4, 25)),
                "fwd_bytes": int(rng.integers(250, 2500)),
                "bwd_bytes": int(rng.integers(350, 7000)),
                "label_binary": 0,
                "label_family": "benign",
                "label_raw": "benign",
                "attack_stage": None,
            })
        elif pattern == 4:
            # Workstation to Internal App Server
            src = str(rng.choice(workstation_ips))
            dst = str(rng.choice(app_hosts))
            flows.append({
                "ts_start": base_time + pd.Timedelta(seconds=float(t)),
                "duration_s": round(float(rng.uniform(0.02, 1.2)), 6),
                "src_ip": src,
                "src_port": int(rng.integers(40000, 65000)),
                "dst_ip": dst,
                "dst_port": 8080,
                "proto": "tcp",
                "fwd_pkts": int(rng.integers(4, 18)),
                "bwd_pkts": int(rng.integers(4, 22)),
                "fwd_bytes": int(rng.integers(350, 2800)),
                "bwd_bytes": int(rng.integers(500, 9500)),
                "label_binary": 0,
                "label_family": "benign",
                "label_raw": "benign",
                "attack_stage": None,
            })
        elif pattern == 5:
            # Workstation to Domain Controller
            src = str(rng.choice(workstation_ips))
            dst = str(rng.choice(dc_hosts))
            port = int(rng.choice([88, 389]))
            flows.append({
                "ts_start": base_time + pd.Timedelta(seconds=float(t)),
                "duration_s": round(float(rng.uniform(0.01, 0.2)), 6),
                "src_ip": src,
                "src_port": int(rng.integers(40000, 65000)),
                "dst_ip": dst,
                "dst_port": port,
                "proto": "tcp",
                "fwd_pkts": int(rng.integers(2, 10)),
                "bwd_pkts": int(rng.integers(2, 10)),
                "fwd_bytes": int(rng.integers(150, 1200)),
                "bwd_bytes": int(rng.integers(200, 2200)),
                "label_binary": 0,
                "label_family": "benign",
                "label_raw": "benign",
                "attack_stage": None,
            })
        else:
            # Internal ICMP health check / ping
            src = str(rng.choice(workstation_ips))
            dst = dns_ip
            flows.append({
                "ts_start": base_time + pd.Timedelta(seconds=float(t)),
                "duration_s": round(float(rng.uniform(0.001, 0.02)), 6),
                "src_ip": src,
                "src_port": pd.NA,
                "dst_ip": dst,
                "dst_port": pd.NA,
                "proto": "icmp",
                "fwd_pkts": int(rng.integers(1, 4)),
                "bwd_pkts": int(rng.integers(1, 4)),
                "fwd_bytes": int(rng.integers(64, 256)),
                "bwd_bytes": int(rng.integers(64, 256)),
                "label_binary": 0,
                "label_family": "benign",
                "label_raw": "benign",
                "attack_stage": None,
            })

    df = pd.DataFrame(flows)
    # Assign identifiers and metadata
    df["dataset"] = "synthetic"
    df["flow_id"] = [f"syn_{i:07d}" for i in range(len(df))]
    df["dup_key"] = make_dup_key(df)

    # Coerce dtypes and sort by (ts_start, flow_id)
    coerced = coerce_flows(df)
    validate_flows(coerced)

    return coerced
