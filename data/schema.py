"""Unified network flow schema, coercion, validation, and key generation."""

import hashlib
from collections import OrderedDict

import pandas as pd

FLOW_COLUMNS: OrderedDict[str, str] = OrderedDict(
    [
        ("flow_id", "string"),
        ("ts_start", "datetime64[ns, UTC]"),
        ("duration_s", "float64"),
        ("src_ip", "string"),
        ("src_port", "Int64"),
        ("dst_ip", "string"),
        ("dst_port", "Int64"),
        ("proto", "string"),
        ("fwd_pkts", "Int64"),
        ("bwd_pkts", "Int64"),
        ("fwd_bytes", "Int64"),
        ("bwd_bytes", "Int64"),
        ("label_binary", "int8"),
        ("label_family", "string"),
        ("label_raw", "string"),
        ("attack_stage", "string"),
        ("dataset", "string"),
        ("dup_key", "string"),
    ]
)


def make_dup_key(df: pd.DataFrame) -> pd.Series:
    """Generate a stable SHA-256 hash string for flow deduplication.

    The key is computed from:
    (src_ip, src_port, dst_ip, dst_port, proto, ts_start, duration_s).
    Used to detect duplicate flows across train/test splits without dropping them.

    Args:
        df: Input DataFrame containing required key columns.

    Returns:
        pd.Series: String series of SHA-256 hex digests.

    Raises:
        ValueError: If any required key column is missing.
    """
    key_cols = [
        "src_ip",
        "src_port",
        "dst_ip",
        "dst_port",
        "proto",
        "ts_start",
        "duration_s",
    ]
    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for dup_key: {missing}")

    if df.empty:
        return pd.Series(dtype="string")

    # Standardize timestamp to ISO UTC string format with microsecond precision
    ts_series = pd.to_datetime(df["ts_start"], utc=True)
    ts_str = ts_series.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    src_ip = df["src_ip"].astype("string").fillna("")
    dst_ip = df["dst_ip"].astype("string").fillna("")
    src_p = df["src_port"].astype("string").fillna("")
    dst_p = df["dst_port"].astype("string").fillna("")
    proto = df["proto"].astype("string").str.lower().fillna("")

    # Format float duration to 9 decimal places (nanosecond precision)
    dur_str = df["duration_s"].map(
        lambda d: f"{float(d):.9f}" if pd.notna(d) else ""
    )

    combined = (
        src_ip
        + "|"
        + src_p
        + "|"
        + dst_ip
        + "|"
        + dst_p
        + "|"
        + proto
        + "|"
        + ts_str
        + "|"
        + dur_str
    )

    return combined.map(lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()).astype("string")


def coerce_flows(df: pd.DataFrame) -> pd.DataFrame:
    """Select, cast, and sort flow columns according to FLOW_COLUMNS.

    Reorders columns to match FLOW_COLUMNS exactly, casts columns to their
    target dtypes, and sorts rows by (ts_start, flow_id).

    Args:
        df: Input DataFrame containing flow records.

    Returns:
        pd.DataFrame: Coerced and sorted DataFrame.

    Raises:
        ValueError: If any required column in FLOW_COLUMNS is missing.
    """
    missing = [c for c in FLOW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required flow columns: {missing}")

    out = df[list(FLOW_COLUMNS.keys())].copy()

    for col, dtype in FLOW_COLUMNS.items():
        if col == "ts_start":
            out["ts_start"] = pd.to_datetime(out["ts_start"], utc=True)
        elif col == "proto":
            out["proto"] = out["proto"].astype("string").str.lower()
        else:
            out[col] = out[col].astype(dtype)

    out = out.sort_values(by=["ts_start", "flow_id"]).reset_index(drop=True)
    return out


def validate_flows(df: pd.DataFrame) -> None:
    """Validate flow records against required integrity constraints.

    Args:
        df: Flow DataFrame to validate.

    Raises:
        ValueError: If any integrity constraint or schema rule is violated.
    """
    missing = [c for c in FLOW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns required for flow validation: {missing}")

    if df.empty:
        # Check empty DataFrame timestamp timezone constraint
        tz = getattr(df["ts_start"].dt, "tz", None)
        if tz is None or str(tz) != "UTC":
            raise ValueError("ts_start must have timezone UTC")
        return

    # 1. ts_start has nulls or is not UTC
    if df["ts_start"].isna().any():
        raise ValueError("ts_start contains null values")
    tz = getattr(df["ts_start"].dt, "tz", None)
    if tz is None or str(tz) != "UTC":
        raise ValueError("ts_start must have timezone UTC")

    # 2. src_ip or dst_ip null/empty
    for col in ["src_ip", "dst_ip"]:
        if df[col].isna().any():
            raise ValueError(f"{col} contains null values")
        if (df[col].astype("string").str.strip() == "").any():
            raise ValueError(f"{col} contains empty values")

    # 3. ports outside 0..65535 (null allowed for icmp)
    proto_lower = df["proto"].astype("string").str.lower()
    is_icmp = proto_lower == "icmp"
    for col in ["src_port", "dst_port"]:
        null_non_icmp = df[col].isna() & ~is_icmp
        if null_non_icmp.any():
            raise ValueError(f"{col} has null values for non-icmp traffic")
        valid_ports = df[col].dropna()
        if ((valid_ports < 0) | (valid_ports > 65535)).any():
            raise ValueError(f"{col} contains port values outside 0..65535")

    # 4. negative duration/packets/bytes
    metrics = ["duration_s", "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes"]
    for col in metrics:
        valid_vals = df[col].dropna()
        if (valid_vals < 0).any():
            raise ValueError(f"{col} contains negative values")

    # 5. label_binary not in {-1, 0, 1}
    if df["label_binary"].isna().any():
        raise ValueError("label_binary contains null values")
    if (~df["label_binary"].isin([-1, 0, 1])).any():
        raise ValueError("label_binary contains values outside {-1, 0, 1}")

    # 6. label_family null
    if df["label_family"].isna().any():
        raise ValueError("label_family contains null values")

    # 7. flow_id not unique
    if df["flow_id"].duplicated().any():
        raise ValueError("flow_id values are not unique")
