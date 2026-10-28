"""CICIDS2017 flow dataset loader into the unified ThreatGraph schema."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.schema import coerce_flows, make_dup_key, validate_flows

logger = logging.getLogger(__name__)

# Protocol mapping: CICIDS2017 uses IANA numbers (6: TCP, 17: UDP, 1: ICMP, 0: HOPOPT)
_PROTO_MAP: dict[int, str] = {
    6: "tcp",
    17: "udp",
    1: "icmp",
    0: "hopopt",
}


def _normalize_proto(val: Any) -> str:
    """Normalize numeric protocol ID or string name to lowercase string."""
    try:
        num = int(float(val))
        return _PROTO_MAP.get(num, str(num))
    except (ValueError, TypeError):
        return str(val).strip().lower()


def _normalize_label(label: str) -> tuple[int, str]:
    """Map CICIDS2017 Label to binary label and normalized family name.

    Returns:
        tuple[int, str]: (label_binary, label_family).
    """
    raw = str(label).strip()
    if raw.upper() == "BENIGN":
        return 0, "benign"

    # Normalize attack family: lowercase, standardize separators to hyphens
    family = (
        raw.lower()
        .replace(" – ", "-")
        .replace(" - ", "-")
        .replace(" ", "-")
        .replace("_", "-")
    )
    while "--" in family:
        family = family.replace("--", "-")

    return 1, family


def load_cicids2017(
    path: str | Path,
    max_rows: int | None = None,
    time_range: tuple[Any, Any] | None = None,
    cache_parquet: str | Path | None = None,
    chunk_size: int = 50000,
) -> pd.DataFrame:
    """Load and process CICIDS2017 CSV flow records into the unified ThreatGraph schema.

    Requirements and handling:
    1. Targets the 'TrafficLabelling' / 'GeneratedLabelledFlows' CSVs with IP columns.
       Rejects the 'MachineLearningCVE' feature-only export with a descriptive error.
    2. Strips leading and trailing whitespace from CSV column names.
    3. Drops repeated internal header rows (e.g. where Label == 'Label').
    4. Handles Infinity, -Infinity, and NaN numeric values.
    5. Converts Flow Duration from microseconds to seconds.
    6. Timezone Assumption: CICIDS2017 was recorded in New Brunswick (ADT/UTC-3),
       but raw timestamps omit timezone offsets. Here timestamps are localized to UTC.
       NOTE: This timezone assumption is an open documentation question for correlation.
    7. Supports streaming/chunked loading, row filtering by time_range, max_rows limit,
       and optional parquet caching.

    Args:
        path: Path to the raw CICIDS2017 CSV file.
        max_rows: Maximum number of valid rows to return.
        time_range: Optional tuple of (start_time, end_time) to filter flows.
        cache_parquet: Optional parquet file path to cache or read processed flows.
        chunk_size: Number of CSV rows per chunk during streaming.

    Returns:
        pd.DataFrame: Validated, coerced DataFrame matching FLOW_COLUMNS.

    Raises:
        ValueError: If file is missing required IP columns or is invalid.
    """
    if cache_parquet is not None:
        cache_path = Path(cache_parquet)
        if cache_path.exists():
            logger.info("Loading cached CICIDS2017 flows from parquet: %s", cache_path)
            cached_df = pd.read_parquet(cache_path)
            return coerce_flows(cached_df)

    input_path = Path(path)
    if not input_path.exists():
        raise ValueError(f"CICIDS2017 file not found: {input_path}")

    # Inspect header on the first row to detect missing IP columns early
    sample_header = pd.read_csv(input_path, nrows=0)
    cleaned_header_cols = [c.strip() for c in sample_header.columns]

    has_src_ip = "Source IP" in cleaned_header_cols or "Src IP" in cleaned_header_cols
    has_dst_ip = "Destination IP" in cleaned_header_cols or "Dst IP" in cleaned_header_cols

    if not (has_src_ip and has_dst_ip):
        raise ValueError(
            f"CICIDS2017 file '{input_path.name}' is missing IP address columns "
            "('Source IP' / 'Destination IP'). This indicates the 'MachineLearningCVE' "
            "feature-only export, which cannot be used for graph-based threat detection. "
            "Please provide the 'GeneratedLabelledFlows' / 'TrafficLabelling' version."
        )

    # Tracking dropped row statistics
    total_raw_rows = 0
    dropped_headers = 0
    dropped_inf_nan = 0
    dropped_invalid_time = 0
    dropped_invalid_metrics = 0

    t_start = pd.to_datetime(time_range[0], utc=True) if time_range is not None else None
    t_end = pd.to_datetime(time_range[1], utc=True) if time_range is not None else None

    processed_chunks: list[pd.DataFrame] = []
    total_collected = 0

    reader = pd.read_csv(
        input_path,
        chunksize=chunk_size,
        low_memory=False,
        dtype=str,
    )

    for chunk in reader:
        total_raw_rows += len(chunk)
        chunk.columns = chunk.columns.str.strip()

        # 1. Filter repeated header rows
        is_header_row = (chunk["Label"].str.strip() == "Label") | (
            chunk["Source IP"].str.strip() == "Source IP"
        )
        if is_header_row.any():
            n_hdr = int(is_header_row.sum())
            dropped_headers += n_hdr
            chunk = chunk[~is_header_row].copy()

        if chunk.empty:
            continue

        # 2. Parse and validate timestamps
        # ASSUMPTION: CICIDS2017 recorded in ADT (UTC-3), but timestamps lack timezone offsets;
        # parsing with UTC=True to standardize into timezone-aware UTC datetime64.
        ts_series = pd.to_datetime(
            chunk["Timestamp"], format="mixed", dayfirst=True, utc=True, errors="coerce"
        )
        bad_time = ts_series.isna()
        if bad_time.any():
            dropped_invalid_time += int(bad_time.sum())
            chunk = chunk[~bad_time].copy()
            ts_series = ts_series[~bad_time]

        if chunk.empty:
            continue

        # Time range filter
        if t_start is not None or t_end is not None:
            time_mask = pd.Series(True, index=chunk.index)
            if t_start is not None:
                time_mask &= ts_series >= t_start
            if t_end is not None:
                time_mask &= ts_series <= t_end
            chunk = chunk[time_mask].copy()
            ts_series = ts_series[time_mask]

        if chunk.empty:
            continue

        # 3. Handle numeric metrics and inf/nan
        numeric_cols = [
            "Flow Duration",
            "Source Port",
            "Destination Port",
            "Total Fwd Packets",
            "Total Backward Packets",
            "Total Length of Fwd Packets",
            "Total Length of Bwd Packets",
        ]

        # Replace textual representations of infinity
        for col in numeric_cols:
            if col in chunk.columns:
                chunk[col] = chunk[col].replace(
                    ["Infinity", "-Infinity", "inf", "-inf", "NaN", "nan"], np.nan
                )
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        bad_numeric = chunk[numeric_cols].isna().any(axis=1)
        if bad_numeric.any():
            dropped_inf_nan += int(bad_numeric.sum())
            chunk = chunk[~bad_numeric].copy()
            ts_series = ts_series[~bad_numeric]

        if chunk.empty:
            continue

        # 4. Check negative durations or counts
        neg_metrics = (
            (chunk["Flow Duration"] < 0)
            | (chunk["Total Fwd Packets"] < 0)
            | (chunk["Total Backward Packets"] < 0)
            | (chunk["Total Length of Fwd Packets"] < 0)
            | (chunk["Total Length of Bwd Packets"] < 0)
        )
        if neg_metrics.any():
            dropped_invalid_metrics += int(neg_metrics.sum())
            chunk = chunk[~neg_metrics].copy()
            ts_series = ts_series[~neg_metrics]

        if chunk.empty:
            continue

        # Build unified chunk columns
        label_raw = chunk["Label"].astype(str).str.strip()
        binary_labels = []
        family_labels = []
        for raw in label_raw:
            b, f = _normalize_label(raw)
            binary_labels.append(b)
            family_labels.append(f)

        chunk_out = pd.DataFrame(index=chunk.index)
        chunk_out["ts_start"] = ts_series
        # Flow Duration is in microseconds in CICIDS2017 -> convert to seconds
        chunk_out["duration_s"] = chunk["Flow Duration"].astype(float) / 1e6
        chunk_out["src_ip"] = chunk["Source IP"].astype(str).str.strip()
        chunk_out["src_port"] = chunk["Source Port"].astype("Int64")
        chunk_out["dst_ip"] = chunk["Destination IP"].astype(str).str.strip()
        chunk_out["dst_port"] = chunk["Destination Port"].astype("Int64")
        chunk_out["proto"] = chunk["Protocol"].map(_normalize_proto)
        chunk_out["fwd_pkts"] = chunk["Total Fwd Packets"].astype("Int64")
        chunk_out["bwd_pkts"] = chunk["Total Backward Packets"].astype("Int64")
        chunk_out["fwd_bytes"] = chunk["Total Length of Fwd Packets"].astype("Int64")
        chunk_out["bwd_bytes"] = chunk["Total Length of Bwd Packets"].astype("Int64")
        chunk_out["label_binary"] = pd.Series(binary_labels, index=chunk.index, dtype="int8")
        chunk_out["label_family"] = pd.Series(family_labels, index=chunk.index, dtype="string")
        chunk_out["label_raw"] = label_raw.astype("string")
        chunk_out["attack_stage"] = pd.Series(
            [None] * len(chunk), index=chunk.index, dtype="string"
        )
        chunk_out["dataset"] = "cicids2017"

        processed_chunks.append(chunk_out)
        total_collected += len(chunk_out)

        if max_rows is not None and total_collected >= max_rows:
            break

    total_dropped = (
        dropped_headers
        + dropped_inf_nan
        + dropped_invalid_time
        + dropped_invalid_metrics
    )
    logger.info(
        "CICIDS2017 stream summary for %s: read %d raw rows, retained %d flows, dropped %d "
        "(headers: %d, inf/nan: %d, invalid time: %d, negative metrics: %d)",
        input_path.name,
        total_raw_rows,
        total_collected,
        total_dropped,
        dropped_headers,
        dropped_inf_nan,
        dropped_invalid_time,
        dropped_invalid_metrics,
    )

    if not processed_chunks:
        # Return empty DataFrame conforming to schema
        empty_df = pd.DataFrame(
            columns=[
                "flow_id", "ts_start", "duration_s", "src_ip", "src_port",
                "dst_ip", "dst_port", "proto", "fwd_pkts", "bwd_pkts",
                "fwd_bytes", "bwd_bytes", "label_binary", "label_family",
                "label_raw", "attack_stage", "dataset", "dup_key",
            ]
        )
        return coerce_flows(empty_df)

    combined = pd.concat(processed_chunks, ignore_index=True)
    if max_rows is not None and len(combined) > max_rows:
        combined = combined.iloc[:max_rows].copy()

    # Generate unique flow IDs and stable deduplication keys
    combined["flow_id"] = [f"cicids_{i:07d}" for i in range(len(combined))]
    combined["dup_key"] = make_dup_key(combined)

    coerced = coerce_flows(combined)
    validate_flows(coerced)

    if cache_parquet is not None:
        cache_path = Path(cache_parquet)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        coerced.to_parquet(cache_path, index=False)
        logger.info("Saved %d coerced CICIDS2017 flows to %s", len(coerced), cache_path)

    return coerced

