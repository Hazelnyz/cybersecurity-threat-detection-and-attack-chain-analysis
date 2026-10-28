"""Unit tests for flow schema coercion, validation, and deduplication keys."""

import pandas as pd
import pytest

from data.schema import FLOW_COLUMNS, coerce_flows, make_dup_key, validate_flows
from data.synthetic import generate_synthetic_flows


@pytest.fixture
def valid_flow_df() -> pd.DataFrame:
    """Fixture providing a small, valid coerced flow DataFrame."""
    return generate_synthetic_flows(seed=42, n_benign=20)


def test_coerce_and_validate_valid_flows(valid_flow_df: pd.DataFrame) -> None:
    """Ensure that valid flows pass coercion and validation without errors."""
    coerced = coerce_flows(valid_flow_df)
    validate_flows(coerced)
    assert list(coerced.columns) == list(FLOW_COLUMNS.keys())


def test_coerce_flows_missing_column(valid_flow_df: pd.DataFrame) -> None:
    """Ensure coerce_flows raises ValueError when required columns are missing."""
    df_missing = valid_flow_df.drop(columns=["flow_id", "proto"])
    with pytest.raises(ValueError, match="Missing required flow columns"):
        coerce_flows(df_missing)


def test_validate_flows_ts_start_null(valid_flow_df: pd.DataFrame) -> None:
    """Ensure validate_flows rejects null values in ts_start."""
    df = valid_flow_df.copy()
    df.loc[0, "ts_start"] = pd.NaT
    with pytest.raises(ValueError, match="ts_start contains null values"):
        validate_flows(df)


def test_validate_flows_ts_start_not_utc(valid_flow_df: pd.DataFrame) -> None:
    """Ensure validate_flows rejects non-UTC timestamps."""
    df = valid_flow_df.copy()
    df["ts_start"] = df["ts_start"].dt.tz_localize(None)  # make tz-naive
    with pytest.raises(ValueError, match="ts_start must have timezone UTC"):
        validate_flows(df)

    df["ts_start"] = valid_flow_df["ts_start"].dt.tz_convert("US/Eastern")
    with pytest.raises(ValueError, match="ts_start must have timezone UTC"):
        validate_flows(df)


def test_validate_flows_src_ip_null_or_empty(valid_flow_df: pd.DataFrame) -> None:
    """Ensure validate_flows rejects null or empty src_ip."""
    df_null = valid_flow_df.copy()
    df_null.loc[0, "src_ip"] = pd.NA
    with pytest.raises(ValueError, match="src_ip contains null values"):
        validate_flows(df_null)

    df_empty = valid_flow_df.copy()
    df_empty.loc[0, "src_ip"] = "   "
    with pytest.raises(ValueError, match="src_ip contains empty values"):
        validate_flows(df_empty)


def test_validate_flows_dst_ip_null_or_empty(valid_flow_df: pd.DataFrame) -> None:
    """Ensure validate_flows rejects null or empty dst_ip."""
    df_null = valid_flow_df.copy()
    df_null.loc[0, "dst_ip"] = pd.NA
    with pytest.raises(ValueError, match="dst_ip contains null values"):
        validate_flows(df_null)

    df_empty = valid_flow_df.copy()
    df_empty.loc[0, "dst_ip"] = ""
    with pytest.raises(ValueError, match="dst_ip contains empty values"):
        validate_flows(df_empty)


def test_validate_flows_port_outside_range(valid_flow_df: pd.DataFrame) -> None:
    """Ensure validate_flows rejects port numbers outside 0..65535."""
    df_neg = valid_flow_df.copy()
    df_neg.loc[0, "src_port"] = -1
    with pytest.raises(ValueError, match="src_port contains port values outside 0..65535"):
        validate_flows(df_neg)

    df_high = valid_flow_df.copy()
    df_high.loc[0, "dst_port"] = 65536
    with pytest.raises(ValueError, match="dst_port contains port values outside 0..65535"):
        validate_flows(df_high)


def test_validate_flows_non_icmp_null_port(valid_flow_df: pd.DataFrame) -> None:
    """Ensure validate_flows rejects null ports for non-ICMP protocols."""
    df = valid_flow_df.copy()
    tcp_idx = df[df["proto"] == "tcp"].index[0]
    df.loc[tcp_idx, "dst_port"] = pd.NA
    with pytest.raises(ValueError, match="dst_port has null values for non-icmp traffic"):
        validate_flows(df)


def test_validate_flows_icmp_null_port_allowed(valid_flow_df: pd.DataFrame) -> None:
    """Ensure validate_flows allows null ports when protocol is ICMP."""
    df = valid_flow_df.copy()
    df.loc[0, "proto"] = "icmp"
    df.loc[0, "src_port"] = pd.NA
    df.loc[0, "dst_port"] = pd.NA
    validate_flows(df)


def test_validate_flows_negative_metrics(valid_flow_df: pd.DataFrame) -> None:
    """Ensure validate_flows rejects negative duration, packets, or bytes."""
    for col in ["duration_s", "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes"]:
        df_neg = valid_flow_df.copy()
        df_neg.loc[0, col] = -1
        with pytest.raises(ValueError, match=f"{col} contains negative values"):
            validate_flows(df_neg)


def test_validate_flows_invalid_label_binary(valid_flow_df: pd.DataFrame) -> None:
    """Ensure validate_flows rejects label_binary outside {-1, 0, 1}."""
    df_invalid = valid_flow_df.copy()
    df_invalid.loc[0, "label_binary"] = 2
    with pytest.raises(ValueError, match="label_binary contains values outside {-1, 0, 1}"):
        validate_flows(df_invalid)

    df_null = valid_flow_df.copy()
    df_null.loc[0, "label_binary"] = pd.NA
    with pytest.raises(ValueError, match="label_binary contains null values"):
        validate_flows(df_null)


def test_validate_flows_null_label_family(valid_flow_df: pd.DataFrame) -> None:
    """Ensure validate_flows rejects null label_family."""
    df = valid_flow_df.copy()
    df.loc[0, "label_family"] = pd.NA
    with pytest.raises(ValueError, match="label_family contains null values"):
        validate_flows(df)


def test_validate_flows_non_unique_flow_id(valid_flow_df: pd.DataFrame) -> None:
    """Ensure validate_flows rejects duplicate flow IDs."""
    df = valid_flow_df.copy()
    df.loc[1, "flow_id"] = df.loc[0, "flow_id"]
    with pytest.raises(ValueError, match="flow_id values are not unique"):
        validate_flows(df)


def test_make_dup_key_stability(valid_flow_df: pd.DataFrame) -> None:
    """Ensure make_dup_key produces deterministic, stable hash strings."""
    key1 = make_dup_key(valid_flow_df)
    key2 = make_dup_key(valid_flow_df)
    pd.testing.assert_series_equal(key1, key2)
    assert len(key1.iloc[0]) == 64  # SHA-256 hex length

