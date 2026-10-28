"""Unit tests for dataset loaders, schema coercion, and dispatching."""

from pathlib import Path

import pandas as pd
import pytest

from data import load_dataset
from data.cicids import load_cicids2017
from data.schema import validate_flows

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_CICIDS_CSV = FIXTURES_DIR / "cicids_sample.csv"
NO_IP_CICIDS_CSV = FIXTURES_DIR / "cicids_no_ip.csv"
FLOW_BYTES_INF_CSV = FIXTURES_DIR / "cicids_flow_bytes_inf.csv"


def test_cicids_whitespace_and_repeated_headers() -> None:
    """Ensure leading whitespace in headers is stripped and internal header rows are dropped."""
    df = load_cicids2017(SAMPLE_CICIDS_CSV)

    # In sample CSV: 7 data rows + 1 repeated header row + 1 inf row = 9 raw rows.
    # Repeated header row and inf row should be dropped -> 6 valid flows retained.
    assert len(df) == 6
    assert "src_ip" in df.columns
    assert "dst_ip" in df.columns

    # Verify no row has header remnants
    assert not (df["src_ip"] == "Source IP").any()
    assert not (df["label_raw"] == "Label").any()


def test_cicids_inf_values_handled() -> None:
    """Ensure rows containing Infinity or NaN metrics are dropped rather than corrupting values."""
    df = load_cicids2017(SAMPLE_CICIDS_CSV)

    # The row with Flow Duration = 'Infinity' must have been dropped
    assert not (df["duration_s"].isin([float("inf"), float("-inf")])).any()
    assert not df["duration_s"].isna().any()


def test_cicids_unit_conversion_microseconds_to_seconds() -> None:
    """Ensure Flow Duration in microseconds is correctly converted to seconds."""
    df = load_cicids2017(SAMPLE_CICIDS_CSV)

    # In fixture row 1: Flow Duration was 1,000,000 microseconds -> should be 1.0 s
    row1 = df[df["src_port"] == 389].iloc[0]
    assert row1["duration_s"] == pytest.approx(1.0, rel=1e-5)

    # In fixture row 2: Flow Duration was 2,000,000 microseconds -> should be 2.0 s
    row2 = df[df["src_port"] == 80].iloc[0]
    assert row2["duration_s"] == pytest.approx(2.0, rel=1e-5)


def test_cicids_label_mapping() -> None:
    """Ensure BENIGN maps to 0/benign and attack classes map to 1 with normalized family names."""
    df = load_cicids2017(SAMPLE_CICIDS_CSV)

    benign_rows = df[df["label_raw"] == "BENIGN"]
    assert len(benign_rows) == 3
    assert (benign_rows["label_binary"] == 0).all()
    assert (benign_rows["label_family"] == "benign").all()

    portscan_row = df[df["label_raw"] == "PortScan"].iloc[0]
    assert portscan_row["label_binary"] == 1
    assert portscan_row["label_family"] == "portscan"

    ddos_row = df[df["label_raw"] == "DDoS"].iloc[0]
    assert ddos_row["label_binary"] == 1
    assert ddos_row["label_family"] == "ddos"

    ftp_row = df[df["label_raw"] == "FTP-Patator"].iloc[0]
    assert ftp_row["label_binary"] == 1
    assert ftp_row["label_family"] == "ftp-patator"


def test_cicids_infinity_in_unused_column_kept() -> None:
    """Ensure rows with Infinity only in unused columns (e.g. Flow Bytes/s) are kept."""
    df = load_cicids2017(FLOW_BYTES_INF_CSV)
    assert len(df) == 1
    assert df.iloc[0]["duration_s"] == pytest.approx(1.0, rel=1e-5)
    validate_flows(df)


def test_cicids_missing_ip_raises_clear_error() -> None:
    """Ensure files lacking IP columns (like MachineLearningCVE export) raise an explicit error."""
    with pytest.raises(
        ValueError, match="missing IP address columns.*MachineLearningCVE"
    ):
        load_cicids2017(NO_IP_CICIDS_CSV)


def test_cicids_validate_flows_passes() -> None:
    """Ensure loader output passes full schema validation."""
    df = load_cicids2017(SAMPLE_CICIDS_CSV)
    validate_flows(df)


def test_cicids_max_rows() -> None:
    """Ensure max_rows limits the number of processed records."""
    df = load_cicids2017(SAMPLE_CICIDS_CSV, max_rows=2)
    assert len(df) == 2
    validate_flows(df)


def test_cicids_time_range() -> None:
    """Ensure time_range filters flows chronologically."""
    start = pd.Timestamp("2017-07-07 08:32:00", tz="UTC")
    end = pd.Timestamp("2017-07-07 08:42:00", tz="UTC")

    df = load_cicids2017(SAMPLE_CICIDS_CSV, time_range=(start, end))
    assert len(df) > 0
    assert (df["ts_start"] >= start).all()
    assert (df["ts_start"] <= end).all()
    validate_flows(df)


def test_cicids_parquet_caching(tmp_path: Path) -> None:
    """Ensure caching to parquet file saves and reloads correctly."""
    cache_file = tmp_path / "cicids_cache.parquet"

    # First call creates cache
    df1 = load_cicids2017(SAMPLE_CICIDS_CSV, cache_parquet=cache_file)
    assert cache_file.exists()

    # Second call reads from cache
    df2 = load_cicids2017(SAMPLE_CICIDS_CSV, cache_parquet=cache_file)
    pd.testing.assert_frame_equal(df1, df2)


def test_load_dataset_dispatcher() -> None:
    """Ensure load_dataset dispatches correctly across all supported and planned datasets."""
    # Synthetic dataset
    df_syn = load_dataset("synthetic", n_benign=20, seed=42)
    assert not df_syn.empty
    validate_flows(df_syn)

    # CICIDS2017 dataset
    df_cicids = load_dataset("cicids2017", path=SAMPLE_CICIDS_CSV)
    assert not df_cicids.empty
    validate_flows(df_cicids)

    # UNSW-NB15 dataset (planned for Phase 2)
    with pytest.raises(NotImplementedError, match="planned for Phase 2"):
        load_dataset("unsw-nb15", path="fake_unsw.csv")

    # CTU-13 dataset (planned for Phase 2)
    with pytest.raises(NotImplementedError, match="planned for Phase 2"):
        load_dataset("ctu13", path="fake_ctu.binetflow")

    # Unknown dataset
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset("unknown_dataset", path="fake.csv")

