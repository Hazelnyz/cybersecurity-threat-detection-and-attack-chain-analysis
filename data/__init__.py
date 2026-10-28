"""ThreatGraph data module providing flow schema, generators, and dataset loaders."""

from pathlib import Path
from typing import Any

import pandas as pd

from data.cicids import load_cicids2017
from data.ctu13 import load_ctu13
from data.schema import FLOW_COLUMNS, coerce_flows, make_dup_key, validate_flows
from data.synthetic import generate_synthetic_flows
from data.unsw import load_unsw_nb15


def load_dataset(
    name: str,
    path: str | Path | None = None,
    **kw: Any,
) -> pd.DataFrame:
    """Load a dataset by name into the unified ThreatGraph flow schema.

    Dispatches to the appropriate loader:
      - "synthetic": Uses generate_synthetic_flows (path is ignored).
      - "cicids2017": Uses load_cicids2017 (path required).
      - "unsw-nb15": Planned for Phase 2 (raises NotImplementedError).
      - "ctu13": Planned for Phase 2 (raises NotImplementedError).

    Args:
        name: Name of the dataset ("synthetic", "cicids2017", "unsw-nb15", "ctu13").
        path: Path to dataset file or folder.
        **kw: Additional parameters passed to the loader.

    Returns:
        pd.DataFrame: Coerced and validated flow DataFrame conforming to FLOW_COLUMNS.

    Raises:
        ValueError: If dataset name is unknown or path is missing when required.
        NotImplementedError: If the dataset loader is planned for Phase 2.
    """
    normalized = name.strip().lower()

    if normalized == "synthetic":
        seed = kw.pop("seed", 42)
        return generate_synthetic_flows(seed=seed, **kw)

    if normalized == "cicids2017":
        if path is None:
            raise ValueError("Path must be provided for dataset 'cicids2017'")
        return load_cicids2017(path=path, **kw)

    if normalized in ("unsw-nb15", "unsw_nb15", "unsw"):
        return load_unsw_nb15(paths=path, **kw)

    if normalized in ("ctu13", "ctu-13", "ctu_13"):
        return load_ctu13(path=path, **kw)

    raise ValueError(
        f"Unknown dataset '{name}'. Expected: 'synthetic', 'cicids2017', 'unsw-nb15', 'ctu13'."
    )


__all__ = [
    "FLOW_COLUMNS",
    "coerce_flows",
    "generate_synthetic_flows",
    "load_cicids2017",
    "load_ctu13",
    "load_dataset",
    "load_unsw_nb15",
    "make_dup_key",
    "validate_flows",
]
