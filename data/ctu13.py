"""CTU-13 binetflow dataset loader into the unified ThreatGraph schema.

CTU-13 Dataset Mapping Specification:
-------------------------------------
Target files: Capture binetflow files (e.g. capture20110810.binetflow).

Columns:
  1. StartTime: Flow start timestamp (e.g. 2011/08/10 09:46:59.607824) -> ts_start (UTC)
  2. Dur: Total duration in seconds -> duration_s
  3. Proto: Protocol name (e.g. tcp, udp, icmp) -> proto (lowercase)
  4. SrcAddr: Source IP address -> src_ip
  5. Sport: Source port (may contain hex strings like '0x0035' or non-numeric) -> src_port
  6. Dir: Direction indicator (e.g. '->', '<->', '<?>')
  7. DstAddr: Destination IP address -> dst_ip
  8. Dport: Destination port (numeric or hex) -> dst_port
  9. State: Transaction state string
  10. sTos: Source type of service
  11. dTos: Destination type of service
  12. TotPkts: Total flow packets -> fwd_pkts (limitation: bwd_pkts set to null/NA as
      binetflow does not record bidirectional packet split)
  13. TotBytes: Total bytes transferred in both directions
  14. SrcBytes: Outbound bytes from source -> fwd_bytes
      Derived bwd_bytes = TotBytes - SrcBytes
  15. Label: Text label containing status and botnet family.
      Mapping rules:
        - Contains 'Botnet' -> label_binary = 1, label_family = 'botnet'
        - Contains 'Normal' -> label_binary = 0, label_family = 'benign'
        - Contains 'Background' -> label_binary = -1, label_family = 'background'
        (Background flows are retained for network graph topology context, but excluded
        from supervised evaluation metrics).
"""

from typing import Any

import pandas as pd


def load_ctu13(
    path: Any,
    max_rows: int | None = None,
    **kw: Any,
) -> pd.DataFrame:
    """Load CTU-13 binetflow capture files into the unified ThreatGraph schema.

    Args:
        path: File path to a .binetflow file.
        max_rows: Optional limit on the number of rows to load.
        **kw: Additional loader parameters.

    Returns:
        pd.DataFrame: Coerced and validated DataFrame.

    Raises:
        NotImplementedError: Scheduled for implementation in Phase 2.
    """
    raise NotImplementedError("planned for Phase 2")
