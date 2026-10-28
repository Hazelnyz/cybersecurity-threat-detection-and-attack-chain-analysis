"""UNSW-NB15 dataset loader into the unified ThreatGraph schema.

UNSW-NB15 Dataset Mapping Specification:
----------------------------------------
Target files: Raw partitions (UNSW-NB15_1.csv through UNSW-NB15_4.csv) without header row.
(The training/testing CSV subsets omit IP addresses and cannot be used for graph construction).

49 Raw Column Names (per NUSW-NB15_features.csv definition):
  1. srcip: Source IP address -> src_ip
  2. sport: Source port number (may contain hex/strings in raw logs) -> src_port
  3. dstip: Destination IP address -> dst_ip
  4. dsport: Destination port number -> dst_port
  5. proto: Transaction protocol -> proto (lowercase)
  6. state: State and its dependent protocol
  7. dur: Record total duration in seconds -> duration_s
  8. sbytes: Source to destination transaction bytes -> fwd_bytes
  9. dbytes: Destination to source transaction bytes -> bwd_bytes
  10. sttl: Source to destination time to live value
  11. dttl: Destination to source time to live value
  12. sloss: Source packets retransmitted or dropped
  13. dloss: Destination packets retransmitted or dropped
  14. service: http, ftp, smtp, ssh, dns, radius, pop3, ssl, snmp, etc.
  15. Sload: Source bits per second
  16. Dload: Destination bits per second
  17. Spkts: Source to destination packet count -> fwd_pkts
  18. Dpkts: Destination to source packet count -> bwd_pkts
  19. swin: Source TCP window advertisement value
  20. dwin: Destination TCP window advertisement value
  21. stcpb: Source TCP base sequence number
  22. dtcpb: Destination TCP base sequence number
  23. smeansz: Mean of the flow packet size transmitted by the src
  24. dmeansz: Mean of the flow packet size transmitted by the dst
  25. trans_depth: Represents pipelined depth into the connection
  26. res_bdy_len: Actual uncompressed content size of data transferred from server
  27. Sjit: Source jitter (mSec)
  28. Djit: Destination jitter (mSec)
  29. Stime: Record start time (epoch timestamp) -> ts_start (UTC)
  30. Ltime: Record last time (epoch timestamp)
  31. Sintpkt: Source interpacket arrival time (mSec)
  32. Dintpkt: Destination interpacket arrival time (mSec)
  33. tcprtt: TCP connection setup round-trip time
  34. synack: TCP connection setup time from SYN to SYN_ACK
  35. ackdat: TCP connection setup time from SYN_ACK to ACK
  36. is_sm_ips_ports: Binary indicator if src and dst IP/ports equal
  37. ct_state_ttl: State and TTL count
  38. ct_flw_http_mthd: Count of HTTP methods
  39. is_ftp_login: Binary indicator for FTP session login
  40. ct_ftp_cmd: Number of FTP commands
  41. ct_srv_src: Connections to same service and IP
  42. ct_srv_dst: Connections to same service and destination IP
  43. ct_dst_ltm: Connections to same destination IP in last 100 flows
  44. ct_src_ltm: Connections to same source IP in last 100 flows
  45. ct_src_dport_ltm: Connections from src IP to dst port in last 100 flows
  46. ct_dst_sport_ltm: Connections to dst IP from src port in last 100 flows
  47. ct_dst_src_ltm: Connections between same src and dst IP in last 100 flows
  48. attack_cat: Attack category name (blank/null for benign) -> label_family
  49. Label: 0 for benign, 1 for attack -> label_binary

Verification note: The 49-column list above is unverified against a local copy of
NUSW-NB15_features.csv in Phase 1 as raw files are not downloaded.
"""

from typing import Any

import pandas as pd


def load_unsw_nb15(
    paths: Any,
    max_rows: int | None = None,
    **kw: Any,
) -> pd.DataFrame:
    """Load UNSW-NB15 raw flow CSVs into the unified ThreatGraph schema.

    Args:
        paths: File path or list of file paths to raw UNSW-NB15 CSV files.
        max_rows: Optional limit on the number of rows to load.
        **kw: Additional loader parameters.

    Returns:
        pd.DataFrame: Coerced and validated DataFrame.

    Raises:
        NotImplementedError: Scheduled for implementation in Phase 2.
    """
    raise NotImplementedError("planned for Phase 2")
