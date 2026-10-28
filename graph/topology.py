"""Simulated enterprise topology and attack-edge probability model."""

import ipaddress
import math
from pathlib import Path

import networkx as nx
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from config import EXPLOIT_FLOOR, MIN_PROB

_DEFAULT_TOPOLOGY_PATH = Path(__file__).parent / "topology.yaml"
_CACHED_TOPOLOGY: "Topology | None" = None


class Vulnerability(BaseModel):
    """Simulated vulnerability with an analyst-assigned exploitability score."""

    id: str
    name: str
    exploitability: float

    @field_validator("exploitability")
    @classmethod
    def validate_exploitability(cls, v: float) -> float:
        """Validate exploitability is strictly in (0, 1]."""
        if not (0.0 < v <= 1.0):
            raise ValueError(f"Vulnerability exploitability must be in (0, 1], got {v}")
        return v


class Service(BaseModel):
    """Network service hosted on a topology node."""

    name: str
    port: int
    proto: str
    vuln: Vulnerability | None = None

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port is in 0..65535."""
        if not (0 <= v <= 65535):
            raise ValueError(f"Service port must be in 0..65535, got {v}")
        return v

    @field_validator("proto")
    @classmethod
    def validate_proto(cls, v: str) -> str:
        """Ensure lowercase protocol."""
        return v.lower()


class Host(BaseModel):
    """Simulated network host."""

    id: str
    ip: str
    segment: str
    os: str
    asset_criticality: int
    services: list[Service] = Field(default_factory=list)

    @field_validator("ip")
    @classmethod
    def validate_ip(cls, v: str) -> str:
        """Validate IP address format."""
        try:
            ipaddress.ip_address(v.strip())
        except ValueError as err:
            raise ValueError(f"Invalid IP address '{v}': {err}") from err
        return v.strip()

    @field_validator("asset_criticality")
    @classmethod
    def validate_criticality(cls, v: int) -> int:
        """Validate asset criticality is an integer in 1..5."""
        if not (1 <= v <= 5):
            raise ValueError(f"Asset criticality must be in 1..5, got {v}")
        return v


class FirewallRule(BaseModel):
    """Firewall access control rule between segments."""

    from_segment: str
    to_segment: str
    ports: list[int]
    proto: str
    action: str = "allow"

    @field_validator("ports")
    @classmethod
    def validate_ports(cls, ports: list[int]) -> list[int]:
        """Validate all rule ports are in 0..65535."""
        for p in ports:
            if not (0 <= p <= 65535):
                raise ValueError(f"Firewall rule port must be in 0..65535, got {p}")
        return ports

    @field_validator("proto")
    @classmethod
    def validate_proto(cls, v: str) -> str:
        """Ensure lowercase protocol."""
        return v.lower()


class Topology(BaseModel):
    """Enterprise network topology model."""

    segments: list[str]
    hosts: list[Host]
    firewall_rules: list[FirewallRule] = Field(default_factory=list)
    crown_jewels: list[str] = Field(default_factory=list)
    entry_points: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_topology_integrity(self) -> "Topology":
        """Validate cross-entity consistency, uniqueness, and reference integrity."""
        seen_ids: set[str] = set()
        seen_ips: set[str] = set()
        segment_set = set(self.segments)

        for host in self.hosts:
            if host.id in seen_ids:
                raise ValueError(f"Duplicate host id: '{host.id}'")
            seen_ids.add(host.id)

            if host.ip in seen_ips:
                raise ValueError(f"Duplicate host IP: '{host.ip}'")
            seen_ips.add(host.ip)

            if host.segment not in segment_set:
                raise ValueError(
                    f"Host '{host.id}' references undefined segment: '{host.segment}'"
                )

        for rule in self.firewall_rules:
            if rule.from_segment not in segment_set:
                raise ValueError(
                    f"Firewall rule references undefined from_segment: '{rule.from_segment}'"
                )
            if rule.to_segment not in segment_set:
                raise ValueError(
                    f"Firewall rule references undefined to_segment: '{rule.to_segment}'"
                )

        for cj in self.crown_jewels:
            if cj not in seen_ids:
                raise ValueError(f"Crown jewel references unknown host: '{cj}'")

        for ep in self.entry_points:
            if ep not in seen_ids:
                raise ValueError(f"Entry point references unknown host: '{ep}'")

        return self

    def ip_to_host(self) -> dict[str, Host]:
        """Build a mapping from IP address to Host model."""
        return ip_to_host(self)

    def reachable_services(self, src_host: Host, dst_host: Host) -> list[Service]:
        """Determine reachable services on dst_host accessible from src_host."""
        return reachable_services(src_host, dst_host, self)

    def base_traversal_probability(self, src_host: Host, dst_host: Host) -> float:
        """Compute base attack traversal probability between two hosts."""
        return base_traversal_probability(src_host, dst_host, self)

    def build_attack_graph(self) -> nx.DiGraph:
        """Construct the directed attack graph for this topology."""
        return build_attack_graph(self)


def load_topology(path: str | Path | None = None) -> Topology:
    """Load and validate enterprise topology from a YAML file.

    Args:
        path: Path to topology.yaml. Defaults to graph/topology.yaml if None.

    Returns:
        Topology: Validated Topology model instance.

    Raises:
        ValueError: If file is not found or fails validation checks.
    """
    target_path = Path(path) if path is not None else _DEFAULT_TOPOLOGY_PATH
    if not target_path.exists():
        raise ValueError(f"Topology configuration file not found: {target_path}")

    with target_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"Topology YAML must define a dictionary root, got {type(data).__name__}"
        )

    try:
        return Topology.model_validate(data)
    except Exception as err:
        raise ValueError(str(err)) from err


def get_default_topology() -> Topology:
    """Retrieve or load the cached default topology.

    Returns:
        Topology: Loaded enterprise topology instance.
    """
    global _CACHED_TOPOLOGY
    if _CACHED_TOPOLOGY is None:
        _CACHED_TOPOLOGY = load_topology(_DEFAULT_TOPOLOGY_PATH)
    return _CACHED_TOPOLOGY


def ip_to_host(topology: Topology) -> dict[str, Host]:
    """Build a mapping from IP address to Host model for the given topology.

    Args:
        topology: Validated enterprise topology.

    Returns:
        dict[str, Host]: Dictionary mapping IP strings to Host objects.
    """
    return {host.ip: host for host in topology.hosts}


def reachable_services(
    src_host: Host,
    dst_host: Host,
    topology: Topology | None = None,
) -> list[Service]:
    """Determine reachable services on dst_host accessible from src_host.

    Intra-segment communication is default-allow (all services reachable).
    Inter-segment communication is default-deny, evaluated against firewall rules.

    Args:
        src_host: Source host.
        dst_host: Destination host.
        topology: Network topology defining firewall rules. If None, uses default topology.

    Returns:
        list[Service]: List of reachable Service instances on dst_host.
    """
    if src_host.segment == dst_host.segment:
        return list(dst_host.services)

    active_topology = topology if topology is not None else get_default_topology()

    reachable: list[Service] = []
    for svc in dst_host.services:
        for rule in active_topology.firewall_rules:
            if (
                rule.action == "allow"
                and rule.from_segment == src_host.segment
                and rule.to_segment == dst_host.segment
                and (rule.proto in (svc.proto, "any", "all"))
                and (svc.port in rule.ports)
            ):
                reachable.append(svc)
                break
    return reachable


def base_traversal_probability(
    src_host: Host,
    dst_host: Host,
    topology: Topology | None = None,
) -> float:
    """Compute base attack traversal probability between two hosts.

    Formula:
        p = reachable(src, dst) * exploitability(dst) * evidence_adjustment

    Documented model note:
        The full model is p = reachable x exploitability x evidence_adjustment,
        and evidence_adjustment is added in a later phase and defaults to 1.0 here.

    reachable(src, dst) is 1 if any service on dst is accessible from src under
    the segment policy and firewall rules, else 0.
    exploitability(dst) is the maximum exploitability across all reachable services
    on dst (using EXPLOIT_FLOOR if none of the reachable services have an explicit vulnerability).

    If reachable is 0 or src == dst, p = 0.0 (no attack edge exists).
    When p > 0, the probability is clamped to [MIN_PROB, 1.0].

    Args:
        src_host: Attacking / source host.
        dst_host: Target / destination host.
        topology: Network topology model.

    Returns:
        float: Traversal probability clamped to [0.0, 1.0].
    """
    if src_host.id == dst_host.id:
        return 0.0

    services = reachable_services(src_host, dst_host, topology)
    if not services:
        return 0.0

    exploitabilities: list[float] = []
    for svc in services:
        if svc.vuln is not None:
            exploitabilities.append(svc.vuln.exploitability)
        else:
            exploitabilities.append(EXPLOIT_FLOOR)

    max_exploit = max(exploitabilities)
    evidence_adjustment = 1.0
    p = 1.0 * max_exploit * evidence_adjustment

    return float(max(MIN_PROB, min(1.0, p)))


def build_attack_graph(topology: Topology) -> nx.DiGraph:
    """Construct a directed attack graph based on simulated reachability and exploitability.

    Nodes represent topology hosts with their metadata.
    Directed edges represent possible attack transitions where traversal probability p > 0,
    with cost = -ln(p), edge_type = 'simulated_reachability', and source = 'simulated'.

    Args:
        topology: Validated enterprise topology.

    Returns:
        nx.DiGraph: Directed graph of simulated attack paths.
    """
    graph = nx.DiGraph()

    for host in topology.hosts:
        graph.add_node(
            host.id,
            ip=host.ip,
            segment=host.segment,
            os=host.os,
            asset_criticality=host.asset_criticality,
        )

    for src in topology.hosts:
        for dst in topology.hosts:
            if src.id == dst.id:
                continue
            p = base_traversal_probability(src, dst, topology)
            if p > 0.0:
                cost = -math.log(p)
                graph.add_edge(
                    src.id,
                    dst.id,
                    p=p,
                    cost=cost,
                    edge_type="simulated_reachability",
                    source="simulated",
                )

    return graph
