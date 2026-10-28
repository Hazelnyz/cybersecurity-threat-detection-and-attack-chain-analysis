"""Configuration settings and global constants for ThreatGraph."""

import os
from dataclasses import dataclass, field

WINDOW_SECONDS: int = 60
EXPLOIT_FLOOR: float = 0.05
MIN_PROB: float = 1e-6
GLOBAL_SEED: int = 42


@dataclass(frozen=True)
class Settings:
    """ThreatGraph runtime configuration settings.

    Attributes are loaded from environment variables with default fallbacks.
    """

    neo4j_uri: str = field(
        default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687")
    )
    neo4j_user: str = field(
        default_factory=lambda: os.getenv("NEO4J_USER", "neo4j")
    )
    neo4j_password: str = field(
        default_factory=lambda: os.getenv("NEO4J_PASSWORD", "change_me")
    )
    postgres_dsn: str = field(
        default_factory=lambda: os.getenv(
            "POSTGRES_DSN",
            "postgresql://threatgraph:change_me@localhost:5432/threatgraph",
        )
    )
    window_seconds: int = WINDOW_SECONDS
    exploit_floor: float = EXPLOIT_FLOOR
    min_prob: float = MIN_PROB
    global_seed: int = GLOBAL_SEED


def get_settings() -> Settings:
    """Return an instance of the ThreatGraph settings.

    Returns:
        Settings: A frozen dataclass populated with current settings.
    """
    return Settings()
