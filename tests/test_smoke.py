"""Smoke tests for ThreatGraph configuration and project setup."""

from config import EXPLOIT_FLOOR, GLOBAL_SEED, MIN_PROB, WINDOW_SECONDS, Settings


def test_settings_load() -> None:
    """Assert that Settings loads properly with default values."""
    settings = Settings()
    assert settings.neo4j_uri == "bolt://localhost:7687"
    assert settings.neo4j_user == "neo4j"
    assert settings.neo4j_password == "change_me"
    assert (
        settings.postgres_dsn
        == "postgresql://threatgraph:change_me@localhost:5432/threatgraph"
    )
    assert settings.window_seconds == 60
    assert settings.exploit_floor == 0.05
    assert settings.min_prob == 1e-6
    assert settings.global_seed == 42


def test_global_constants() -> None:
    """Assert that top-level constants have the expected values."""
    assert WINDOW_SECONDS == 60
    assert EXPLOIT_FLOOR == 0.05
    assert MIN_PROB == 1e-6
    assert GLOBAL_SEED == 42

