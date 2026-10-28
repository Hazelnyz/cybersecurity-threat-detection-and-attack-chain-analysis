-- ThreatGraph PostgreSQL Schema Initialization (Phase 1)
-- Plain DDL definitions with IF NOT EXISTS

CREATE TABLE IF NOT EXISTS flows (
    flow_id VARCHAR(64) PRIMARY KEY,
    ts_start TIMESTAMPTZ NOT NULL,
    duration_s DOUBLE PRECISION NOT NULL,
    src_ip VARCHAR(45) NOT NULL,
    src_port INTEGER,
    dst_ip VARCHAR(45) NOT NULL,
    dst_port INTEGER,
    proto VARCHAR(16) NOT NULL,
    fwd_pkts BIGINT,
    bwd_pkts BIGINT,
    fwd_bytes BIGINT,
    bwd_bytes BIGINT,
    label_binary SMALLINT NOT NULL,
    label_family VARCHAR(64) NOT NULL,
    label_raw VARCHAR(128),
    attack_stage VARCHAR(64),
    dataset VARCHAR(64) NOT NULL,
    dup_key VARCHAR(64) NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    window_id INTEGER NOT NULL,
    node_ip VARCHAR(45) NOT NULL,
    model_name VARCHAR(64) NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS attack_findings (
    id VARCHAR(64) PRIMARY KEY,
    window_id INTEGER NOT NULL,
    technique_id VARCHAR(32) NOT NULL,
    status VARCHAR(32) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    reason TEXT NOT NULL,
    evidence JSONB NOT NULL
);

