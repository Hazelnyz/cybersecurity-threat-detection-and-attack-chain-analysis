# ThreatGraph

**ThreatGraph** is a graph-based cyber threat detection and attack-path analysis framework designed for enterprise network telemetry. ThreatGraph transforms continuous network flow records into time-windowed communication graphs, evaluates reachability and vulnerability posture across simulated enterprise topologies, and computes optimal multi-hop adversary attack paths using graph search algorithms.

---

## Progress Tracker

- [x] **Phase 1: Foundation, Data Loading, Graph Schema, & Simulated Topology**
  - Project scaffold and environment configuration (Python 3.11, strict ruff linting, zero-dependency offline test suite).
  - Unified network flow schema (`FLOW_COLUMNS`), dtype coercion, validation, and SHA-256 deduplication key generation.
  - Deterministic synthetic 4-stage kill-chain generator (`recon` $\to$ `initial_access` $\to$ `lateral_movement` $\to$ `exfiltration`) with realistic time gaps and benign hard negatives.
  - Simulated 18-host enterprise topology across 5 security zones with CVSS-inspired exploitability scoring and firewall rule evaluation.
  - Real dataset ingestion: chunked CICIDS2017 loader with repeated header stripping, UTC normalization, and parquet caching; UNSW-NB15 and CTU-13 specifications.
  - Tumbling time-windowed graph builder ($[t_{\text{start}}, t_{\text{end}})$) with strict temporal leakage assertions, observed communication edges (`connects_to`), structural edges (`hosted_on`), and leak-free feature allowlists (`FEATURE_COLUMNS_HOST`, `FEATURE_COLUMNS_EDGE`).
  - Persistence layer: Docker Compose specification (Neo4j 5 + PostgreSQL 16), PostgreSQL DDL (`flows`, `predictions`, `attack_findings`), and parameterized Neo4j batch loader with uniqueness constraints.
- [ ] **Phase 2: Graph Neural Networks (GNNs)**
- [ ] **Phase 3: Bayesian Threat Reasoning & MITRE ATT&CK Mapping**
- [ ] **Phase 4: Fuzzy Logic & Rule Engine**
- [x] **Phase 5: Attack-Path Search & Reinforcement Learning** *(Partly Done)*
  - [x] Min-heap Dijkstra search algorithm on directed attack graphs with additive negative log-probability edge costs ($c = -\ln(p)$).
  - [x] A* search algorithm with an admissible and consistent reverse unweighted BFS hop heuristic ($h(n) = c_{\min} \cdot \text{hops}(n, \text{target})$).
  - [x] Attack-path evaluation demo script comparing Dijkstra and A* across crown jewel targets.
  - [ ] Reinforcement learning agent for adaptive multi-stage lateral movement modeling (planned).
- [ ] **Phase 6: Explainability & API**

---

## Quick Start

### 1. Installation

Ensure you are using Python 3.11. Install core dependencies and development tools:

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

### 2. Run the Verification Test Suite

All tests run completely offline without requiring dataset downloads or running databases:

```bash
python -m pytest
```

### 3. Run the Attack-Path Search Demo

Execute Dijkstra and A* path search algorithms from the external adversary (`attacker-ext`) to enterprise crown jewels (`dc-01`, `db-01`):

```bash
python scripts/demo_attack_path.py
```

### 4. Optional: Run the Synthetic Flow & Window Loader Demo

Generate multi-stage synthetic attack flows, partition them into tumbling 60-second graph windows, and preview graph statistics:

```bash
python scripts/load_demo.py
```

---

## Documentation

For the complete schema specification, feature allowlists, mathematical proofs of heuristic admissibility, and pipeline architecture diagrams, see:
- [`docs/graph_schema.md`](docs/graph_schema.md)

