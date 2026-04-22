# Incident Intelligence System — CLAUDE.md

## Project Purpose

An ML-based incident intelligence pipeline that automates operational incident triage. It ingests raw signals from multiple heterogeneous sources (alerts, tickets, logs), applies NLP and ML processing, classifies incidents into categories, detects anomalous activity, groups related events through semantic clustering, and surfaces actionable intelligence.

**Problem it solves**: Operations teams are flooded with noisy cross-source data. This system automatically normalises, categorises, correlates, and prioritises incidents for engineer attention.

---

## Pipeline Overview

```
data/raw/incidents.csv
        ↓
[data_generator]    Generate synthetic 150-signal dataset (dev/test only)
        ↓
[normaliser]        Standardise schema, token replacement, temporal features
        ↓
[preprocess]        Text cleaning, tokenisation, lemmatisation, stopword removal
        ↓
[classifier]        Category prediction (dual-backend: transformer / classical)
[similarity]        Pairwise cosine similarity matrix (dual-backend)
        ↓
[anomaly]           Z-score anomaly detection on hourly time-series
        ↓
[correlator]        Union-Find clustering: similarity + time + category gates
        ↓
[scorer]            Composite confidence score (source + classifier + anomaly)
        ↓
[evaluate]          Metrics report (precision / recall / F1)
[visualise / app]   Streamlit dashboard
```

Each stage serialises output to `data/processed/` so individual modules can be run in isolation.

---

## Directory Structure

```
incident-intelligence/
├── config.py                        # All tunable parameters (single source of truth)
├── main.py                          # CLI orchestration (stub)
├── app.py                           # Streamlit dashboard entry point (stub)
├── requirements.txt
├── src/
│   ├── data_generator.py            # Synthetic data generation ✓
│   ├── data_loader.py               # CSV/JSON ingestion (stub)
│   ├── normaliser.py                # Schema normalisation + text token replacement ✓
│   ├── preprocess.py                # NLP tokenisation / lemmatisation ✓
│   ├── classifier.py                # Incident category classifier, dual-backend ✓
│   ├── similarity.py                # Pairwise similarity matrix, dual-backend ✓
│   ├── anomaly.py                   # Z-score anomaly scoring ✓
│   ├── correlator.py                # Union-Find signal clustering ✓
│   ├── scorer.py                    # Composite confidence scoring (stub)
│   ├── evaluate.py                  # Pipeline evaluation metrics (stub)
│   └── visualise.py                 # Charts + dashboard components (stub)
├── data/
│   ├── raw/
│   │   └── incidents.csv            # Input: 150 signals × 6 cols
│   └── processed/
│       ├── processed_incidents.csv  # After normalise + preprocess (12 cols)
│       ├── scored_incidents.csv     # After anomaly scoring (16 cols)
│       ├── similarity_pairs.json    # All pairs above threshold
│       └── correlation_results.json # Correlated groups with metadata
├── models/                          # Serialised ML artifacts (auto-created)
│   ├── classifier_classical.pkl
│   ├── classifier_transformer.pkl
│   ├── label_encoder.pkl
│   ├── tfidf_vectorizer.pkl
│   └── transformer_model_name.txt
├── tests/                           # Stubs — not yet implemented
│   ├── test_preprocess.py
│   ├── test_classifier.py
│   ├── test_similarity.py
│   └── test_correlator.py
└── notebooks/
    └── transformer_experiments.ipynb  # Research notebook (stub)
```

---

## Incident Categories (5 total)

| Category | Example signals |
|---|---|
| Authentication Failure | SSO/token/credential issues, login errors, MFA failures |
| Network Outage | Connectivity loss, latency spikes, BGP/DNS issues, packet loss |
| Deployment Failure | CI/CD failures, pod crashes, config errors, rollback triggers |
| Performance Degradation | Slow queries, high CPU/memory, timeouts, cache issues |
| Security Breach | Brute force attempts, privilege escalation, SQL injection, data exfiltration |

---

## Module Reference

### data_generator.py
Generates 150 synthetic signals (8 incidents × 5 categories × 3 signals each) with realistic templates per source type (alert/ticket/log). Adds 20% vocabulary mixing to simulate real-world cross-category ambiguity.

**Output schema**: `signal_id, source_type, category, text, timestamp, incident_group`

### normaliser.py
11-step regex-based normalisation applied to raw text:
1. Lowercase
2. Strip log-level prefixes (CRITICAL, ERROR, WARN, etc.)
3. Percentages → `PCTVAL`
4. Times → `TIMEVAL`
5. Version strings → `VERVAL`
6. IP addresses → `IPADDR`
7. 3+ digit integers → `NUMVAL`
8. Exception class names → `EXCEPTION`
9-11. Strip special characters, collapse whitespace

Also adds: `source_weight`, `hour`, `day_of_week`, `is_business_hours` (08:00–18:00 Mon–Fri).

**Source reliability weights**: alert=1.0, ticket=0.8, log=0.6

### preprocess.py
NLP processing on `normalised_text`:
- Builds NLTK stopword list **minus** 23 incident-critical words (`failed`, `down`, `timeout`, `error`, etc.)
- Preserves special tokens (`PCTVAL`, `TIMEVAL`, `IPADDR`, etc.) verbatim
- Discards single-character tokens
- Lemmatises remaining tokens (noun mode)

**Output column**: `preprocessed_text`

### classifier.py
Dual-backend incident classifier using `LogisticRegression` on:
- **Transformer**: `[CLS]` embeddings from `distilbert-base-uncased` (max 128 tokens)
- **Classical**: TF-IDF vectors (max 3000 features, unigrams+bigrams, `sublinear_tf=True`)

`predict()` returns: `predicted_category`, `confidence`, `all_probabilities`, `backend_used`

Stratified 80/20 train/test split. Serialises artifacts to `models/`.

### similarity.py
Computes an n×n pairwise cosine similarity matrix:
- **Transformer**: Dense embeddings via `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Classical**: Sparse TF-IDF cosine similarity

`compute_similarity_matrix()` returns `(matrix, backend_used)` tuple — always unpack both values.

`find_similar_pairs(similarity_matrix, signal_ids)` extracts upper-triangle pairs above `SIMILARITY_THRESHOLD`, sorted descending by score.

### anomaly.py
Z-score anomaly detection on hourly signal volume:
- Computes global mean/std across 24 hourly buckets
- Per-signal z-score clipped to `[-3, 3]`, normalised to `[0, 1]`
- Off-hours multiplier: 1.3× for signals outside business hours (capped at 1.0)

**Output columns**: `z_score`, `anomaly_score`, `is_anomalous`, `time_context`

**Threshold**: `ANOMALY_ZSCORE_THRESHOLD = 2.0`

### correlator.py
Union-Find clustering with three gates per candidate pair:
1. Similarity ≥ `SIMILARITY_THRESHOLD`
2. Same `predicted_category` from classifier
3. Timestamps within `CORRELATION_TIME_WINDOW_MINUTES`

Singleton groups are discarded. Each surviving group gets a composite `confidence_score`:

```
anomaly_boost  = 1.0 + group.max_anomaly_score
confidence     = (max_source_weight   × 0.3
               +  max_classifier_conf × 0.4
               +  min(anomaly_boost, 2.0) × 0.3 × 0.5)
confidence     = min(confidence, 1.0)
```

`evaluate_correlation()` reports precision/recall/F1 at the signal-pair level against ground-truth `incident_group`.

**Output schema per group**: `group_id, signal_ids, predicted_category, signal_count, source_types, time_span_minutes, avg_similarity, max_anomaly_score, confidence_score, signals_detail`

---

## Configuration (`config.py`)

All parameters live in one place. Key values:

```python
# Paths
RAW_DATA_PATH                  = "data/raw/incidents.csv"
PROCESSED_PATH                 = "data/processed/"
OUTPUTS_PATH                   = "data/outputs/"

# ML
TFIDF_MAX_FEATURES             = 3000
TFIDF_NGRAM_RANGE              = (1, 2)
TEST_SIZE                      = 0.2
RANDOM_STATE                   = 42

# Backends
DEFAULT_BACKEND                = "transformer"   # falls back to "classical" if torch unavailable
FALLBACK_BACKEND               = "classical"
CLASSIFIER_MODEL               = "distilbert-base-uncased"
SIMILARITY_MODEL               = "all-MiniLM-L6-v2"

# Similarity & correlation
SIMILARITY_THRESHOLD           = 0.35
CORRELATION_TIME_WINDOW_MINUTES = 30

# Anomaly
ANOMALY_ZSCORE_THRESHOLD       = 2.0

# Confidence scoring weights
CONFIDENCE_WEIGHTS = {
    "source_weight":   0.3,
    "classifier_conf": 0.4,
    "anomaly_boost":   0.3,
}

SOURCE_WEIGHTS = {"alert": 1.0, "ticket": 0.8, "log": 0.6}
```

---

## How to Run

### Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

# Download NLTK data (one-time)
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

### Full pipeline via CLI (stub — partial)

```bash
python main.py --generate          # Generate synthetic data first
python main.py                     # Run with transformer backend
python main.py --backend classical # Run with classical backend (no GPU needed)
```

### Run individual modules (all are directly executable)

```bash
# 1. Generate synthetic dataset → data/raw/incidents.csv
python src/data_generator.py

# 2. Normalise + preprocess → data/processed/processed_incidents.csv
python src/normaliser.py
python src/preprocess.py

# 3. Train classifier (both backends) → models/
python src/classifier.py

# 4. Compute similarity matrix → data/processed/similarity_pairs.json
python src/similarity.py

# 5. Score anomalies → data/processed/scored_incidents.csv
python src/anomaly.py

# 6. Correlate signals → data/processed/correlation_results.json + metrics
python src/correlator.py
```

### Dashboard (stub)

```bash
streamlit run app.py
```

### Tests (stubs — not yet implemented)

```bash
pytest tests/
```

---

## Implementation Status

| Module | Status |
|---|---|
| data_generator.py | Done |
| normaliser.py | Done |
| preprocess.py | Done |
| classifier.py | Done |
| similarity.py | Done |
| anomaly.py | Done |
| correlator.py | Done |
| data_loader.py | Stub |
| scorer.py | Stub |
| evaluate.py | Stub |
| visualise.py | Stub |
| app.py | Stub |
| main.py | Stub |
| tests/ | Stubs |
| notebooks/ | Stubs |

---

## Dependencies

```
pandas==2.2.2            # Data manipulation
numpy==1.26.4            # Numerics
scikit-learn==1.4.2      # TF-IDF, LogisticRegression, cosine_similarity
nltk==3.8.1              # Tokenisation, lemmatisation, stopwords
sentence-transformers==3.0.0  # Transformer similarity embeddings
transformers==4.41.0     # HuggingFace model loading
torch==2.3.0             # PyTorch backend for transformers
streamlit==1.35.0        # Web dashboard
matplotlib==3.9.0        # Plotting
seaborn==0.13.2          # Statistical visualisation
pytest==8.2.0            # Testing
```

The transformer stack (`sentence-transformers`, `transformers`, `torch`) is optional. All modules automatically fall back to the classical backend if these are not installed.

---

## Key Design Decisions

- **Dual-backend architecture**: Transformer for accuracy when GPU/packages available; classical for robustness everywhere else. Fallback is automatic and transparent.
- **Token replacement in normaliser**: Ephemeral values (IPs, timestamps, percentages) are replaced with semantic tokens (`IPADDR`, `TIMEVAL`, etc.) to keep the classifier vocabulary stable across datasets.
- **Incident-critical stopword allowlist**: 23 operationally meaningful words (`failed`, `timeout`, `down`, `crash`, etc.) are excluded from NLTK's stopword list so they survive preprocessing.
- **Union-Find for correlation**: O(α(n)) amortised merge with three validation gates — prevents false groupings from similarity alone.
- **Off-hours anomaly boost (1.3×)**: Reflects that incidents outside business hours carry higher operational risk.
- **Composite confidence score**: Source reliability (30%) + classifier confidence (40%) + anomaly boost (30%).
- **CSV-first intermediate storage**: Every pipeline stage saves to CSV so any stage can be re-run from disk without rerunning the full pipeline.
- **`compute_similarity_matrix` returns a tuple**: Always unpack as `matrix, backend = compute_similarity_matrix(...)`. Passing the tuple directly to `find_similar_pairs` causes an `AttributeError`.
