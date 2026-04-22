# ML-Based Incident Intelligence System

## Overview

A machine learning pipeline that ingests operational incidents from heterogeneous sources (alerts, tickets, logs), classifies them by category, detects anomalies, groups related events into correlation clusters, and surfaces actionable intelligence through an interactive dashboard.

## Problem Statement

Operations teams are flooded with incidents from multiple monitoring systems using different formats and reliability levels. Manually triaging, deduplicating, and correlating these events is slow and error-prone. This system automates the intelligence layer: it normalises cross-source data, categorises incidents with confidence scores, flags anomalous spikes, and clusters related events so engineers can focus on root causes rather than noise.

## Architecture

```
data/raw/incidents.csv
        │
        ▼
 [data_loader]          — reads and validates raw CSV
        │
        ▼
  [normaliser]          — standardises schema, applies source weights
        │
        ▼
  [preprocess]          — text cleaning, TF-IDF or transformer prep
        │
   ┌────┴────┐
   ▼         ▼
[classifier] [similarity]   — category prediction | pairwise similarity
   │         │
   └────┬────┘
        │
      [anomaly]          — z-score spike detection on time-series
        │
   [correlator]          — temporal + semantic clustering
        │
    [scorer]             — composite confidence score
        │
   ┌────┴────┐
   ▼         ▼
[evaluate] [visualise]   — metrics report | Streamlit dashboard
```

## ML Components

### Classification — Dual Backend

| Backend | Model | Features |
|---|---|---|
| `classical` | Logistic Regression / Random Forest | TF-IDF (up to 3000 features, unigrams + bigrams) |
| `transformer` | DistilBERT (`distilbert-base-uncased`) | Raw description strings, fine-tuned on incident categories |

The system starts with `DEFAULT_BACKEND = "transformer"`. If the transformer model is unavailable (e.g., no GPU, import error), it automatically falls back to `FALLBACK_BACKEND = "classical"`. Both backends expose the same interface: a list of predicted labels and a parallel list of confidence scores in [0, 1].

### Similarity — Dual Backend

| Backend | Model | Method |
|---|---|---|
| `classical` | TF-IDF vectors | Cosine similarity on sparse matrix |
| `transformer` | `all-MiniLM-L6-v2` (sentence-transformers) | Dense embedding cosine similarity |

Pairs whose similarity score exceeds `SIMILARITY_THRESHOLD = 0.5` are passed to the correlator for cluster merging. The transformer backend captures semantic similarity beyond keyword overlap (e.g., "login failed" ≈ "authentication error"), while the classical backend is faster and has no model download requirement.

## Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (required for classical preprocessing)
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

## How to Run

**Generate synthetic data and run the full pipeline:**
```bash
python main.py --generate
```

**Run pipeline on existing data (transformer backend):**
```bash
python main.py
```

**Run pipeline with classical backend:**
```bash
python main.py --backend classical
```

**Launch the interactive dashboard:**
```bash
streamlit run app.py
```

**Run tests:**
```bash
pytest tests/
```

## Project Structure

```
incident-intelligence/
├── data/
│   ├── raw/            — source CSV files (incidents.csv)
│   ├── processed/      — preprocessed feature artifacts
│   └── outputs/        — evaluation reports, charts, model files
├── src/
│   ├── data_generator.py   — synthetic incident generation
│   ├── data_loader.py      — CSV ingestion and validation
│   ├── normaliser.py       — cross-source schema normalisation
│   ├── preprocess.py       — text cleaning and vectorisation
│   ├── classifier.py       — dual-backend category classification
│   ├── similarity.py       — dual-backend semantic similarity
│   ├── anomaly.py          — z-score anomaly detection
│   ├── correlator.py       — temporal + semantic clustering
│   ├── scorer.py           — composite confidence scoring
│   ├── evaluate.py         — pipeline evaluation and reporting
│   └── visualise.py        — charts and Streamlit components
├── notebooks/
│   └── transformer_experiments.ipynb   — Track B: transformer tuning
├── tests/
│   ├── test_preprocess.py
│   ├── test_classifier.py
│   ├── test_similarity.py
│   └── test_correlator.py
├── app.py          — Streamlit dashboard entry point
├── main.py         — CLI pipeline entry point
├── config.py       — all constants and hyperparameters
└── requirements.txt
```
