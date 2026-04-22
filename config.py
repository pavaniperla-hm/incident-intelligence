# Data settings
RAW_DATA_PATH = "data/raw/incidents.csv"
PROCESSED_PATH = "data/processed/"
OUTPUTS_PATH = "data/outputs/"

# Incident categories
INCIDENT_CATEGORIES = [
    "Authentication Failure",
    "Network Outage",
    "Deployment Failure",
    "Performance Degradation",
    "Security Breach"
]

# Source types and their reliability weights
SOURCE_WEIGHTS = {
    "alert":  1.0,
    "ticket": 0.8,
    "log":    0.6
}

# ML settings
TFIDF_MAX_FEATURES = 3000
TFIDF_NGRAM_RANGE = (1, 2)
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Similarity settings
SIMILARITY_THRESHOLD = 0.25
CORRELATION_TIME_WINDOW_MINUTES = 30

# Anomaly settings
ANOMALY_ZSCORE_THRESHOLD = 2.0

# Confidence score weights
CONFIDENCE_WEIGHTS = {
    "source_weight":    0.3,
    "classifier_conf":  0.4,
    "anomaly_boost":    0.3
}

# Transformer model names
SIMILARITY_MODEL = "all-MiniLM-L6-v2"
CLASSIFIER_MODEL = "distilbert-base-uncased"

# Backend selection: "transformer" or "classical"
DEFAULT_BACKEND = "transformer"
FALLBACK_BACKEND = "classical"
