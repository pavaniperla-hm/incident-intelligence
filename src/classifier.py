"""
classifier.py — Incident category classification (dual-backend).

Purpose:
    Assigns each incident to one of the predefined categories using either a
    classical ML model (Logistic Regression on TF-IDF features) or a
    transformer model (DistilBERT CLS-token embeddings).  The transformer
    backend is used when ``DEFAULT_BACKEND = "transformer"`` in config and
    the ``transformers`` / ``torch`` packages are installed; otherwise the
    classical backend is used automatically.

Pipeline position: After preprocess; feeds confidence scores into scorer.

Inputs:
    - Classical backend  : ``preprocessed_text`` column (str) from preprocess
    - Transformer backend: same text column — raw strings are tokenised
                           internally by the HuggingFace tokenizer
    - config.INCIDENT_CATEGORIES   — ordered list of valid category labels
    - config.CLASSIFIER_MODEL      — HuggingFace model ID for transformer path
    - config.DEFAULT_BACKEND / FALLBACK_BACKEND
    - config.TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TEST_SIZE, RANDOM_STATE

Outputs:
    - Trained artefacts saved to config.PROCESSED_PATH:
        classifier_{backend}.pkl   — fitted LogisticRegression
        label_encoder.pkl          — fitted LabelEncoder
        tfidf_vectorizer.pkl       — fitted TfidfVectorizer (classical only)
        transformer_model_name.txt — model ID string (transformer only)
    - ``predict()`` returns a dict with predicted category, confidence score,
      full probability distribution, and which backend was used.
"""

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Path fix — allows running this file directly as a script from any directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (  # noqa: E402
    CLASSIFIER_MODEL,
    DEFAULT_BACKEND,
    INCIDENT_CATEGORIES,
    PROCESSED_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
)

# ---------------------------------------------------------------------------
# Optional transformer imports — availability flag set at import time
# ---------------------------------------------------------------------------
try:
    from transformers import AutoModel, AutoTokenizer
    import torch
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def get_tfidf_features(
    texts: list,
    vectorizer: TfidfVectorizer = None,
    fit: bool = True,
) -> tuple:
    """Build or apply a TF-IDF feature matrix from preprocessed text strings.

    When ``fit=True`` a new :class:`~sklearn.feature_extraction.text.TfidfVectorizer`
    is created with project-wide settings (sublinear TF scaling, unigrams +
    bigrams, capped vocabulary size) and fitted on ``texts``.  When
    ``fit=False`` the supplied ``vectorizer`` is used to transform ``texts``
    without refitting — this is the correct path for test/inference data to
    prevent data leakage.

    Args:
        texts:      List of preprocessed incident description strings.
        vectorizer: An already-fitted ``TfidfVectorizer``.  Required when
                    ``fit=False``; ignored when ``fit=True``.
        fit:        Whether to fit a new vectorizer (``True``) or transform
                    only (``False``).

    Returns:
        Tuple ``(X_matrix, vectorizer)`` where ``X_matrix`` is a scipy sparse
        matrix of shape ``(n_samples, n_features)`` and ``vectorizer`` is the
        fitted ``TfidfVectorizer`` instance.

    Raises:
        ValueError: If ``fit=False`` but no ``vectorizer`` is provided.
    """
    if not fit and vectorizer is None:
        raise ValueError("A fitted vectorizer must be supplied when fit=False.")

    if fit:
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            sublinear_tf=True,
            min_df=1,
        )
        X_matrix = vectorizer.fit_transform(texts)
        print(f"TF-IDF vocabulary: {len(vectorizer.vocabulary_)} terms")
    else:
        X_matrix = vectorizer.transform(texts)

    return X_matrix, vectorizer


def get_transformer_features(
    texts: list,
    model_name: str = None,
) -> np.ndarray:
    """Extract CLS-token embeddings from a pre-trained transformer model.

    Each text is independently tokenised (max 128 tokens, truncated / padded)
    and passed through the model in ``torch.no_grad()`` mode.  The embedding
    of the ``[CLS]`` token (index 0 of ``last_hidden_state``) is used as the
    fixed-size sentence representation, following the standard BERT convention.

    Args:
        texts:      List of incident description strings (normalised or
                    preprocessed — either works; transformer tokenizers handle
                    raw text directly).
        model_name: HuggingFace model identifier.  Defaults to
                    ``config.CLASSIFIER_MODEL`` (``"distilbert-base-uncased"``).

    Returns:
        NumPy array of shape ``(n_texts, hidden_size)`` — one embedding row
        per input text.

    Raises:
        RuntimeError: If ``TRANSFORMER_AVAILABLE`` is ``False`` (i.e.
                      ``transformers`` or ``torch`` is not installed).
    """
    if not TRANSFORMER_AVAILABLE:
        raise RuntimeError(
            "Transformer backend not available — install transformers and torch."
        )

    model_name = model_name or CLASSIFIER_MODEL
    print(f"Loading transformer model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    embeddings = []
    total = len(texts)

    for i, text in enumerate(texts, start=1):
        inputs = tokenizer(
            text,
            max_length=128,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = model(**inputs)
        cls_embedding = output.last_hidden_state[:, 0, :].numpy()
        embeddings.append(cls_embedding[0])

        if i % 25 == 0:
            print(f"Extracting features: {i}/{total}...")

    return np.vstack(embeddings)


def train_classifier(X_train, y_train) -> LogisticRegression:
    """Fit a multinomial Logistic Regression classifier.

    Uses L-BFGS so that output probabilities across all five categories sum to
    exactly 1.0 — a requirement for the confidence scoring stage.  The
    ``multi_class`` parameter was removed in scikit-learn 1.4; lbfgs handles
    multiclass natively.

    Args:
        X_train: Feature matrix for training — either a scipy sparse matrix
                 (classical backend) or a dense NumPy array (transformer).
        y_train: Encoded integer label array of shape ``(n_samples,)``.

    Returns:
        A fitted :class:`~sklearn.linear_model.LogisticRegression` instance.
    """
    clf = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        C=1.0,
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)
    print(f"Classifier trained on {X_train.shape[0]} samples")
    return clf


def predict(
    text: str,
    classifier: LogisticRegression,
    vectorizer_or_embedder,
    label_encoder: LabelEncoder,
    backend: str = "classical",
) -> dict:
    """Classify a single raw incident description and return a scored result.

    Runs the text through the same preprocessing and feature extraction that
    was used during training, then returns the predicted category alongside
    the full probability distribution so callers can assess confidence.

    Args:
        text:                  Raw or normalised incident description string.
        classifier:            Fitted :class:`~sklearn.linear_model.LogisticRegression`.
        vectorizer_or_embedder: Fitted ``TfidfVectorizer`` (classical backend)
                               or the HuggingFace model name string (transformer
                               backend — the model is reloaded on demand).
        label_encoder:         Fitted :class:`~sklearn.preprocessing.LabelEncoder`.
        backend:               ``"classical"`` or ``"transformer"``.

    Returns:
        Dict with keys:

        - ``predicted_category`` — decoded string label.
        - ``confidence``         — float probability of the top class (0–1).
        - ``all_probabilities``  — ``{category: probability}`` for all five
                                   categories.
        - ``backend_used``       — the backend string that was actually used.
    """
    from src.preprocess import preprocess_text  # local import avoids circular deps

    cleaned = preprocess_text(text)

    if backend == "transformer":
        X = get_transformer_features([cleaned], model_name=vectorizer_or_embedder)
    else:
        X, _ = get_tfidf_features([cleaned], vectorizer=vectorizer_or_embedder, fit=False)

    proba = classifier.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    predicted_label = label_encoder.inverse_transform([pred_idx])[0]
    all_proba = {
        label_encoder.inverse_transform([i])[0]: round(float(p), 4)
        for i, p in enumerate(proba)
    }

    return {
        "predicted_category": predicted_label,
        "confidence": round(float(proba[pred_idx]), 4),
        "all_probabilities": all_proba,
        "backend_used": backend,
    }


def run_training_pipeline(df: pd.DataFrame, backend: str = None) -> dict:
    """Execute the end-to-end training pipeline for a given backend.

    Steps:

    1. Resolve the backend (param → config default → fallback).
    2. Encode labels with :class:`~sklearn.preprocessing.LabelEncoder` fitted
       on the full ``INCIDENT_CATEGORIES`` list so encoding is stable across
       runs even if a category is absent from a small dataset.
    3. Stratified train / test split (``TEST_SIZE``, ``RANDOM_STATE``).
    4. Extract features for train set (fit) and test set (transform only).
    5. Train :class:`~sklearn.linear_model.LogisticRegression`.
    6. Serialise all artefacts to ``PROCESSED_PATH``.

    Args:
        df:      Preprocessed signals DataFrame containing at least
                 ``preprocessed_text`` and ``category`` columns.
        backend: ``"classical"`` or ``"transformer"``.  ``None`` falls back to
                 ``config.DEFAULT_BACKEND``.

    Returns:
        Dict with keys:
            ``classifier``    — fitted LogisticRegression.
            ``label_encoder`` — fitted LabelEncoder.
            ``vectorizer``    — fitted TfidfVectorizer (classical) or model
                               name string (transformer).
            ``X_test``        — held-out feature matrix.
            ``y_test``        — held-out encoded label array.
            ``backend``       — backend string actually used.
    """
    # --- Resolve backend ---------------------------------------------------
    resolved_backend = backend or DEFAULT_BACKEND
    if resolved_backend == "transformer" and not TRANSFORMER_AVAILABLE:
        print(
            "WARNING: transformer backend requested but transformers/torch are "
            "not installed. Falling back to classical backend."
        )
        resolved_backend = "classical"

    print(f"\n=== Training pipeline — backend: {resolved_backend} ===")

    # --- Label encoding ----------------------------------------------------
    le = LabelEncoder()
    le.fit(INCIDENT_CATEGORIES)  # fit on full category list for stability
    y = le.transform(df["category"].values)

    # --- Train / test split (stratified) -----------------------------------
    texts = df["preprocessed_text"].tolist()
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Train size: {len(X_train_texts)}  |  Test size: {len(X_test_texts)}")

    # --- Feature extraction ------------------------------------------------
    vectorizer = None
    if resolved_backend == "transformer":
        X_train = get_transformer_features(X_train_texts, model_name=CLASSIFIER_MODEL)
        X_test = get_transformer_features(X_test_texts, model_name=CLASSIFIER_MODEL)
    else:
        X_train, vectorizer = get_tfidf_features(X_train_texts, fit=True)
        X_test, _ = get_tfidf_features(X_test_texts, vectorizer=vectorizer, fit=False)

    # --- Train -------------------------------------------------------------
    clf = train_classifier(X_train, y_train)

    # --- Persist artefacts -------------------------------------------------
    os.makedirs(PROCESSED_PATH, exist_ok=True)

    clf_path = os.path.join(PROCESSED_PATH, f"classifier_{resolved_backend}.pkl")
    le_path = os.path.join(PROCESSED_PATH, "label_encoder.pkl")

    with open(clf_path, "wb") as f:
        pickle.dump(clf, f)
    with open(le_path, "wb") as f:
        pickle.dump(le, f)

    if resolved_backend == "classical":
        vec_path = os.path.join(PROCESSED_PATH, "tfidf_vectorizer.pkl")
        with open(vec_path, "wb") as f:
            pickle.dump(vectorizer, f)
        print(f"Saved: {clf_path}")
        print(f"Saved: {vec_path}")
    else:
        model_name_path = os.path.join(PROCESSED_PATH, "transformer_model_name.txt")
        with open(model_name_path, "w") as f:
            f.write(CLASSIFIER_MODEL)
        vectorizer = CLASSIFIER_MODEL  # return the name string
        print(f"Saved: {clf_path}")
        print(f"Saved: {model_name_path}")

    print(f"Saved: {le_path}")

    return {
        "classifier": clf,
        "label_encoder": le,
        "vectorizer": vectorizer,
        "X_test": X_test,
        "y_test": y_test,
        "backend": resolved_backend,
    }


def load_classifier(backend: str = "classical") -> tuple:
    """Load serialised classifier artefacts from disk.

    Loads the classifier and label encoder (common to both backends) plus the
    backend-specific feature extractor: a TfidfVectorizer for the classical
    backend or the model name string for the transformer backend.

    Args:
        backend: ``"classical"`` or ``"transformer"``.

    Returns:
        Tuple ``(classifier, label_encoder, vectorizer_or_model_name)``.

    Raises:
        FileNotFoundError: If any required artefact file is missing from
                           ``config.PROCESSED_PATH``.
    """
    clf_path = os.path.join(PROCESSED_PATH, f"classifier_{backend}.pkl")
    le_path = os.path.join(PROCESSED_PATH, "label_encoder.pkl")

    for path in (clf_path, le_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Artefact not found: {path}\n"
                f"Run run_training_pipeline(backend='{backend}') first."
            )

    with open(clf_path, "rb") as f:
        clf = pickle.load(f)
    with open(le_path, "rb") as f:
        le = pickle.load(f)

    if backend == "classical":
        vec_path = os.path.join(PROCESSED_PATH, "tfidf_vectorizer.pkl")
        if not os.path.exists(vec_path):
            raise FileNotFoundError(
                f"Vectorizer not found: {vec_path}\n"
                f"Run run_training_pipeline(backend='classical') first."
            )
        with open(vec_path, "rb") as f:
            vectorizer = pickle.load(f)
    else:
        model_name_path = os.path.join(PROCESSED_PATH, "transformer_model_name.txt")
        if not os.path.exists(model_name_path):
            raise FileNotFoundError(
                f"Model name file not found: {model_name_path}\n"
                f"Run run_training_pipeline(backend='transformer') first."
            )
        with open(model_name_path, "r") as f:
            vectorizer = f.read().strip()

    return clf, le, vectorizer


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _project_root = os.path.join(os.path.dirname(__file__), "..")
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    # --- Load or build processed data -------------------------------------
    processed_csv = os.path.join(_project_root, "data", "processed", "processed_incidents.csv")
    if not os.path.exists(processed_csv):
        print("Processed data not found — running full preprocessing pipeline...")
        raw_csv = os.path.join(_project_root, "data", "raw", "incidents.csv")
        df_raw = pd.read_csv(raw_csv)
        from src.normaliser import normalise_dataframe
        from src.preprocess import preprocess_dataframe
        df_norm = normalise_dataframe(df_raw.copy())
        df = preprocess_dataframe(df_norm)
        os.makedirs(os.path.dirname(processed_csv), exist_ok=True)
        df.to_csv(processed_csv, index=False)
    else:
        df = pd.read_csv(processed_csv)

    print(f"Loaded {len(df)} signals from {processed_csv}")

    # --- Classical backend (always runs) ----------------------------------
    result_classical = run_training_pipeline(df, backend="classical")

    sample_text = (
        "Multiple users cannot log in — authentication service returning 401 errors"
    )
    pred = predict(
        sample_text,
        result_classical["classifier"],
        result_classical["vectorizer"],
        result_classical["label_encoder"],
        backend="classical",
    )
    print(f"\nSample prediction:")
    print(f"  Predicted : {pred['predicted_category']} (confidence: {pred['confidence']:.2f})")
    print(f"  Backend   : {pred['backend_used']}")
    print(f"  All probs : {json.dumps(pred['all_probabilities'], indent=4)}")

    # --- Transformer backend (optional) -----------------------------------
    if TRANSFORMER_AVAILABLE:
        print("\n=== Running transformer backend ===")
        result_transformer = run_training_pipeline(df, backend="transformer")
        pred_t = predict(
            sample_text,
            result_transformer["classifier"],
            result_transformer["vectorizer"],
            result_transformer["label_encoder"],
            backend="transformer",
        )
        print(f"\nSample prediction (transformer):")
        print(f"  Predicted : {pred_t['predicted_category']} (confidence: {pred_t['confidence']:.2f})")
        print(f"  Backend   : {pred_t['backend_used']}")
    else:
        print("\nTransformer backend not available — skipping")
        print("Install with: pip install transformers torch")
