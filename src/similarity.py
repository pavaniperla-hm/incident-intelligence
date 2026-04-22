"""
similarity.py — Incident deduplication and similarity detection (dual-backend).

Purpose:
    Computes pairwise semantic similarity between incident descriptions to
    identify near-duplicate or closely related events.  Supports two backends:
    classical (TF-IDF cosine similarity on sparse vectors) and transformer
    (sentence-transformer dense embeddings + cosine similarity).  The
    transformer backend captures semantic similarity beyond keyword overlap —
    e.g. "login failed" ≈ "authentication error" — while the classical backend
    is lightweight and has no model-download requirement.

Pipeline position: After preprocess; feeds into correlator.

Inputs:
    - List of preprocessed or normalised text strings (one per signal)
    - List of signal_id strings aligned to the text list
    - config.SIMILARITY_MODEL     — sentence-transformers model ID
    - config.SIMILARITY_THRESHOLD — minimum score to flag a pair as similar
    - config.DEFAULT_BACKEND      — "transformer" or "classical"

Outputs:
    - Symmetric NumPy ndarray of shape (n, n) — pairwise cosine scores
    - List of pair dicts with keys signal_a, signal_b, similarity,
      above_threshold — saved to PROCESSED_PATH/similarity_pairs.json
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Path fix — allows running this file directly as a script from any directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (  # noqa: E402
    DEFAULT_BACKEND,
    PROCESSED_PATH,
    SIMILARITY_MODEL,
    SIMILARITY_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Optional sentence-transformers import
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False

# ---------------------------------------------------------------------------
# Module-level model cache — the model loads once per Python session
# ---------------------------------------------------------------------------
_transformer_model_cache: dict = {}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def get_transformer_model(model_name: str = None):
    """Return a loaded SentenceTransformer model, using a module-level cache.

    On the first call for a given model name the model is downloaded (or read
    from the local HuggingFace cache) and stored in ``_transformer_model_cache``.
    Subsequent calls with the same name return the cached object immediately,
    avoiding repeated disk I/O and model initialisation overhead.

    Args:
        model_name: sentence-transformers model identifier.  Defaults to
                    ``config.SIMILARITY_MODEL`` (``"all-MiniLM-L6-v2"``).

    Returns:
        A loaded :class:`sentence_transformers.SentenceTransformer` instance.

    Raises:
        RuntimeError: If ``SENTENCE_TRANSFORMER_AVAILABLE`` is ``False``
                      (i.e. the ``sentence-transformers`` package is not
                      installed).
    """
    if not SENTENCE_TRANSFORMER_AVAILABLE:
        raise RuntimeError(
            "Transformer backend not available — install sentence-transformers:\n"
            "  pip install sentence-transformers"
        )

    model_name = model_name or SIMILARITY_MODEL

    if model_name in _transformer_model_cache:
        print(f"Using cached model: {model_name}")
        return _transformer_model_cache[model_name]

    model = SentenceTransformer(model_name)
    _transformer_model_cache[model_name] = model
    print(f"Loaded similarity model: {model_name}")
    return model


def compute_tfidf_similarity(texts: list) -> np.ndarray:
    """Compute an n×n pairwise cosine-similarity matrix using TF-IDF vectors.

    Fits a :class:`~sklearn.feature_extraction.text.TfidfVectorizer` on all
    ``texts`` (so the vocabulary is derived from the whole corpus, not a
    train split), then computes the full pairwise cosine similarity matrix
    using sklearn's optimised sparse implementation.

    Args:
        texts: List of preprocessed incident description strings.

    Returns:
        Dense NumPy ndarray of shape ``(n, n)`` with float32 scores in
        ``[0.0, 1.0]``.  The matrix is symmetric with ones on the diagonal.
    """
    vectorizer = TfidfVectorizer(min_df=1)
    X = vectorizer.fit_transform(texts)
    matrix = cosine_similarity(X, X)
    return matrix.astype(np.float32)


def compute_transformer_similarity(
    texts: list,
    model_name: str = None,
) -> np.ndarray:
    """Compute an n×n pairwise cosine-similarity matrix using dense embeddings.

    Encodes every text string into a fixed-size embedding vector using a
    sentence-transformer model, then computes all pairwise cosine similarities
    in one batched operation via ``util.cos_sim``.  This backend captures
    semantic relationships (synonyms, paraphrases) that TF-IDF misses.

    Args:
        texts:      List of incident description strings — raw, normalised, or
                    preprocessed text all work; the sentence-transformer
                    tokenizer handles the raw strings directly.
        model_name: sentence-transformers model ID.  Defaults to
                    ``config.SIMILARITY_MODEL``.

    Returns:
        Dense NumPy ndarray of shape ``(n, n)`` with float32 cosine scores in
        ``[-1.0, 1.0]`` (in practice ``[0.0, 1.0]`` for non-negative text
        embeddings).  Symmetric with ones on the diagonal.
    """
    model = get_transformer_model(model_name)
    embeddings = model.encode(texts, show_progress_bar=False)
    sim_tensor = util.cos_sim(embeddings, embeddings)
    return np.array(sim_tensor, dtype=np.float32)


def compute_similarity_matrix(
    texts: list,
    backend: str = None,
) -> tuple:
    """Compute the pairwise similarity matrix using the selected backend.

    Resolves the backend (parameter → config default → automatic fallback),
    delegates to the appropriate compute function, and returns both the matrix
    and the name of the backend actually used.

    Args:
        texts:   List of incident description strings.
        backend: ``"transformer"`` or ``"classical"``.  Defaults to
                 ``config.DEFAULT_BACKEND``.

    Returns:
        Tuple ``(similarity_matrix, backend_used)`` where
        ``similarity_matrix`` is a NumPy ndarray of shape ``(n, n)`` and
        ``backend_used`` is the string name of the backend that ran.

    Side effects:
        Prints a one-line summary including matrix dimensions and backend name.
    """
    resolved = backend or DEFAULT_BACKEND

    if resolved == "transformer" and not SENTENCE_TRANSFORMER_AVAILABLE:
        print(
            "WARNING: transformer backend requested but sentence-transformers "
            "is not installed.  Falling back to classical backend."
        )
        resolved = "classical"

    if resolved == "transformer":
        matrix = compute_transformer_similarity(texts)
    else:
        matrix = compute_tfidf_similarity(texts)

    n = len(texts)
    print(f"Similarity matrix computed ({n}×{n}) using {resolved} backend")
    return matrix, resolved


def find_similar_pairs(
    similarity_matrix: np.ndarray,
    signal_ids: list,
    threshold: float = None,
) -> list:
    """Extract all signal pairs whose similarity score meets the threshold.

    Iterates over the upper triangle of the similarity matrix (i < j) to
    avoid duplicate pairs and self-comparisons, collects every pair at or
    above ``threshold``, and returns them sorted by similarity descending.

    Args:
        similarity_matrix: Square NumPy ndarray of shape ``(n, n)`` as
                           returned by :func:`compute_similarity_matrix`.
        signal_ids:        List of signal ID strings aligned to the matrix
                           rows/columns (e.g. ``["INC-0001-S0", ...]``).
        threshold:         Minimum similarity score to include a pair.
                           Defaults to ``config.SIMILARITY_THRESHOLD``.

    Returns:
        List of dicts sorted by ``similarity`` descending, each containing:

        - ``signal_a``        — signal_id of the first signal.
        - ``signal_b``        — signal_id of the second signal.
        - ``similarity``      — cosine score rounded to 4 decimal places.
        - ``above_threshold`` — always ``True`` (pairs below threshold are
                               excluded; field retained for schema consistency).

    Side effects:
        Prints the count of pairs found and the threshold used.
    """
    threshold = threshold if threshold is not None else SIMILARITY_THRESHOLD
    n = similarity_matrix.shape[0]
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            score = float(similarity_matrix[i, j])
            if score >= threshold:
                pairs.append({
                    "signal_a": signal_ids[i],
                    "signal_b": signal_ids[j],
                    "similarity": round(score, 4),
                    "above_threshold": True,
                })

    pairs.sort(key=lambda p: p["similarity"], reverse=True)
    print(f"Found {len(pairs)} pairs above threshold {threshold}")
    return pairs


def compare_backends(
    texts: list,
    signal_ids: list,
    df: pd.DataFrame = None,
    pairs_to_show: int = 8,
) -> pd.DataFrame:
    """Compare TF-IDF and transformer similarity scores side by side.

    Computes both similarity matrices, then selects a representative sample
    of ``pairs_to_show`` pairs that includes both same-incident and
    cross-category pairs for a meaningful comparison.  When ``df`` is supplied
    the ``incident_group`` and ``category`` columns are used to annotate each
    pair; otherwise those columns are left as ``None``.

    Args:
        texts:         List of incident description strings.
        signal_ids:    List of signal ID strings aligned to ``texts``.
        df:            Optional source DataFrame with ``signal_id``,
                       ``incident_group``, and ``category`` columns for
                       annotating pairs.  If ``None`` annotations are skipped.
        pairs_to_show: Maximum number of pairs to include in the output table.

    Returns:
        :class:`pandas.DataFrame` with columns:
        ``signal_a``, ``signal_b``, ``tfidf_score``, ``transformer_score``,
        ``same_incident``, ``same_category``.

    Side effects:
        Prints the comparison table.
    """
    # Compute both matrices.
    tfidf_matrix = compute_tfidf_similarity(texts)

    if SENTENCE_TRANSFORMER_AVAILABLE:
        transformer_matrix = compute_transformer_similarity(texts)
    else:
        print("sentence-transformers not available — transformer column will be None.")
        transformer_matrix = None

    # Build lookup maps from signal_id → incident_group and category.
    incident_map: dict = {}
    category_map: dict = {}
    if df is not None:
        for _, row in df.iterrows():
            incident_map[row["signal_id"]] = row.get("incident_group", "")
            category_map[row["signal_id"]] = row.get("category", "")

    # Collect all upper-triangle pairs with annotations.
    n = len(signal_ids)
    all_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            sid_a = signal_ids[i]
            sid_b = signal_ids[j]
            tfidf_score = round(float(tfidf_matrix[i, j]), 4)
            trans_score = (
                round(float(transformer_matrix[i, j]), 4)
                if transformer_matrix is not None
                else None
            )
            same_incident = (
                incident_map.get(sid_a) == incident_map.get(sid_b)
                if incident_map
                else None
            )
            same_category = (
                category_map.get(sid_a) == category_map.get(sid_b)
                if category_map
                else None
            )
            all_pairs.append({
                "signal_a": sid_a,
                "signal_b": sid_b,
                "tfidf_score": tfidf_score,
                "transformer_score": trans_score,
                "same_incident": same_incident,
                "same_category": same_category,
            })

    # Select a balanced sample: prefer same-incident pairs first, then
    # cross-category pairs so the comparison table is maximally informative.
    same_inc = [p for p in all_pairs if p["same_incident"]]
    cross_cat = [p for p in all_pairs if not p["same_incident"] and not p["same_category"]]
    same_cat_diff_inc = [
        p for p in all_pairs if not p["same_incident"] and p["same_category"]
    ]

    # Sort each group by transformer score descending (or tfidf if no transformer).
    score_key = "transformer_score" if transformer_matrix is not None else "tfidf_score"
    for group in (same_inc, cross_cat, same_cat_diff_inc):
        group.sort(key=lambda p: p[score_key] or 0.0, reverse=True)

    half = pairs_to_show // 2
    selected = same_inc[:half] + cross_cat[: pairs_to_show - half]
    # Fill remaining slots if either bucket was short.
    if len(selected) < pairs_to_show:
        selected += same_cat_diff_inc[: pairs_to_show - len(selected)]
    selected = selected[:pairs_to_show]

    result_df = pd.DataFrame(selected)

    print("\n=== Backend Comparison (sample pairs) ===")
    with pd.option_context("display.float_format", "{:.4f}".format, "display.max_colwidth", 14):
        print(result_df.to_string(index=False))

    return result_df


def save_similarity_results(pairs: list, path: str = None) -> None:
    """Serialise the similar-pairs list to a JSON file.

    Args:
        pairs: List of pair dicts as returned by :func:`find_similar_pairs`.
        path:  Destination file path.  Defaults to
               ``PROCESSED_PATH/similarity_pairs.json``.

    Side effects:
        Creates parent directories if needed, writes the file, and prints a
        confirmation line.
    """
    if path is None:
        path = os.path.join(PROCESSED_PATH, "similarity_pairs.json")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(pairs, f, indent=2)

    print(f"Saved {len(pairs)} pairs to {path}")


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _project_root = os.path.join(os.path.dirname(__file__), "..")
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    processed_csv = os.path.join(
        _project_root, "data", "processed", "processed_incidents.csv"
    )
    df_all = pd.read_csv(processed_csv)
    texts_all = df_all["preprocessed_text"].tolist()
    ids_all = df_all["signal_id"].tolist()

    # --- Full-corpus backend comparison ------------------------------------
    print("\n=== Backend Comparison ===")
    compare_backends(texts_all, ids_all, df=df_all, pairs_to_show=8)

    # --- Focused 9-signal demo: INC-0001, INC-0002, INC-0003 ---------------
    focus_ids = [
        sid for sid in ids_all
        if any(sid.startswith(inc) for inc in ("INC-0001-", "INC-0002-", "INC-0003-"))
    ]
    focus_mask = df_all["signal_id"].isin(focus_ids)
    focus_df = df_all[focus_mask].reset_index(drop=True)
    focus_texts = focus_df["preprocessed_text"].tolist()
    focus_signal_ids = focus_df["signal_id"].tolist()

    tfidf_mat = compute_tfidf_similarity(focus_texts)
    if SENTENCE_TRANSFORMER_AVAILABLE:
        trans_mat = compute_transformer_similarity(focus_texts)
    else:
        trans_mat = None

    incident_of = {
        row["signal_id"]: row["incident_group"]
        for _, row in focus_df.iterrows()
    }

    same_pairs = []
    cross_pairs = []
    n_focus = len(focus_signal_ids)
    for i in range(n_focus):
        for j in range(i + 1, n_focus):
            sid_a = focus_signal_ids[i]
            sid_b = focus_signal_ids[j]
            t_score = round(float(tfidf_mat[i, j]), 3)
            tr_score = (
                round(float(trans_mat[i, j]), 3) if trans_mat is not None else "N/A"
            )
            entry = (sid_a, sid_b, t_score, tr_score)
            if incident_of.get(sid_a) == incident_of.get(sid_b):
                same_pairs.append(entry)
            else:
                cross_pairs.append(entry)

    print("\n=== Same-incident pairs (should score HIGH) ===")
    for sid_a, sid_b, t, tr in same_pairs:
        print(f"{sid_a} vs {sid_b}  |  TF-IDF: {t:.3f}  |  Transformer: {tr}")

    print("\n=== Cross-incident pairs (should score LOW) ===")
    for sid_a, sid_b, t, tr in cross_pairs[:6]:  # limit output length
        print(f"{sid_a} vs {sid_b}  |  TF-IDF: {t:.3f}  |  Transformer: {tr}")

    # --- Full-corpus similar pairs + save ----------------------------------
    print("\n=== Finding similar pairs across all 150 signals ===")
    if SENTENCE_TRANSFORMER_AVAILABLE:
        full_matrix, backend_used = compute_similarity_matrix(texts_all, backend="transformer")
    else:
        full_matrix, backend_used = compute_similarity_matrix(texts_all, backend="classical")

    pairs = find_similar_pairs(full_matrix, ids_all)
    save_similarity_results(pairs)
