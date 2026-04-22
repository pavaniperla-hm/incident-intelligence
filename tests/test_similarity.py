"""
Tests for src/similarity.py.

Covers matrix shape, self-similarity, low scores for unrelated texts, and
high scores for semantically similar auth-failure descriptions using the
transformer backend.  The transformer test is skipped when sentence-transformers
is not installed.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.similarity import (  # noqa: E402
    SENTENCE_TRANSFORMER_AVAILABLE,
    compute_similarity_matrix,
    compute_tfidf_similarity,
)

_requires_transformer = pytest.mark.skipif(
    not SENTENCE_TRANSFORMER_AVAILABLE,
    reason="sentence-transformers not installed",
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_similarity_matrix_shape():
    """Verify compute_similarity_matrix returns a square (n x n) matrix."""
    texts = ["text alpha", "text beta", "text gamma", "text delta"]
    matrix, _ = compute_similarity_matrix(texts, backend="classical")
    assert matrix.shape == (len(texts), len(texts))


def test_identical_texts_score_1():
    """Verify cosine similarity of a text against itself is approximately 1.0."""
    texts = [
        "authentication service login failed token expired",
        "authentication service login failed token expired",
    ]
    # Both classical and transformer are tested; classical is always available.
    matrix = compute_tfidf_similarity(texts)
    # Off-diagonal [0,1] is the score between two identical texts.
    assert abs(matrix[0, 1] - 1.0) < 1e-4, (
        f"Expected ~1.0 for identical texts, got {matrix[0, 1]:.6f}"
    )


def test_unrelated_texts_score_low():
    """Verify semantically unrelated texts produce TF-IDF cosine similarity below 0.4."""
    texts = [
        "authentication login failed credentials invalid",
        "kubernetes deployment pipeline pod crash rollback",
    ]
    matrix = compute_tfidf_similarity(texts)
    score = float(matrix[0, 1])
    assert score < 0.4, (
        f"Expected score < 0.4 for unrelated texts, got {score:.4f}"
    )


@_requires_transformer
def test_similar_texts_score_high():
    """Verify semantically similar auth-failure texts score above 0.5 with transformer backend."""
    texts = [
        "token validation failed login error",
        "jwt expired authentication error",
    ]
    matrix, backend = compute_similarity_matrix(texts, backend="transformer")
    assert backend == "transformer"
    score = float(matrix[0, 1])
    assert score > 0.5, (
        f"Expected transformer similarity > 0.5 for related auth-failure texts, "
        f"got {score:.4f}"
    )
