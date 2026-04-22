"""
Tests for src/classifier.py.

Covers artefact loading, predict() output schema, confidence range, and
correct classification of a canonical Authentication Failure signal.
Requires trained artefacts in data/processed/ — run `python main.py train`
or `python src/classifier.py` first.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classifier import load_classifier, predict  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level fixture: load once, reuse across all tests in this file.
# ---------------------------------------------------------------------------

try:
    _CLF, _LE, _VEC = load_classifier(backend="classical")
    _ARTEFACTS_AVAILABLE = True
except FileNotFoundError:
    _ARTEFACTS_AVAILABLE = False

_requires_artefacts = pytest.mark.skipif(
    not _ARTEFACTS_AVAILABLE,
    reason="Classical classifier artefacts not found — run `python main.py train` first",
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@_requires_artefacts
def test_classical_backend_loads():
    """Verify load_classifier returns non-None objects for the classical backend."""
    clf, le, vec = load_classifier(backend="classical")
    assert clf is not None
    assert le is not None
    assert vec is not None


@_requires_artefacts
def test_predict_returns_required_keys():
    """Verify predict() output dict contains all four required keys."""
    result = predict(
        "authentication service is down",
        _CLF, _VEC, _LE,
        backend="classical",
    )
    for key in ("predicted_category", "confidence", "all_probabilities", "backend_used"):
        assert key in result, f"Missing key {key!r} in predict() output"


@_requires_artefacts
def test_predict_confidence_between_0_and_1():
    """Verify predict() confidence is a float in the closed interval [0, 1]."""
    result = predict(
        "memory usage spike high cpu timeout response slow",
        _CLF, _VEC, _LE,
        backend="classical",
    )
    conf = result["confidence"]
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 1.0


@_requires_artefacts
def test_auth_failure_classified_correctly():
    """Verify a canonical auth failure text is predicted as Authentication Failure."""
    result = predict(
        "users cannot log in authentication failed token invalid",
        _CLF, _VEC, _LE,
        backend="classical",
    )
    assert result["predicted_category"] == "Authentication Failure", (
        f"Expected 'Authentication Failure', got {result['predicted_category']!r} "
        f"(confidence {result['confidence']:.2f})"
    )
