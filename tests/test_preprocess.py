"""
Tests for src/normaliser.py and src/preprocess.py.

Covers the 11-step regex normalisation pipeline (normalise_text) and the
NLP preprocessing pipeline (preprocess_text): token preservation, stopword
handling, and lemmatisation.
"""

import os
import sys

import pytest

# Ensure project root is on sys.path so src.* imports work when pytest is
# invoked from any working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.normaliser import normalise_text  # noqa: E402
from src.preprocess import preprocess_text, PRESERVE_TOKENS  # noqa: E402


def test_clean_text_removes_log_prefix():
    """Verify that a CRITICAL: log-level prefix is stripped by normalise_text."""
    result = normalise_text("CRITICAL: some important text here", "alert")
    # After lowercase + log-level strip, "critical:" should be gone.
    assert "critical" not in result
    assert "important" in result


def test_pctval_replacement():
    """Verify percentage values are replaced with the PCTVAL semantic token."""
    result = normalise_text("error rate dropped 97%", "alert")
    assert "PCTVAL" in result
    # The raw number must not survive as a bare digit string.
    assert "97" not in result


def test_preserve_tokens_survive_preprocessing():
    """Verify PCTVAL, TIMEVAL, IPADDR, and NUMVAL are kept verbatim by preprocess_text."""
    # Feed a string that already contains normaliser output tokens.
    text = "host IPADDR failed at TIMEVAL cpu PCTVAL connections NUMVAL"
    result = preprocess_text(text)
    for token in ("IPADDR", "TIMEVAL", "PCTVAL", "NUMVAL"):
        assert token in result, f"Expected preserve-token {token!r} in output: {result!r}"


def test_incident_critical_words_kept():
    """Verify 'failed', 'error', and 'unauthorized' survive stopword filtering."""
    result = preprocess_text("service failed with error unauthorized access")
    assert "failed" in result
    assert "error" in result
    assert "unauthorized" in result
