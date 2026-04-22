"""
Tests for src/correlator.py.

Covers the time-window gate, signal lookup construction, confidence score
bounds, and end-to-end group purity against the ground-truth incident_group
labels in the saved correlation results.  The purity test is skipped when
pipeline artefacts are not present.
"""

import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.correlator import (  # noqa: E402
    build_signal_lookup,
    compute_confidence_score,
    signals_within_time_window,
)

# Absolute path to the project root — used to load on-disk artefacts.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CORR_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "correlation_results.json")
_SCORED_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "scored_incidents.csv")

_artefacts_available = os.path.exists(_CORR_PATH) and os.path.exists(_SCORED_PATH)
_requires_artefacts = pytest.mark.skipif(
    not _artefacts_available,
    reason=(
        "Pipeline artefacts not found -- "
        "run `python main.py train && python main.py correlate` first"
    ),
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_time_window_check():
    """Verify signals 15 min apart pass the window gate and 45 min apart fail it."""
    base = pd.Timestamp("2024-01-01 10:00:00")
    close = pd.Timestamp("2024-01-01 10:15:00")  # 15 min — within 30-min window
    far = pd.Timestamp("2024-01-01 10:45:00")    # 45 min — outside 30-min window

    assert signals_within_time_window(base, close, window_minutes=30) is True
    assert signals_within_time_window(base, far, window_minutes=30) is False


def test_build_signal_lookup_keys():
    """Verify every signal_id in the DataFrame appears as a key in the lookup dict."""
    df = pd.DataFrame({
        "signal_id": ["SIG-001", "SIG-002", "SIG-003"],
        "text": ["alpha", "beta", "gamma"],
        "source_type": ["alert", "ticket", "log"],
    })
    lookup = build_signal_lookup(df)
    for sid in df["signal_id"]:
        assert sid in lookup, f"Expected {sid!r} in lookup dict"


def test_confidence_score_between_0_and_1():
    """Verify compute_confidence_score returns a float in the closed interval [0, 1]."""
    df = pd.DataFrame({
        "signal_id": ["S1", "S2"],
        "source_type": ["alert", "ticket"],
        "anomaly_score": [0.5, 0.3],
    })
    group = {
        "signal_ids": ["S1", "S2"],
        "max_anomaly_score": 0.5,
    }
    predictions = {
        "S1": {"confidence": 0.8},
        "S2": {"confidence": 0.6},
    }
    score = compute_confidence_score(group, df, predictions)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0, f"Confidence {score:.4f} is outside [0, 1]"


@_requires_artefacts
def test_perfect_groups_have_no_mixing():
    """Verify every correlated group contains signals from a single incident_group."""
    with open(_CORR_PATH) as f:
        groups = json.load(f)

    df = pd.read_csv(_SCORED_PATH)
    true_group_of = dict(zip(df["signal_id"], df["incident_group"]))

    fragmented = []
    for g in groups:
        true_incs = {true_group_of.get(sid) for sid in g["signal_ids"]}
        # A perfect group maps to exactly one ground-truth incident.
        if len(true_incs) > 1:
            fragmented.append(g["group_id"])

    assert len(fragmented) == 0, (
        f"{len(fragmented)} fragmented group(s) found: {fragmented}"
    )
