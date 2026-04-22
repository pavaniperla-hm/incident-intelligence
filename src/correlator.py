"""
correlator.py — Temporal and semantic incident correlation.

Purpose:
    Groups related signals into correlated incident clusters by combining
    semantic similarity (from similarity.py) with temporal proximity and
    category agreement.  Uses a Union-Find (disjoint set) algorithm for
    efficient incremental merging — each pair that passes all three gates
    (similarity threshold, time window, same predicted category) is merged
    into a common group.  Singleton groups are discarded; only multi-signal
    correlated groups are returned.

Pipeline position: After similarity and anomaly; feeds into scorer.

Inputs:
    - scored_incidents.csv — DataFrame with all preprocess + anomaly columns
    - similarity_pairs.json — list of (signal_a, signal_b, similarity) dicts
    - predictions dict — {signal_id: predict() output} from classifier
    - config.SIMILARITY_THRESHOLD         — minimum score to attempt a merge
    - config.CORRELATION_TIME_WINDOW_MINUTES — maximum time gap for merging
    - config.CONFIDENCE_WEIGHTS           — weights for composite score formula
    - config.SOURCE_WEIGHTS               — used for max source weight lookup

Outputs:
    - List of group dicts with keys: group_id, signal_ids, predicted_category,
      signal_count, source_types, time_span_minutes, avg_similarity,
      max_anomaly_score, confidence_score, signals_detail
    - Saved to PROCESSED_PATH/correlation_results.json
"""

import importlib
import json
import os
import sys
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path fix — allows running this file directly as a script from any directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (  # noqa: E402
    CONFIDENCE_WEIGHTS,
    CORRELATION_TIME_WINDOW_MINUTES,
    PROCESSED_PATH,
    SIMILARITY_THRESHOLD,
    SOURCE_WEIGHTS,
)


# ---------------------------------------------------------------------------
# Internal Union-Find implementation
# ---------------------------------------------------------------------------

class _UnionFind:
    """Lightweight Union-Find (disjoint set union) with path compression."""

    def __init__(self, elements):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        # Union by rank.
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def groups(self):
        """Return a dict mapping root → set of members."""
        buckets: dict = {}
        for e in self.parent:
            root = self.find(e)
            buckets.setdefault(root, set()).add(e)
        return buckets


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def signals_within_time_window(
    ts_a: pd.Timestamp,
    ts_b: pd.Timestamp,
    window_minutes: int = None,
) -> bool:
    """Check whether two signal timestamps fall within the correlation window.

    Args:
        ts_a:           Timestamp of the first signal.
        ts_b:           Timestamp of the second signal.
        window_minutes: Maximum allowed gap in minutes.  Defaults to
                        ``config.CORRELATION_TIME_WINDOW_MINUTES``.

    Returns:
        ``True`` if ``|ts_a - ts_b| ≤ window_minutes``, ``False`` otherwise.
    """
    window = window_minutes if window_minutes is not None else CORRELATION_TIME_WINDOW_MINUTES
    diff = abs((pd.Timestamp(ts_a) - pd.Timestamp(ts_b)).total_seconds()) / 60.0
    return diff <= window


def build_signal_lookup(df: pd.DataFrame) -> dict:
    """Build an O(1) lookup dict from signal_id to all column values.

    Converts each DataFrame row into a plain dict so that individual signal
    properties (timestamp, source_type, anomaly_score, etc.) can be retrieved
    in constant time during the correlation pass without repeated DataFrame
    indexing.

    Args:
        df: Scored signals DataFrame with a ``signal_id`` column.

    Returns:
        Dict mapping ``signal_id`` → dict of ``{column: value}`` for every
        column in ``df``.
    """
    return {row["signal_id"]: row.to_dict() for _, row in df.iterrows()}


def compute_confidence_score(
    group: dict,
    df: pd.DataFrame,
    predictions: dict,
) -> float:
    """Compute a composite confidence score for a correlated group.

    Combines three evidence signals using the weights from
    ``config.CONFIDENCE_WEIGHTS``:

    - **source_weight**   — reliability of the best source in the group
      (alert > ticket > log per ``config.SOURCE_WEIGHTS``).
    - **classifier_conf** — the classifier's highest probability score for
      any signal in the group.
    - **anomaly_boost**   — a multiplier derived from the group's peak anomaly
      score; anomalous incidents receive a higher composite confidence.

    Formula::

        anomaly_boost  = 1.0 + group['max_anomaly_score']
        confidence     = (
            max_source_weight   * CONFIDENCE_WEIGHTS['source_weight']
          + max_classifier_conf * CONFIDENCE_WEIGHTS['classifier_conf']
          + min(anomaly_boost, 2.0) * CONFIDENCE_WEIGHTS['anomaly_boost'] * 0.5
        )
        confidence = min(round(confidence, 4), 1.0)

    Args:
        group:       Group dict as partially built by :func:`correlate_signals`
                     (must already contain ``signal_ids`` and
                     ``max_anomaly_score``).
        df:          Scored signals DataFrame with ``source_type`` column.
        predictions: Dict mapping ``signal_id`` → ``predict()`` output dict
                     from classifier.

    Returns:
        Composite confidence score clipped to ``[0.0, 1.0]``.
    """
    signal_ids = group["signal_ids"]

    # Max source weight across signals in group.
    source_weights = [
        SOURCE_WEIGHTS.get(
            df.loc[df["signal_id"] == sid, "source_type"].values[0], 0.0
        )
        for sid in signal_ids
        if sid in df["signal_id"].values
    ]
    max_source_weight = max(source_weights) if source_weights else 0.0

    # Max classifier confidence across predictions for signals in group.
    confs = [
        predictions[sid]["confidence"]
        for sid in signal_ids
        if sid in predictions
    ]
    max_classifier_conf = max(confs) if confs else 0.0

    anomaly_boost = 1.0 + group["max_anomaly_score"]

    confidence = (
        max_source_weight * CONFIDENCE_WEIGHTS["source_weight"]
        + max_classifier_conf * CONFIDENCE_WEIGHTS["classifier_conf"]
        + min(anomaly_boost, 2.0) * CONFIDENCE_WEIGHTS["anomaly_boost"] * 0.5
    )
    return min(round(float(confidence), 4), 1.0)


def correlate_signals(
    df: pd.DataFrame,
    similarity_pairs: list,
    predictions: dict,
) -> list:
    """Group related signals into correlated incident clusters.

    Uses a Union-Find algorithm to merge signals incrementally.  For each
    candidate pair from ``similarity_pairs``, three gates must all pass before
    the pair is merged:

    1. **Similarity** — score ≥ ``config.SIMILARITY_THRESHOLD``.
    2. **Category agreement** — both signals share the same predicted category.
    3. **Temporal proximity** — timestamps within
       ``config.CORRELATION_TIME_WINDOW_MINUTES``.

    After all pairs are processed, singleton groups (signals with no
    correlated partner) are discarded.  Each surviving group is enriched with
    metadata and a composite confidence score.

    Args:
        df:               Scored signals DataFrame containing ``signal_id``,
                          ``timestamp``, ``source_type``, ``text``,
                          ``anomaly_score``, and ``incident_group`` columns.
        similarity_pairs: List of pair dicts as returned by
                          ``similarity.find_similar_pairs``, each with keys
                          ``signal_a``, ``signal_b``, ``similarity``.
        predictions:      Dict mapping ``signal_id`` → ``predict()`` output
                          dict from ``classifier.predict``.

    Returns:
        List of group dicts sorted by ``confidence_score`` descending, each
        containing:

        - ``group_id``           — sequential ID string, e.g. ``"CORR-001"``.
        - ``signal_ids``         — list of signal_id strings in the group.
        - ``predicted_category`` — modal predicted category across the group.
        - ``signal_count``       — number of signals.
        - ``source_types``       — sorted list of unique source types.
        - ``time_span_minutes``  — float, max − min timestamp gap in minutes.
        - ``avg_similarity``     — mean similarity of within-group pairs.
        - ``max_anomaly_score``  — highest anomaly_score in the group.
        - ``confidence_score``   — composite score from
                                   :func:`compute_confidence_score`.
        - ``signals_detail``     — list of ``{signal_id, source_type,
                                   text_snippet}`` dicts (first 80 chars).
    """
    signal_ids = df["signal_id"].tolist()
    lookup = build_signal_lookup(df)
    uf = _UnionFind(signal_ids)

    # Track which pairs were actually merged (for avg_similarity calculation).
    merged_pairs: list = []

    for pair in similarity_pairs:
        sid_a = pair["signal_a"]
        sid_b = pair["signal_b"]
        score = pair["similarity"]

        # Gate 1 — similarity threshold (pairs list may include sub-threshold
        # entries if caller passed a custom list; we re-check for safety).
        if score < SIMILARITY_THRESHOLD:
            continue

        # Gate 2 — category agreement.
        cat_a = predictions.get(sid_a, {}).get("predicted_category")
        cat_b = predictions.get(sid_b, {}).get("predicted_category")
        if cat_a is None or cat_b is None or cat_a != cat_b:
            continue

        # Gate 3 — temporal proximity.
        ts_a = pd.Timestamp(lookup[sid_a]["timestamp"])
        ts_b = pd.Timestamp(lookup[sid_b]["timestamp"])
        if not signals_within_time_window(ts_a, ts_b):
            continue

        uf.union(sid_a, sid_b)
        merged_pairs.append((sid_a, sid_b, score))

    # Collect non-singleton groups.
    raw_groups = {
        root: members
        for root, members in uf.groups().items()
        if len(members) > 1
    }

    # Build within-group similarity lookup for avg_similarity.
    pair_score_map = {
        (p[0], p[1]): p[2] for p in merged_pairs
    }
    pair_score_map.update({(p[1], p[0]): p[2] for p in merged_pairs})

    results = []
    for rank, (_, members) in enumerate(
        sorted(raw_groups.items(), key=lambda x: len(x[1]), reverse=True),
        start=1,
    ):
        member_list = sorted(members)

        # Timestamps.
        timestamps = [pd.Timestamp(lookup[sid]["timestamp"]) for sid in member_list]
        min_ts, max_ts = min(timestamps), max(timestamps)
        time_span = (max_ts - min_ts).total_seconds() / 60.0

        # Most common predicted category.
        cats = [
            predictions.get(sid, {}).get("predicted_category", "Unknown")
            for sid in member_list
        ]
        predicted_category = Counter(cats).most_common(1)[0][0]

        # Unique source types.
        source_types = sorted(
            {lookup[sid]["source_type"] for sid in member_list}
        )

        # Max anomaly score.
        anomaly_scores = [
            float(lookup[sid].get("anomaly_score", 0.0)) for sid in member_list
        ]
        max_anomaly = max(anomaly_scores)

        # Average similarity of within-group pairs.
        within_scores = []
        for i, sa in enumerate(member_list):
            for sb in member_list[i + 1 :]:
                s = pair_score_map.get((sa, sb)) or pair_score_map.get((sb, sa))
                if s is not None:
                    within_scores.append(s)
        avg_sim = round(float(np.mean(within_scores)), 4) if within_scores else 0.0

        # Signals detail.
        signals_detail = [
            {
                "signal_id": sid,
                "source_type": lookup[sid]["source_type"],
                "text_snippet": str(lookup[sid].get("text", ""))[:80],
            }
            for sid in member_list
        ]

        group: dict = {
            "group_id": f"CORR-{rank:03d}",
            "signal_ids": member_list,
            "predicted_category": predicted_category,
            "signal_count": len(member_list),
            "source_types": source_types,
            "time_span_minutes": round(time_span, 2),
            "avg_similarity": avg_sim,
            "max_anomaly_score": round(max_anomaly, 4),
            "confidence_score": 0.0,  # placeholder — computed next
            "signals_detail": signals_detail,
        }
        group["confidence_score"] = compute_confidence_score(group, df, predictions)
        results.append(group)

    # Sort by confidence descending; re-assign group_ids in final order.
    results.sort(key=lambda g: g["confidence_score"], reverse=True)
    for i, g in enumerate(results, start=1):
        g["group_id"] = f"CORR-{i:03d}"

    return results


def evaluate_correlation(
    correlated_groups: list,
    df: pd.DataFrame,
) -> dict:
    """Evaluate correlated groups against ground-truth ``incident_group`` labels.

    For each correlated group, finds which true incident_groups its signals
    belong to.  A **perfect group** is one where all signals share the same
    true incident_group.  A **fragmented group** mixes signals from different
    true incidents.

    Precision and recall are computed at the signal-pair level:

    - **Precision** — of all pairs grouped together, what fraction truly
      co-belong (same ``incident_group``).
    - **Recall** — of all true co-belonging pairs in the dataset, what
      fraction were recovered by correlation.

    Args:
        correlated_groups: List of group dicts from :func:`correlate_signals`.
        df:                Scored signals DataFrame with ``signal_id`` and
                           ``incident_group`` columns.

    Returns:
        Dict with keys: ``total_groups``, ``perfect_groups``,
        ``fragmented_groups``, ``signals_in_groups``, ``precision``,
        ``recall``, ``f1``.

    Side effects:
        Prints a formatted evaluation report.
    """
    true_group_of = dict(zip(df["signal_id"], df["incident_group"]))

    perfect = 0
    fragmented = 0
    tp_pairs = 0  # pairs correctly grouped (same true incident)
    fp_pairs = 0  # pairs incorrectly grouped (different true incidents)
    signals_in_groups = 0

    for g in correlated_groups:
        sids = g["signal_ids"]
        signals_in_groups += len(sids)
        true_incs = {true_group_of.get(sid) for sid in sids}

        if len(true_incs) == 1:
            perfect += 1
        else:
            fragmented += 1

        # Count pairwise TP / FP.
        for i, sa in enumerate(sids):
            for sb in sids[i + 1 :]:
                if true_group_of.get(sa) == true_group_of.get(sb):
                    tp_pairs += 1
                else:
                    fp_pairs += 1

    # True co-belonging pairs in the full dataset.
    all_sids = df["signal_id"].tolist()
    total_true_pairs = 0
    true_group_list = [true_group_of.get(s) for s in all_sids]
    group_counts = Counter(true_group_list)
    for cnt in group_counts.values():
        total_true_pairs += cnt * (cnt - 1) // 2

    precision = tp_pairs / (tp_pairs + fp_pairs) if (tp_pairs + fp_pairs) > 0 else 0.0
    recall = tp_pairs / total_true_pairs if total_true_pairs > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print("\n=== Correlation Evaluation ===")
    print(f"  Total correlated groups  : {len(correlated_groups)}")
    print(f"  Perfect groups           : {perfect}")
    print(f"  Fragmented groups        : {fragmented}")
    print(f"  Signals in groups        : {signals_in_groups}")
    print(f"  Pair precision           : {precision:.4f}")
    print(f"  Pair recall              : {recall:.4f}")
    print(f"  F1                       : {f1:.4f}")

    return {
        "total_groups": len(correlated_groups),
        "perfect_groups": perfect,
        "fragmented_groups": fragmented,
        "signals_in_groups": signals_in_groups,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def save_correlation_results(groups: list, path: str = None) -> None:
    """Serialise correlated groups to a JSON file.

    Converts any ``pandas.Timestamp`` or ``datetime`` values to ISO-8601
    strings before serialisation so the output is pure JSON-compatible.

    Args:
        groups: List of group dicts from :func:`correlate_signals`.
        path:   Destination file path.  Defaults to
                ``PROCESSED_PATH/correlation_results.json``.

    Side effects:
        Creates parent directories if needed, writes the file, and prints a
        confirmation line.
    """
    if path is None:
        path = os.path.join(PROCESSED_PATH, "correlation_results.json")

    def _default(obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(groups, f, indent=2, default=_default)

    print(f"Saved {len(groups)} correlated groups to {path}")


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _project_root = os.path.join(os.path.dirname(__file__), "..")
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    from src.classifier import load_classifier, predict  # noqa: E402
    from src.similarity import compute_similarity_matrix, find_similar_pairs  # noqa: E402

    # --- Load data ---------------------------------------------------------
    scored_csv = os.path.join(
        _project_root, "data", "processed", "scored_incidents.csv"
    )
    df = pd.read_csv(scored_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # --- Compute similarity pairs on the fly -------------------------------
    import config
    importlib.reload(config)
    sim_matrix, _ = compute_similarity_matrix(df["preprocessed_text"].tolist(), backend="transformer")
    similarity_pairs = find_similar_pairs(sim_matrix, df["signal_id"].tolist(), threshold=config.SIMILARITY_THRESHOLD)

    # --- Load classifier and generate predictions --------------------------
    clf, le, vectorizer = load_classifier(backend="classical")

    print("Generating predictions for all signals...")
    predictions: dict = {}
    for _, row in df.iterrows():
        sid = row["signal_id"]
        text = str(row.get("preprocessed_text", row.get("text", "")))
        predictions[sid] = predict(text, clf, vectorizer, le, backend="classical")

    # --- Correlate ---------------------------------------------------------
    groups = correlate_signals(df, similarity_pairs, predictions)

    # --- Summary -----------------------------------------------------------
    total_signals = len(df)
    total_groups = len(groups)
    signals_in_groups = sum(g["signal_count"] for g in groups)
    avg_per_group = signals_in_groups / total_groups if total_groups else 0.0
    avg_conf = (
        sum(g["confidence_score"] for g in groups) / total_groups
        if total_groups else 0.0
    )

    print("\n=== Correlation Results ===")
    print(f"Total signals:          {total_signals}")
    print(f"Correlated groups:      {total_groups}")
    print(f"Signals in groups:      {signals_in_groups}")
    print(f"Avg signals per group:  {avg_per_group:.1f}")
    print(f"Avg confidence score:   {avg_conf:.4f}")

    # --- Top 3 groups detail -----------------------------------------------
    print("\n=== Top 3 Highest-Confidence Groups ===")
    for g in groups[:3]:
        print(f"\nGroup {g['group_id']}  |  {g['predicted_category']}  |  confidence: {g['confidence_score']:.2f}")
        print(f"Sources: {', '.join(g['source_types'])}")
        print(f"Time span: {g['time_span_minutes']:.1f} minutes")
        print("Signals:")
        for sig in g["signals_detail"]:
            print(f"  {sig['signal_id']} [{sig['source_type']}]  \"{sig['text_snippet'][:80]}...\"")

    # --- Evaluate ----------------------------------------------------------
    evaluate_correlation(groups, df)

    # --- Save --------------------------------------------------------------
    save_correlation_results(groups)
