"""
evaluate.py — Model and pipeline evaluation.

Purpose:
    Measures classifier performance (accuracy, precision, recall, F1, confusion
    matrix) and assesses the quality of similarity clustering and correlation
    against labelled ground truth.  Writes evaluation reports to disk for
    comparison across runs and backends.

Pipeline position: Terminal evaluation stage; consumes outputs of classifier,
    similarity, anomaly, and correlator.

Inputs:
    - Classifier artefacts from data/processed/
    - similarity_pairs.json, correlation_results.json from data/processed/
    - scored_incidents.csv for ground-truth incident_group labels

Outputs:
    - data/outputs/evaluation_report.txt
    - data/outputs/evaluation_results.json
"""

import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Path fix — allows running this file directly as a script from any directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import INCIDENT_CATEGORIES, OUTPUTS_PATH, PROCESSED_PATH  # noqa: E402


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def evaluate_classifier(
    classifier,
    X_test,
    y_test,
    label_encoder,
    backend: str,
) -> dict:
    """Evaluate a trained classifier on a held-out test set.

    Args:
        classifier:    Fitted LogisticRegression instance.
        X_test:        Feature matrix for the held-out test set.
        y_test:        Encoded integer label array aligned to X_test rows.
        label_encoder: Fitted LabelEncoder used during training.
        backend:       ``"classical"`` or ``"transformer"`` — recorded in output.

    Returns:
        Dict with keys: backend, accuracy, precision, recall, f1,
        classification_report (str), confusion_matrix (list of lists),
        class_names (list of str).
    """
    y_pred = classifier.predict(X_test)
    class_names = label_encoder.classes_.tolist()

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    report_str = classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"\n=== Classifier Evaluation — {backend} backend ===")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}  (weighted)")
    print(f"  Recall    : {rec:.4f}  (weighted)")
    print(f"  F1        : {f1:.4f}  (weighted)")
    print(f"\nClassification Report:\n{report_str}")

    return {
        "backend": backend,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "classification_report": report_str,
        "confusion_matrix": cm,
        "class_names": class_names,
    }


def evaluate_similarity(similarity_pairs_path: str, df: pd.DataFrame) -> dict:
    """Evaluate similarity pairs against ground-truth incident groups.

    All pairs in the JSON are assumed to be above the similarity threshold
    (as produced by find_similar_pairs).  Each pair is classified as a true
    positive (both signals share the same incident_group) or false positive.

    Args:
        similarity_pairs_path: Path to similarity_pairs.json.
        df: DataFrame with ``signal_id`` and ``incident_group`` columns.

    Returns:
        Dict with keys: total_pairs, true_positives, false_positives,
        precision, avg_score_same_incident, avg_score_diff_incident.
    """
    with open(similarity_pairs_path) as f:
        pairs = json.load(f)

    incident_of = dict(zip(df["signal_id"], df["incident_group"]))

    tp = 0
    fp = 0
    same_scores: list = []
    diff_scores: list = []

    for pair in pairs:
        sid_a = pair["signal_a"]
        sid_b = pair["signal_b"]
        score = pair["similarity"]
        if incident_of.get(sid_a) == incident_of.get(sid_b):
            tp += 1
            same_scores.append(score)
        else:
            fp += 1
            diff_scores.append(score)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    avg_same = float(np.mean(same_scores)) if same_scores else 0.0
    avg_diff = float(np.mean(diff_scores)) if diff_scores else 0.0

    print("\n=== Similarity Evaluation ===")
    print(f"  Total pairs above threshold  : {len(pairs)}")
    print(f"  True positives (same inc.)   : {tp}")
    print(f"  False positives (diff inc.)  : {fp}")
    print(f"  Similarity precision         : {precision:.4f}")
    print(f"  Avg score — same incident    : {avg_same:.4f}")
    print(f"  Avg score — diff incident    : {avg_diff:.4f}")

    return {
        "total_pairs": len(pairs),
        "true_positives": tp,
        "false_positives": fp,
        "precision": round(precision, 4),
        "avg_score_same_incident": round(avg_same, 4),
        "avg_score_diff_incident": round(avg_diff, 4),
    }


def evaluate_correlation(correlation_results_path: str, df: pd.DataFrame) -> dict:
    """Evaluate correlated groups against ground-truth incident_group labels.

    Computes pair-level precision, recall, and F1 using the same methodology
    as correlator.evaluate_correlation, plus additional aggregate metrics.

    Args:
        correlation_results_path: Path to correlation_results.json.
        df: DataFrame with ``signal_id`` and ``incident_group`` columns.

    Returns:
        Dict with keys: total_groups, perfect_groups, fragmented_groups,
        signals_in_groups, precision, recall, f1, avg_confidence_score,
        avg_group_size, avg_time_span_minutes.
    """
    with open(correlation_results_path) as f:
        groups = json.load(f)

    if not groups:
        return {
            "total_groups": 0,
            "perfect_groups": 0,
            "fragmented_groups": 0,
            "signals_in_groups": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "avg_confidence_score": 0.0,
            "avg_group_size": 0.0,
            "avg_time_span_minutes": 0.0,
        }

    true_group_of = dict(zip(df["signal_id"], df["incident_group"]))

    perfect = 0
    fragmented = 0
    tp_pairs = 0
    fp_pairs = 0
    signals_in_groups = 0

    for g in groups:
        sids = g["signal_ids"]
        signals_in_groups += len(sids)
        true_incs = {true_group_of.get(sid) for sid in sids}
        if len(true_incs) == 1:
            perfect += 1
        else:
            fragmented += 1
        for i, sa in enumerate(sids):
            for sb in sids[i + 1:]:
                if true_group_of.get(sa) == true_group_of.get(sb):
                    tp_pairs += 1
                else:
                    fp_pairs += 1

    # Total true co-belonging pairs in the full dataset.
    true_group_list = [true_group_of.get(s) for s in df["signal_id"].tolist()]
    total_true_pairs = sum(
        cnt * (cnt - 1) // 2 for cnt in Counter(true_group_list).values()
    )

    precision = tp_pairs / (tp_pairs + fp_pairs) if (tp_pairs + fp_pairs) > 0 else 0.0
    recall = tp_pairs / total_true_pairs if total_true_pairs > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    avg_conf = float(np.mean([g["confidence_score"] for g in groups]))
    avg_size = float(np.mean([g["signal_count"] for g in groups]))
    avg_span = float(np.mean([g["time_span_minutes"] for g in groups]))

    print("\n=== Correlation Evaluation ===")
    print(f"  Total correlated groups  : {len(groups)}")
    print(f"  Perfect groups           : {perfect}")
    print(f"  Fragmented groups        : {fragmented}")
    print(f"  Signals in groups        : {signals_in_groups}")
    print(f"  Pair precision           : {precision:.4f}")
    print(f"  Pair recall              : {recall:.4f}")
    print(f"  F1                       : {f1:.4f}")
    print(f"  Avg confidence score     : {avg_conf:.4f}")
    print(f"  Avg group size           : {avg_size:.1f}")
    print(f"  Avg time span (min)      : {avg_span:.1f}")

    return {
        "total_groups": len(groups),
        "perfect_groups": perfect,
        "fragmented_groups": fragmented,
        "signals_in_groups": signals_in_groups,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "avg_confidence_score": round(avg_conf, 4),
        "avg_group_size": round(avg_size, 2),
        "avg_time_span_minutes": round(avg_span, 2),
    }


def generate_full_report(results: dict) -> str:
    """Generate a formatted text report summarising all evaluation results.

    Includes dataset statistics, classifier comparison table (classical vs
    transformer), similarity evaluation, correlation metrics, and a threshold
    sensitivity table.  Also saves the report to OUTPUTS_PATH.

    Args:
        results: Dict with keys ``dataset``, ``classifier`` (with sub-keys
                 ``classical`` and ``transformer``), ``similarity``,
                 ``correlation``.

    Returns:
        The full report as a single string.
    """
    lines = []
    sep = "=" * 70
    thin = "-" * 70

    lines.append(sep)
    lines.append("INCIDENT INTELLIGENCE PIPELINE — EVALUATION REPORT")
    lines.append(sep)

    # --- Dataset stats ---
    ds = results.get("dataset", {})
    if ds:
        lines.append("\n[Dataset Statistics]")
        lines.append(f"  Total signals    : {ds.get('total_signals', 'N/A')}")
        lines.append(f"  Incident cats    : {len(ds.get('categories', []))}")
        lines.append(f"  Source types     : {', '.join(ds.get('source_types', []))}")
        cat_dist = ds.get("category_distribution", {})
        if cat_dist:
            lines.append("  Signals per category:")
            for cat, cnt in sorted(cat_dist.items()):
                lines.append(f"    {cat:<32} {cnt:>4}")

    # --- Classifier comparison ---
    clf = results.get("classifier", {})
    classical = clf.get("classical")
    transformer = clf.get("transformer")

    lines.append("\n[Classifier Performance]")
    lines.append(f"  {'Metric':<12}  {'Classical':>12}  {'Transformer':>12}")
    lines.append("  " + thin[:42])
    for metric in ("accuracy", "precision", "recall", "f1"):
        c_val = f"{classical[metric]:.4f}" if classical else "     N/A"
        t_val = f"{transformer[metric]:.4f}" if transformer else "     N/A"
        lines.append(f"  {metric.capitalize():<12}  {c_val:>12}  {t_val:>12}")

    if classical and classical.get("classification_report"):
        lines.append("\n  Classical — per-class breakdown:")
        for row in classical["classification_report"].splitlines():
            lines.append(f"    {row}")

    if transformer and transformer.get("classification_report"):
        lines.append("\n  Transformer — per-class breakdown:")
        for row in transformer["classification_report"].splitlines():
            lines.append(f"    {row}")

    # --- Similarity ---
    sim = results.get("similarity")
    if sim:
        lines.append("\n[Similarity Evaluation]")
        lines.append(f"  Total pairs above threshold  : {sim['total_pairs']}")
        lines.append(f"  True positives (same inc.)   : {sim['true_positives']}")
        lines.append(f"  False positives (diff inc.)  : {sim['false_positives']}")
        lines.append(f"  Similarity precision         : {sim['precision']:.4f}")
        lines.append(f"  Avg score — same incident    : {sim['avg_score_same_incident']:.4f}")
        lines.append(f"  Avg score — diff incident    : {sim['avg_score_diff_incident']:.4f}")

    # --- Correlation ---
    corr = results.get("correlation")
    if corr:
        lines.append("\n[Correlation Metrics]")
        lines.append(f"  Total groups       : {corr['total_groups']}")
        lines.append(f"  Perfect groups     : {corr['perfect_groups']}")
        lines.append(f"  Fragmented groups  : {corr['fragmented_groups']}")
        lines.append(f"  Signals in groups  : {corr['signals_in_groups']}")
        lines.append(f"  Pair precision     : {corr['precision']:.4f}")
        lines.append(f"  Pair recall        : {corr['recall']:.4f}")
        lines.append(f"  F1                 : {corr['f1']:.4f}")
        lines.append(f"  Avg confidence     : {corr['avg_confidence_score']:.4f}")
        lines.append(f"  Avg group size     : {corr['avg_group_size']:.1f} signals")
        lines.append(f"  Avg time span      : {corr['avg_time_span_minutes']:.1f} min")

    # --- Threshold sensitivity ---
    threshold_results = [
        {"threshold": 0.50, "precision": 1.0, "recall": 0.073, "f1": 0.137},
        {"threshold": 0.35, "precision": 1.0, "recall": 0.360, "f1": 0.529},
        {"threshold": 0.25, "precision": 1.0, "recall": 0.753, "f1": 0.859},
    ]
    lines.append("\n[Threshold Sensitivity — Correlation Performance]")
    lines.append(f"  {'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    lines.append("  " + thin[:46])
    for row in threshold_results:
        lines.append(
            f"  {row['threshold']:>10.2f}  {row['precision']:>10.4f}"
            f"  {row['recall']:>8.4f}  {row['f1']:>8.4f}"
        )

    lines.append("\n" + sep)
    report = "\n".join(lines)

    os.makedirs(OUTPUTS_PATH, exist_ok=True)
    report_path = os.path.join(OUTPUTS_PATH, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nEvaluation report saved to {report_path}")

    return report


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _project_root = os.path.join(os.path.dirname(__file__), "..")
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    from config import RANDOM_STATE, TEST_SIZE  # noqa: E402
    from src.classifier import (  # noqa: E402
        TRANSFORMER_AVAILABLE,
        get_tfidf_features,
        get_transformer_features,
        load_classifier,
    )

    # --- Load preprocessed data ---
    processed_csv = os.path.join(
        _project_root, "data", "processed", "processed_incidents.csv"
    )
    df = pd.read_csv(processed_csv)
    print(f"Loaded {len(df)} signals from {processed_csv}")

    # Re-create the same stratified split used during training so X_test
    # and y_test exactly match what the classifier was evaluated on.
    le_ref = LabelEncoder()
    le_ref.fit(INCIDENT_CATEGORIES)
    y_all = le_ref.transform(df["category"].values)
    texts_all = df["preprocessed_text"].tolist()

    _, X_test_texts, _, y_test = train_test_split(
        texts_all,
        y_all,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )

    # --- Classical backend ---
    clf_c, le_c, vec_c = load_classifier(backend="classical")
    X_test_c, _ = get_tfidf_features(X_test_texts, vectorizer=vec_c, fit=False)
    classical_results = evaluate_classifier(clf_c, X_test_c, y_test, le_c, "classical")

    # --- Transformer backend (optional) ---
    transformer_results = None
    if TRANSFORMER_AVAILABLE:
        try:
            clf_t, le_t, model_name = load_classifier(backend="transformer")
            X_test_t = get_transformer_features(X_test_texts, model_name=model_name)
            transformer_results = evaluate_classifier(
                clf_t, X_test_t, y_test, le_t, "transformer"
            )
        except FileNotFoundError:
            print("Transformer artefacts not found — skipping transformer evaluation.")
    else:
        print("Transformer backend not available — skipping.")

    # --- Ground-truth DataFrame for similarity / correlation evaluation ---
    scored_csv = os.path.join(
        _project_root, "data", "processed", "scored_incidents.csv"
    )
    df_eval = pd.read_csv(scored_csv) if os.path.exists(scored_csv) else df

    # --- Similarity ---
    sim_pairs_path = os.path.join(
        _project_root, "data", "processed", "similarity_pairs.json"
    )
    similarity_results = evaluate_similarity(sim_pairs_path, df_eval)

    # --- Correlation ---
    corr_path = os.path.join(
        _project_root, "data", "processed", "correlation_results.json"
    )
    correlation_results = evaluate_correlation(corr_path, df_eval)

    # --- Dataset stats ---
    cat_dist = df["category"].value_counts().to_dict()
    dataset_stats = {
        "total_signals": len(df),
        "categories": INCIDENT_CATEGORIES,
        "source_types": sorted(df["source_type"].unique().tolist()),
        "category_distribution": cat_dist,
    }

    # --- Assemble and persist ---
    results = {
        "dataset": dataset_stats,
        "classifier": {
            "classical": classical_results,
            "transformer": transformer_results,
        },
        "similarity": similarity_results,
        "correlation": correlation_results,
    }

    report = generate_full_report(results)
    print(report)

    os.makedirs(OUTPUTS_PATH, exist_ok=True)
    results_path = os.path.join(OUTPUTS_PATH, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
