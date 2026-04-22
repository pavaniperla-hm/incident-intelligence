"""
visualise.py — Dashboard and report visualisation.

Purpose:
    Generates static PNG charts that surface incident trends, category
    distributions, classifier performance, similarity quality, correlation
    confidence, and threshold sensitivity to end users.

Pipeline position: Terminal display stage; consumes outputs of all prior modules.

Inputs:
    - processed_incidents.csv or scored_incidents.csv
    - data/outputs/evaluation_results.json
    - data/processed/similarity_pairs.json
    - data/processed/correlation_results.json

Outputs:
    - data/outputs/signal_distribution.png
    - data/outputs/classifier_comparison.png
    - data/outputs/confusion_matrices.png
    - data/outputs/similarity_comparison.png
    - data/outputs/threshold_sensitivity.png
    - data/outputs/confidence_distribution.png
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Path fix — allows running this file directly as a script from any directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import INCIDENT_CATEGORIES, OUTPUTS_PATH  # noqa: E402

plt.style.use("seaborn-v0_8-whitegrid")

# Short labels used on confusion matrix axes to avoid overflow.
_SHORT_LABELS = ["Auth", "Network", "Deploy", "Perf", "Security"]

# One colour per source type for stacked bars.
_SOURCE_COLOURS = {"alert": "#2196F3", "ticket": "#FF9800", "log": "#4CAF50"}

# One colour per category for multi-category charts.
_CAT_COLOURS = [
    "#E53935",  # Auth Failure   — red
    "#1E88E5",  # Network Outage — blue
    "#43A047",  # Deploy Failure — green
    "#FB8C00",  # Perf Degrad.   — orange
    "#8E24AA",  # Security Breach — purple
]


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------


def plot_topic_distribution(df: pd.DataFrame) -> None:
    """Horizontal stacked bar chart: signal counts per category by source type.

    Args:
        df: DataFrame with at least ``category`` and ``source_type`` columns.
    """
    source_types = ["alert", "ticket", "log"]
    counts = (
        df.groupby(["category", "source_type"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=source_types, fill_value=0)
    )
    # Sort categories by total descending for readability.
    counts["_total"] = counts.sum(axis=1)
    counts = counts.sort_values("_total").drop(columns="_total")

    fig, ax = plt.subplots(figsize=(10, 6))
    left = np.zeros(len(counts))
    for src in source_types:
        if src in counts.columns:
            vals = counts[src].values
            ax.barh(
                counts.index,
                vals,
                left=left,
                color=_SOURCE_COLOURS[src],
                label=src.capitalize(),
                edgecolor="white",
                linewidth=0.5,
            )
            left += vals

    ax.set_xlabel("Signal Count")
    ax.set_title("Signal Distribution by Incident Category and Source Type")
    ax.legend(title="Source Type", loc="lower right")
    ax.set_xlim(0, left.max() * 1.12)

    # Value labels at end of each bar.
    for i, total in enumerate(left):
        ax.text(total + 0.3, i, str(int(total)), va="center", fontsize=9)

    plt.tight_layout()
    os.makedirs(OUTPUTS_PATH, exist_ok=True)
    out = os.path.join(OUTPUTS_PATH, "signal_distribution.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_classifier_comparison(results: dict) -> None:
    """Grouped bar chart comparing classical vs transformer classifier metrics.

    Args:
        results: Eval results dict with a ``classifier`` key containing
                 ``classical`` and ``transformer`` sub-dicts.
    """
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = [m.capitalize() for m in metrics]
    clf = results.get("classifier", {})
    classical = clf.get("classical") or {}
    transformer = clf.get("transformer") or {}

    c_vals = [classical.get(m, 0.0) for m in metrics]
    t_vals = [transformer.get(m, 0.0) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_c = ax.bar(
        x - width / 2, c_vals, width,
        label="Classical", color="#1E88E5", edgecolor="white"
    )
    bars_t = ax.bar(
        x + width / 2, t_vals, width,
        label="Transformer", color="#43A047", edgecolor="white"
    )

    for bar in (*bars_c, *bars_t):
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.01,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Classifier Performance — Classical vs Transformer Backend")
    ax.legend()

    plt.tight_layout()
    os.makedirs(OUTPUTS_PATH, exist_ok=True)
    out = os.path.join(OUTPUTS_PATH, "classifier_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_confusion_matrices(results: dict) -> None:
    """Side-by-side seaborn heatmap confusion matrices for both backends.

    Args:
        results: Eval results dict with a ``classifier`` key.
    """
    clf = results.get("classifier", {})
    backends = []
    for key in ("classical", "transformer"):
        entry = clf.get(key)
        if entry and entry.get("confusion_matrix"):
            backends.append((key.capitalize(), entry))

    if not backends:
        print("No confusion matrix data available — skipping plot.")
        return

    n_plots = len(backends)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for ax, (title, entry) in zip(axes, backends):
        cm = np.array(entry["confusion_matrix"])
        n_classes = cm.shape[0]
        tick_labels = _SHORT_LABELS[:n_classes]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=tick_labels,
            yticklabels=tick_labels,
            ax=ax,
            cbar=False,
        )
        ax.set_title(f"{title} Backend")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    fig.suptitle("Confusion Matrices", fontsize=13, y=1.02)
    plt.tight_layout()
    os.makedirs(OUTPUTS_PATH, exist_ok=True)
    out = os.path.join(OUTPUTS_PATH, "confusion_matrices.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_similarity_comparison(df: pd.DataFrame) -> None:
    """Two-panel violin plot: TF-IDF vs transformer similarity distributions.

    Left panel shows TF-IDF cosine scores; right panel shows transformer cosine
    scores.  Each panel splits pairs into same-incident vs cross-incident so
    the separation gap between backends is visible.

    Args:
        df: DataFrame with ``signal_id``, ``incident_group``, and
            ``preprocessed_text`` columns.
    """
    from src.similarity import (  # local import — optional dep  # noqa: E402
        SENTENCE_TRANSFORMER_AVAILABLE,
        compute_tfidf_similarity,
        compute_transformer_similarity,
    )

    texts = df["preprocessed_text"].tolist()
    signal_ids = df["signal_id"].tolist()
    incident_of = dict(zip(df["signal_id"], df["incident_group"]))

    tfidf_mat = compute_tfidf_similarity(texts)
    transformer_mat = compute_transformer_similarity(texts) if SENTENCE_TRANSFORMER_AVAILABLE else None

    rows = []
    n = len(signal_ids)
    for i in range(n):
        for j in range(i + 1, n):
            sid_a, sid_b = signal_ids[i], signal_ids[j]
            label = "Same Incident" if incident_of.get(sid_a) == incident_of.get(sid_b) else "Cross Incident"
            rows.append({
                "pair_type": label,
                "tfidf": float(tfidf_mat[i, j]),
                "transformer": float(transformer_mat[i, j]) if transformer_mat is not None else None,
            })

    pair_df = pd.DataFrame(rows)

    n_panels = 2 if transformer_mat is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), sharey=False)
    if n_panels == 1:
        axes = [axes]

    palette = {"Same Incident": "#1E88E5", "Cross Incident": "#E53935"}

    sns.violinplot(
        data=pair_df,
        x="pair_type",
        y="tfidf",
        palette=palette,
        ax=axes[0],
        inner="box",
        cut=0,
    )
    axes[0].set_title("TF-IDF Similarity")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Cosine Similarity Score")

    if transformer_mat is not None:
        sns.violinplot(
            data=pair_df,
            x="pair_type",
            y="transformer",
            palette=palette,
            ax=axes[1],
            inner="box",
            cut=0,
        )
        axes[1].set_title("Transformer Similarity")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Cosine Similarity Score")

    fig.suptitle(
        "TF-IDF vs Transformer Similarity — Same-Incident vs Cross-Incident Pairs",
        fontsize=11,
    )
    plt.tight_layout()
    os.makedirs(OUTPUTS_PATH, exist_ok=True)
    out = os.path.join(OUTPUTS_PATH, "similarity_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_threshold_sensitivity(threshold_results: list) -> None:
    """Line chart of precision, recall, and F1 across similarity thresholds.

    Args:
        threshold_results: List of dicts with keys ``threshold``, ``precision``,
                           ``recall``, ``f1``.
    """
    thresholds = [r["threshold"] for r in threshold_results]
    precisions = [r["precision"] for r in threshold_results]
    recalls = [r["recall"] for r in threshold_results]
    f1s = [r["f1"] for r in threshold_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precisions, "o-", color="#E53935", label="Precision", linewidth=2)
    ax.plot(thresholds, recalls, "s-", color="#1E88E5", label="Recall", linewidth=2)
    ax.plot(thresholds, f1s, "^-", color="#43A047", label="F1", linewidth=2)

    ax.set_xlabel("Similarity Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Correlation Performance vs Similarity Threshold")
    ax.set_xticks(thresholds)
    ax.set_ylim(0, 1.08)
    ax.legend()

    for x_val, p, r, f in zip(thresholds, precisions, recalls, f1s):
        ax.annotate(f"{p:.3f}", (x_val, p), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="#E53935")
        ax.annotate(f"{r:.3f}", (x_val, r), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="#1E88E5")
        ax.annotate(f"{f:.3f}", (x_val, f), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="#43A047")

    plt.tight_layout()
    os.makedirs(OUTPUTS_PATH, exist_ok=True)
    out = os.path.join(OUTPUTS_PATH, "threshold_sensitivity.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_confidence_distribution(correlation_results_path: str) -> None:
    """Histogram of confidence scores across correlated groups, coloured by category.

    Args:
        correlation_results_path: Path to correlation_results.json.
    """
    with open(correlation_results_path) as f:
        groups = json.load(f)

    if not groups:
        print("No correlated groups found — skipping confidence distribution plot.")
        return

    group_df = pd.DataFrame([
        {"confidence_score": g["confidence_score"], "category": g["predicted_category"]}
        for g in groups
    ])

    cats_present = [c for c in INCIDENT_CATEGORIES if c in group_df["category"].values]
    cat_colour_map = dict(zip(INCIDENT_CATEGORIES, _CAT_COLOURS))

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 15)

    for cat in cats_present:
        subset = group_df[group_df["category"] == cat]["confidence_score"].values
        if len(subset) > 0:
            ax.hist(
                subset,
                bins=bins,
                alpha=0.65,
                label=cat,
                color=cat_colour_map.get(cat, "#999999"),
                edgecolor="white",
            )

    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Group Count")
    ax.set_title("Incident Confidence Score Distribution")
    ax.set_xlim(0, 1)
    ax.legend(title="Category", fontsize=8, title_fontsize=9)

    plt.tight_layout()
    os.makedirs(OUTPUTS_PATH, exist_ok=True)
    out = os.path.join(OUTPUTS_PATH, "confidence_distribution.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def generate_all_visualisations(df: pd.DataFrame, eval_results: dict) -> None:
    """Run all six plot functions in sequence.

    Args:
        df:           Processed/scored signals DataFrame.
        eval_results: Dict from evaluate.generate_full_report or loaded from
                      evaluation_results.json.
    """
    _project_root = os.path.join(os.path.dirname(__file__), "..")
    corr_path = os.path.join(
        _project_root, "data", "processed", "correlation_results.json"
    )

    threshold_results = [
        {"threshold": 0.50, "precision": 1.0, "recall": 0.073, "f1": 0.137},
        {"threshold": 0.35, "precision": 1.0, "recall": 0.360, "f1": 0.529},
        {"threshold": 0.25, "precision": 1.0, "recall": 0.753, "f1": 0.859},
    ]

    plot_topic_distribution(df)
    plot_classifier_comparison(eval_results)
    plot_confusion_matrices(eval_results)
    plot_similarity_comparison(df)
    plot_threshold_sensitivity(threshold_results)
    plot_confidence_distribution(corr_path)

    print(f"\nAll visualisations saved to {OUTPUTS_PATH}")


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _project_root = os.path.join(os.path.dirname(__file__), "..")
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    # --- Load signals DataFrame ---
    scored_csv = os.path.join(
        _project_root, "data", "processed", "scored_incidents.csv"
    )
    processed_csv = os.path.join(
        _project_root, "data", "processed", "processed_incidents.csv"
    )
    if os.path.exists(scored_csv):
        df = pd.read_csv(scored_csv)
    else:
        df = pd.read_csv(processed_csv)
    print(f"Loaded {len(df)} signals")

    # --- Load or generate evaluation results ---
    results_path = os.path.join(OUTPUTS_PATH, "evaluation_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            eval_results = json.load(f)
        print(f"Loaded evaluation results from {results_path}")
    else:
        print(
            f"evaluation_results.json not found at {results_path}\n"
            "Run src/evaluate.py first to generate evaluation results."
        )
        sys.exit(1)

    generate_all_visualisations(df, eval_results)
    print("Done — check data/outputs/ for all charts.")
