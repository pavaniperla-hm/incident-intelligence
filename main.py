"""
main.py — CLI entry point for the Incident Intelligence pipeline.

Usage:
    python main.py generate   — generate synthetic dataset
    python main.py train      — normalise, preprocess, train both backends
    python main.py correlate  — run correlator with current config threshold
    python main.py evaluate   — evaluate all pipeline components, save report
    python main.py visualise  — regenerate all charts to data/outputs/
    python main.py run-all    — execute all of the above in sequence
"""

import argparse
import json
import os
import sys

# Ensure imports resolve from the project root regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Change to the project root so all relative config paths work correctly.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import (  # noqa: E402
    INCIDENT_CATEGORIES,
    OUTPUTS_PATH,
    PROCESSED_PATH,
    RAW_DATA_PATH,
    RANDOM_STATE,
    SIMILARITY_THRESHOLD,
    TEST_SIZE,
)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _header(name: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Running: {name}")
    print(f"{'=' * 60}")


def _footer() -> None:
    print(f"{'=' * 60}")
    print("  Done")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Individual commands
# ---------------------------------------------------------------------------

def cmd_generate() -> None:
    """Generate a fresh synthetic 150-signal dataset to data/raw/incidents.csv."""
    _header("generate")

    from src.data_generator import generate_dataset, save_dataset  # noqa: E402

    df = generate_dataset()
    save_dataset(df)

    _footer()


def cmd_train() -> None:
    """Normalise, preprocess, train classifiers, score anomalies, compute similarity."""
    _header("train")

    import pandas as pd
    from src.normaliser import normalise_dataframe  # noqa: E402
    from src.preprocess import preprocess_dataframe  # noqa: E402
    from src.classifier import (  # noqa: E402
        TRANSFORMER_AVAILABLE,
        run_training_pipeline,
    )
    from src.anomaly import score_all_signals  # noqa: E402
    from src.similarity import (  # noqa: E402
        SENTENCE_TRANSFORMER_AVAILABLE,
        compute_similarity_matrix,
        find_similar_pairs,
        save_similarity_results,
    )

    # 1. Load raw incidents.
    print(f"\nLoading raw data from {RAW_DATA_PATH}...")
    df_raw = pd.read_csv(RAW_DATA_PATH)
    print(f"  {len(df_raw)} signals loaded")

    # 2. Normalise schema + text tokens.
    print("\nNormalising...")
    df_norm = normalise_dataframe(df_raw.copy())

    # 3. NLP preprocessing (tokenise, lemmatise, stopword filter).
    print("\nPreprocessing...")
    df_proc = preprocess_dataframe(df_norm)

    # 4. Persist processed DataFrame.
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    proc_path = os.path.join(PROCESSED_PATH, "processed_incidents.csv")
    df_proc.to_csv(proc_path, index=False)
    print(f"\nSaved processed data -> {proc_path}")

    # 5. Train classical classifier (always runs; no GPU required).
    print("\nTraining classical backend...")
    run_training_pipeline(df_proc, backend="classical")

    # 6. Train transformer classifier only when the stack is available.
    if TRANSFORMER_AVAILABLE:
        print("\nTraining transformer backend...")
        run_training_pipeline(df_proc, backend="transformer")
    else:
        print("\nTransformer stack not installed -- skipping transformer training.")

    # 7. Anomaly scoring.
    print("\nScoring anomalies...")
    df_scored = score_all_signals(df_proc)
    scored_path = os.path.join(PROCESSED_PATH, "scored_incidents.csv")
    df_scored.to_csv(scored_path, index=False)
    print(f"Saved scored data -> {scored_path}")

    # 8. Similarity matrix + similar pairs.
    print("\nComputing similarity matrix...")
    backend = "transformer" if SENTENCE_TRANSFORMER_AVAILABLE else "classical"
    sim_matrix, backend_used = compute_similarity_matrix(
        df_scored["preprocessed_text"].tolist(), backend=backend
    )
    pairs = find_similar_pairs(sim_matrix, df_scored["signal_id"].tolist())
    save_similarity_results(pairs)
    print(f"Similarity computed using {backend_used} backend -- {len(pairs)} pairs above threshold")

    _footer()


def cmd_correlate() -> None:
    """Run the correlator with the threshold from config and print a summary."""
    _header("correlate")

    import importlib
    import pandas as pd
    import config  # noqa: E402
    from src.classifier import load_classifier, predict  # noqa: E402
    from src.similarity import (  # noqa: E402
        SENTENCE_TRANSFORMER_AVAILABLE,
        compute_similarity_matrix,
        find_similar_pairs,
    )
    from src.correlator import (  # noqa: E402
        correlate_signals,
        evaluate_correlation,
        save_correlation_results,
    )

    # Reload config so any runtime edits to SIMILARITY_THRESHOLD are picked up.
    importlib.reload(config)
    threshold = config.SIMILARITY_THRESHOLD
    print(f"\nUsing SIMILARITY_THRESHOLD = {threshold}")

    # Load scored incidents (preferred) or fall back to processed.
    scored_path = os.path.join(PROCESSED_PATH, "scored_incidents.csv")
    processed_path = os.path.join(PROCESSED_PATH, "processed_incidents.csv")
    csv_path = scored_path if os.path.exists(scored_path) else processed_path
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"Loaded {len(df)} signals from {csv_path}")

    # Compute similarity matrix using the best available backend.
    print("\nComputing similarity matrix...")
    backend = "transformer" if SENTENCE_TRANSFORMER_AVAILABLE else "classical"
    sim_matrix, backend_used = compute_similarity_matrix(
        df["preprocessed_text"].tolist(), backend=backend
    )
    similarity_pairs = find_similar_pairs(
        sim_matrix, df["signal_id"].tolist(), threshold=threshold
    )

    # Generate predictions for every signal using the classical classifier.
    print("\nGenerating predictions...")
    clf, le, vectorizer = load_classifier(backend="classical")
    predictions: dict = {}
    for _, row in df.iterrows():
        sid = row["signal_id"]
        text = str(row.get("preprocessed_text", row.get("text", "")))
        predictions[sid] = predict(text, clf, vectorizer, le, backend="classical")

    # Run Union-Find correlation.
    print("\nCorrelating signals...")
    groups = correlate_signals(df, similarity_pairs, predictions)

    # Print summary.
    total_groups = len(groups)
    signals_in_groups = sum(g["signal_count"] for g in groups)
    avg_conf = (
        sum(g["confidence_score"] for g in groups) / total_groups
        if total_groups else 0.0
    )
    print(f"\n=== Correlation Summary ===")
    print(f"  Signals            : {len(df)}")
    print(f"  Correlated groups  : {total_groups}")
    print(f"  Signals in groups  : {signals_in_groups}")
    print(f"  Avg confidence     : {avg_conf:.4f}")

    # Evaluate against ground truth.
    evaluate_correlation(groups, df)

    # Save correlation_results.json.
    save_correlation_results(groups)

    _footer()


def cmd_evaluate() -> None:
    """Load all artefacts, run full evaluation, save report and results JSON."""
    _header("evaluate")

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    from src.classifier import (  # noqa: E402
        TRANSFORMER_AVAILABLE,
        get_tfidf_features,
        get_transformer_features,
        load_classifier,
    )
    from src.evaluate import (  # noqa: E402
        evaluate_classifier,
        evaluate_correlation,
        evaluate_similarity,
        generate_full_report,
    )

    # Load processed data.
    processed_path = os.path.join(PROCESSED_PATH, "processed_incidents.csv")
    df = pd.read_csv(processed_path)
    print(f"\nLoaded {len(df)} signals from {processed_path}")

    # Reconstruct the exact held-out test split used during training.
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

    # Classical classifier evaluation.
    clf_c, le_c, vec_c = load_classifier(backend="classical")
    X_test_c, _ = get_tfidf_features(X_test_texts, vectorizer=vec_c, fit=False)
    classical_results = evaluate_classifier(clf_c, X_test_c, y_test, le_c, "classical")

    # Transformer classifier evaluation (optional).
    transformer_results = None
    if TRANSFORMER_AVAILABLE:
        try:
            clf_t, le_t, model_name = load_classifier(backend="transformer")
            X_test_t = get_transformer_features(X_test_texts, model_name=model_name)
            transformer_results = evaluate_classifier(
                clf_t, X_test_t, y_test, le_t, "transformer"
            )
        except FileNotFoundError:
            print("Transformer artefacts not found -- skipping transformer evaluation.")

    # Ground-truth DataFrame for similarity / correlation evaluation.
    scored_path = os.path.join(PROCESSED_PATH, "scored_incidents.csv")
    df_eval = pd.read_csv(scored_path) if os.path.exists(scored_path) else df

    # Similarity and correlation evaluation.
    sim_pairs_path = os.path.join(PROCESSED_PATH, "similarity_pairs.json")
    similarity_results = evaluate_similarity(sim_pairs_path, df_eval)

    corr_path = os.path.join(PROCESSED_PATH, "correlation_results.json")
    correlation_results = evaluate_correlation(corr_path, df_eval)

    # Assemble results dict.
    cat_dist = df["category"].value_counts().to_dict()
    results = {
        "dataset": {
            "total_signals": len(df),
            "categories": INCIDENT_CATEGORIES,
            "source_types": sorted(df["source_type"].unique().tolist()),
            "category_distribution": cat_dist,
        },
        "classifier": {
            "classical": classical_results,
            "transformer": transformer_results,
        },
        "similarity": similarity_results,
        "correlation": correlation_results,
    }

    # Generate and print the full text report.
    report = generate_full_report(results)
    print(report)

    # Persist evaluation results JSON.
    os.makedirs(OUTPUTS_PATH, exist_ok=True)
    results_path = os.path.join(OUTPUTS_PATH, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results JSON saved -> {results_path}")

    _footer()


def cmd_visualise() -> None:
    """Regenerate all PNG charts to data/outputs/."""
    _header("visualise")

    import pandas as pd
    from src.visualise import generate_all_visualisations  # noqa: E402

    # Load the most complete signals DataFrame available.
    scored_path = os.path.join(PROCESSED_PATH, "scored_incidents.csv")
    processed_path = os.path.join(PROCESSED_PATH, "processed_incidents.csv")
    csv_path = scored_path if os.path.exists(scored_path) else processed_path
    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} signals from {csv_path}")

    # Load evaluation results (required by classifier comparison charts).
    results_path = os.path.join(OUTPUTS_PATH, "evaluation_results.json")
    if not os.path.exists(results_path):
        print(
            f"WARNING: {results_path} not found.\n"
            "         Run 'python main.py evaluate' first for full chart output."
        )
        eval_results: dict = {}
    else:
        with open(results_path) as f:
            eval_results = json.load(f)

    generate_all_visualisations(df, eval_results)

    _footer()


def cmd_run_all() -> None:
    """Execute every pipeline stage in sequence."""
    _header("run-all")
    print("  Stages: generate -> train -> correlate -> evaluate -> visualise")
    print(f"{'=' * 60}")

    cmd_generate()
    cmd_train()
    cmd_correlate()
    cmd_evaluate()
    cmd_visualise()

    print(f"\n{'=' * 60}")
    print("  run-all complete -- all pipeline stages finished.")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Usage text
# ---------------------------------------------------------------------------

_COMMANDS = {
    "generate":  "Generate synthetic 150-signal dataset -> data/raw/incidents.csv",
    "train":     "Normalise, preprocess, train classifiers, score anomalies, compute similarity",
    "correlate": "Cluster related signals using the current config SIMILARITY_THRESHOLD",
    "evaluate":  "Evaluate classifiers + similarity + correlation; save report + JSON",
    "visualise": "Regenerate all PNG charts -> data/outputs/",
    "run-all":   "Execute all stages in sequence (generate -> train -> correlate -> evaluate -> visualise)",
}


def print_usage() -> None:
    print("\nIncident Intelligence System -- Pipeline CLI")
    print("=" * 60)
    print("\nUsage:  python main.py <command>\n")
    print("Commands:")
    for cmd, desc in _COMMANDS.items():
        print(f"  {cmd:<12}  {desc}")
    print("\nExamples:")
    print("  python main.py generate    # create fresh synthetic dataset")
    print("  python main.py train       # full training pipeline")
    print("  python main.py run-all     # end-to-end in one shot")
    print("  python main.py evaluate    # evaluate and save report")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Incident Intelligence System -- pipeline CLI",
        add_help=True,
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=list(_COMMANDS.keys()),
        help="Pipeline stage to run (omit to see usage)",
    )
    args = parser.parse_args()

    if args.command is None:
        print_usage()
        return

    dispatch = {
        "generate":  cmd_generate,
        "train":     cmd_train,
        "correlate": cmd_correlate,
        "evaluate":  cmd_evaluate,
        "visualise": cmd_visualise,
        "run-all":   cmd_run_all,
    }
    dispatch[args.command]()


if __name__ == "__main__":
    main()
