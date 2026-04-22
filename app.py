"""
app.py — Streamlit dashboard for the Incident Intelligence System.

Run with:
    streamlit run app.py
"""

import sys
import os

# Must be first so all src.* imports resolve correctly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Page config must come before any other st.* call.
st.set_page_config(
    page_title="Incident Intelligence System",
    page_icon="🚨",
    layout="wide",
)

from config import (  # noqa: E402
    CORRELATION_TIME_WINDOW_MINUTES,
    DEFAULT_BACKEND,
    INCIDENT_CATEGORIES,
    OUTPUTS_PATH,
    PROCESSED_PATH,
    SIMILARITY_THRESHOLD,
)
from src.anomaly import compute_hourly_baseline, score_signal_anomaly  # noqa: E402
from src.classifier import TRANSFORMER_AVAILABLE, load_classifier, predict  # noqa: E402
from src.normaliser import normalise_text  # noqa: E402
from src.preprocess import preprocess_text  # noqa: E402
from src.similarity import SENTENCE_TRANSFORMER_AVAILABLE  # noqa: E402


# ---------------------------------------------------------------------------
# Cached resource loader — runs once per Python session
# ---------------------------------------------------------------------------

@st.cache_resource
def load_all_artefacts() -> dict:
    """Load and cache every heavy ML artefact needed by the dashboard."""
    arts: dict = {}

    # --- Classical classifier (required for classification + fallback sim) ---
    try:
        clf_c, le, vec_c = load_classifier(backend="classical")
        arts.update(clf_c=clf_c, le=le, vec_c=vec_c, clf_c_ok=True)
    except FileNotFoundError as exc:
        arts["clf_c_ok"] = False
        arts["clf_c_error"] = str(exc)

    # --- Transformer classifier (optional) ---
    arts["clf_t_ok"] = False
    if TRANSFORMER_AVAILABLE:
        try:
            clf_t, _, model_name = load_classifier(backend="transformer")
            arts.update(clf_t=clf_t, clf_t_model=model_name, clf_t_ok=True)
        except FileNotFoundError:
            pass

    # --- Signals DataFrame (prefer scored, fall back to processed) ---
    arts["df_ok"] = False
    for candidate in (
        os.path.join(PROCESSED_PATH, "scored_incidents.csv"),
        os.path.join(PROCESSED_PATH, "processed_incidents.csv"),
    ):
        if os.path.exists(candidate):
            df = pd.read_csv(candidate)
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            # CSV may load is_business_hours as string; coerce to bool.
            if df["is_business_hours"].dtype == object:
                df["is_business_hours"] = df["is_business_hours"].map(
                    {"True": True, "False": False, True: True, False: False}
                )
            arts["df"] = df
            arts["df_ok"] = True
            break

    # --- MiniLM sentence-transformer + pre-computed signal embeddings ---
    arts["sim_ok"] = False
    if SENTENCE_TRANSFORMER_AVAILABLE and arts["df_ok"]:
        try:
            from src.similarity import get_transformer_model  # noqa: E402
            sim_model = get_transformer_model()
            arts["sim_model"] = sim_model
            # Pre-compute once; reused for every new signal query.
            arts["embeddings"] = sim_model.encode(
                arts["df"]["preprocessed_text"].tolist(),
                show_progress_bar=False,
            ).astype(np.float32)
            arts["sim_ok"] = True
        except Exception:
            pass

    # --- Pre-computed TF-IDF matrix for classical similarity fallback ---
    if arts.get("clf_c_ok") and arts["df_ok"]:
        arts["tfidf_mat"] = arts["vec_c"].transform(
            arts["df"]["preprocessed_text"].tolist()
        )

    # --- Hourly baseline for anomaly scoring new signals ---
    if arts["df_ok"]:
        arts["baseline"] = compute_hourly_baseline(arts["df"])

    # --- Correlation results ---
    corr_path = os.path.join(PROCESSED_PATH, "correlation_results.json")
    arts["corr_ok"] = False
    if os.path.exists(corr_path):
        with open(corr_path) as f:
            arts["corr_results"] = json.load(f)
        arts["corr_ok"] = True

    # --- Evaluation results ---
    eval_path = os.path.join(OUTPUTS_PATH, "evaluation_results.json")
    arts["eval_ok"] = False
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            arts["eval_results"] = json.load(f)
        arts["eval_ok"] = True

    return arts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Short category name for coloured badges in the Incident Feed.
_CAT_COLOUR = {
    "Authentication Failure": "red",
    "Network Outage":         "blue",
    "Deployment Failure":     "green",
    "Performance Degradation": "orange",
    "Security Breach":        "violet",
}

# Hardcoded threshold experiment results (used in Analytics + Evaluation tabs).
_THRESHOLD_TABLE = [
    {"Threshold": 0.50, "Precision": 1.000, "Recall": 0.073, "F1": 0.137},
    {"Threshold": 0.35, "Precision": 1.000, "Recall": 0.360, "F1": 0.529},
    {"Threshold": 0.25, "Precision": 1.000, "Recall": 0.753, "F1": 0.859},
]


def _show_missing(filename: str, command: str) -> None:
    """Render a consistent warning for a missing pipeline output file."""
    st.warning(
        f"**{filename}** not found.  "
        f"Run `{command}` from the project root to generate it."
    )


# ---------------------------------------------------------------------------
# Load artefacts (cached)
# ---------------------------------------------------------------------------

arts = load_all_artefacts()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🚨 Incident Intelligence System")
st.sidebar.markdown("**ML-Based Proactive Issue Detection**")
st.sidebar.divider()

# Current configuration pulled from config.py.
st.sidebar.markdown("**⚙️ Configuration**")
st.sidebar.markdown(f"- Similarity threshold: `{SIMILARITY_THRESHOLD}`")
st.sidebar.markdown(f"- Time window: `{CORRELATION_TIME_WINDOW_MINUTES} min`")
st.sidebar.markdown(f"- Default backend: `{DEFAULT_BACKEND}`")

st.sidebar.divider()
st.sidebar.markdown("**📦 Dataset**")
st.sidebar.markdown("- 150 signals total")
st.sidebar.markdown("- 5 incident categories")
st.sidebar.markdown("- 3 source types (alert, ticket, log)")

st.sidebar.divider()
# Numbered pipeline step list.
st.sidebar.markdown("**🔄 Pipeline Steps**")
for i, step in enumerate(
    [
        "Data Ingestion",
        "Normalisation",
        "Preprocessing",
        "Classification",
        "Similarity",
        "Anomaly Scoring",
        "Correlation",
        "Dashboard",
    ],
    start=1,
):
    st.sidebar.markdown(f"{i}. {step}")

st.sidebar.divider()
# Category filter — used by the Incident Feed tab.
st.sidebar.markdown("**🔽 Incident Feed Filters**")
cat_filter = st.sidebar.multiselect(
    "Filter by Category",
    options=INCIDENT_CATEGORIES,
    default=INCIDENT_CATEGORIES,
    key="cat_filter",
)

st.sidebar.divider()
st.sidebar.info("Master's Project — Software Engineering")


# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Analyse Signal", "📋 Incident Feed", "📊 Analytics", "📈 Evaluation"]
)


# ===========================================================================
# TAB 1 — Analyse Signal
# ===========================================================================

with tab1:
    st.title("🔍 Analyse Signal")
    st.markdown(
        "Paste any raw incident signal text to run it through the full pipeline: "
        "normalise → preprocess → classify → similarity → anomaly."
    )

    # Gate: classifier must be trained before this tab is useful.
    if not arts.get("clf_c_ok"):
        _show_missing("classifier_classical.pkl", "python src/classifier.py")
        st.stop()

    # --- Input area ---------------------------------------------------------
    input_text = st.text_area(
        "Paste incident signal text",
        placeholder=(
            "e.g. CRITICAL: Authentication service returning 401 errors for "
            "all SSO users since 14:32 UTC. Token validation failure rate at "
            "94%. Incident logged by on-call engineer."
        ),
        height=130,
        key="signal_input",
    )

    # Metadata columns.
    col_src, col_cat, col_be = st.columns(3)
    with col_src:
        source_type = st.selectbox("Source type", ["alert", "ticket", "log"])
    with col_cat:
        expected_cat = st.selectbox(
            "Expected category (optional, for validation)",
            ["—"] + INCIDENT_CATEGORIES,
        )
    with col_be:
        backend_opts = ["classical"]
        if arts.get("clf_t_ok"):
            backend_opts.append("transformer")
        backend_choice = st.selectbox("Classification backend", backend_opts)

    analyse_clicked = st.button("Analyse Signal", type="primary")

    # Run pipeline when button is clicked and text is present.
    if analyse_clicked and input_text.strip():
        with st.spinner("Running pipeline…"):
            raw = input_text.strip()

            # Step 1 — Normalise raw text.
            normalised = normalise_text(raw, source_type)

            # Step 2 — Preprocess normalised text (tokenise, lemmatise, filter).
            preprocessed = preprocess_text(normalised)

            # Step 3 — Classify.
            if backend_choice == "transformer" and arts.get("clf_t_ok"):
                clf_result = predict(
                    normalised,
                    arts["clf_t"],
                    arts["clf_t_model"],
                    arts["le"],
                    backend="transformer",
                )
            else:
                clf_result = predict(
                    normalised,
                    arts["clf_c"],
                    arts["vec_c"],
                    arts["le"],
                    backend="classical",
                )

            # Step 4 — Similarity against all 150 existing signals.
            if arts.get("sim_ok"):
                # Preferred: MiniLM dense embeddings.
                new_emb = arts["sim_model"].encode(
                    [preprocessed], show_progress_bar=False
                ).astype(np.float32)
                sim_scores = cosine_similarity(new_emb, arts["embeddings"])[0]
            elif arts.get("tfidf_mat") is not None:
                # Fallback: TF-IDF cosine similarity.
                new_vec = arts["vec_c"].transform([preprocessed])
                sim_scores = cosine_similarity(new_vec, arts["tfidf_mat"])[0]
            else:
                sim_scores = None

            # Build top-5 similar signals table.
            top_df = None
            if sim_scores is not None and arts["df_ok"]:
                top_idx = np.argsort(sim_scores)[::-1][:5]
                top_df = (
                    arts["df"]
                    .iloc[top_idx][["signal_id", "source_type", "category"]]
                    .copy()
                    .reset_index(drop=True)
                )
                top_df["similarity"] = np.round(sim_scores[top_idx], 4)

            # Step 5 — Anomaly score using current wall-clock time as the
            # signal's arrival time (no timestamp in user input).
            now = datetime.now()
            is_biz = 8 <= now.hour < 18 and now.weekday() < 5
            mock_signal = pd.Series(
                {
                    "signal_id": "NEW",
                    "hour": now.hour,
                    "is_business_hours": is_biz,
                }
            )
            anomaly_info = score_signal_anomaly(
                mock_signal, arts["baseline"], arts["df"]
            )

        # Persist results so they survive widget interactions.
        st.session_state["analysis"] = {
            "raw_input": raw,
            "normalised": normalised,
            "preprocessed": preprocessed,
            "clf_result": clf_result,
            "top_df": top_df,
            "anomaly_info": anomaly_info,
        }

    # --- Display results (persist across interactions via session_state) ----
    if st.session_state.get("analysis"):
        res = st.session_state["analysis"]
        clf = res["clf_result"]
        anomaly = res["anomaly_info"]
        top_df = res["top_df"]

        st.divider()
        st.subheader("Pipeline Results")

        # Three headline metric columns.
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            correct_flag = ""
            if expected_cat != "—":
                correct_flag = " ✅" if clf["predicted_category"] == expected_cat else " ❌"
            st.metric(
                "Predicted Category",
                clf["predicted_category"] + correct_flag,
                delta=f"confidence {clf['confidence']:.1%}",
            )
        with mc2:
            st.metric(
                "Anomaly Score",
                f"{anomaly['anomaly_score']:.3f}",
                delta=anomaly["time_context"].replace("_", " "),
                delta_color="off",
            )
        with mc3:
            if top_df is not None and len(top_df):
                best = top_df.iloc[0]
                st.metric(
                    "Top Similar Signal",
                    best["signal_id"],
                    delta=f"similarity {best['similarity']:.3f}",
                )
            else:
                st.metric("Top Similar Signal", "N/A")

        # Normalised text before / after expander.
        with st.expander("Normalised text"):
            c_raw, c_norm = st.columns(2)
            with c_raw:
                st.markdown("**Raw input:**")
                st.code(res["raw_input"], language=None)
            with c_norm:
                st.markdown("**After normalisation:**")
                st.code(res["normalised"], language=None)

        # Top 5 similar signals table.
        with st.expander("Top 5 similar signals"):
            if top_df is not None:
                st.dataframe(top_df, use_container_width=True, hide_index=True)
            else:
                st.warning("Similarity model not available — run the pipeline first.")

        # Category probability bar chart.
        with st.expander("All category probabilities"):
            probs = clf["all_probabilities"]
            prob_df = pd.DataFrame(
                {"Probability": list(probs.values())},
                index=list(probs.keys()),
            )
            st.bar_chart(prob_df)

        # Grouping hint — find which correlated group the best match belongs to.
        if top_df is not None and len(top_df) and arts.get("corr_ok"):
            best_sid = top_df.iloc[0]["signal_id"]
            matching_group = next(
                (g for g in arts["corr_results"] if best_sid in g["signal_ids"]),
                None,
            )
            if matching_group:
                st.info(
                    f"If this signal were added to the system, it would likely be "
                    f"grouped with: **{matching_group['group_id']}** "
                    f"({matching_group['predicted_category']}, "
                    f"confidence: {matching_group['confidence_score']:.2f})"
                )
            else:
                cat = top_df.iloc[0]["category"]
                st.info(
                    f"If this signal were added to the system, it would likely be "
                    f"grouped with: signals related to **{cat}**"
                )


# ===========================================================================
# TAB 2 — Incident Feed
# ===========================================================================

with tab2:
    st.title("📋 Incident Feed")
    st.markdown("Signals grouped by the ML correlation engine, sorted by confidence.")

    if not arts.get("corr_ok"):
        _show_missing("correlation_results.json", "python src/correlator.py")
    else:
        all_groups = arts["corr_results"]

        # Apply the category filter from the sidebar multiselect.
        filtered = [g for g in all_groups if g["predicted_category"] in cat_filter]

        if not filtered:
            st.info("No groups match the selected category filter.")
        else:
            # Summary metrics row.
            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Correlated Groups", len(filtered))
            sm2.metric("Signals Correlated", sum(g["signal_count"] for g in filtered))
            avg_conf = float(np.mean([g["confidence_score"] for g in filtered]))
            sm3.metric("Avg Confidence", f"{avg_conf:.3f}")
            avg_sz = float(np.mean([g["signal_count"] for g in filtered]))
            sm4.metric("Avg Group Size", f"{avg_sz:.1f}")

            st.divider()

            # One expander per group, sorted by confidence descending.
            for grp in sorted(filtered, key=lambda g: g["confidence_score"], reverse=True):
                header = (
                    f"**{grp['group_id']}**  ·  {grp['predicted_category']}  "
                    f"·  confidence: {grp['confidence_score']:.2f}"
                )
                with st.expander(header):
                    # Confidence progress bar.
                    st.progress(
                        float(grp["confidence_score"]),
                        text=f"Confidence: {grp['confidence_score']:.2f}",
                    )

                    # Coloured category badge.
                    colour = _CAT_COLOUR.get(grp["predicted_category"], "gray")
                    st.markdown(f":{colour}[**{grp['predicted_category']}**]")

                    # Group metadata.
                    meta_a, meta_b = st.columns(2)
                    meta_a.markdown(f"**Time span:** {grp['time_span_minutes']:.1f} min")
                    meta_b.markdown(f"**Source types:** {', '.join(grp['source_types'])}")

                    # Signal table with source, snippet, and group avg similarity.
                    sig_rows = [
                        {
                            "signal_id": s["signal_id"],
                            "source_type": s["source_type"],
                            "text_snippet": s["text_snippet"][:100],
                            "avg_similarity": grp["avg_similarity"],
                        }
                        for s in grp["signals_detail"]
                    ]
                    st.dataframe(
                        pd.DataFrame(sig_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

                    # Ground-truth incident_group caption for validation.
                    if arts["df_ok"]:
                        sids = grp["signal_ids"]
                        true_incs = (
                            arts["df"]
                            .loc[arts["df"]["signal_id"].isin(sids), "incident_group"]
                            .unique()
                            .tolist()
                        )
                        st.caption(
                            f"Ground-truth incident group(s): "
                            f"{', '.join(str(x) for x in sorted(true_incs))}"
                        )


# ===========================================================================
# TAB 3 — Analytics
# ===========================================================================

with tab3:
    st.title("📊 Dataset and Model Analytics")

    # Each PNG with a caption explaining what it shows.
    charts = [
        (
            "signal_distribution.png",
            "Signal count per incident category, stacked by source type "
            "(alert / ticket / log).",
            "python src/visualise.py",
        ),
        (
            "classifier_comparison.png",
            "Classical vs transformer backend — accuracy, precision, recall, F1.",
            "python src/visualise.py",
        ),
        (
            "confusion_matrices.png",
            "True vs predicted category for both classifier backends.",
            "python src/visualise.py",
        ),
        (
            "similarity_comparison.png",
            "TF-IDF vs transformer similarity score distributions: "
            "same-incident pairs should score much higher than cross-incident pairs.",
            "python src/visualise.py",
        ),
        (
            "threshold_sensitivity.png",
            "Correlation precision, recall, and F1 as the similarity threshold "
            "is swept from 0.25 to 0.50.",
            "python src/visualise.py",
        ),
    ]

    for fname, caption, cmd in charts:
        img_path = os.path.join(OUTPUTS_PATH, fname)
        if os.path.exists(img_path):
            st.image(img_path, caption=caption, use_column_width=True)
        else:
            _show_missing(fname, cmd)
        st.divider()

    # Classifier comparison table from evaluation results.
    if arts.get("eval_ok"):
        clf_data = arts["eval_results"].get("classifier", {})
        clf_rows = []
        for be in ("classical", "transformer"):
            entry = clf_data.get(be)
            if entry:
                clf_rows.append(
                    {
                        "Backend": be.capitalize(),
                        "Accuracy": entry["accuracy"],
                        "Precision": entry["precision"],
                        "Recall": entry["recall"],
                        "F1": entry["f1"],
                    }
                )
        if clf_rows:
            st.subheader("Classifier Metrics Table")
            st.dataframe(
                pd.DataFrame(clf_rows).set_index("Backend"),
                use_container_width=True,
            )

    # Threshold sensitivity table (hardcoded experiment results).
    st.subheader("Threshold Sensitivity Table")
    st.dataframe(
        pd.DataFrame(_THRESHOLD_TABLE).set_index("Threshold"),
        use_container_width=True,
    )


# ===========================================================================
# TAB 4 — Evaluation
# ===========================================================================

with tab4:
    st.title("📈 Model Evaluation Report")

    if not arts.get("eval_ok"):
        _show_missing("evaluation_results.json", "python src/evaluate.py")
    else:
        eval_res = arts["eval_results"]

        # --- Classifier comparison with conditional formatting ---
        st.subheader("Classifier Performance")
        clf_data = eval_res.get("classifier", {})
        clf_rows = []
        for be in ("classical", "transformer"):
            entry = clf_data.get(be)
            if entry:
                clf_rows.append(
                    {
                        "Backend": be.capitalize(),
                        "Accuracy": entry["accuracy"],
                        "Precision": entry["precision"],
                        "Recall": entry["recall"],
                        "F1": entry["f1"],
                    }
                )
        if clf_rows:
            clf_eval_df = pd.DataFrame(clf_rows).set_index("Backend")
            # Highlight the better value in each metric column.
            st.dataframe(
                clf_eval_df.style.highlight_max(axis=0, color="#d4f1d4"),
                use_container_width=True,
            )

        st.divider()

        # --- Correlation metrics as st.metric widgets ---
        st.subheader("Correlation Metrics")
        corr = eval_res.get("correlation", {})
        cm1, cm2, cm3, cm4 = st.columns(4)
        cm1.metric("Total Groups", corr.get("total_groups", "N/A"))
        cm2.metric("Pair Precision", f"{corr.get('precision', 0.0):.4f}")
        cm3.metric("Pair Recall", f"{corr.get('recall', 0.0):.4f}")
        cm4.metric("F1", f"{corr.get('f1', 0.0):.4f}")

        # Additional aggregate stats.
        ca1, ca2, ca3 = st.columns(3)
        ca1.metric("Avg Confidence", f"{corr.get('avg_confidence_score', 0.0):.4f}")
        ca2.metric("Avg Group Size", f"{corr.get('avg_group_size', 0.0):.1f}")
        ca3.metric("Avg Time Span", f"{corr.get('avg_time_span_minutes', 0.0):.1f} min")

        st.divider()

        # --- Similarity evaluation ---
        sim = eval_res.get("similarity")
        if sim:
            st.subheader("Similarity Evaluation")
            ss1, ss2, ss3 = st.columns(3)
            ss1.metric("Total Pairs (above threshold)", sim.get("total_pairs", 0))
            ss2.metric("True Positives (same incident)", sim.get("true_positives", 0))
            ss3.metric("Similarity Precision", f"{sim.get('precision', 0.0):.4f}")
            sc1, sc2 = st.columns(2)
            sc1.metric("Avg Score — Same Incident", f"{sim.get('avg_score_same_incident', 0.0):.4f}")
            sc2.metric("Avg Score — Cross Incident", f"{sim.get('avg_score_diff_incident', 0.0):.4f}")

        st.divider()

        # --- Threshold sensitivity as interactive dataframe ---
        st.subheader("Threshold Sensitivity")
        st.dataframe(
            pd.DataFrame(_THRESHOLD_TABLE).set_index("Threshold"),
            use_container_width=True,
        )

        st.divider()

        # --- Insight paragraph ---
        st.info(
            "**Why classical outperforms transformer on classification**: "
            "With only 150 signals (30 per category), the dataset is too small "
            "to leverage DistilBERT's deep contextual representations effectively — "
            "the model was pre-trained on millions of documents and cannot fine-tune "
            "meaningfully on 120 training examples. TF-IDF bigrams capture the "
            "task-specific vocabulary (e.g. `login_failed`, `pod_crash`) very well "
            "at this scale, and logistic regression generalises cleanly to the 30-example "
            "test set. However, the MiniLM sentence-transformer wins decisively on "
            "similarity: its semantic embeddings detect paraphrase relationships and "
            "cross-category vocabulary mixing that TF-IDF completely misses, which "
            "is exactly why transformer similarity pairs produce tighter same-incident "
            "clusters in the correlation stage."
        )

        # --- Full raw evaluation report ---
        report_path = os.path.join(OUTPUTS_PATH, "evaluation_report.txt")
        if os.path.exists(report_path):
            with open(report_path) as f:
                report_text = f.read()
            with st.expander("Full evaluation report"):
                st.text(report_text)
        else:
            _show_missing("evaluation_report.txt", "python src/evaluate.py")
