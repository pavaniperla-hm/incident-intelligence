"""
anomaly.py — Statistical anomaly detection on incident time-series.

Purpose:
    Detects anomalous spikes in incident volume using z-score analysis over
    hourly time buckets.  Each signal is scored 0–1 based on how unusual the
    activity level was during its arrival hour, with an additional multiplier
    applied to off-hours signals (outside 08:00–17:59, Mon–Fri) to reflect the
    higher operational severity of incidents that occur outside business hours.

Pipeline position: Parallel to classifier/similarity; feeds into scorer.

Inputs:
    - Preprocessed pandas DataFrame from preprocess / normaliser with columns:
      signal_id, hour, is_business_hours (added by normaliser.parse_timestamps)
    - config.ANOMALY_ZSCORE_THRESHOLD — z-score cutoff for is_anomalous flag

Outputs:
    - DataFrame with four new columns appended:
        z_score       — standardised hourly activity score (clipped ±3)
        anomaly_score — normalised score in [0.0, 1.0] with off-hours boost
        is_anomalous  — bool flag: True when |z_score| ≥ threshold
        time_context  — "business_hours" or "off_hours"
    - Saved to data/processed/scored_incidents.csv when run as a script
"""

import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path fix — allows running this file directly as a script from any directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ANOMALY_ZSCORE_THRESHOLD, PROCESSED_PATH  # noqa: E402


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def compute_hourly_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the mean and standard deviation of per-hour signal counts.

    Groups all signals by their ``hour`` column (integer 0–23), counts how
    many signals arrived in each hour bucket across the full dataset, then
    calculates the global mean and std of those counts.  Hours with zero
    signals are included so the baseline reflects the full 24-hour cycle.

    Args:
        df: DataFrame containing at least an ``hour`` column (integer 0–23)
            as produced by ``normaliser.parse_timestamps``.

    Returns:
        DataFrame with columns:
            ``hour``  — integer 0–23
            ``count`` — number of signals that arrived in that hour
            ``mean``  — scalar mean of counts broadcast to every row
            ``std``   — scalar std of counts broadcast to every row

    Side effects:
        Prints a one-line summary of mean and std.
    """
    # Count signals per hour; reindex to ensure all 24 hours are represented.
    counts = (
        df.groupby("hour")
        .size()
        .reindex(range(24), fill_value=0)
        .reset_index()
    )
    counts.columns = ["hour", "count"]

    global_mean = float(counts["count"].mean())
    global_std = float(counts["count"].std(ddof=0))  # population std

    counts["mean"] = global_mean
    counts["std"] = global_std

    print(
        f"Hourly baseline computed — avg {global_mean:.1f} signals/hour, "
        f"std {global_std:.1f}"
    )
    return counts


def compute_zscore(current_count: int, mean: float, std: float) -> float:
    """Calculate a clipped z-score for a single hourly signal count.

    Measures how many standard deviations the observed count deviates from
    the hourly mean.  The result is clipped to [−3, 3] so that extreme
    outliers do not dominate downstream normalisation, and rounded to 4
    decimal places for readability in reports.

    Args:
        current_count: Number of signals observed in the target hour.
        mean:          Mean hourly signal count across the full dataset.
        std:           Standard deviation of hourly signal counts.

    Returns:
        Z-score as a float in [−3.0, 3.0], or ``0.0`` when ``std`` is zero
        (i.e. all hours have identical counts — no meaningful deviation).
    """
    if std == 0.0:
        return 0.0
    raw = (current_count - mean) / std
    clipped = float(np.clip(raw, -3.0, 3.0))
    return round(clipped, 4)


def score_signal_anomaly(
    signal: pd.Series,
    baseline: pd.DataFrame,
    df: pd.DataFrame,
) -> dict:
    """Compute an anomaly score for a single signal row.

    Looks up the baseline statistics for the signal's arrival hour, counts
    how many other signals arrived in that same hour (using the full
    DataFrame), computes the z-score, normalises it to [0, 1], and applies
    a 1.3× multiplier for off-hours signals to reflect their higher
    operational severity.

    Args:
        signal:   A single row from the processed signals DataFrame as a
                  ``pandas.Series``.  Must contain ``signal_id``, ``hour``,
                  and ``is_business_hours`` fields.
        baseline: The hourly baseline DataFrame returned by
                  :func:`compute_hourly_baseline`.
        df:       The full signals DataFrame used to count how many signals
                  share the same hour as ``signal``.

    Returns:
        Dict with keys:

        - ``signal_id``         — string identifier of the signal.
        - ``hour``              — integer hour (0–23) of the signal.
        - ``signals_this_hour`` — total signals in this hour across the dataset.
        - ``z_score``           — clipped z-score in [−3, 3].
        - ``anomaly_score``     — normalised score in [0.0, 1.0]; off-hours
                                  signals receive a 1.3× boost (capped at 1.0).
        - ``is_anomalous``      — ``True`` when ``|z_score| ≥ ANOMALY_ZSCORE_THRESHOLD``.
        - ``time_context``      — ``"business_hours"`` or ``"off_hours"``.
    """
    hour = int(signal["hour"])
    signals_this_hour = int((df["hour"] == hour).sum())

    row = baseline[baseline["hour"] == hour].iloc[0]
    mean = float(row["mean"])
    std = float(row["std"])

    z = compute_zscore(signals_this_hour, mean, std)
    anomaly_score = min(abs(z) / 3.0, 1.0)

    is_anomalous = abs(z) >= ANOMALY_ZSCORE_THRESHOLD

    time_context = (
        "business_hours" if signal["is_business_hours"] else "off_hours"
    )

    # Off-hours multiplier: incidents outside business hours are more severe.
    if time_context == "off_hours":
        anomaly_score = min(anomaly_score * 1.3, 1.0)

    return {
        "signal_id": signal["signal_id"],
        "hour": hour,
        "signals_this_hour": signals_this_hour,
        "z_score": z,
        "anomaly_score": round(anomaly_score, 4),
        "is_anomalous": is_anomalous,
        "time_context": time_context,
    }


def score_all_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute anomaly scores for every signal in the DataFrame.

    Builds the hourly baseline once, then iterates over every row to produce
    an anomaly score dict.  The four scoring columns are merged back into
    the DataFrame.

    Args:
        df: Preprocessed signals DataFrame with ``signal_id``, ``hour``, and
            ``is_business_hours`` columns (output of normaliser + preprocess).

    Returns:
        The input DataFrame with four new columns appended:
        ``z_score``, ``anomaly_score``, ``is_anomalous``, ``time_context``.

    Side effects:
        Prints a summary: total anomalous signals, mean anomaly score by
        time context, and the hour with the highest mean anomaly score.
    """
    baseline = compute_hourly_baseline(df)

    scores = [
        score_signal_anomaly(row, baseline, df)
        for _, row in df.iterrows()
    ]
    scores_df = pd.DataFrame(scores).set_index("signal_id")

    df = df.copy()
    df = df.set_index("signal_id")
    for col in ("z_score", "anomaly_score", "is_anomalous", "time_context"):
        df[col] = scores_df[col]
    df = df.reset_index()

    # --- Summary -----------------------------------------------------------
    n_anomalous = int(df["is_anomalous"].sum())
    bh_mask = df["time_context"] == "business_hours"
    oh_mask = df["time_context"] == "off_hours"
    avg_bh = df.loc[bh_mask, "anomaly_score"].mean() if bh_mask.any() else 0.0
    avg_oh = df.loc[oh_mask, "anomaly_score"].mean() if oh_mask.any() else 0.0

    peak_hour = (
        df.groupby("hour")["anomaly_score"]
        .mean()
        .idxmax()
    )

    print(f"\nAnomaly scoring complete:")
    print(f"  Anomalous signals      : {n_anomalous} / {len(df)}")
    print(f"  Avg score (biz hours)  : {avg_bh:.4f}")
    print(f"  Avg score (off hours)  : {avg_oh:.4f}")
    print(f"  Peak anomaly hour      : {peak_hour:02d}:00")

    return df


def get_incident_anomaly_score(incident_group: str, df: pd.DataFrame) -> float:
    """Return the maximum anomaly score across all signals in an incident group.

    Aggregates signal-level anomaly scores to an incident-level urgency score
    by taking the maximum — one highly anomalous signal is sufficient to flag
    the whole incident.  Used by the scorer to apply an anomaly boost to the
    composite confidence score.

    Args:
        incident_group: The ``incident_group`` identifier string, e.g.
                        ``"INC-0007"``.
        df:             Scored signals DataFrame containing ``incident_group``
                        and ``anomaly_score`` columns.

    Returns:
        Maximum ``anomaly_score`` float across all signals in the group, or
        ``0.0`` if the group is not found.
    """
    mask = df["incident_group"] == incident_group
    if not mask.any():
        return 0.0
    return float(df.loc[mask, "anomaly_score"].max())


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
    df = pd.read_csv(processed_csv)

    # Ensure normaliser-derived columns are present; rebuild if missing.
    if "hour" not in df.columns or "is_business_hours" not in df.columns:
        print("Adding temporal columns via normaliser...")
        from src.normaliser import normalise_dataframe
        df = normalise_dataframe(df)

    # Ensure is_business_hours is boolean (CSV read may load it as string).
    if df["is_business_hours"].dtype == object:
        df["is_business_hours"] = df["is_business_hours"].map(
            {"True": True, "False": False, True: True, False: False}
        )

    df_scored = score_all_signals(df)

    # --- Top 5 most anomalous signals --------------------------------------
    top5 = df_scored.nlargest(5, "anomaly_score")[
        ["signal_id", "hour", "time_context", "z_score", "anomaly_score", "is_anomalous"]
    ]
    print("\n=== Top 5 most anomalous signals ===")
    print(top5.to_string(index=False))

    # --- Business hours vs off-hours example comparison --------------------
    print("\n=== Anomaly score examples ===")
    bh_sample = df_scored[df_scored["time_context"] == "business_hours"].iloc[0]
    oh_sample = df_scored[df_scored["time_context"] == "off_hours"].iloc[0]

    bh_label = "normal" if not bh_sample["is_anomalous"] else "anomalous"
    oh_label = "normal" if not oh_sample["is_anomalous"] else "anomalous"

    print(
        f"Business hours signal ({bh_sample['hour']:02d}:00):  "
        f"anomaly_score = {bh_sample['anomaly_score']:.2f}  [{bh_label}]"
    )
    print(
        f"Off-hours signal      ({oh_sample['hour']:02d}:00):  "
        f"anomaly_score = {oh_sample['anomaly_score']:.2f}  [{oh_label}]"
    )

    # --- Save --------------------------------------------------------------
    out_path = os.path.join(
        _project_root, "data", "processed", "scored_incidents.csv"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_scored.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path} — {len(df_scored)} rows, {len(df_scored.columns)} columns")
