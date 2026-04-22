"""
normaliser.py — Cross-source incident normalisation.

Purpose:
    Unifies the schema and cleans structural noise from raw signals before any
    ML processing.  Applies regex-based text normalisation to replace volatile
    numeric tokens (percentages, IPs, timestamps, version numbers, large
    integers, exception class names) with stable semantic tokens so that the
    TF-IDF vocabulary and transformer embeddings are not polluted by ephemeral
    values.  Also enriches the DataFrame with source-reliability weights and
    timestamp-derived features.

Pipeline position: After data_loader, before preprocess.

Inputs:
    - Raw pandas DataFrame from data_generator / data_loader with columns:
      signal_id, source_type, category, text, timestamp, incident_group
    - config.SOURCE_WEIGHTS — dict mapping source type → reliability weight
    - config.INCIDENT_CATEGORIES — list of valid category label strings

Outputs:
    - Normalised pandas DataFrame with additional columns:
        normalised_text  — cleaned description ready for vectorisation
        source_weight    — float reliability weight for the signal's source
        hour             — integer hour of day (0–23)
        day_of_week      — integer 0 (Monday) – 6 (Sunday)
        is_business_hours — bool, True if hour ∈ [8, 18) and weekday
"""

import os
import re
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path fix — allows running this file directly as a script from any directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import INCIDENT_CATEGORIES, SOURCE_WEIGHTS  # noqa: E402

# ---------------------------------------------------------------------------
# Compiled regex patterns (compiled once at import time for performance)
# ---------------------------------------------------------------------------

# Log-level prefixes at start of string or after a newline, followed by
# optional colon and whitespace.
_RE_LOG_LEVEL = re.compile(
    r"(?:^|(?<=\n))"
    r"(?:critical|error|warn(?:ing)?|alert|fatal|info)"
    r":?\s*",
    re.IGNORECASE,
)

# Percentage values — must come before the general integer replacement.
_RE_PERCENTAGE = re.compile(r"\d+\.?\d*%")

# Wall-clock time strings (HH:MM or HH:MM:SS).
_RE_TIME = re.compile(r"\b\d{2}:\d{2}(?::\d{2})?\b")

# Version strings like v1, v2.3, v1.0.4.
_RE_VERSION = re.compile(r"\bv\d+[\d.]*\b")

# IPv4 addresses.
_RE_IP = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")

# Standalone integers with 3 or more digits (after all other numeric patterns
# have already been replaced, so e.g. port numbers in IPs are already gone).
_RE_INTEGER = re.compile(r"\b\d{3,}\b")

# Java / Python exception and error class names.
_RE_EXCEPTION = re.compile(r"\b[A-Z][a-zA-Z]+(?:Exception|Error)\b")

# Characters to remove — keep alphanumeric, space, hyphen, slash, underscore.
_RE_SPECIAL = re.compile(r"[^\w\s\-/]")

# Collapse runs of whitespace.
_RE_SPACES = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def normalise_text(text: str, source_type: str) -> str:  # noqa: ARG001
    """Clean a single incident description string by replacing volatile tokens.

    Applies the following transformations in order so that later rules do not
    accidentally re-match artefacts introduced by earlier replacements:

    1. Lowercase the entire string.
    2. Strip leading log-level prefixes (CRITICAL, ERROR, WARN, WARNING, ALERT,
       FATAL, INFO) including optional trailing colon and whitespace.
    3. Replace percentage values (e.g. ``97%``, ``12.5%``) with ``PCTVAL``.
    4. Replace wall-clock time strings (``HH:MM`` or ``HH:MM:SS``) with
       ``TIMEVAL``.
    5. Replace version strings (``v1``, ``v2.3.1``) with ``VERVAL``.
    6. Replace IPv4 addresses with ``IPADDR``.
    7. Replace standalone integers of three or more digits with ``NUMVAL``.
    8. Replace Java / Python exception / error class names (e.g.
       ``NullPointerException``, ``OutOfMemoryError``) with ``EXCEPTION``.
    9. Remove all characters except alphanumerics, spaces, hyphens (``-``),
       forward slashes (``/``), and underscores (``_``).
    10. Collapse multiple consecutive whitespace characters into a single space.
    11. Strip leading and trailing whitespace.

    The ``source_type`` parameter is accepted for interface consistency with
    ``normalise_dataframe`` but is not used in the current implementation; all
    sources receive identical text cleaning.

    Args:
        text:        Raw incident description string.
        source_type: Source of the signal — one of ``"alert"``, ``"ticket"``,
                     ``"log"``.  Reserved for future source-specific rules.

    Returns:
        Cleaned, normalised string ready for downstream vectorisation.
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. Lowercase.
    text = text.lower()

    # 2. Strip log-level prefixes.
    text = _RE_LOG_LEVEL.sub("", text)

    # 3. Percentages → PCTVAL (before general integer replacement).
    text = _RE_PERCENTAGE.sub("PCTVAL", text)

    # 4. Time strings → TIMEVAL (before general integer replacement).
    text = _RE_TIME.sub("TIMEVAL", text)

    # 5. Version strings → VERVAL (before IP / integer replacement).
    text = _RE_VERSION.sub("VERVAL", text)

    # 6. IP addresses → IPADDR (before integer replacement to avoid partial
    #    matches on the numeric octets).
    text = _RE_IP.sub("IPADDR", text)

    # 7. Standalone 3+ digit integers → NUMVAL.
    text = _RE_INTEGER.sub("NUMVAL", text)

    # 8. Exception / Error class names → EXCEPTION.
    text = _RE_EXCEPTION.sub("EXCEPTION", text)

    # 9. Remove unwanted special characters.
    text = _RE_SPECIAL.sub(" ", text)

    # 10. Collapse whitespace.
    text = _RE_SPACES.sub(" ", text)

    # 11. Strip.
    return text.strip()


def add_source_weight(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``source_weight`` column based on the signal's source type.

    Maps each value in ``df["source_type"]`` to its reliability weight using
    ``SOURCE_WEIGHTS`` from config.  Unknown source types receive a weight of
    ``0.0`` so they are down-ranked rather than raising an error.

    Args:
        df: DataFrame containing at least a ``source_type`` column.

    Returns:
        The same DataFrame with a new ``source_weight`` float column appended.
        The original DataFrame is modified in-place and also returned.
    """
    df["source_weight"] = df["source_type"].map(SOURCE_WEIGHTS).fillna(0.0)
    return df


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the ``timestamp`` column and add temporal feature columns.

    Ensures the ``timestamp`` column is a proper ``datetime64`` dtype, then
    derives three new columns that capture temporal context relevant to
    anomaly detection and scoring:

    - ``hour``              — integer hour of the day (0–23).
    - ``day_of_week``       — integer weekday (0 = Monday … 6 = Sunday).
    - ``is_business_hours`` — ``True`` if ``hour ∈ [8, 18)`` **and**
                              ``day_of_week ∈ {0, 1, 2, 3, 4}`` (Mon–Fri).

    Args:
        df: DataFrame containing a ``timestamp`` column parseable by
            ``pandas.to_datetime``.

    Returns:
        The same DataFrame with the ``timestamp`` column coerced to
        ``datetime64[ns]`` and three new columns added.  The original
        DataFrame is modified in-place and also returned.
    """
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_business_hours"] = (
        df["hour"].between(8, 17) & df["day_of_week"].between(0, 4)
    )
    return df


def validate_schema(df: pd.DataFrame) -> bool:
    """Validate that the DataFrame conforms to the expected raw-signal schema.

    Performs four checks and prints a pass/fail report for each:

    1. **Required columns** — all of ``signal_id``, ``source_type``,
       ``category``, ``text``, ``timestamp``, ``incident_group`` must be
       present.
    2. **source_type values** — every value must be one of
       ``{"alert", "ticket", "log"}``.
    3. **category values** — every value must be in ``INCIDENT_CATEGORIES``
       from config.
    4. **No nulls** — none of the required columns may contain null values.

    Args:
        df: DataFrame to validate.

    Returns:
        ``True`` if every check passes, ``False`` otherwise.
    """
    required_columns = [
        "signal_id", "source_type", "category", "text", "timestamp", "incident_group",
    ]
    valid_sources = {"alert", "ticket", "log"}
    checks: dict[str, bool] = {}

    # Check 1 — required columns present.
    missing = [c for c in required_columns if c not in df.columns]
    checks["Required columns present"] = len(missing) == 0
    if missing:
        checks["Required columns present — missing"] = False  # type: ignore[assignment]

    # Check 2 — source_type values.
    if "source_type" in df.columns:
        bad_sources = set(df["source_type"].unique()) - valid_sources
        checks["source_type values valid"] = len(bad_sources) == 0
    else:
        checks["source_type values valid"] = False

    # Check 3 — category values.
    if "category" in df.columns:
        bad_categories = set(df["category"].unique()) - set(INCIDENT_CATEGORIES)
        checks["category values valid"] = len(bad_categories) == 0
    else:
        checks["category values valid"] = False

    # Check 4 — no nulls in required columns.
    if not missing:
        null_counts = df[required_columns].isnull().sum()
        checks["No nulls in required columns"] = bool(null_counts.sum() == 0)
    else:
        checks["No nulls in required columns"] = False

    # Print validation report.
    print("\n=== Schema Validation Report ===")
    all_passed = True
    for check_name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check_name}")
        if not passed:
            all_passed = False

    if all_passed:
        print(f"  All checks passed — {len(df)} rows, {len(df.columns)} columns\n")
    else:
        print("  One or more validation checks failed.\n")

    return all_passed


def normalise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrate the full normalisation pipeline for a raw signals DataFrame.

    Calls sub-functions in the correct order:

    1. :func:`validate_schema` — raises ``ValueError`` if any check fails so
       that corrupted data never reaches the ML stages.
    2. :func:`parse_timestamps` — coerce timestamps and add temporal features.
    3. :func:`add_source_weight` — add reliability weight per source type.
    4. :func:`normalise_text` — apply row-wise text cleaning, storing the
       result in a new ``normalised_text`` column (the original ``text``
       column is preserved for inspection / debugging).

    Args:
        df: Raw pandas DataFrame produced by data_generator / data_loader.

    Returns:
        Enriched DataFrame with four new columns appended:
        ``normalised_text``, ``source_weight``, ``hour``, ``day_of_week``,
        ``is_business_hours``.

    Raises:
        ValueError: If :func:`validate_schema` returns ``False``.
    """
    if not validate_schema(df):
        raise ValueError(
            "Schema validation failed — fix the input DataFrame before normalising."
        )

    df = parse_timestamps(df)
    df = add_source_weight(df)

    df["normalised_text"] = df.apply(
        lambda row: normalise_text(row["text"], row["source_type"]), axis=1
    )

    print(f"Normalisation complete — {len(df)} signals processed")
    return df


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _project_root = os.path.join(os.path.dirname(__file__), "..")
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    raw_path = os.path.join(_project_root, "data", "raw", "incidents.csv")
    df_raw = pd.read_csv(raw_path)

    df_norm = normalise_dataframe(df_raw.copy())

    # --- Before / after comparison — one example per source type -----------
    print("=== Before / After Normalisation (one example per source type) ===\n")
    for source in ["alert", "ticket", "log"]:
        sample = df_raw[df_raw["source_type"] == source].iloc[0]
        original = sample["text"]
        normalised = normalise_text(original, source)
        print(f"=== Before normalisation ===")
        print(f"[{source}]  {original}\n")
        print(f"=== After normalisation ===")
        print(f"[{source}]  {normalised}\n")
        print("-" * 70 + "\n")

    # --- New columns preview -----------------------------------------------
    print("=== New columns — source_weight, hour, is_business_hours (5 rows) ===")
    preview_cols = ["signal_id", "source_type", "source_weight", "hour", "is_business_hours"]
    print(df_norm[preview_cols].head(5).to_string(index=False))
