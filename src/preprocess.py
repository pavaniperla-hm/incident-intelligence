"""
preprocess.py — NLP tokenisation, stopword removal, and lemmatisation.

Purpose:
    Applies NLP preprocessing to normalised incident text before vectorisation
    or transformer encoding.  Tokenises each signal's ``normalised_text``,
    removes stopwords (with an explicit allowlist of incident-critical terms),
    lemmatises ordinary tokens, and preserves the semantic replacement tokens
    produced by the normaliser (PCTVAL, TIMEVAL, etc.) verbatim so they remain
    as first-class vocabulary items for downstream TF-IDF or transformer models.

Pipeline position: After normaliser, before classifier and similarity.

Inputs:
    - Normalised pandas DataFrame from normaliser with a ``normalised_text``
      column (output of ``normaliser.normalise_dataframe``).
    - config.INCIDENT_CATEGORIES — used only for documentation / validation.

Outputs:
    - DataFrame with a new ``preprocessed_text`` column containing
      space-joined lemmatised token strings.
    - Saved to data/processed/processed_incidents.csv when run as a script.
"""

import os
import sys
from collections import Counter

import pandas as pd

# ---------------------------------------------------------------------------
# Path fix — allows running this file directly as a script from any directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# NLTK — quiet downloads so the module can be imported in any environment
# ---------------------------------------------------------------------------
import nltk  # noqa: E402

for _resource in ("stopwords", "punkt", "punkt_tab", "wordnet", "omw-1.4"):
    nltk.download(_resource, quiet=True)

from nltk.stem import WordNetLemmatizer       # noqa: E402
from nltk.tokenize import word_tokenize       # noqa: E402
from nltk.corpus import stopwords as _sw_corpus  # noqa: E402

from config import INCIDENT_CATEGORIES  # noqa: E402  # noqa: F401 (used in docstring)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Tokens produced by the normaliser that must survive preprocessing unchanged.
PRESERVE_TOKENS = {
    "PCTVAL", "TIMEVAL", "NUMVAL", "VERVAL",
    "IPADDR", "EXCEPTION",
}

# Words that NLTK marks as stopwords but carry real meaning in incident context.
# Removing them from the stopword set ensures they are not discarded.
INCIDENT_CRITICAL_WORDS = {
    "down", "up", "failed", "failure", "error", "invalid",
    "not", "no", "cannot", "unable", "unauthorized", "blocked",
    "dropped", "exceeded", "denied", "refused", "timeout",
    "critical", "warning", "high", "low", "slow", "fast",
}

# Module-level lemmatiser instance — shared across calls to avoid repeated
# construction (WordNetLemmatizer is stateless so sharing is safe).
_LEMMATIZER = WordNetLemmatizer()

# Lazy cache for the stopword set — built once on first call, reused thereafter.
_STOPWORDS_CACHE: set = None


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def build_stopwords() -> set:
    """Build a domain-aware stopword set for incident text.

    Loads the NLTK English stopword list and removes every word that appears
    in ``INCIDENT_CRITICAL_WORDS``.  This ensures that incident-relevant
    terms like ``"failed"``, ``"timeout"``, and ``"unauthorized"`` are never
    silently discarded during tokenisation.

    Returns:
        A ``set`` of lowercase stopword strings with incident-critical terms
        excluded.

    Side effects:
        Prints a one-line summary: the size of the final set and how many
        NLTK default stopwords were retained / excluded.

    Example::

        sw = build_stopwords()
        assert "failed" not in sw   # preserved
        assert "the" in sw          # still removed
    """
    global _STOPWORDS_CACHE
    if _STOPWORDS_CACHE is not None:
        return _STOPWORDS_CACHE

    base = set(_sw_corpus.words("english"))
    removed = base & INCIDENT_CRITICAL_WORDS
    domain_stopwords = base - INCIDENT_CRITICAL_WORDS
    print(
        f"Stopword list: {len(domain_stopwords)} words "
        f"(NLTK default minus {len(removed)} incident-critical terms)"
    )
    _STOPWORDS_CACHE = domain_stopwords
    return domain_stopwords


def tokenise_and_filter(text: str, stopwords: set) -> list:
    """Tokenise a single normalised text string and apply filtering rules.

    Processing order for each token:

    1. If the token is in ``PRESERVE_TOKENS`` — keep it exactly as-is
       (no casing change, no lemmatisation).
    2. If the token is a single character — discard it.
    3. If the lowercase form of the token is in ``stopwords`` — discard it.
    4. Otherwise — lemmatise with ``WordNetLemmatizer.lemmatize`` (noun mode,
       which is NLTK's default and appropriate for technical vocabulary) and
       keep the result.

    Args:
        text:      A normalised incident description string (output of
                   ``normaliser.normalise_text``).
        stopwords: Set of stopword strings to filter out, as returned by
                   :func:`build_stopwords`.

    Returns:
        Ordered list of retained token strings.  Preserve-tokens retain their
        original uppercase form; all other tokens are in lowercase after
        lemmatisation.

    Example::

        tokens = tokenise_and_filter(
            "auth_error_rate exceeded threshold PCTVAL",
            build_stopwords(),
        )
        # → ["auth_error_rate", "exceeded", "threshold", "PCTVAL"]
    """
    raw_tokens = word_tokenize(text)
    result = []

    for token in raw_tokens:
        # Rule 1 — preserve normaliser tokens verbatim.
        if token in PRESERVE_TOKENS:
            result.append(token)
            continue

        # Rule 2 — discard single characters (punctuation remnants, initials).
        if len(token) <= 1:
            continue

        # Rule 3 — discard stopwords (compare lowercase so "The" == "the").
        if token.lower() in stopwords:
            continue

        # Rule 4 — lemmatise and keep (output is already lowercase from
        # normaliser, but lemmatize handles any case correctly).
        result.append(_LEMMATIZER.lemmatize(token.lower()))

    return result


def preprocess_text(text: str, stopwords: set = None) -> str:
    """Preprocess a single incident description string end-to-end.

    Convenience wrapper that optionally builds stopwords if none are supplied,
    delegates to :func:`tokenise_and_filter`, then joins the token list back
    into a whitespace-separated string ready for TF-IDF or transformer input.

    Args:
        text:      Normalised incident description string.
        stopwords: Pre-built stopword set from :func:`build_stopwords`.  If
                   ``None``, stopwords are built on every call — callers
                   processing many rows should build the set once and pass it
                   in to avoid redundant construction.

    Returns:
        Space-joined string of filtered, lemmatised tokens.

    Example::

        result = preprocess_text(
            "auth_error_rate exceeded threshold PCTVAL",
        )
        # → "auth_error_rate exceeded threshold PCTVAL"
    """
    if stopwords is None:
        stopwords = build_stopwords()
    tokens = tokenise_and_filter(text, stopwords)
    return " ".join(tokens)


def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = "normalised_text",
) -> pd.DataFrame:
    """Apply NLP preprocessing to every row of a normalised signals DataFrame.

    Builds the stopword set once, then iterates over ``text_column`` row by
    row, calling :func:`preprocess_text` for each entry.  Progress is printed
    every 50 rows so long-running batches remain observable.

    Args:
        df:          Normalised pandas DataFrame containing ``text_column``.
        text_column: Name of the column holding normalised text to preprocess.
                     Defaults to ``"normalised_text"`` (the column produced by
                     ``normaliser.normalise_dataframe``).

    Returns:
        The same DataFrame with a new ``preprocessed_text`` column appended.
        The source ``text_column`` is preserved unchanged.

    Raises:
        KeyError: If ``text_column`` is not present in ``df``.
    """
    stopwords = build_stopwords()
    total = len(df)
    preprocessed = []

    for i, text in enumerate(df[text_column], start=1):
        preprocessed.append(preprocess_text(text, stopwords))
        if i % 50 == 0:
            print(f"Processed {i}/{total}...")

    df = df.copy()
    df["preprocessed_text"] = preprocessed
    print(f"Preprocessing complete — {total} signals")
    return df


def get_token_stats(
    df: pd.DataFrame,
    text_column: str = "preprocessed_text",
) -> dict:
    """Compute vocabulary statistics over the preprocessed text column.

    Calculates per-signal token counts and a global frequency distribution,
    then prints a formatted report.  Useful for sanity-checking vocabulary
    size and confirming that normaliser tokens survive preprocessing.

    Metrics computed:

    - Average token count per signal (rounded to 2 decimal places).
    - Minimum and maximum token count across all signals.
    - Top 10 most frequent tokens across the entire corpus.
    - Occurrence count of each token in ``PRESERVE_TOKENS``.

    Args:
        df:          DataFrame containing a preprocessed text column.
        text_column: Name of the column to analyse.  Defaults to
                     ``"preprocessed_text"``.

    Returns:
        Dict with keys:
            ``avg_tokens``      — float, mean tokens per signal.
            ``min_tokens``      — int, shortest signal token count.
            ``max_tokens``      — int, longest signal token count.
            ``top_10_tokens``   — list of (token, count) tuples.
            ``preserve_counts`` — dict mapping each PRESERVE_TOKEN → int count.
    """
    all_tokens: list = []
    token_counts: list = []

    for text in df[text_column]:
        tokens = text.split()
        token_counts.append(len(tokens))
        all_tokens.extend(tokens)

    counter = Counter(all_tokens)
    top_10 = counter.most_common(10)
    preserve_counts = {tok: counter.get(tok, 0) for tok in sorted(PRESERVE_TOKENS)}

    avg_tokens = round(sum(token_counts) / len(token_counts), 2) if token_counts else 0.0
    min_tokens = min(token_counts) if token_counts else 0
    max_tokens = max(token_counts) if token_counts else 0

    print("\n=== Token Statistics ===")
    print(f"  Signals analysed  : {len(df)}")
    print(f"  Avg tokens/signal : {avg_tokens}")
    print(f"  Min tokens        : {min_tokens}")
    print(f"  Max tokens        : {max_tokens}")
    print(f"  Unique tokens     : {len(counter)}")
    print("\n  Top 10 tokens:")
    for rank, (token, count) in enumerate(top_10, start=1):
        print(f"    {rank:2d}. {token:<30s} {count}")
    print("\n  Preserve-token occurrences:")
    for token, count in preserve_counts.items():
        print(f"    {token:<12s} {count}")
    print()

    return {
        "avg_tokens": avg_tokens,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "top_10_tokens": top_10,
        "preserve_counts": preserve_counts,
    }


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _project_root = os.path.join(os.path.dirname(__file__), "..")
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    from src.normaliser import normalise_dataframe  # noqa: E402

    raw_path = os.path.join(_project_root, "data", "raw", "incidents.csv")
    df_raw = pd.read_csv(raw_path)
    df_norm = normalise_dataframe(df_raw.copy())
    df_proc = preprocess_dataframe(df_norm)

    # --- Before / after comparison — one example per source type -----------
    print("\n=== Before / After Preprocessing (one example per source type) ===\n")
    for source in ["alert", "ticket", "log"]:
        mask = df_norm["source_type"] == source
        idx = df_norm[mask].index[0]
        norm_text = df_norm.loc[idx, "normalised_text"]
        prep_text = df_proc.loc[idx, "preprocessed_text"]
        print(f"=== After normalisation ===")
        print(f"[{source}]  {norm_text}\n")
        print(f"=== After preprocessing ===")
        print(f"[{source}]  {prep_text}\n")
        print("-" * 70 + "\n")

    # --- Token statistics --------------------------------------------------
    get_token_stats(df_proc)

    # --- Save processed DataFrame ------------------------------------------
    out_path = os.path.join(_project_root, "data", "processed", "processed_incidents.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_proc.to_csv(out_path, index=False)
    print(f"Saved to {out_path} — {len(df_proc)} rows, {len(df_proc.columns)} columns")
