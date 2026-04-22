"""
scorer.py — Composite confidence score computation.

Purpose:
    Combines signals from the classifier (category confidence), anomaly detector
    (anomaly boost flag), and source normaliser (source reliability weight) into
    a single composite confidence score for each incident.  Higher scores
    indicate higher certainty about the incident's category and significance.

Pipeline position: After classifier, anomaly, and correlator; feeds into evaluate
    and visualise.

Inputs:
    - pandas Series of classifier confidence scores from classifier
    - pandas Series of anomaly boolean flags from anomaly
    - pandas Series of source_weight values from normaliser
    - config.CONFIDENCE_WEIGHTS — dict with keys: source_weight, classifier_conf,
      anomaly_boost

Outputs:
    - pandas Series of composite confidence scores in [0, 1], one per incident
    - Scores appended to the main DataFrame as a `confidence_score` column
"""

# TODO: implement
