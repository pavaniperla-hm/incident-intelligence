"""
data_loader.py — Raw incident data ingestion.

Purpose:
    Reads raw incident data from disk (CSV or other formats) and returns a
    validated pandas DataFrame ready for downstream preprocessing.

Pipeline position: First stage — feeds into normaliser and preprocess.

Inputs:
    - File path (default: config.RAW_DATA_PATH → data/raw/incidents.csv)
    - Supported formats: CSV (primary), JSON (secondary)

Outputs:
    - pandas DataFrame with raw incident records; schema:
      incident_id, timestamp, source, category, description, severity
    - Raises FileNotFoundError or ValueError on malformed input
"""

# TODO: implement
