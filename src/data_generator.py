"""
data_generator.py — Synthetic incident data generation.

Purpose:
    Generates synthetic incident records for development, testing, and
    experimentation when real operational data is unavailable.  Produces
    correlated multi-signal bursts per incident (alert + ticket + log) so
    the dataset reflects realistic operational noise, including deliberate
    cross-category vocabulary mixing to simulate ambiguity.

Pipeline position: Upstream of all other stages (pre-ingestion).

Inputs:
    - config constants: INCIDENT_CATEGORIES, SOURCE_WEIGHTS, RAW_DATA_PATH
    - Optional parameters: num_incidents_per_category, signals_per_incident

Outputs:
    - pandas DataFrame with columns:
      signal_id, source_type, category, text, timestamp, incident_group
    - CSV written to RAW_DATA_PATH (data/raw/incidents.csv)
"""

import os
import sys
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path fix — allows running this file directly as a script from any directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import RAW_DATA_PATH, INCIDENT_CATEGORIES, SOURCE_WEIGHTS  # noqa: E402

# ---------------------------------------------------------------------------
# Vocabulary banks
# ---------------------------------------------------------------------------

CATEGORY_VOCABULARY = {
    "Authentication Failure": {
        "alert": [
            "CRITICAL: login_success_rate dropped from {ok}% to {fail}% in last {mins} minutes",
            "ALERT: auth_error_rate exceeded threshold — current value {fail}% up from baseline {ok}%",
            "WARNING: {fail} failed authentication attempts detected in {mins} minute window",
            "CRITICAL: SSO service returning 401 errors — {fail}% of requests failing",
            "ALERT: token_validation_failures spiked to {fail} per minute — normal is {ok}",
            "CRITICAL: MFA service timeout rate {fail}% — investigation required",
        ],
        "ticket": [
            "Users unable to log in since {time} — receiving invalid credentials error",
            "Multiple users reporting authentication failures — token validation not working",
            "Cannot access internal portal — getting 401 Unauthorized since this morning",
            "SSO login broken for {fail} users — IT please investigate urgently",
            "Password reset not working — users locked out of system since {time}",
            "VPN authentication failing for remote workers — JWT token expired error",
        ],
        "log": [
            "ERROR AuthService TokenValidationException: JWT signature invalid at {time}",
            "WARN auth-gateway 401 Unauthorized /api/v2/login — {fail} occurrences in {mins}m",
            "ERROR sso-service Connection refused to identity-provider at {time}",
            "CRITICAL auth-middleware NullPointerException in validateToken() line 847",
            "ERROR oauth2-handler Token expired: issued {mins} minutes ago exceeds limit",
            "WARN ldap-connector Authentication bind failed for user pool — retrying",
        ],
    },
    "Network Outage": {
        "alert": [
            "CRITICAL: packet_loss_rate {fail}% on core router — threshold is {ok}%",
            "ALERT: network_latency_ms spiked to {fail}ms — baseline {ok}ms",
            "CRITICAL: {fail} downstream services unreachable — network partition detected",
            "WARNING: BGP session dropped with peer {ok} — rerouting traffic",
            "CRITICAL: datacenter east-{ok} connectivity lost — failover initiated",
            "ALERT: DNS resolution failures {fail}% — external connectivity degraded",
        ],
        "ticket": [
            "Cannot reach external services since {time} — complete outage for team",
            "Internet connectivity down in office building {ok} — no network access",
            "API calls to external partners timing out — network issue suspected since {time}",
            "VPN connection dropping every {mins} minutes — unable to work remotely",
            "Database connections failing — network between app and DB unreachable",
            "All microservices returning 503 — suspected network partition in cluster",
        ],
        "log": [
            "ERROR network-monitor ConnectionTimeoutException host unreachable at {time}",
            "WARN load-balancer upstream {fail} of {ok} nodes marked unhealthy",
            "ERROR dns-resolver NXDOMAIN for internal service mesh — {mins} retries failed",
            "CRITICAL router-agent Interface GigabitEthernet0/{ok} down — link failure",
            "ERROR tcp-handler Connection reset by peer — {fail} retries exhausted",
            "WARN service-mesh Circuit breaker OPEN for {fail} services simultaneously",
        ],
    },
    "Deployment Failure": {
        "alert": [
            "CRITICAL: deployment pipeline {ok} failed at stage {fail} — rollback triggered",
            "ALERT: error_rate spiked to {fail}% immediately after release v{ok}",
            "WARNING: health checks failing for {fail} of {ok} new pods after deploy",
            "CRITICAL: canary deployment showing {fail}% error rate — halting rollout",
            "ALERT: container startup failures {fail} in last {mins} minutes after push",
            "CRITICAL: kubernetes rollout failed — {fail} pods in CrashLoopBackOff",
        ],
        "ticket": [
            "Production deployment at {time} caused service outage — need rollback",
            "New release broke authentication flow — users getting 500 errors since deploy",
            "Deployment to prod failed — application not starting after update at {time}",
            "Config change pushed at {time} caused database connection failures",
            "Docker container crashing on startup after latest push — logs show OOM error",
            "CI/CD pipeline deployment stuck for {mins} minutes — manual intervention needed",
        ],
        "log": [
            "ERROR kubernetes PodInitializing timeout after {mins}m — CrashLoopBackOff",
            "FATAL application startup failed: missing environment variable DATABASE_URL",
            "ERROR helm-deploy Release {ok} failed: timed out waiting for condition",
            "CRITICAL docker-runtime Container exited with code {fail} immediately on start",
            "ERROR config-loader Failed to parse configuration: invalid YAML at line {fail}",
            "WARN rollout-controller Deployment {ok} exceeded progress deadline {mins}m",
        ],
    },
    "Performance Degradation": {
        "alert": [
            "WARNING: api_response_time_p99 is {fail}ms — SLA threshold is {ok}ms",
            "ALERT: cpu_utilization at {fail}% on {ok} nodes — autoscaling triggered",
            "CRITICAL: database_query_time_avg {fail}ms — {ok}x slower than baseline",
            "WARNING: memory_usage {fail}% — approaching OOM threshold on {ok} pods",
            "ALERT: throughput dropped from {ok} to {fail} requests/sec — capacity issue",
            "CRITICAL: cache_hit_rate dropped to {fail}% — database under heavy load",
        ],
        "ticket": [
            "Application extremely slow since {time} — pages taking {fail} seconds to load",
            "API response times unacceptable — {fail}ms average reported by users since {time}",
            "Dashboard loading very slowly — timeout errors appearing for heavy reports",
            "Database queries taking {fail} seconds — was {ok} seconds yesterday",
            "System unusable during peak hours — {fail} second response times",
            "Memory leak suspected — application slowing down every {mins} minutes",
        ],
        "log": [
            "WARN query-executor Slow query detected: {fail}ms for SELECT on orders table",
            "ERROR jvm-monitor GC overhead limit exceeded — heap usage {fail}%",
            "WARN connection-pool All {ok} connections in use — requests queuing",
            "CRITICAL thread-pool {fail} threads blocked — deadlock suspected at {time}",
            "WARN cache-manager Redis response time {fail}ms — cache performance degraded",
            "ERROR memory-manager OutOfMemoryError: Java heap space after {mins} minutes",
        ],
    },
    "Security Breach": {
        "alert": [
            "CRITICAL: {fail} failed login attempts from IP {ok} — brute force detected",
            "ALERT: unusual data access pattern — {fail}GB exported in {mins} minutes",
            "CRITICAL: privilege escalation attempt detected for user account at {time}",
            "WARNING: {fail} API calls from unrecognised IP range in last {mins} minutes",
            "CRITICAL: sensitive endpoint accessed {fail} times outside business hours",
            "ALERT: SQL injection pattern detected in {fail} requests — WAF blocking",
        ],
        "ticket": [
            "Suspicious login from foreign IP at {time} — account may be compromised",
            "User reported receiving password reset email they did not request at {time}",
            "Unexpected admin account created at {time} — security team please investigate",
            "Unusual file access pattern detected — {fail} sensitive files accessed in {mins}m",
            "Possible data exfiltration — large upload to external endpoint at {time}",
            "Security scanner flagged {fail} critical vulnerabilities in production at {time}",
        ],
        "log": [
            "CRITICAL security-monitor Brute force: {fail} auth failures from {ok} in {mins}m",
            "ERROR waf-handler SQL injection blocked: malicious payload in request at {time}",
            "WARN audit-logger Privilege escalation: user {ok} accessed admin endpoint",
            "CRITICAL ids-system Port scan detected from external IP {ok} at {time}",
            "ERROR access-control Unauthorised access to /api/admin/users — blocked at {time}",
            "WARN data-monitor Anomalous export: {fail}MB from sensitive table in {mins}m",
        ],
    },
}

_TIME_WINDOWS = [5, 10, 15, 20, 30]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def fill_template(template: str) -> str:
    """Fill a text template with realistic random placeholder values.

    Replacements:
        {ok}   — healthy baseline integer (90–99), e.g. a success percentage
                 or healthy node count.
        {fail} — degraded/failure integer (2–45), e.g. an error count or
                 failure percentage.
        {mins} — a realistic time-window integer sampled from
                 [5, 10, 15, 20, 30] minutes.
        {time} — a wall-clock time string in HH:MM format chosen uniformly
                 across a 24-hour day, e.g. "09:47".

    Args:
        template: A string containing zero or more of the placeholders above.

    Returns:
        The template with every placeholder replaced by its random value.
        Placeholders that appear more than once receive independent values
        on each occurrence.
    """
    result = template

    # Replace each occurrence independently so repeated placeholders vary.
    while "{ok}" in result:
        result = result.replace("{ok}", str(random.randint(90, 99)), 1)

    while "{fail}" in result:
        result = result.replace("{fail}", str(random.randint(2, 45)), 1)

    while "{mins}" in result:
        result = result.replace("{mins}", str(random.choice(_TIME_WINDOWS)), 1)

    while "{time}" in result:
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        result = result.replace("{time}", f"{hour:02d}:{minute:02d}", 1)

    return result


def generate_incident_burst(
    incident_id: str,
    category: str,
    base_time: datetime,
    num_signals: int = 3,
) -> list:
    """Generate a correlated burst of signals that all belong to one incident.

    Each burst selects ``num_signals`` distinct source types (alert, ticket,
    log), picks a random template from the vocabulary for that category and
    source, fills it with realistic values, and scatters the signals across a
    realistic detection window (0–25 minutes after ``base_time``).

    A 20 % vocabulary-mixing step appends one sentence from a randomly chosen
    *different* category to simulate real-world ambiguity where, for example,
    a deployment failure also triggers authentication errors.

    Args:
        incident_id: Unique identifier string for this incident group, e.g.
            ``"INC-0042"``.
        category: One of the five incident categories defined in
            ``INCIDENT_CATEGORIES``.
        base_time: The earliest possible timestamp for signals in this burst.
        num_signals: Number of signals to generate (default 3).  Must not
            exceed the number of available source types (3).

    Returns:
        A list of ``num_signals`` dicts, each with keys:
            signal_id   — unique identifier, e.g. ``"INC-0042-S0"``
            source_type — one of ``"alert"``, ``"ticket"``, ``"log"``
            category    — the true incident category
            text        — the filled (and possibly mixed) description string
            timestamp   — a ``datetime`` within 25 minutes of ``base_time``
            incident_group — same as ``incident_id`` for all signals in burst
    """
    source_types = random.sample(["alert", "ticket", "log"], k=num_signals)
    other_categories = [c for c in INCIDENT_CATEGORIES if c != category]
    signals = []

    for idx, source_type in enumerate(source_types):
        template = random.choice(CATEGORY_VOCABULARY[category][source_type])
        text = fill_template(template)

        # Vocabulary mixing: 20 % chance to append a sentence from another category.
        if random.random() < 0.20:
            noise_category = random.choice(other_categories)
            noise_source = random.choice(["alert", "ticket", "log"])
            noise_template = random.choice(
                CATEGORY_VOCABULARY[noise_category][noise_source]
            )
            noise_sentence = fill_template(noise_template)
            text = f"{text}. Additionally: {noise_sentence}"

        # Scatter within a 25-minute burst window.
        offset_minutes = random.uniform(0, 25)
        timestamp = base_time + timedelta(minutes=offset_minutes)

        signals.append(
            {
                "signal_id": f"{incident_id}-S{idx}",
                "source_type": source_type,
                "category": category,
                "text": text,
                "timestamp": timestamp,
                "incident_group": incident_id,
            }
        )

    return signals


def generate_dataset(
    num_incidents_per_category: int = 8,
    signals_per_incident: int = 3,
) -> pd.DataFrame:
    """Generate a full synthetic incident dataset spanning 30 days.

    For each of the five incident categories, ``num_incidents_per_category``
    incident bursts are created, each producing ``signals_per_incident``
    signals.  Burst base-times are distributed randomly across January 2024
    to give a realistic time-series shape.  The final DataFrame is sorted
    chronologically by timestamp.

    Minimum dataset size with defaults:
        5 categories × 8 incidents × 3 signals = 120 signals.

    Args:
        num_incidents_per_category: Number of distinct incident bursts per
            category (default 8).
        signals_per_incident: Number of signals (rows) per incident burst
            (default 3, max 3 due to three source types).

    Returns:
        pandas DataFrame with columns:
            signal_id, source_type, category, text, timestamp, incident_group
        Rows are sorted ascending by timestamp.
    """
    start_date = datetime(2024, 1, 1)
    period_days = 30

    all_signals: list = []
    incident_counter = 0

    for category in INCIDENT_CATEGORIES:
        for _ in range(num_incidents_per_category):
            incident_counter += 1
            incident_id = f"INC-{incident_counter:04d}"

            # Random base time within the 30-day window.
            offset_seconds = random.uniform(0, period_days * 24 * 3600)
            base_time = start_date + timedelta(seconds=offset_seconds)

            burst = generate_incident_burst(
                incident_id=incident_id,
                category=category,
                base_time=base_time,
                num_signals=signals_per_incident,
            )
            all_signals.extend(burst)

    df = pd.DataFrame(all_signals)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    total_signals = len(df)
    total_incidents = df["incident_group"].nunique()
    total_categories = df["category"].nunique()
    print(
        f"Generated {total_signals} signals across "
        f"{total_incidents} incidents ({total_categories} categories)"
    )

    return df


def save_dataset(df: pd.DataFrame, path: str = None) -> None:
    """Serialise the dataset to a CSV file.

    Creates any missing parent directories before writing.  If ``path`` is
    not provided, the destination defaults to ``RAW_DATA_PATH`` from config
    (``data/raw/incidents.csv`` relative to the project root).

    Args:
        df:   The DataFrame returned by :func:`generate_dataset`.
        path: Destination file path.  Defaults to ``RAW_DATA_PATH``.
    """
    if path is None:
        path = RAW_DATA_PATH

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False)

    num_rows, num_cols = df.shape
    print(f"Dataset saved to {path} — {num_rows} rows, {num_cols} columns")


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Ensure project root is on the path when running as a script.
    _project_root = os.path.join(os.path.dirname(__file__), "..")
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    random.seed(42)
    np.random.seed(42)

    df = generate_dataset(num_incidents_per_category=10, signals_per_incident=3)
    save_dataset(df)

    print("\n--- First 3 rows ---")
    with pd.option_context("display.max_colwidth", 80):
        print(df.head(3).to_string(index=False))

    print("\n--- Category distribution ---")
    print(df["category"].value_counts().to_string())

    print("\n--- Source type distribution ---")
    print(df["source_type"].value_counts().to_string())
