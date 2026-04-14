"""
BIRE Configuration Module

Central location for all constants used across the pipeline.

Why this exists:
- Prevents hardcoding values throughout the codebase
- Makes experiments reproducible and tunable
- Keeps modeling assumptions explicit (e.g., window sizes, excluded columns)

This is especially important in clinical pipelines where small changes
can significantly affect downstream model behavior.
"""
RESAMPLE_FREQ = "5min"
WINDOW_SIZE = 6


SIGNAL_COLS = [
    "heart_rate",
    "resp_rate",
    "spo2",
    "temperature",
    "sbp",
    "dbp"
]

VALID_RANGES = {
    "heart_rate": (20, 260),
    "resp_rate": (5, 80),
    "spo2": (40, 100),
    "temperature": (30, 43),
    "sbp": (50, 250),
    "dbp": (30, 150)
}

# ------------------------------------------------------------------
# LEAKAGE PREVENTION (CRITICAL)
# ------------------------------------------------------------------
# These rules ensure that no future or target information leaks into
# the feature set. This is essential for realistic model evaluation,
# especially in time-series clinical prediction tasks.
# Columns to exclude from modeling
# These are identifiers or direct leakage sources
EXCLUDE_COLS = {
    "patient_id",
    "timestamp",
    "event_now",  # current event flag (leakage)
    "target",     # prediction target (must never be a feature)
}

# Patterns used to automatically detect leakage features
# Any column containing these substrings will be excluded
LEAKAGE_PATTERNS = [
    "target",
    "future",
    "lead",
    "next",
]
