# src/bire/config/settings.py
# This module contains centralized configuration settings for the BIRE intelligence system.
"""
BIRE Centralized Configuration System

This module contains centralized architectural configuration values
used across the BIRE intelligence system.

Purpose:
- improve maintainability
- reduce duplicated configuration logic
- standardize operational behavior
- support scalable architecture management
- improve reproducibility across notebooks and backend systems
"""

# =========================================================
# GLOBAL SYSTEM SETTINGS
# =========================================================

RANDOM_STATE = 42

RESAMPLE_FREQ = "5min"

DEFAULT_ROLLING_WINDOW = 6

# =========================================================
# EVENT / TARGET SETTINGS
# =========================================================

PREDICTION_HORIZONS = [15, 30, 60]

DEFAULT_TARGET_HORIZON = 60

# =========================================================
# GSS SETTINGS
# =========================================================

GSS_DEFAULT_COOLDOWN = 12

WATCH_THRESHOLD = 0.42

ESCALATE_THRESHOLD = 0.55

URGENT_THRESHOLD = 0.80

# =========================================================
# CONFIDENCE SETTINGS
# =========================================================

CONFIDENCE_WEIGHTS = {
    "horizon_stress": 0.40,
    "volatility_stress": 0.25,
    "acceleration_stress": 0.25,
    "evidence_stress": 0.10,
}

CONFIDENCE_POWER = 1.0

HIGH_CONFIDENCE_THRESHOLD = 0.95

MODERATE_CONFIDENCE_THRESHOLD = 0.80

# =========================================================
# MONITOR SETTINGS
# =========================================================

MONITOR_VOLATILITY_WINDOW = 6

RECOVERY_STABILITY_WINDOW = 12

# =========================================================
# EXPORT SETTINGS
# =========================================================

EXPORT_BASE_DIR = "outputs"

EXPORT_CSV_DIR = "csv"

EXPORT_LOG_DIR = "logs"

EXPORT_AUDIT_DIR = "audits"

# =========================================================
# LOGGING SETTINGS
# =========================================================

ENABLE_SYSTEM_LOGGING = True

ENABLE_AUDIT_EXPORTS = True

ENABLE_POLICY_AUDITS = True