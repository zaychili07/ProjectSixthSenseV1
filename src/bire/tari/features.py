# ============================================================
# BIRE OS — Trajectory-Aware Recovery Intelligence (TARI)
# File: src/bire/tari/features.py
# Chapter: 66
#
# Created: 2026-07-03
# Updated: 2026-07-03
#
# CHANGED:
# - Initialized TARI feature module
# - Prepared backend space for recovery governance features
# - Added framework for recovery trust formation
# - Separated TARI feature creation from scoring utilities
#
# Purpose:
# Backend helper for TARI recovery governance,
# recovery trust formation, recovery validation,
# and recovery evidence integration.
# ============================================================

import pandas as pd

def add_tari_recovery_trust_state(df):
    """
    Establish initial TARI recovery trust state.

    TARI governs recovery by accumulating
    completed intelligence across multiple
    intelligence layers.

    Recovery trust is earned.
    """

    out = df.copy()

    required_cols = [
    "ati_recovery_state",
    "recovery_authenticity_state",
    "hidden_instability_score",
    "lpmr_recovery_pattern_memory_state",
]

    missing_cols = [
        col for col in required_cols
        if col not in out.columns
    ]

    if missing_cols:
        out["tari_missing_inputs"] = ", ".join(missing_cols)
        return out

    out["tari_recovery_trust_state"] = (
        "RECOVERY_TRUST_UNVERIFIED"
    )

    # Recovery appears authentic
    authentic = (
        out["ati_recovery_state"]
        .eq("RESILIENT_RECOVERY")
    )

    # Hidden instability low
    stable = (
        out["hidden_instability_score"] < 0.30
    )

    # Previous recovery memory earned
    experienced = (
        out[
            "lpmr_recovery_pattern_memory_state"
        ].eq(
            "RECOVERY_PATTERN_MEMORY_EARNED"
        )
    )

    out.loc[
        authentic &
        stable &
        experienced,
        "tari_recovery_trust_state"
    ] = "RECOVERY_TRUST_PROBABLE"

    return out