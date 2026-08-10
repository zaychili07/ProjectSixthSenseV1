
# ============================================================
# BIRE OS — Initial CMR Retrieval Mechanics
# File: src/bire/noise/cmr.py
# Chapter: 62.6
#
# Created: 2026-05-30
# Updated: 2026-05-30
#
# Purpose:
# CMR stores noise, uncertainty, fragmentation, volatility,
# and low-trust signals that NCL does not fully pass forward,
# but preserves for later retrieval by intelligence layers.
# ============================================================
import pandas as pd


def build_collective_memory_repository(
    ncl_df: pd.DataFrame,
    patient_col: str = "patient_id",
    encounter_col: str = "encounter_id",
) -> pd.DataFrame:

    keep_cols = [
        patient_col,
        encounter_col,
        "ncl_pass",
        "ncl_signal_state",
        "ncl_pass_forward_flag",
        "ncl_pass_forward_type",
        "signal_quality_score",
        "continuity_score",
        "fragmentation_score",
        "volatility_score",
        "missingness_score",
        "ncl_reliability_score",
        "confidence_modifier",
        "forecasting_modifier",
        "escalation_modifier",
        "trajectory_modifier",
        "replay_modifier",
        "hidden_burden_modifier",
]

    available_cols = [col for col in keep_cols if col in ncl_df.columns]

    cmr_df = ncl_df.loc[
        ~ncl_df["ncl_pass_forward_flag"],
        available_cols
    ].copy()

    """
    Build the Collective Memory Repository.

    CMR stores noise, uncertainty, fragmentation, volatility,
    and low-trust signals that NCL does not fully pass forward,
    but preserves for later retrieval by intelligence layers.
    """

    df = ncl_df.copy()

    if "ncl_pass_forward_flag" not in df.columns:
        raise ValueError(
            "CMR requires 'ncl_pass_forward_flag'. "
            "Run NCL pass-forward logic before building CMR."
        )

    cmr_df = df.loc[~df["ncl_pass_forward_flag"]].copy()

    cmr_df["cmr_archive_flag"] = True

    cmr_df["cmr_retrieval_priority"] = (
        cmr_df.get("fragmentation_score", 0).fillna(0)
        + cmr_df.get("volatility_score", 0).fillna(0)
        + cmr_df.get("missingness_score", 0).fillna(0)
        + (1 - cmr_df.get("ncl_reliability_score", 1).fillna(1))
    )

    cmr_df["cmr_noise_quality"] = "LOW_IMPACT_NOISE"

    cmr_df.loc[
        cmr_df["cmr_retrieval_priority"] >= 0.75,
        "cmr_noise_quality"
    ] = "MODERATE_RETRIEVABLE_NOISE"

    cmr_df.loc[
        cmr_df["cmr_retrieval_priority"] >= 1.25,
        "cmr_noise_quality"
    ] = "HIGH_RETRIEVABLE_NOISE"

    cmr_df.loc[
        cmr_df["cmr_retrieval_priority"] >= 1.75,
        "cmr_noise_quality"
    ] = "CRITICAL_RECONSTRUCTION_NOISE"

    return cmr_df