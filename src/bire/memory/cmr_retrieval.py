# ============================================================
# BIRE OS — CMR Retrieval Philosophy
# File: src/bire/memory/cmr_retrieval.py
# Chapter: 62.5
#
# Created: 2026-05-26
# Updated: 2026-05-26
#
# Purpose:
# Backend helper for CMR retrieval governance,
# historical uncertainty correlation,
# and replay-aware ecosystem memory retrieval.
# ============================================================

import pandas as pd


def build_cmr_retrieval_framework() -> pd.DataFrame:
    """
    Build the Collective Memory Repository retrieval framework.

    Returns
    -------
    pd.DataFrame
        CMR retrieval governance table.
    """

    return pd.DataFrame(
        {
            "retrieval_state": [

                "LOW_RELEVANCE_RETRIEVAL",

                "MODERATE_RELEVANCE_RETRIEVAL",

                "HIGH_RELEVANCE_RETRIEVAL",

                "REPLAY_REINFORCEMENT_RETRIEVAL",

                "SUPPRESSED_ESCALATION_RETRIEVAL",

                "FORECAST_REINFORCEMENT_RETRIEVAL"
            ],

            "retrieval_state_description": [

                "Archived instability weakly correlates with current concern.",

                "Historical instability partially reinforces current interpretation.",

                "Archived instability strongly correlates with current operational concern.",

                "Replay continuity strengthens instability interpretation.",

                "Prior suppression history increases escalation caution.",

                "Prior forecasting instability increases forecasting awareness."
            ],

            "operational_behavior": [

                "Minimal impact on active interpretation.",

                "Moderate reinforcement of operational concern.",

                "Strong reinforcement of operational concern awareness.",

                "Increase replay continuity confidence weighting.",

                "Increase escalation moderation awareness.",

                "Increase forecasting caution and instability awareness."
            ],

            "ecosystem_impact": [

                "Minimal retrieval influence.",

                "Moderate historical uncertainty influence.",

                "Strong retrospective instability reinforcement.",

                "Strengthens replay continuity reasoning.",

                "Strengthens suppression auditability.",

                "Strengthens forecasting traceability."
            ],

            "core_retrieval_principle": [

                "Weak retrieval should remain informative",

                "Moderate retrieval should reinforce awareness",

                "Strong retrieval should strengthen caution",

                "Replay retrieval should preserve continuity awareness",

                "Suppression retrieval should preserve escalation explainability",

                "Forecast retrieval should preserve forecasting auditability"
            ]
        }
    )

# ============================================================
# BIRE OS — Initial CMR Retrieval Mechanics
# File: src/bire/memory/cmr_retrieval.py
# Chapter: 62.6
#
# Created: 2026-05-26
# Updated: 2026-05-26
#
# Purpose:
# Initial operational CMR retrieval mechanics for archived
# uncertainty comparison, retrieval requests, and ecosystem
# memory reinforcement.
# ============================================================


def build_initial_cmr_archive(
    ncl_df: pd.DataFrame,
    reliability_col: str = "ncl_reliability_score",
    state_col: str = "ncl_signal_state",
    patient_col: str = "patient_id",
) -> pd.DataFrame:
    """
    Build an initial CMR archive from NCL output.

    The archive preserves rows where reliability degradation,
    volatility, uncertainty, or non-perfect trust may be useful
    for future retrieval.
    """

    out = ncl_df.copy()

    archive_df = out[
        (out[reliability_col] < 0.99)
        | (out[state_col] != "RELIABLE_SIGNAL")
        | (out.get("visibility_modifier", 0) > 0.01)
    ].copy()

    archive_df["cmr_archive_reason"] = "NCL_RELIABILITY_OR_UNCERTAINTY_FRAGMENT"
    archive_df["cmr_source_layer"] = "NCL"
    archive_df["cmr_active_interpretation"] = False

    return archive_df.reset_index(drop=True)


def retrieve_from_cmr(
    current_df: pd.DataFrame,
    cmr_archive_df: pd.DataFrame,
    patient_col: str = "patient_id",
    care_mode_col: str = "care_mode",
) -> pd.DataFrame:
    """
    Perform initial CMR retrieval.

    For each current row, retrieve archived uncertainty fragments
    for the same patient and care mode.
    """

    current = current_df.copy()
    archive = cmr_archive_df.copy()

    retrieval_summary = (
        archive.groupby([patient_col, care_mode_col])
        .agg(
            cmr_retrieved_fragments=("cmr_archive_reason", "count"),
            avg_retrieved_reliability=("ncl_reliability_score", "mean"),
            avg_retrieved_volatility=("volatility_score", "mean"),
            avg_retrieved_visibility=("visibility_modifier", "mean"),
        )
        .reset_index()
    )

    out = current.merge(
        retrieval_summary,
        on=[patient_col, care_mode_col],
        how="left",
    )

    fill_cols = [
        "cmr_retrieved_fragments",
        "avg_retrieved_reliability",
        "avg_retrieved_volatility",
        "avg_retrieved_visibility",
    ]

    for col in fill_cols:
        out[col] = out[col].fillna(0)

    out["cmr_retrieval_triggered"] = (
        out["cmr_retrieved_fragments"] > 0
    )

    out["cmr_retrieval_relevance"] = out.apply(
        lambda row: (
            "HIGH_RELEVANCE_RETRIEVAL"
            if row["cmr_retrieved_fragments"] >= 5
            else "MODERATE_RELEVANCE_RETRIEVAL"
            if row["cmr_retrieved_fragments"] >= 2
            else "LOW_RELEVANCE_RETRIEVAL"
            if row["cmr_retrieved_fragments"] == 1
            else "NO_RETRIEVAL_MATCH"
        ),
        axis=1,
    )

    return out