# ============================================================
# BIRE OS — Initial NCL Installation Layer
# File: src/bire/noise/ncl.py
# Chapter: 61.9
#
# Created: 2026-05-26
# Updated: 2026-05-30
# ADDED: lines: 184-213, def add_ncl_pass_forward_logic
# Reason: This gives NCL the threshold responsibility.
# Purpose:
# Initial operational NCL installation for reliability scoring,
# signal state classification, and downstream modifier generation.
# ============================================================

import pandas as pd
import numpy as np


def apply_initial_ncl_layer(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    sequence_col: str = "encounter_sequence",
) -> pd.DataFrame:
    """
    Apply the first operational Noise Complaint Layer.

    This function creates:
    - signal_quality_score
    - continuity_score
    - fragmentation_score
    - volatility_score
    - missingness_score
    - ncl_reliability_score
    - ncl_signal_state
    - confidence_modifier
    - forecasting_modifier
    - escalation_modifier
    - trajectory_modifier
    - replay_modifier
    - hidden_burden_modifier
    - visibility_modifier

    Parameters
    ----------
    df : pd.DataFrame
        Input trajectory dataframe.
    patient_col : str
        Patient identifier column.
    sequence_col : str
        Longitudinal ordering column.

    Returns
    -------
    pd.DataFrame
        Dataframe with initial NCL reliability fields attached.
    """

    out = df.copy()

    required_cols = [
        patient_col,
        sequence_col,
        "hidden_instability_score",
        "deterioration_tendency",
        "deceptive_stability_index",
        "hospital_system_failure_pressure",
        "stabilization_durability_score",
    ]

    missing_required = [col for col in required_cols if col not in out.columns]
    if missing_required:
        raise ValueError(
            f"Missing required columns for NCL installation: {missing_required}"
        )

    out = out.sort_values([patient_col, sequence_col]).reset_index(drop=True)

    signal_cols = [
        "hidden_instability_score",
        "deterioration_tendency",
        "deceptive_stability_index",
        "hospital_system_failure_pressure",
        "stabilization_durability_score",
    ]

    # --------------------------------------------------------
    # Missingness scoring
    # --------------------------------------------------------
    out["missingness_score"] = (
        out[signal_cols]
        .isna()
        .mean(axis=1)
        .round(4)
    )

    # --------------------------------------------------------
    # Signal quality score
    # Higher is better.
    # --------------------------------------------------------
    out["signal_quality_score"] = (
        1.0 - out["missingness_score"]
    ).clip(0, 1).round(4)

    # --------------------------------------------------------
    # Continuity / fragmentation scoring
    # --------------------------------------------------------
    out["previous_sequence"] = (
        out.groupby(patient_col)[sequence_col]
        .shift(1)
    )

    out["sequence_gap"] = (
        out[sequence_col] - out["previous_sequence"]
    )

    out["fragmentation_score"] = np.where(
        out["previous_sequence"].isna(),
        0.0,
        np.where(out["sequence_gap"] > 1, 1.0, 0.0),
    )

    out["continuity_score"] = (
        1.0 - out["fragmentation_score"]
    ).clip(0, 1).round(4)

    # --------------------------------------------------------
    # Volatility scoring
    # Uses row-to-row absolute movement across core signals.
    # Higher volatility = less reliable interpretation.
    # --------------------------------------------------------
    for col in signal_cols:
        out[f"{col}_delta"] = (
            out.groupby(patient_col)[col]
            .diff()
            .abs()
        )

    delta_cols = [f"{col}_delta" for col in signal_cols]

    out["volatility_score"] = (
        out[delta_cols]
        .mean(axis=1)
        .fillna(0)
        .clip(0, 1)
        .round(4)
    )

    # --------------------------------------------------------
    # NCL reliability score
    # Higher is better.
    # --------------------------------------------------------
    out["ncl_reliability_score"] = (
        out["signal_quality_score"] * 0.35
        + out["continuity_score"] * 0.25
        + (1.0 - out["fragmentation_score"]) * 0.15
        + (1.0 - out["volatility_score"]) * 0.15
        + (1.0 - out["missingness_score"]) * 0.10
    ).clip(0, 1).round(4)

    # --------------------------------------------------------
    # Signal state classification
    # --------------------------------------------------------
    def classify_signal_state(row):
        if row["ncl_reliability_score"] >= 0.90:
            return "RELIABLE_SIGNAL"
        if row["ncl_reliability_score"] >= 0.70:
            return "PARTIAL_SIGNAL"
        if row["fragmentation_score"] >= 1.0:
            return "FRAGMENTED_SIGNAL"
        if row["volatility_score"] >= 0.20:
            return "UNSTABLE_SIGNAL"
        if row["ncl_reliability_score"] >= 0.50:
            return "NOISY_SIGNAL"
        if row["ncl_reliability_score"] >= 0.30:
            return "LOW_CONFIDENCE_SIGNAL"
        return "CRITICAL_RELIABILITY_DEGRADATION"

    out["ncl_signal_state"] = out.apply(classify_signal_state, axis=1)

    # --------------------------------------------------------
    # Downstream modifiers
    # Higher = stronger trust.
    # Lower = more cautious downstream behavior.
    # --------------------------------------------------------
    out["confidence_modifier"] = out["ncl_reliability_score"].round(4)

    out["forecasting_modifier"] = (
        out["ncl_reliability_score"] * 0.80
        + out["continuity_score"] * 0.20
    ).clip(0, 1).round(4)

    out["escalation_modifier"] = (
        out["ncl_reliability_score"] * 0.70
        + (1.0 - out["volatility_score"]) * 0.30
    ).clip(0, 1).round(4)

    out["trajectory_modifier"] = (
        out["ncl_reliability_score"] * 0.60
        + (1.0 - out["volatility_score"]) * 0.40
    ).clip(0, 1).round(4)

    out["replay_modifier"] = (
        out["ncl_reliability_score"] * 0.60
        + out["continuity_score"] * 0.40
    ).clip(0, 1).round(4)

    out["hidden_burden_modifier"] = (
        out["ncl_reliability_score"] * 0.75
        + out["signal_quality_score"] * 0.25
    ).clip(0, 1).round(4)

    out["visibility_modifier"] = (
        1.0 - out["ncl_reliability_score"]
    ).clip(0, 1).round(4)

# --------------------------------------------------
# NCL Pass Forward Decision
# --------------------------------------------------

    out["ncl_pass_forward_flag"] = out["ncl_signal_state"].isin(
    [
        "RELIABLE_SIGNAL",
        "PARTIAL_SIGNAL",
    ]

    )

    out["ncl_pass_forward_type"] = "WITHHELD_TO_CMR"

    out.loc[
        out["ncl_signal_state"] == "RELIABLE_SIGNAL",
        "ncl_pass_forward_type"
        ] = "PASSED_RELIABLE_SIGNAL"

    out.loc[
    out["ncl_signal_state"] == "PARTIAL_SIGNAL",
    "ncl_pass_forward_type"

    ] = "PASSED_PARTIAL_SIGNAL"

    return out

def add_ncl_pass_forward_logic(
    df,
    reliability_col="ncl_reliability_score",
    signal_state_col="ncl_signal_state",
    pass_threshold=0.90,
    qualified_threshold=0.75,
):
    out = df.copy()

    out["ncl_pass_forward_type"] = "WITHHELD_TO_CMR"

    out.loc[
        (out[reliability_col] >= pass_threshold)
        & (out[signal_state_col] == "RELIABLE_SIGNAL"),
        "ncl_pass_forward_type",
    ] = "PASSED_RELIABLE_SIGNAL"

    out.loc[
        (out[reliability_col] >= qualified_threshold)
        & (out[reliability_col] < pass_threshold)
        & (out[signal_state_col] == "PARTIAL_SIGNAL"),
        "ncl_pass_forward_type",
    ] = "PASSED_PARTIAL_SIGNAL"

    out["ncl_pass_forward_flag"] = out["ncl_pass_forward_type"].isin(
        [
            "PASSED_RELIABLE_SIGNAL",
            "PASSED_PARTIAL_SIGNAL",
        ]
    )

    return out



def apply_ncl_multipass_layer(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    sequence_col: str = "encounter_sequence",
    ncl_pass: str = "PASS_1_RAW_SIGNAL",
) -> pd.DataFrame:
    """
    Apply NCL as a reusable reliability-hardening checkpoint.

    Pass 1:
        Raw signal reliability check after BIL.

    Pass 2:
        Post-feature / trajectory reliability check after feature engineering.

    Notes
    -----
    Pass 2 is NOT stricter than Pass 1.
    It applies the same reliability logic at a later checkpoint.

    Returns
    -------
    pd.DataFrame
        Dataframe with NCL reliability fields and pass metadata.
    """

    out = apply_initial_ncl_layer(
        df=df,
        patient_col=patient_col,
        sequence_col=sequence_col,
    )

    out["ncl_pass"] = ncl_pass

    return out