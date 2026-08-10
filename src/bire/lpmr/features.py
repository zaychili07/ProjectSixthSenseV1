# ============================================================
# BIRE OS — Longitudinal Patient Memory Retention (LPMR)
# File: src/bire/lpmr/features.py
# Chapter: 65
#
# Created: 2026-06-20
# Updated: 2026-06-20
#
# CHANGED:
# - Initialized LPMR feature creation module
# - Added support for patient-specific memory retention outputs
# - Prepared backend space for longitudinal experience retention
# - Separated LPMR feature generation from scoring utilities
#
# Purpose:
# Backend helper for creating LPMR memory retention
# outputs, patient-specific trajectory memory features,
# and longitudinal experience retention signals.
# ============================================================

import pandas as pd


def add_lpmr_recovery_pattern_memory(df):
    """
    Form LPMR recovery pattern memory from repeated ATI recovery states.

    LPMR does not reinterpret ATI.
    LPMR checks whether ATI recovery observations repeat enough
    to become longitudinal patient memory.
    """

    out = df.copy()

    required_cols = [
        "patient_id",
        "ati_recovery_state",
    ]

    missing_cols = [
        col for col in required_cols
        if col not in out.columns
    ]

    if missing_cols:
        out["lpmr_missing_inputs"] = ", ".join(missing_cols)
        return out

    recovery_counts = (
        out.groupby(["patient_id", "ati_recovery_state"])
        .size()
        .reset_index(name="lpmr_recovery_state_count")
    )

    out = out.merge(
        recovery_counts,
        on=["patient_id", "ati_recovery_state"],
        how="left",
    )

    out["lpmr_recovery_pattern_memory_state"] = (
        "RECOVERY_MEMORY_NOT_YET_EARNED"
    )

    out.loc[
        out["lpmr_recovery_state_count"].ge(2),
        "lpmr_recovery_pattern_memory_state"
    ] = "RECOVERY_PATTERN_MEMORY_EARNED"

    out["lpmr_recovery_state_memory_label"] = (
        out["ati_recovery_state"].astype(str)
        + "_ATI_LPMR"
    )

    return out

def add_lpmr_rebound_pattern_memory(df):
    """
    Form LPMR rebound pattern memory from repeated
    ATI rebound observations.

    LPMR does not reinterpret ATI.
    LPMR determines whether rebound observations
    have earned longitudinal memory.
    """

    out = df.copy()

    required_cols = [
        "patient_id",
        "ati_recovery_state",
    ]

    missing_cols = [
        col for col in required_cols
        if col not in out.columns
    ]

    if missing_cols:
        out["lpmr_missing_inputs"] = ", ".join(missing_cols)
        return out

    rebound_counts = (
        out[
            out["ati_recovery_state"]
            .eq("REBOUND_VULNERABILITY")
        ]
        .groupby("patient_id")
        .size()
        .reset_index(name="lpmr_rebound_state_count")
    )

    out = out.merge(
        rebound_counts,
        on="patient_id",
        how="left"
    )

    out["lpmr_rebound_state_count"] = (
        out["lpmr_rebound_state_count"]
        .fillna(0)
        .astype(int)
    )

    out["lpmr_rebound_pattern_memory_state"] = (
        "REBOUND_MEMORY_NOT_YET_EARNED"
    )

    out.loc[
        out["lpmr_rebound_state_count"] >= 2,
        "lpmr_rebound_pattern_memory_state"
    ] = "REBOUND_PATTERN_MEMORY_EARNED"

    out["lpmr_rebound_memory_label"] = (
        "REBOUND_VULNERABILITY_ATI_LPMR"
    )

    return out

def add_lpmr_stabilization_pattern_memory(df):
    """
    Form LPMR stabilization pattern memory from
    repeated ATI stabilization observations.
    """

    out = df.copy()

    required_cols = [
        "patient_id",
        "ati_recovery_state",
    ]

    missing_cols = [
        col for col in required_cols
        if col not in out.columns
    ]

    if missing_cols:
        out["lpmr_missing_inputs"] = ", ".join(missing_cols)
        return out

    stabilization_states = [
        "FRAGILE_STABILIZATION",
        "TEMPORARY_NORMALIZATION",
    ]

    stabilization_counts = (
        out[
            out["ati_recovery_state"]
            .isin(stabilization_states)
        ]
        .groupby("patient_id")
        .size()
        .reset_index(
            name="lpmr_stabilization_state_count"
        )
    )

    out = out.merge(
        stabilization_counts,
        on="patient_id",
        how="left",
    )

    out["lpmr_stabilization_state_count"] = (
        out["lpmr_stabilization_state_count"]
        .fillna(0)
        .astype(int)
    )

    out["lpmr_stabilization_pattern_memory_state"] = (
        "STABILIZATION_MEMORY_NOT_YET_EARNED"
    )

    out.loc[
        out["lpmr_stabilization_state_count"] >= 2,
        "lpmr_stabilization_pattern_memory_state"
    ] = "STABILIZATION_PATTERN_MEMORY_EARNED"

    out["lpmr_stabilization_memory_label"] = (
        out["ati_recovery_state"]
        .astype(str)
        + "_ATI_LPMR"
    )

    return out

def add_lpmr_compensation_pattern_memory(df):
    """
    Form LPMR compensation pattern memory from
    repeated HVI compensation observations.
    """

    out = df.copy()

    required_cols = [
        "patient_id",
        "compensation_reserve_score",
    ]

    missing_cols = [
        col for col in required_cols
        if col not in out.columns
    ]

    if missing_cols:
        out["lpmr_missing_inputs"] = ", ".join(missing_cols)
        return out

    compensation_events = (
        out["compensation_reserve_score"] < 0.50
    )

    compensation_counts = (
        out.loc[compensation_events]
        .groupby("patient_id")
        .size()
        .reset_index(
            name="lpmr_compensation_state_count"
        )
    )

    out = out.merge(
        compensation_counts,
        on="patient_id",
        how="left"
    )

    out["lpmr_compensation_state_count"] = (
        out["lpmr_compensation_state_count"]
        .fillna(0)
        .astype(int)
    )

    out["lpmr_compensation_pattern_memory_state"] = (
        "COMPENSATION_MEMORY_NOT_YET_EARNED"
    )

    out.loc[
        out["lpmr_compensation_state_count"] >= 2,
        "lpmr_compensation_pattern_memory_state"
    ] = "COMPENSATION_PATTERN_MEMORY_EARNED"

    out["lpmr_compensation_memory_label"] = (
        "COMPENSATION_PATTERN_HVI_LPMR"
    )

    return out

def add_lpmr_instability_pattern_memory(df):
    """
    Form LPMR instability pattern memory from repeated
    HVI/ATI instability observations.
    """

    out = df.copy()

    required_cols = [
        "patient_id",
        "hidden_instability_score",
    ]

    missing_cols = [
        col for col in required_cols
        if col not in out.columns
    ]

    if missing_cols:
        out["lpmr_missing_inputs"] = ", ".join(missing_cols)
        return out

    instability_events = (
        out["hidden_instability_score"] >= 0.50
    )

    instability_counts = (
        out.loc[instability_events]
        .groupby("patient_id")
        .size()
        .reset_index(
            name="lpmr_instability_state_count"
        )
    )

    out = out.merge(
        instability_counts,
        on="patient_id",
        how="left",
    )

    out["lpmr_instability_state_count"] = (
        out["lpmr_instability_state_count"]
        .fillna(0)
        .astype(int)
    )

    out["lpmr_instability_pattern_memory_state"] = (
        "INSTABILITY_MEMORY_NOT_YET_EARNED"
    )

    out.loc[
        out["lpmr_instability_state_count"] >= 2,
        "lpmr_instability_pattern_memory_state"
    ] = "INSTABILITY_PATTERN_MEMORY_EARNED"

    out["lpmr_instability_memory_label"] = (
        "INSTABILITY_PATTERN_HVI_LPMR"
    )

    return out

def add_lpmr_progressive_baseline_drift_memory(df):
    """
    Form LPMR progressive baseline drift memory.
    """

    out = df.copy()

    required_cols = [
        "patient_id",
        "longitudinal_continuity_risk",
    ]

    missing_cols = [
        col for col in required_cols
        if col not in out.columns
    ]

    if missing_cols:
        out["lpmr_missing_inputs"] = ", ".join(missing_cols)
        return out

    drift_events = (
        out["longitudinal_continuity_risk"] > 0.50
    )

    drift_counts = (
        out.loc[drift_events]
        .groupby("patient_id")
        .size()
        .reset_index(
            name="lpmr_baseline_drift_count"
        )
    )

    out = out.merge(
        drift_counts,
        on="patient_id",
        how="left",
    )

    out["lpmr_baseline_drift_count"] = (
        out["lpmr_baseline_drift_count"]
        .fillna(0)
        .astype(int)
    )

    out["lpmr_baseline_drift_memory_state"] = (
        "BASELINE_DRIFT_MEMORY_NOT_YET_EARNED"
    )

    out.loc[
        out["lpmr_baseline_drift_count"] >= 2,
        "lpmr_baseline_drift_memory_state"
    ] = "BASELINE_DRIFT_MEMORY_EARNED"

    out["lpmr_baseline_drift_memory_label"] = (
        "BASELINE_DRIFT_HVI_LPMR"
    )

    return out

def add_lpmr_patient_trajectory_memory(df):
    """
    Form LPMR patient trajectory memory from
    repeated ATI trajectory behavior.
    """

    out = df.copy()

    required_cols = [
        "patient_id",
        "ati_trajectory_behavior",
    ]

    missing_cols = [
        col for col in required_cols
        if col not in out.columns
    ]

    if missing_cols:
        out["lpmr_missing_inputs"] = ", ".join(missing_cols)
        return out

    trajectory_counts = (
        out.groupby(
            [
                "patient_id",
                "ati_trajectory_behavior",
            ]
        )
        .size()
        .reset_index(
            name="lpmr_patient_trajectory_count"
        )
    )

    out = out.merge(
        trajectory_counts,
        on=[
            "patient_id",
            "ati_trajectory_behavior",
        ],
        how="left",
    )

    out["lpmr_patient_trajectory_count"] = (
        out["lpmr_patient_trajectory_count"]
        .fillna(0)
        .astype(int)
    )

    out["lpmr_patient_trajectory_memory_state"] = (
        "PATIENT_TRAJECTORY_MEMORY_NOT_YET_EARNED"
    )

    out.loc[
        out["lpmr_patient_trajectory_count"] >= 2,
        "lpmr_patient_trajectory_memory_state"
    ] = "PATIENT_TRAJECTORY_MEMORY_EARNED"

    out["lpmr_patient_trajectory_memory_label"] = (
        out["ati_trajectory_behavior"].astype(str)
        + "_ATI_LPMR"
    )

    return out


def add_lpmr_hidden_deterioration_memory(df):
    """
    Form LPMR hidden deterioration memory from repeated
    hidden deterioration observations.

    LPMR does not detect or interpret hidden deterioration.
    LPMR determines whether hidden deterioration observations
    have earned longitudinal memory.
    """

    out = df.copy()

    required_cols = [
        "patient_id",
        "hidden_instability_score",
    ]

    missing_cols = [
        col for col in required_cols
        if col not in out.columns
    ]

    if missing_cols:
        out["lpmr_missing_inputs"] = ", ".join(missing_cols)
        return out

    hidden_deterioration_events = (
        out["hidden_instability_score"] >= 0.50
    )

    hidden_deterioration_counts = (
        out.loc[hidden_deterioration_events]
        .groupby("patient_id")
        .size()
        .reset_index(
            name="lpmr_hidden_deterioration_count"
        )
    )

    out = out.merge(
        hidden_deterioration_counts,
        on="patient_id",
        how="left",
    )

    out["lpmr_hidden_deterioration_count"] = (
        out["lpmr_hidden_deterioration_count"]
        .fillna(0)
        .astype(int)
    )

    out["lpmr_hidden_deterioration_memory_state"] = (
        "HIDDEN_DETERIORATION_MEMORY_NOT_YET_EARNED"
    )

    out.loc[
        out["lpmr_hidden_deterioration_count"] >= 2,
        "lpmr_hidden_deterioration_memory_state"
    ] = "HIDDEN_DETERIORATION_MEMORY_EARNED"

    out["lpmr_hidden_deterioration_memory_label"] = (
        "HIDDEN_DETERIORATION_HVI_LPMR"
    )

    return out

def add_lpmr_repeated_relapse_memory(df):
    """
    Form LPMR repeated relapse memory.

    LPMR does not interpret relapse.
    LPMR determines whether repeated relapse
    observations have earned longitudinal memory.
    """

    out = df.copy()

    required_cols = [
        "patient_id",
        "ati_recovery_state",
    ]

    missing_cols = [
        col for col in required_cols
        if col not in out.columns
    ]

    if missing_cols:
        out["lpmr_missing_inputs"] = ", ".join(missing_cols)
        return out

    relapse_states = [
        "DECEPTIVE_RECOVERY",
        "REBOUND_VULNERABILITY",
    ]

    relapse_counts = (
        out[
            out["ati_recovery_state"].isin(relapse_states)
        ]
        .groupby("patient_id")
        .size()
        .reset_index(
            name="lpmr_repeated_relapse_count"
        )
    )

    out = out.merge(
        relapse_counts,
        on="patient_id",
        how="left",
    )

    out["lpmr_repeated_relapse_count"] = (
        out["lpmr_repeated_relapse_count"]
        .fillna(0)
        .astype(int)
    )

    out["lpmr_repeated_relapse_memory_state"] = (
        "RELAPSE_MEMORY_NOT_YET_EARNED"
    )

    out.loc[
        out["lpmr_repeated_relapse_count"] >= 2,
        "lpmr_repeated_relapse_memory_state"
    ] = "RELAPSE_MEMORY_EARNED"

    out["lpmr_repeated_relapse_memory_label"] = (
        "RELAPSE_PATTERN_ATI_LPMR"
    )

    return out

def add_lpmr_stabilization_durability_memory(df):
    """
    Form LPMR stabilization durability memory from
    repeated stabilization durability observations.
    """

    out = df.copy()

    required_cols = [
        "patient_id",
        "stabilization_durability_score",
    ]

    missing_cols = [
        col for col in required_cols
        if col not in out.columns
    ]

    if missing_cols:
        out["lpmr_missing_inputs"] = ", ".join(missing_cols)
        return out

    durability_events = (
        out["stabilization_durability_score"] < 0.50
    )

    durability_counts = (
        out.loc[durability_events]
        .groupby("patient_id")
        .size()
        .reset_index(
            name="lpmr_stabilization_durability_count"
        )
    )

    out = out.merge(
        durability_counts,
        on="patient_id",
        how="left",
    )

    out["lpmr_stabilization_durability_count"] = (
        out["lpmr_stabilization_durability_count"]
        .fillna(0)
        .astype(int)
    )

    out["lpmr_stabilization_durability_memory_state"] = (
        "STABILIZATION_DURABILITY_MEMORY_NOT_YET_EARNED"
    )

    out.loc[
        out["lpmr_stabilization_durability_count"] >= 2,
        "lpmr_stabilization_durability_memory_state"
    ] = "STABILIZATION_DURABILITY_MEMORY_EARNED"

    out["lpmr_stabilization_durability_memory_label"] = (
        "STABILIZATION_DURABILITY_HVI_LPMR"
    )

    return out