"""
BIRE OS Synthetic Ecosystem Audit & Observability Engine

Chapter 53 doctrine:
No more babying BIRE OS.

Purpose:
Audit whether the synthetic healthcare world is sufficiently realistic,
stressful, fragmented, contradictory, deceptive, uncertain, and operationally
complex for BIRE OS.

This module provides:
- ecosystem observability
- manifest generation
- schema auditing
- quality auditing
- realism validation
- contradiction auditing
- false reassurance auditing
- self-doubt auditing
- synthetic regret auditing
- learning signal analysis
- operational pressure analysis
- uncertainty auditing
- exportable ecosystem governance artifacts

This is the ecosystem inspector.

We Detect What Others Miss.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# Utility
# =========================================================


def _safe_mean(series):
    if series is None or len(series) == 0:
        return 0.0
    return round(float(pd.to_numeric(series, errors="coerce").mean()), 4)


def _safe_max(series):
    if series is None or len(series) == 0:
        return 0.0
    return round(float(pd.to_numeric(series, errors="coerce").max()), 4)


def _safe_min(series):
    if series is None or len(series) == 0:
        return 0.0
    return round(float(pd.to_numeric(series, errors="coerce").min()), 4)


def _safe_rate(df, col):
    if col not in df.columns or len(df) == 0:
        return 0.0
    return round(float(df[col].fillna(False).astype(bool).mean()), 4)


def _safe_count(df, col):
    if col not in df.columns or len(df) == 0:
        return 0
    return int(df[col].fillna(False).astype(bool).sum())


def _value_counts(df, col, top_n=None):
    if col not in df.columns:
        return {}
    vc = df[col].astype(str).value_counts(dropna=False)
    if top_n is not None:
        vc = vc.head(top_n)
    return vc.to_dict()


def _numeric_summary(df, columns, threshold=0.60):
    summary = {}

    for col in columns:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            summary[col] = {
                "mean": _safe_mean(values),
                "min": _safe_min(values),
                "max": _safe_max(values),
                "high_cases": int((values >= threshold).sum()),
                "high_rate": round(float((values >= threshold).mean()), 4),
            }

    return summary


def _flag_summary(df, columns):
    summary = {}

    for col in columns:
        if col in df.columns:
            summary[col] = {
                "count": _safe_count(df, col),
                "rate": _safe_rate(df, col),
            }

    return summary


def _explode_pipe_values(series):
    return (
        series
        .fillna("none")
        .astype(str)
        .str.split(" | ")
        .explode()
        .str.strip()
    )


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if pd.isna(obj):
        return None
    return str(obj)


# =========================================================
# Core Audit
# =========================================================


def generate_ecosystem_audit(df):
    """
    Generate full ecosystem audit summary.

    Parameters
    ----------
    df:
        Final Chapter 53 ecosystem dataframe, usually outcomes_df.

    Returns
    -------
    dict
        Dictionary containing:
        - audit
        - schema_df
        - contradiction_df
        - high_risk_df
        - high_uncertainty_df
        - high_regret_df
        - high_self_doubt_df
        - false_reassurance_df
        - upgrade_pressure_df
    """

    audit = {}

    # =====================================================
    # Dataset metadata
    # =====================================================

    audit["rows"] = int(len(df))
    audit["columns"] = int(len(df.columns))

    audit["unique_patients"] = int(df["patient_id"].nunique()) if "patient_id" in df.columns else 0
    audit["unique_encounters"] = int(df["encounter_id"].nunique()) if "encounter_id" in df.columns else 0

    audit["ecosystem_scale"] = {
        "rows": audit["rows"],
        "columns": audit["columns"],
        "unique_patients": audit["unique_patients"],
        "unique_encounters": audit["unique_encounters"],
    }

    # =====================================================
    # Schema report
    # =====================================================

    schema_report = []

    for col in df.columns:
        schema_report.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "missing_values": int(df[col].isna().sum()),
                "missing_percent": round(float(df[col].isna().mean() * 100), 2),
                "unique_values": int(df[col].astype(str).nunique(dropna=False)),
            }
        )

    schema_df = pd.DataFrame(schema_report)

    audit["schema_columns"] = list(df.columns)

    audit["schema_health"] = {
        "total_columns": int(len(df.columns)),
        "columns_with_missing": int((schema_df["missing_values"] > 0).sum()),
        "columns_over_25pct_missing": int((schema_df["missing_percent"] >= 25).sum()),
        "columns_over_50pct_missing": int((schema_df["missing_percent"] >= 50).sum()),
    }

    # =====================================================
    # Distributions
    # =====================================================

    distribution_columns = [
        "care_mode",
        "condition_profile",
        "hospital_state",
        "system_strain_state",
        "bed_flow_state",
        "care_location_state",
        "information_trust_state",
        "operational_continuity_state",
        "outcome_category",
        "final_outcome",
        "synthetic_ground_truth_label",
        "recovery_trust_state",
        "failure_mode",
        "resource_constraint_state",
        "escalation_queue_state",
    ]

    audit["distributions"] = {}

    for col in distribution_columns:
        if col in df.columns:
            audit["distributions"][col] = _value_counts(df, col)

    # Keep older top-level keys for compatibility.
    if "care_mode" in df.columns:
        audit["care_mode_distribution"] = _value_counts(df, "care_mode")

    if "condition_profile" in df.columns:
        audit["condition_distribution"] = _value_counts(df, "condition_profile")

    if "final_outcome" in df.columns:
        audit["outcome_distribution"] = _value_counts(df, "final_outcome")

    if "system_strain_state" in df.columns:
        audit["system_strain_distribution"] = _value_counts(df, "system_strain_state")

    if "recovery_trust_state" in df.columns:
        audit["recovery_trust_distribution"] = _value_counts(df, "recovery_trust_state")

    # =====================================================
    # Numeric pressure summaries
    # =====================================================

    uncertainty_cols = [
        "intervention_uncertainty_score",
        "handoff_uncertainty_score",
        "outcome_uncertainty_score",
        "outcome_self_contradiction_score",
        "synthetic_regret_score",
    ]

    audit["uncertainty_summary"] = _numeric_summary(
        df,
        uncertainty_cols,
        threshold=0.60,
    )

    severity_cols = [
        "ecosystem_severity_score",
        "longitudinal_complexity_score",
        "physiologic_consequence_pressure",
        "therapeutic_consequence_pressure",
        "operational_consequence_pressure",
        "continuity_consequence_pressure",
        "preventability_pressure_score",
    ]

    audit["severity_and_consequence_summary"] = _numeric_summary(
        df,
        severity_cols,
        threshold=0.65,
    )

    operational_cols = [
        "hospital_pressure_score",
        "operational_instability_index",
        "operational_cascade_risk",
        "hospital_system_failure_pressure",
        "resource_dependent_deterioration_risk",
        "monitoring_blind_spot_score",
        "escalation_queue_saturation_score",
        "surveillance_fatigue_score",
        "nurse_patient_ratio",
        "documentation_backlog_score",
    ]

    audit["operational_pressure_summary"] = _numeric_summary(
        df,
        operational_cols,
        threshold=0.60,
    )

    handoff_cols = [
        "handoff_uncertainty_score",
        "longitudinal_continuity_risk",
        "operational_memory_decay_score",
        "handoff_deception_pressure_score",
        "handoff_delayed_escalation_risk",
        "handoff_reescalation_pressure",
    ]

    audit["handoff_continuity_summary"] = _numeric_summary(
        df,
        handoff_cols,
        threshold=0.60,
    )

    recovery_cols = [
        "recovery_quality_score",
        "discharge_instability_score",
        "readmission_risk",
        "recurrent_deterioration_risk",
        "outcome_confidence_score",
    ]

    audit["recovery_and_readmission_summary"] = _numeric_summary(
        df,
        recovery_cols,
        threshold=0.60,
    )

    # Backward compatibility.
    if "longitudinal_complexity_score" in df.columns:
        audit["complexity_summary"] = {
            "mean": _safe_mean(df["longitudinal_complexity_score"]),
            "max": _safe_max(df["longitudinal_complexity_score"]),
            "high_complexity_cases": int((df["longitudinal_complexity_score"] >= 0.65).sum()),
        }

    if "ecosystem_severity_score" in df.columns:
        audit["ecosystem_severity_summary"] = {
            "mean": _safe_mean(df["ecosystem_severity_score"]),
            "max": _safe_max(df["ecosystem_severity_score"]),
            "high_severity_cases": int((df["ecosystem_severity_score"] >= 0.70).sum()),
        }

    # =====================================================
    # Flag summaries
    # =====================================================

    deception_flags = [
        "deceived_deterioration_outcome",
        "false_reassurance_outcome_flag",
        "operational_false_reassurance_flag",
        "false_reassurance_from_handoff_flag",
        "masked_physiology_failure",
        "false_normalization_pattern",
        "dependency_misinterpreted_as_recovery",
        "confidence_without_truth",
    ]

    audit["deception_and_false_reassurance_flags"] = _flag_summary(
        df,
        deception_flags,
    )

    hidden_deterioration_flags = [
        "missed_hidden_deterioration_signal",
        "hidden_instability_persistence",
        "active_deterioration_during_recovery_label",
        "premature_resolution_flag",
        "recovery_claim_contradicted_flag",
        "contradictory_recovery_signature",
        "stabilized_but_not_safe",
        "trajectory_reversal_after_recovery",
    ]

    audit["hidden_deterioration_flags"] = _flag_summary(
        df,
        hidden_deterioration_flags,
    )

    operational_flags = [
        "icu_overflow_flag",
        "er_overcrowding_flag",
        "hallway_care_flag",
        "icu_level_care_outside_icu_flag",
        "delayed_escalation_flag",
        "delayed_reassessment_flag",
        "reassessment_after_intervention_gap_flag",
        "operational_cascade_flag",
        "operationally_amplified_harm",
        "bire_operations_attention_required_flag",
    ]

    audit["operational_failure_flags"] = _flag_summary(
        df,
        operational_flags,
    )

    continuity_flags = [
        "handoff_degradation_occurred",
        "treatment_dependency_visibility_loss",
        "longitudinal_memory_loss_flag",
        "escalation_memory_loss_flag",
        "reassessment_plan_loss_flag",
        "hidden_trend_loss_flag",
        "continuity_rescue_flag",
        "bire_continuity_skepticism_required_flag",
        "handoff_attention_required_flag",
    ]

    audit["continuity_failure_flags"] = _flag_summary(
        df,
        continuity_flags,
    )

    self_interrogation_flags = [
        "bire_self_doubt_required_flag",
        "synthetic_regret_signal",
        "avoidable_risk_signal",
        "systemically_missed_decline",
        "late_truth_recognition",
        "silent_failure_accumulation",
        "care_pathway_fragility",
        "escalation_without_resolution",
        "recovery_without_resilience",
    ]

    audit["self_interrogation_flags"] = _flag_summary(
        df,
        self_interrogation_flags,
    )

    # =====================================================
    # Learning signal analysis
    # =====================================================

    learning_signal_summary = {}

    if "bire_learning_signal" in df.columns:
        exploded = _explode_pipe_values(df["bire_learning_signal"])
        learning_signal_summary = exploded.value_counts().to_dict()

    audit["learning_signal_distribution"] = learning_signal_summary

    upgrade_area_summary = {}

    if "recommended_system_upgrade_area" in df.columns:
        upgrade_series = (
            df["recommended_system_upgrade_area"]
            .apply(lambda x: x if isinstance(x, list) else [str(x)])
            .explode()
            .astype(str)
            .str.strip()
        )
        upgrade_area_summary = upgrade_series.value_counts().to_dict()

    audit["recommended_upgrade_area_distribution"] = upgrade_area_summary

    # =====================================================
    # Realism / pressure scorecard
    # =====================================================

    def _exists_rate(col):
        return _safe_rate(df, col)

    pressure_score_components = {
        "hidden_deterioration_pressure": _exists_rate("missed_hidden_deterioration_signal"),
        "false_reassurance_pressure": _exists_rate("false_reassurance_outcome_flag"),
        "self_doubt_pressure": _exists_rate("bire_self_doubt_required_flag"),
        "synthetic_regret_pressure": _exists_rate("synthetic_regret_signal"),
        "operational_pressure": _exists_rate("bire_operations_attention_required_flag"),
        "continuity_pressure": _exists_rate("handoff_attention_required_flag"),
        "recovery_contradiction_pressure": _exists_rate("recovery_claim_contradicted_flag"),
        "avoidable_risk_pressure": _exists_rate("avoidable_risk_signal"),
    }

    ecosystem_pressure_score = round(
        float(np.mean(list(pressure_score_components.values()))),
        4,
    )

    if ecosystem_pressure_score >= 0.45:
        ecosystem_pressure_grade = "EXTREME_PROVING_GROUND"
    elif ecosystem_pressure_score >= 0.30:
        ecosystem_pressure_grade = "STRONG_PROVING_GROUND"
    elif ecosystem_pressure_score >= 0.18:
        ecosystem_pressure_grade = "MODERATE_PROVING_GROUND"
    else:
        ecosystem_pressure_grade = "UNDERPOWERED_SYNTHETIC_WORLD"

    audit["ecosystem_pressure_scorecard"] = {
        "components": pressure_score_components,
        "ecosystem_pressure_score": ecosystem_pressure_score,
        "ecosystem_pressure_grade": ecosystem_pressure_grade,
        "doctrine": "The environment should expose BIRE OS weaknesses.",
    }

    # =====================================================
    # Contradiction audits — vectorized, memory-safe
    # =====================================================

    contradiction_frames = []

    base_cols = [
        col for col in ["patient_id", "encounter_id"]
        if col in df.columns
    ]

    def _make_contradiction(mask, issue, severity):
        if mask is None or int(mask.sum()) == 0:
            return pd.DataFrame(
                columns=["patient_id", "encounter_id", "issue", "severity"]
            )

        out = df.loc[mask, base_cols].copy()
        out["issue"] = issue
        out["severity"] = severity
        return out

    if "care_mode" in df.columns and "medications_administered" in df.columns:
        meds_text = df["medications_administered"].astype(str).str.lower()

        mask = (
            (df["care_mode"].astype(str) == "OUTPATIENT")
            & meds_text.str.contains("vasopressor", na=False)
        )

        contradiction_frames.append(
            _make_contradiction(mask, "outpatient_vasopressor_usage", "high")
        )

    flag_issue_map = {
        "recovery_claim_contradicted_flag": (
            "recovery_claim_contradicted_by_instability",
            "high",
        ),
        "confidence_without_truth": (
            "confidence_without_truth",
            "critical",
        ),
        "stabilized_but_not_safe": (
            "stabilized_but_not_safe",
            "high",
        ),
        "icu_level_care_outside_icu_flag": (
            "icu_level_need_outside_icu",
            "critical",
        ),
        "active_deterioration_during_recovery_label": (
            "active_deterioration_during_recovery",
            "critical",
        ),
    }

    for col, (issue, severity) in flag_issue_map.items():
        if col in df.columns:
            mask = df[col].fillna(False).astype(bool)
            contradiction_frames.append(
                _make_contradiction(mask, issue, severity)
            )

    if contradiction_frames:
        contradiction_df = pd.concat(
            contradiction_frames,
            ignore_index=True,
        )
    else:
        contradiction_df = pd.DataFrame(
            columns=["patient_id", "encounter_id", "issue", "severity"]
        )

    audit["contradiction_case_count"] = int(len(contradiction_df))

    if len(contradiction_df) > 0:
        audit["contradiction_issue_distribution"] = (
            contradiction_df["issue"].value_counts().to_dict()
        )
        audit["contradiction_severity_distribution"] = (
            contradiction_df["severity"].value_counts().to_dict()
        )
    else:
        audit["contradiction_issue_distribution"] = {}
        audit["contradiction_severity_distribution"] = {}

    # =====================================================
    # Missingness audit
    # =====================================================

    missingness_summary = (
        df.isna()
        .mean()
        .sort_values(ascending=False)
        .to_dict()
    )

    audit["missingness_summary"] = {
        k: round(float(v), 4)
        for k, v in missingness_summary.items()
    }

    # =====================================================
    # High-value case subsets — memory safer than df.copy()
    # =====================================================

    if "ecosystem_severity_score" in df.columns:
        high_risk_df = (
            df.loc[df["ecosystem_severity_score"] >= 0.70]
            .sort_values("ecosystem_severity_score", ascending=False)
        )
    else:
        high_risk_df = pd.DataFrame()

    if "outcome_uncertainty_score" in df.columns:
        high_uncertainty_df = (
            df.loc[df["outcome_uncertainty_score"] >= 0.50]
            .sort_values("outcome_uncertainty_score", ascending=False)
        )
    else:
        high_uncertainty_df = pd.DataFrame()

    if "synthetic_regret_score" in df.columns:
        high_regret_df = (
            df.loc[df["synthetic_regret_score"] >= 0.35]
            .sort_values("synthetic_regret_score", ascending=False)
        )
    else:
        high_regret_df = pd.DataFrame()

    if "bire_self_doubt_required_flag" in df.columns:
        high_self_doubt_df = df.loc[
            df["bire_self_doubt_required_flag"].fillna(False).astype(bool)
        ]

        sort_cols = [
            col for col in [
                "outcome_self_contradiction_score",
                "outcome_uncertainty_score",
            ]
            if col in high_self_doubt_df.columns
        ]

        if sort_cols:
            high_self_doubt_df = high_self_doubt_df.sort_values(
                sort_cols,
                ascending=False,
            )
    else:
        high_self_doubt_df = pd.DataFrame()

    if "false_reassurance_outcome_flag" in df.columns:
        false_reassurance_df = df.loc[
            df["false_reassurance_outcome_flag"].fillna(False).astype(bool)
        ]

        if "outcome_self_contradiction_score" in false_reassurance_df.columns:
            false_reassurance_df = false_reassurance_df.sort_values(
                "outcome_self_contradiction_score",
                ascending=False,
            )
    else:
        false_reassurance_df = pd.DataFrame()

    if "recommended_system_upgrade_area" in df.columns:
        upgrade_pressure_df = df.loc[
            df["recommended_system_upgrade_area"].astype(str)
            != "['baseline_monitoring_validation']"
        ]
    else:
        upgrade_pressure_df = pd.DataFrame()

    # =====================================================
    # HVI readiness audit
    # =====================================================

    hvi_ready_cols = [
        "hidden_instability_score",
        "missed_hidden_deterioration_signal",
        "hidden_instability_persistence",
        "masked_physiology_failure",
        "stabilized_but_not_safe",
    ]

    hvi_present = [col for col in hvi_ready_cols if col in df.columns]

    audit["hvi_readiness_audit"] = {
        "hvi_signal_columns_present": hvi_present,
        "hvi_signal_columns_missing": [
            col for col in hvi_ready_cols if col not in df.columns
        ],
        "hvi_pressure_rate": max(
            [
                _safe_rate(df, col)
                for col in [
                    "missed_hidden_deterioration_signal",
                    "hidden_instability_persistence",
                    "masked_physiology_failure",
                    "stabilized_but_not_safe",
                ]
                if col in df.columns
            ]
            or [0.0]
        ),
        "recommendation": "HVI_hidden_vitals_intelligence should be added to BIRE OS roadmap.",
    }

    # =====================================================
    # Manifest
    # =====================================================

    manifest = {
        "ecosystem_name": "BIRE_OS_SYNTHETIC_HEALTHCARE_WORLD",
        "ecosystem_version": "chapter_53_resilient_intelligence_proving_ground",
        "measurement_system": "US_HOSPITAL_STANDARD",
        "temperature_unit": "fahrenheit",
        "blood_pressure_unit": "mmHg",
        "spo2_unit": "percent",
        "glucose_unit": "mg_dL",
        "creatinine_unit": "mg_dL",
        "lactate_unit": "mmol_L",
        "generated_columns": list(df.columns),
        "synthetic_doctrine": "No more babying BIRE OS.",
        "mission": "Break it now, so humans do not pay later.",
        "core_question": (
            "Did the synthetic ecosystem expose where BIRE OS must become more resilient?"
        ),
    }

    audit["manifest"] = manifest

    return {
        "audit": audit,
        "schema_df": schema_df,
        "contradiction_df": contradiction_df,
        "high_risk_df": high_risk_df,
        "high_uncertainty_df": high_uncertainty_df,
        "high_regret_df": high_regret_df,
        "high_self_doubt_df": high_self_doubt_df,
        "false_reassurance_df": false_reassurance_df,
        "upgrade_pressure_df": upgrade_pressure_df,
    }


# =========================================================
# Export Engine
# =========================================================


def save_ecosystem_audit(
    ecosystem_outputs,
    output_dir="outputs/chapter_53_audit",
):
    """
    Save ecosystem audit artifacts.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audit = ecosystem_outputs["audit"]
    schema_df = ecosystem_outputs["schema_df"]
    contradiction_df = ecosystem_outputs["contradiction_df"]
    high_risk_df = ecosystem_outputs["high_risk_df"]
    high_uncertainty_df = ecosystem_outputs["high_uncertainty_df"]
    high_regret_df = ecosystem_outputs["high_regret_df"]
    high_self_doubt_df = ecosystem_outputs["high_self_doubt_df"]
    false_reassurance_df = ecosystem_outputs["false_reassurance_df"]
    upgrade_pressure_df = ecosystem_outputs["upgrade_pressure_df"]

    audit_path = output_dir / "ecosystem_audit.json"

    with open(audit_path, "w") as f:
        json.dump(
            audit,
            f,
            indent=4,
            default=_json_default,
        )

    schema_df.to_csv(output_dir / "schema_report.csv", index=False)
    contradiction_df.to_csv(output_dir / "contradiction_cases.csv", index=False)
    high_risk_df.to_csv(output_dir / "high_risk_cases.csv", index=False)
    high_uncertainty_df.to_csv(output_dir / "high_uncertainty_cases.csv", index=False)
    high_regret_df.to_csv(output_dir / "high_regret_cases.csv", index=False)
    high_self_doubt_df.to_csv(output_dir / "high_self_doubt_cases.csv", index=False)
    false_reassurance_df.to_csv(output_dir / "false_reassurance_cases.csv", index=False)
    upgrade_pressure_df.to_csv(output_dir / "upgrade_pressure_cases.csv", index=False)

    return str(audit_path)
