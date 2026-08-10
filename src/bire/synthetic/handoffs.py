"""
BIRE OS Synthetic Handoff Degradation &
Operational Continuity Intelligence Engine

Chapter 53 doctrine:
No more babying BIRE OS.

Purpose:
Simulate how healthcare transitions degrade operational memory,
context, trust, escalation continuity, treatment awareness, reassessment
planning, hidden-trend preservation, and evidence reliability.

This module does not only ask:
"Did a bad handoff happen?"

It asks:
- What was lost?
- Was the lost context clinically meaningful?
- Did the handoff create false reassurance?
- Did hidden instability get summarized as stability?
- Did reassessment disappear?
- Did escalation history decay?
- Did BIRE OS preserve memory better than the operational workflow?

The hospital may forget.
BIRE OS should not.

We Detect What Others Miss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bire.synthetic.config import SYNTHETIC_ECOSYSTEM_CONFIG


SHIFT_TYPES = [
    "day_shift",
    "night_shift",
    "weekend_shift",
    "holiday_shift",
    "shift_change_window",
    "overnight_cross_cover",
]

STAFFING_PRESSURE_LEVELS = [
    "normal",
    "elevated",
    "high",
    "critical",
    "unsafe_fragmented",
]

HANDOFF_CONTEXTS = [
    "routine_shift_handoff",
    "rapid_transfer_handoff",
    "er_to_inpatient_handoff",
    "inpatient_to_icu_handoff",
    "icu_to_stepdown_handoff",
    "stepdown_to_floor_handoff",
    "discharge_transition",
    "readmission_transition",
    "cross_cover_transition",
    "consult_service_transition",
    "boarding_transition",
]

HANDOFF_FAILURE_TYPES = {
    "allergy_not_transferred": 0.10,
    "medication_reconciliation_incomplete": 0.18,
    "diagnosis_context_missing": 0.16,
    "imaging_findings_not_acknowledged": 0.12,
    "intervention_history_missing": 0.16,
    "post_event_status_not_updated": 0.14,
    "delayed_chart_sync": 0.22,
    "incomplete_discharge_summary": 0.14,
    "handoff_note_fragmented": 0.20,
    "pending_lab_not_followed_up": 0.13,
    "repeat_imaging_context_missing": 0.10,
    "treatment_dependency_not_transferred": 0.16,
    "readmission_history_overlooked": 0.14,
    "provider_interpretation_mismatch": 0.13,
    "cross_team_documentation_conflict": 0.14,
    "summary_compression_loss": 0.20,
    "escalation_history_missing": 0.17,
    "reassessment_plan_missing": 0.16,
    "oxygen_support_context_missing": 0.13,
    "vasopressor_dependency_not_communicated": 0.10,
    "sedation_context_missing": 0.09,
    "false_recovery_concern_not_transferred": 0.12,
    "hidden_instability_trend_lost": 0.15,
    "family_or_nursing_concern_not_carried_forward": 0.10,
    "pending_consult_recommendation_lost": 0.11,
    "abnormal_trend_summarized_as_stable": 0.12,
    "diagnostic_uncertainty_minimized": 0.10,
    "imaging_delay_not_communicated": 0.11,
    "lab_delay_not_communicated": 0.11,
    "discharge_risk_understated": 0.10,
    "rebound_risk_not_communicated": 0.12,
    "bire_alert_context_not_reviewed": 0.10,
    "monitoring_requirement_not_transferred": 0.13,
    "support_dependency_documented_as_stable": 0.12,
    "near_miss_history_not_carried_forward": 0.10,
    "care_team_assumption_mismatch": 0.12,
    "delayed_reassessment_after_transfer": 0.12,
}

HANDOFF_SEVERITY_MAP = {
    "allergy_not_transferred": 5,
    "medication_reconciliation_incomplete": 5,
    "diagnosis_context_missing": 3,
    "imaging_findings_not_acknowledged": 4,
    "intervention_history_missing": 5,
    "post_event_status_not_updated": 5,
    "delayed_chart_sync": 2,
    "incomplete_discharge_summary": 2,
    "handoff_note_fragmented": 3,
    "pending_lab_not_followed_up": 4,
    "repeat_imaging_context_missing": 3,
    "treatment_dependency_not_transferred": 6,
    "readmission_history_overlooked": 5,
    "provider_interpretation_mismatch": 4,
    "cross_team_documentation_conflict": 5,
    "summary_compression_loss": 4,
    "escalation_history_missing": 6,
    "reassessment_plan_missing": 5,
    "oxygen_support_context_missing": 4,
    "vasopressor_dependency_not_communicated": 7,
    "sedation_context_missing": 5,
    "false_recovery_concern_not_transferred": 6,
    "hidden_instability_trend_lost": 6,
    "family_or_nursing_concern_not_carried_forward": 4,
    "pending_consult_recommendation_lost": 4,
    "abnormal_trend_summarized_as_stable": 6,
    "diagnostic_uncertainty_minimized": 5,
    "imaging_delay_not_communicated": 4,
    "lab_delay_not_communicated": 4,
    "discharge_risk_understated": 5,
    "rebound_risk_not_communicated": 6,
    "bire_alert_context_not_reviewed": 5,
    "monitoring_requirement_not_transferred": 6,
    "support_dependency_documented_as_stable": 7,
    "near_miss_history_not_carried_forward": 5,
    "care_team_assumption_mismatch": 5,
    "delayed_reassessment_after_transfer": 6,
}


def _clip(value, low=0.0, high=1.0):
    return float(np.clip(value, low, high))


def _safe_float(row, col, default=0.0):
    value = row.get(col, default)
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(row, col, default=0):
    value = row.get(col, default)
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return default


def _safe_bool(row, col, default=False):
    value = row.get(col, default)
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    return bool(value)


def _choose_handoff_context(row, rng, rapid_transfer_flag=False):
    care_mode = str(row.get("care_mode", "OUTPATIENT"))
    transition_pathway = str(row.get("transition_pathway", row.get("transfer_pathway_text", "")))
    discharge_disposition = str(row.get("discharge_disposition", ""))

    readmission_flag = _safe_bool(row, "readmission_flag", False)
    bounceback_flag = _safe_bool(row, "bounceback_flag", False)
    failed_discharge_flag = _safe_bool(row, "failed_discharge_flag", False)

    if rapid_transfer_flag:
        return "rapid_transfer_handoff"

    if readmission_flag or bounceback_flag or failed_discharge_flag:
        return "readmission_transition"

    if "ICU" in transition_pathway and "INPATIENT" in transition_pathway:
        return "inpatient_to_icu_handoff"

    if "ER" in transition_pathway and "INPATIENT" in transition_pathway:
        return "er_to_inpatient_handoff"

    if "stepdown" in transition_pathway.lower():
        return "icu_to_stepdown_handoff"

    if "home" in discharge_disposition:
        return "discharge_transition"

    if care_mode == "ICU":
        return rng.choice(
            ["routine_shift_handoff", "consult_service_transition", "cross_cover_transition"],
            p=[0.50, 0.25, 0.25],
        )

    if "ER" in care_mode:
        return rng.choice(
            ["er_to_inpatient_handoff", "boarding_transition", "routine_shift_handoff"],
            p=[0.45, 0.35, 0.20],
        )

    return rng.choice(
        ["routine_shift_handoff", "cross_cover_transition", "consult_service_transition"],
        p=[0.60, 0.25, 0.15],
    )


def _continuity_failure_domains(failures):
    failure_set = set(failures)
    domains = []

    medication_failures = {
        "medication_reconciliation_incomplete",
        "allergy_not_transferred",
        "oxygen_support_context_missing",
        "vasopressor_dependency_not_communicated",
        "sedation_context_missing",
        "treatment_dependency_not_transferred",
        "intervention_history_missing",
        "support_dependency_documented_as_stable",
    }

    evidence_failures = {
        "pending_lab_not_followed_up",
        "imaging_findings_not_acknowledged",
        "repeat_imaging_context_missing",
        "diagnosis_context_missing",
        "pending_consult_recommendation_lost",
        "diagnostic_uncertainty_minimized",
        "imaging_delay_not_communicated",
        "lab_delay_not_communicated",
    }

    lifecycle_failures = {
        "post_event_status_not_updated",
        "escalation_history_missing",
        "reassessment_plan_missing",
        "false_recovery_concern_not_transferred",
        "readmission_history_overlooked",
        "hidden_instability_trend_lost",
        "rebound_risk_not_communicated",
        "bire_alert_context_not_reviewed",
        "monitoring_requirement_not_transferred",
        "near_miss_history_not_carried_forward",
        "delayed_reassessment_after_transfer",
    }

    communication_failures = {
        "handoff_note_fragmented",
        "summary_compression_loss",
        "provider_interpretation_mismatch",
        "cross_team_documentation_conflict",
        "abnormal_trend_summarized_as_stable",
        "family_or_nursing_concern_not_carried_forward",
        "discharge_risk_understated",
        "care_team_assumption_mismatch",
    }

    if failure_set & medication_failures:
        domains.append("MEDICATION_AND_TREATMENT_CONTEXT_LOSS")
    if failure_set & evidence_failures:
        domains.append("EVIDENCE_FOLLOWUP_CONTEXT_LOSS")
    if failure_set & lifecycle_failures:
        domains.append("LIFECYCLE_AND_ESCALATION_MEMORY_LOSS")
    if failure_set & communication_failures:
        domains.append("COMMUNICATION_AND_SUMMARY_DEGRADATION")

    if not domains:
        domains.append("NO_MAJOR_CONTINUITY_DOMAIN_LOSS")

    return sorted(set(domains))


def _lost_critical_context(failures):
    mapping = {
        "allergy_not_transferred": "allergy_context",
        "medication_reconciliation_incomplete": "medication_context",
        "diagnosis_context_missing": "diagnostic_context",
        "imaging_findings_not_acknowledged": "imaging_context",
        "intervention_history_missing": "intervention_history",
        "post_event_status_not_updated": "post_event_status",
        "pending_lab_not_followed_up": "pending_labs",
        "treatment_dependency_not_transferred": "treatment_dependency",
        "readmission_history_overlooked": "readmission_history",
        "escalation_history_missing": "escalation_history",
        "reassessment_plan_missing": "reassessment_plan",
        "oxygen_support_context_missing": "oxygen_support_context",
        "vasopressor_dependency_not_communicated": "vasopressor_dependency",
        "sedation_context_missing": "sedation_context",
        "false_recovery_concern_not_transferred": "false_recovery_concern",
        "hidden_instability_trend_lost": "hidden_instability_trend",
        "family_or_nursing_concern_not_carried_forward": "bedside_concern",
        "pending_consult_recommendation_lost": "pending_consult_recommendation",
        "abnormal_trend_summarized_as_stable": "abnormal_trend",
        "diagnostic_uncertainty_minimized": "diagnostic_uncertainty",
        "imaging_delay_not_communicated": "imaging_delay",
        "lab_delay_not_communicated": "lab_delay",
        "discharge_risk_understated": "discharge_risk",
        "rebound_risk_not_communicated": "rebound_risk",
        "bire_alert_context_not_reviewed": "bire_alert_context",
        "monitoring_requirement_not_transferred": "monitoring_requirement",
        "support_dependency_documented_as_stable": "support_dependency",
        "near_miss_history_not_carried_forward": "near_miss_history",
        "care_team_assumption_mismatch": "care_team_assumption",
        "delayed_reassessment_after_transfer": "reassessment_timing",
    }

    context = [mapping[f] for f in failures if f in mapping]
    if not context:
        context.append("none")
    return sorted(set(context))


def _information_trust_state(handoff_severity_score):
    if handoff_severity_score >= 32:
        return "critical_operational_fragmentation"
    if handoff_severity_score >= 20:
        return "high_operational_fragmentation"
    if handoff_severity_score >= 9:
        return "partial_operational_fragmentation"
    return "stable_operational_continuity"


def _operational_continuity_state(handoff_deception_pressure_score):
    if handoff_deception_pressure_score >= 0.72:
        return "CONTINUITY_FAILURE"
    if handoff_deception_pressure_score >= 0.55:
        return "CONTINUITY_CRITICAL"
    if handoff_deception_pressure_score >= 0.38:
        return "CONTINUITY_UNSTABLE"
    if handoff_deception_pressure_score >= 0.20:
        return "CONTINUITY_DEGRADED"
    return "CONTINUITY_STABLE"


def _recognition_delay_minutes(
    handoff_context,
    staffing_pressure_score,
    handoff_severity_score,
    handoff_uncertainty_score,
    operational_memory_decay_score,
    false_reassurance_from_handoff_flag,
    rng,
):
    base = {
        "routine_shift_handoff": 35,
        "rapid_transfer_handoff": 75,
        "er_to_inpatient_handoff": 85,
        "inpatient_to_icu_handoff": 55,
        "icu_to_stepdown_handoff": 95,
        "stepdown_to_floor_handoff": 105,
        "discharge_transition": 240,
        "readmission_transition": 180,
        "cross_cover_transition": 120,
        "consult_service_transition": 100,
        "boarding_transition": 150,
    }.get(handoff_context, 90)

    delay = base
    delay += staffing_pressure_score * 260
    delay += handoff_severity_score * 5
    delay += handoff_uncertainty_score * 180
    delay += operational_memory_decay_score * 120
    delay += float(false_reassurance_from_handoff_flag) * 90
    delay += rng.normal(0, 25)

    return int(max(0, round(delay)))


def _failure_adjustment(
    failure,
    row,
    handoff_context,
    staffing_pressure_score,
    fragility,
    intervention_uncertainty,
    confidence_instability,
    data_trust_pressure,
    provider_change_count,
    cross_specialty_transition,
    recovery_authenticity_state,
    current_lifecycle_state,
    hidden_instability_score,
    rebound_risk,
    treatment_masking_risk,
    monitoring_blind_spot_proxy,
):
    adjusted = HANDOFF_FAILURE_TYPES[failure]
    adjusted += staffing_pressure_score * 0.40
    adjusted += fragility * 0.05
    adjusted += intervention_uncertainty * 0.06
    adjusted += confidence_instability * 0.05
    adjusted += data_trust_pressure * 0.05

    if provider_change_count >= 4:
        adjusted += 0.05

    if cross_specialty_transition:
        adjusted += 0.04

    if handoff_context in ["rapid_transfer_handoff", "inpatient_to_icu_handoff", "boarding_transition"]:
        adjusted += 0.04

    support_dependent_states = [
        "SUPPORT_DEPENDENT_STABILITY",
        "ARTIFICIAL_RECOVERY_PATTERN",
        "FALSE_RECOVERY_PATTERN",
        "RECOVERY_NOT_TRUSTWORTHY",
    ]

    if failure in [
        "treatment_dependency_not_transferred",
        "vasopressor_dependency_not_communicated",
        "oxygen_support_context_missing",
        "false_recovery_concern_not_transferred",
        "rebound_risk_not_communicated",
        "support_dependency_documented_as_stable",
    ] and recovery_authenticity_state in support_dependent_states:
        adjusted += 0.13

    if failure in [
        "post_event_status_not_updated",
        "escalation_history_missing",
        "reassessment_plan_missing",
        "bire_alert_context_not_reviewed",
        "monitoring_requirement_not_transferred",
        "delayed_reassessment_after_transfer",
    ] and current_lifecycle_state in [
        "ESCALATE",
        "URGENT",
        "MONITOR",
        "RE-ESCALATE",
        "CRITICAL",
        "DE_ESCALATION_MONITOR",
        "POST_EVENT_MONITOR",
    ]:
        adjusted += 0.13

    if failure in [
        "hidden_instability_trend_lost",
        "abnormal_trend_summarized_as_stable",
        "diagnostic_uncertainty_minimized",
    ] and hidden_instability_score >= 0.45:
        adjusted += 0.13

    if failure in [
        "rebound_risk_not_communicated",
        "false_recovery_concern_not_transferred",
        "discharge_risk_understated",
    ] and rebound_risk >= 0.40:
        adjusted += 0.10

    if failure in [
        "support_dependency_documented_as_stable",
        "treatment_dependency_not_transferred",
        "intervention_history_missing",
    ] and treatment_masking_risk >= 0.45:
        adjusted += 0.11

    if failure in [
        "monitoring_requirement_not_transferred",
        "delayed_reassessment_after_transfer",
        "reassessment_plan_missing",
    ] and monitoring_blind_spot_proxy >= 0.45:
        adjusted += 0.10

    return _clip(adjusted, 0, 0.94)


def generate_handoff_degradation(df, random_seed=None):
    """
    Generate operational continuity degradation and BIRE memory preservation signals.

    Parameters
    ----------
    df:
        DataFrame from intervention layer.

    Returns
    -------
    pd.DataFrame
        Input rows with handoff / continuity intelligence columns added.
    """

    if random_seed is None:
        random_seed = SYNTHETIC_ECOSYSTEM_CONFIG["random_seed"]

    rng = np.random.default_rng(random_seed)
    updated_rows = []

    for row in df.to_dict("records"):
        care_mode = str(row.get("care_mode", "OUTPATIENT"))
        current_lifecycle_state = row.get(
            "current_lifecycle_state",
            row.get("bire_state", "UNKNOWN"),
        )
        post_event_state = row.get("post_event_state", "NO_EVENT")

        fragility = _safe_float(row, "fragility_score", 0.0)
        deterioration = _safe_float(row, "deterioration_tendency", 0.0)
        recovery_resilience = _safe_float(row, "recovery_resilience", 0.5)

        intervention_uncertainty = _safe_float(row, "intervention_uncertainty_score", 0.0)
        rebound_risk = _safe_float(row, "rebound_deterioration_risk", 0.0)
        treatment_masking_risk = _safe_float(row, "treatment_masking_risk", 0.0)
        stabilization_durability = _safe_float(row, "stabilization_durability_score", 0.5)
        confidence_instability = _safe_float(row, "therapeutic_confidence_instability", 0.0)
        post_intervention_uncertainty = _safe_float(row, "post_intervention_uncertainty_pressure", 0.0)

        hidden_instability_score = _safe_float(row, "hidden_instability_score", 0.0)
        silent_collapse_pressure = _safe_float(row, "silent_collapse_pressure", 0.0)
        data_trust_pressure = _safe_float(row, "data_trust_pressure_score", 0.0)
        diagnosis_pressure = _safe_float(row, "diagnosis_risk_pressure_score", 0.0)
        imaging_pressure = _safe_float(row, "imaging_deception_pressure_score", 0.0)
        medication_pressure = _safe_float(row, "therapeutic_system_pressure_score", 0.0)

        medication_count = _safe_int(row, "medication_count", 0)
        diagnosis_count = _safe_int(row, "diagnosis_count", 0)

        intervention_trust_state = row.get("intervention_trust_state", "response_uncertain")
        recovery_authenticity_state = row.get(
            "recovery_authenticity_state",
            "PARTIAL_RECOVERY_UNCERTAIN",
        )

        treatment_dependency_visibility_loss_existing = _safe_bool(
            row,
            "treatment_dependency_visibility_loss",
            False,
        )

        shift_type = rng.choice(
            SHIFT_TYPES,
            p=[0.38, 0.25, 0.13, 0.05, 0.11, 0.08],
        )

        staffing_pressure_level = rng.choice(
            STAFFING_PRESSURE_LEVELS,
            p=[0.40, 0.25, 0.18, 0.11, 0.06],
        )

        staffing_pressure_score_map = {
            "normal": 0.00,
            "elevated": 0.08,
            "high": 0.18,
            "critical": 0.32,
            "unsafe_fragmented": 0.45,
        }
        staffing_pressure_score = staffing_pressure_score_map[staffing_pressure_level]

        if care_mode == "ICU":
            provider_change_count = int(rng.integers(3, 10))
        elif care_mode == "INPATIENT":
            provider_change_count = int(rng.integers(2, 7))
        elif "ER" in care_mode:
            provider_change_count = int(rng.integers(1, 5))
        elif care_mode == "OUTPATIENT":
            provider_change_count = int(rng.integers(0, 3))
        else:
            provider_change_count = int(rng.integers(1, 5))

        consult_service_count = int(rng.integers(0, 6))
        if diagnosis_count >= 7:
            consult_service_count += int(rng.integers(1, 4))
        if fragility >= 0.70 or hidden_instability_score >= 0.50:
            consult_service_count += int(rng.integers(0, 3))

        cross_specialty_transition = bool(consult_service_count >= 2)

        boarding_delay_flag = bool(
            rng.random()
            < _clip(
                0.04
                + staffing_pressure_score
                + float("ER" in care_mode) * 0.08
                + hidden_instability_score * 0.04,
                0,
                0.85,
            )
        )

        er_hallway_care_flag = bool(
            "ER" in care_mode
            and rng.random()
            < _clip(0.03 + staffing_pressure_score + rebound_risk * 0.05, 0, 0.85)
        )

        rapid_transfer_flag = bool(
            rng.random()
            < _clip(
                0.02
                + staffing_pressure_score
                + rebound_risk * 0.09
                + hidden_instability_score * 0.07
                + deterioration * 0.05,
                0,
                0.80,
            )
        )

        handoff_context = _choose_handoff_context(
            row=row,
            rng=rng,
            rapid_transfer_flag=rapid_transfer_flag,
        )

        monitoring_blind_spot_proxy = _clip(
            data_trust_pressure * 0.20
            + staffing_pressure_score * 0.30
            + provider_change_count * 0.035
            + hidden_instability_score * 0.18
            + treatment_masking_risk * 0.16
        )

        degradation_probability = 0.05

        if care_mode == "ICU":
            degradation_probability += 0.20
        elif care_mode == "INPATIENT":
            degradation_probability += 0.14
        elif "ER" in care_mode:
            degradation_probability += 0.13

        if handoff_context in [
            "rapid_transfer_handoff",
            "er_to_inpatient_handoff",
            "inpatient_to_icu_handoff",
            "readmission_transition",
            "boarding_transition",
            "icu_to_stepdown_handoff",
            "discharge_transition",
        ]:
            degradation_probability += 0.13

        if shift_type in [
            "night_shift",
            "weekend_shift",
            "holiday_shift",
            "shift_change_window",
            "overnight_cross_cover",
        ]:
            degradation_probability += 0.08

        degradation_probability += fragility * 0.12
        degradation_probability += deterioration * 0.06
        degradation_probability += hidden_instability_score * 0.10
        degradation_probability += intervention_uncertainty * 0.18
        degradation_probability += rebound_risk * 0.16
        degradation_probability += treatment_masking_risk * 0.15
        degradation_probability += confidence_instability * 0.14
        degradation_probability += post_intervention_uncertainty * 0.12
        degradation_probability += data_trust_pressure * 0.10
        degradation_probability += staffing_pressure_score
        degradation_probability += min(provider_change_count, 8) * 0.015
        degradation_probability += min(medication_count, 10) * 0.010
        degradation_probability += min(diagnosis_count, 10) * 0.010

        degradation_probability = _clip(degradation_probability, 0, 0.97)
        handoff_degradation_occurred = bool(rng.random() < degradation_probability)

        handoff_failures = []

        if handoff_degradation_occurred:
            for failure in HANDOFF_FAILURE_TYPES:
                adjusted_probability = _failure_adjustment(
                    failure=failure,
                    row=row,
                    handoff_context=handoff_context,
                    staffing_pressure_score=staffing_pressure_score,
                    fragility=fragility,
                    intervention_uncertainty=intervention_uncertainty,
                    confidence_instability=confidence_instability,
                    data_trust_pressure=data_trust_pressure,
                    provider_change_count=provider_change_count,
                    cross_specialty_transition=cross_specialty_transition,
                    recovery_authenticity_state=recovery_authenticity_state,
                    current_lifecycle_state=current_lifecycle_state,
                    hidden_instability_score=hidden_instability_score,
                    rebound_risk=rebound_risk,
                    treatment_masking_risk=treatment_masking_risk,
                    monitoring_blind_spot_proxy=monitoring_blind_spot_proxy,
                )

                if rng.random() < adjusted_probability:
                    handoff_failures.append(failure)

        documentation_conflict_flag = bool(
            rng.random()
            < _clip(
                0.03
                + consult_service_count * 0.02
                + data_trust_pressure * 0.10
                + staffing_pressure_score * 0.25
                + treatment_masking_risk * 0.05,
                0,
                0.80,
            )
        )

        provider_interpretation_disagreement = bool(
            documentation_conflict_flag
            and rng.random() < _clip(0.70 + cross_specialty_transition * 0.10, 0, 0.90)
        )

        summary_compression_score = round(
            _clip(
                provider_change_count * 0.08
                + staffing_pressure_score
                + float(cross_specialty_transition) * 0.08
                + float(handoff_context == "discharge_transition") * 0.07
                + float(handoff_context == "boarding_transition") * 0.06
                + data_trust_pressure * 0.08,
                0,
                1,
            ),
            4,
        )

        critical_context_omission_probability = round(
            _clip(
                summary_compression_score * 0.55
                + confidence_instability * 0.12
                + intervention_uncertainty * 0.10
                + data_trust_pressure * 0.08
                + treatment_masking_risk * 0.08
                + hidden_instability_score * 0.07,
                0,
                1,
            ),
            4,
        )

        treatment_dependency_visibility_loss = bool(
            treatment_dependency_visibility_loss_existing
            or "treatment_dependency_not_transferred" in handoff_failures
            or "support_dependency_documented_as_stable" in handoff_failures
            or (
                (
                    intervention_trust_state == "treatment_supported_stability"
                    or recovery_authenticity_state
                    in [
                        "SUPPORT_DEPENDENT_STABILITY",
                        "ARTIFICIAL_RECOVERY_PATTERN",
                        "FALSE_RECOVERY_PATTERN",
                    ]
                )
                and rng.random()
                < _clip(
                    0.12
                    + staffing_pressure_score
                    + rebound_risk * 0.10
                    + treatment_masking_risk * 0.18,
                    0,
                    0.88,
                )
            )
        )

        longitudinal_memory_loss_flag = bool(
            rng.random()
            < _clip(
                0.04
                + provider_change_count * 0.02
                + float(handoff_context in ["readmission_transition", "discharge_transition"]) * 0.08
                + staffing_pressure_score * 0.22
                + summary_compression_score * 0.12,
                0,
                0.82,
            )
        )

        escalation_memory_loss_flag = bool(
            "escalation_history_missing" in handoff_failures
            or (
                current_lifecycle_state
                in ["ESCALATE", "URGENT", "MONITOR", "RE-ESCALATE", "CRITICAL"]
                and rng.random()
                < _clip(
                    0.04 + staffing_pressure_score + summary_compression_score * 0.22,
                    0,
                    0.75,
                )
            )
        )

        reassessment_plan_loss_flag = bool(
            "reassessment_plan_missing" in handoff_failures
            or "delayed_reassessment_after_transfer" in handoff_failures
            or (
                post_event_state != "NO_EVENT"
                and rng.random()
                < _clip(
                    0.05
                    + staffing_pressure_score
                    + intervention_uncertainty * 0.20
                    + treatment_masking_risk * 0.12,
                    0,
                    0.75,
                )
            )
        )

        hidden_trend_loss_flag = bool(
            "hidden_instability_trend_lost" in handoff_failures
            or "abnormal_trend_summarized_as_stable" in handoff_failures
            or (
                hidden_instability_score >= 0.50
                and rng.random()
                < _clip(
                    0.05
                    + summary_compression_score * 0.30
                    + data_trust_pressure * 0.15,
                    0,
                    0.70,
                )
            )
        )

        false_reassurance_from_handoff_flag = bool(
            (
                "abnormal_trend_summarized_as_stable" in handoff_failures
                or "false_recovery_concern_not_transferred" in handoff_failures
                or "discharge_risk_understated" in handoff_failures
                or "support_dependency_documented_as_stable" in handoff_failures
            )
            or (
                hidden_instability_score >= 0.45
                and summary_compression_score >= 0.35
                and rng.random() < 0.38
            )
            or (
                treatment_masking_risk >= 0.50
                and stabilization_durability <= 0.45
                and rng.random() < 0.42
            )
        )

        delayed_reassessment_flag = bool(
            reassessment_plan_loss_flag
            or "delayed_reassessment_after_transfer" in handoff_failures
            or (
                monitoring_blind_spot_proxy >= 0.50
                and hidden_instability_score >= 0.40
                and rng.random() < 0.38
            )
        )

        delayed_escalation_flag = bool(
            escalation_memory_loss_flag
            or (
                false_reassurance_from_handoff_flag
                and hidden_instability_score >= 0.42
            )
            or (
                rebound_risk >= 0.45
                and staffing_pressure_score >= 0.18
                and rng.random() < 0.35
            )
        )

        lost_critical_context = _lost_critical_context(handoff_failures)
        continuity_failure_domains = _continuity_failure_domains(handoff_failures)

        handoff_severity_score = 0
        for failure in handoff_failures:
            handoff_severity_score += HANDOFF_SEVERITY_MAP.get(failure, 1)

        if documentation_conflict_flag:
            handoff_severity_score += 3
        if provider_interpretation_disagreement:
            handoff_severity_score += 2
        if treatment_dependency_visibility_loss:
            handoff_severity_score += 5
        if longitudinal_memory_loss_flag:
            handoff_severity_score += 4
        if escalation_memory_loss_flag:
            handoff_severity_score += 5
        if reassessment_plan_loss_flag:
            handoff_severity_score += 4
        if hidden_trend_loss_flag:
            handoff_severity_score += 5
        if false_reassurance_from_handoff_flag:
            handoff_severity_score += 5
        if delayed_reassessment_flag:
            handoff_severity_score += 4
        if delayed_escalation_flag:
            handoff_severity_score += 5

        hidden_instability_amplification = 0.0

        amplification_failures = [
            "post_event_status_not_updated",
            "intervention_history_missing",
            "medication_reconciliation_incomplete",
            "pending_lab_not_followed_up",
            "treatment_dependency_not_transferred",
            "escalation_history_missing",
            "reassessment_plan_missing",
            "hidden_instability_trend_lost",
            "false_recovery_concern_not_transferred",
            "abnormal_trend_summarized_as_stable",
            "rebound_risk_not_communicated",
            "bire_alert_context_not_reviewed",
            "monitoring_requirement_not_transferred",
            "support_dependency_documented_as_stable",
            "near_miss_history_not_carried_forward",
            "delayed_reassessment_after_transfer",
        ]

        for failure in amplification_failures:
            if failure in handoff_failures:
                hidden_instability_amplification += 0.085

        hidden_instability_amplification += treatment_dependency_visibility_loss * 0.18
        hidden_instability_amplification += documentation_conflict_flag * 0.10
        hidden_instability_amplification += escalation_memory_loss_flag * 0.12
        hidden_instability_amplification += reassessment_plan_loss_flag * 0.10
        hidden_instability_amplification += hidden_trend_loss_flag * 0.16
        hidden_instability_amplification += false_reassurance_from_handoff_flag * 0.14
        hidden_instability_amplification += delayed_reassessment_flag * 0.10
        hidden_instability_amplification += delayed_escalation_flag * 0.10

        hidden_instability_amplification = round(_clip(hidden_instability_amplification, 0, 1), 4)

        operational_memory_decay_score = round(
            _clip(
                float(longitudinal_memory_loss_flag) * 0.18
                + float(escalation_memory_loss_flag) * 0.20
                + float(reassessment_plan_loss_flag) * 0.17
                + float(hidden_trend_loss_flag) * 0.19
                + float(false_reassurance_from_handoff_flag) * 0.15
                + float(delayed_reassessment_flag) * 0.12
                + float(delayed_escalation_flag) * 0.12
                + summary_compression_score * 0.18
                + staffing_pressure_score * 0.10,
                0,
                1,
            ),
            4,
        )

        handoff_uncertainty_score = round(
            _clip(
                handoff_severity_score * 0.034
                + hidden_instability_amplification
                + staffing_pressure_score * 0.24
                + data_trust_pressure * 0.12
                + diagnosis_pressure * 0.08
                + imaging_pressure * 0.08
                + medication_pressure * 0.07
                + operational_memory_decay_score * 0.16,
                0,
                1,
            ),
            4,
        )

        longitudinal_continuity_risk = round(
            _clip(
                rebound_risk * 0.20
                + handoff_uncertainty_score * 0.34
                + operational_memory_decay_score * 0.20
                + fragility * 0.15
                + hidden_instability_score * 0.12
                + staffing_pressure_score * 0.12
                + confidence_instability * 0.10,
                0,
                1,
            ),
            4,
        )

        handoff_deception_pressure_score = round(
            _clip(
                hidden_instability_amplification * 0.27
                + operational_memory_decay_score * 0.23
                + confidence_instability * 0.14
                + treatment_masking_risk * 0.13
                + intervention_uncertainty * 0.09
                + float(documentation_conflict_flag) * 0.08
                + float(treatment_dependency_visibility_loss) * 0.13
                + float(false_reassurance_from_handoff_flag) * 0.18,
                0,
                1,
            ),
            4,
        )

        operational_continuity_score = round(
            _clip(
                1.0
                - handoff_severity_score * 0.018
                - staffing_pressure_score * 0.22
                - summary_compression_score * 0.10
                - handoff_deception_pressure_score * 0.20,
                0,
                1,
            ),
            4,
        )

        information_trust_state = _information_trust_state(handoff_severity_score)
        operational_continuity_state = _operational_continuity_state(handoff_deception_pressure_score)

        time_to_recognition_delay_minutes = _recognition_delay_minutes(
            handoff_context=handoff_context,
            staffing_pressure_score=staffing_pressure_score,
            handoff_severity_score=handoff_severity_score,
            handoff_uncertainty_score=handoff_uncertainty_score,
            operational_memory_decay_score=operational_memory_decay_score,
            false_reassurance_from_handoff_flag=false_reassurance_from_handoff_flag,
            rng=rng,
        )

        bire_memory_preserved_flag = bool(
            (
                hidden_instability_score >= 0.45
                or rebound_risk >= 0.45
                or confidence_instability >= 0.40
                or treatment_masking_risk >= 0.45
                or current_lifecycle_state
                in ["ESCALATE", "URGENT", "MONITOR", "RE-ESCALATE", "CRITICAL"]
            )
            and (
                operational_memory_decay_score < 0.78
                or rng.random() < 0.58
            )
        )

        continuity_rescue_flag = bool(
            bire_memory_preserved_flag
            and (
                longitudinal_memory_loss_flag
                or escalation_memory_loss_flag
                or hidden_trend_loss_flag
                or false_reassurance_from_handoff_flag
                or delayed_reassessment_flag
                or delayed_escalation_flag
            )
        )

        handoff_delayed_escalation_risk = round(
            _clip(
                handoff_uncertainty_score * 0.28
                + operational_memory_decay_score * 0.24
                + float(false_reassurance_from_handoff_flag) * 0.18
                + hidden_instability_score * 0.14
                + rebound_risk * 0.10
                + staffing_pressure_score * 0.08,
                0,
                1,
            ),
            4,
        )

        handoff_reescalation_pressure = round(
            _clip(
                rebound_risk * 0.24
                + hidden_instability_amplification * 0.23
                + operational_memory_decay_score * 0.19
                + intervention_uncertainty * 0.13
                + treatment_masking_risk * 0.10
                + float(false_reassurance_from_handoff_flag) * 0.13,
                0,
                1,
            ),
            4,
        )

        handoff_false_recovery_pressure_score = round(
            _clip(
                float(false_reassurance_from_handoff_flag) * 0.25
                + treatment_masking_risk * 0.20
                + rebound_risk * 0.18
                + (1 - stabilization_durability) * 0.15
                + hidden_instability_score * 0.14
                + summary_compression_score * 0.08,
                0,
                1,
            ),
            4,
        )

        requires_operational_reassessment = bool(
            handoff_severity_score >= 7
            or hidden_instability_amplification >= 0.30
            or handoff_uncertainty_score >= 0.50
            or operational_memory_decay_score >= 0.45
            or false_reassurance_from_handoff_flag
            or delayed_reassessment_flag
        )

        bire_continuity_skepticism_required_flag = bool(
            information_trust_state
            in [
                "high_operational_fragmentation",
                "critical_operational_fragmentation",
            ]
            or handoff_deception_pressure_score >= 0.45
            or treatment_dependency_visibility_loss
            or hidden_trend_loss_flag
            or false_reassurance_from_handoff_flag
        )

        handoff_attention_required_flag = bool(
            requires_operational_reassessment
            or bire_continuity_skepticism_required_flag
            or longitudinal_continuity_risk >= 0.50
            or handoff_delayed_escalation_risk >= 0.50
        )

        row["shift_type"] = shift_type
        row["staffing_pressure_level"] = staffing_pressure_level
        row["staffing_pressure_score"] = round(float(staffing_pressure_score), 4)
        row["handoff_context"] = handoff_context

        row["provider_change_count"] = int(provider_change_count)
        row["consult_service_count"] = int(consult_service_count)
        row["cross_specialty_transition"] = bool(cross_specialty_transition)

        row["boarding_delay_flag"] = bool(boarding_delay_flag)
        row["er_hallway_care_flag"] = bool(er_hallway_care_flag)
        row["rapid_transfer_flag"] = bool(rapid_transfer_flag)

        row["handoff_degradation_probability"] = round(degradation_probability, 4)
        row["handoff_degradation_occurred"] = bool(handoff_degradation_occurred)
        row["handoff_failures"] = handoff_failures
        row["lost_critical_context"] = lost_critical_context
        row["continuity_failure_domains"] = continuity_failure_domains

        row["documentation_conflict_flag"] = bool(documentation_conflict_flag)
        row["provider_interpretation_disagreement"] = bool(provider_interpretation_disagreement)
        row["summary_compression_score"] = summary_compression_score
        row["critical_context_omission_probability"] = critical_context_omission_probability

        row["treatment_dependency_visibility_loss"] = bool(treatment_dependency_visibility_loss)
        row["longitudinal_memory_loss_flag"] = bool(longitudinal_memory_loss_flag)
        row["escalation_memory_loss_flag"] = bool(escalation_memory_loss_flag)
        row["reassessment_plan_loss_flag"] = bool(reassessment_plan_loss_flag)
        row["hidden_trend_loss_flag"] = bool(hidden_trend_loss_flag)
        row["false_reassurance_from_handoff_flag"] = bool(false_reassurance_from_handoff_flag)

        row["delayed_reassessment_flag"] = bool(delayed_reassessment_flag)
        row["delayed_escalation_flag"] = bool(delayed_escalation_flag)

        row["handoff_severity_score"] = int(handoff_severity_score)
        row["operational_continuity_score"] = operational_continuity_score
        row["operational_continuity_state"] = operational_continuity_state
        row["information_trust_state"] = information_trust_state

        row["hidden_instability_amplification"] = hidden_instability_amplification
        row["handoff_uncertainty_score"] = handoff_uncertainty_score
        row["longitudinal_continuity_risk"] = longitudinal_continuity_risk
        row["operational_memory_decay_score"] = operational_memory_decay_score
        row["handoff_deception_pressure_score"] = handoff_deception_pressure_score
        row["time_to_recognition_delay_minutes"] = int(time_to_recognition_delay_minutes)

        row["bire_memory_preserved_flag"] = bool(bire_memory_preserved_flag)
        row["continuity_rescue_flag"] = bool(continuity_rescue_flag)

        row["handoff_delayed_escalation_risk"] = handoff_delayed_escalation_risk
        row["handoff_reescalation_pressure"] = handoff_reescalation_pressure
        row["handoff_false_recovery_pressure_score"] = handoff_false_recovery_pressure_score

        row["requires_operational_reassessment"] = bool(requires_operational_reassessment)
        row["bire_continuity_skepticism_required_flag"] = bool(bire_continuity_skepticism_required_flag)
        row["handoff_attention_required_flag"] = bool(handoff_attention_required_flag)

        updated_rows.append(row)

    return pd.DataFrame(updated_rows)