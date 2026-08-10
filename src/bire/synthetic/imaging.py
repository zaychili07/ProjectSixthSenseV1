"""
BIRE OS Synthetic Imaging Intelligence Engine

Chapter 53 doctrine:
No more babying BIRE OS.

Purpose:
Generate imaging evidence that behaves like real hospital imaging:
- delayed
- contradictory
- incomplete
- evolving
- operationally buried
- falsely reassuring
- trajectory-sensitive
- acknowledgment-dependent
- discordant with physiology, labs, diagnoses, treatment, and operations

This is NOT simple imaging generation.
This is Synthetic Imaging Intelligence Warfare.

Doctrine:
We Detect What Others Miss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bire.synthetic.config import SYNTHETIC_ECOSYSTEM_CONFIG


IMAGING_TYPES = [
    "none",
    "portable_chest_xray",
    "chest_xray",
    "ct_head",
    "ct_chest",
    "ct_abdomen_pelvis",
    "ct_pulmonary_embolism",
    "mri_brain",
    "mri_spine",
    "ultrasound",
    "doppler_ultrasound",
    "echocardiogram",
    "repeat_ct",
    "repeat_xray",
]

IMAGING_FINDINGS = [
    "normal",
    "mild_abnormality",
    "critical_finding",
    "incidental_finding",
    "ambiguous_finding",
    "progressive_worsening",
    "partial_resolution",
    "new_complication",
    "chronic_abnormality",
    "subtle_early_abnormality",
    "possible_occult_process",
    "underappreciated_instability",
    "possible_false_negative",
]

RADIOLOGY_INTERPRETATION_STATES = [
    "clear_read",
    "equivocal_read",
    "limited_interpretation",
    "subtle_finding_buried",
    "possible_false_reassurance",
    "radiology_undercalls_severity",
    "radiology_overcalls_severity",
    "discordant_preliminary_and_final_read",
]

IMAGING_TRAJECTORY_STATES = [
    "stable_imaging",
    "slow_progression",
    "rapid_progression",
    "occult_worsening_before_visibility",
    "imaging_lagging_clinical_decline",
    "false_imaging_reassurance",
    "post_treatment_apparent_improvement",
    "worsening_despite_reassuring_scan",
]


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


def _weighted_choice(rng, weights):
    keys = list(weights.keys())
    probs = np.array([max(float(weights[k]), 0.001) for k in keys], dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(keys, p=probs))


def _choose_imaging_type(row, rng):
    care_mode = str(row.get("care_mode", row.get("highest_acuity_mode", "OUTPATIENT")))
    condition_profile = str(row.get("condition_profile", ""))
    event_type = str(row.get("event_type", "NONE"))
    presentation = str(row.get("presentation_profile", ""))

    if "pancreatitis" in condition_profile or "abdominal_pain" in presentation:
        return str(rng.choice(["ct_abdomen_pelvis", "ultrasound", "repeat_ct"], p=[0.52, 0.30, 0.18]))

    if "copd" in condition_profile or "respiratory" in event_type or "shortness_of_breath" in presentation:
        return str(rng.choice(["portable_chest_xray", "ct_chest", "ct_pulmonary_embolism", "repeat_xray"], p=[0.42, 0.25, 0.13, 0.20]))

    if "chf" in condition_profile or "hemodynamic" in event_type:
        return str(rng.choice(["portable_chest_xray", "echocardiogram", "ct_chest"], p=[0.44, 0.38, 0.18]))

    if "arrhythmia" in condition_profile or "arrhythmia" in event_type or "chest_pain" in presentation:
        return str(rng.choice(["echocardiogram", "ct_chest", "ct_pulmonary_embolism"], p=[0.48, 0.32, 0.20]))

    if "neurologic" in condition_profile or "altered_mental_status" in presentation or "dizziness" in presentation:
        return str(rng.choice(["ct_head", "mri_brain"], p=[0.58, 0.42]))

    if care_mode == "ICU":
        return str(rng.choice(["portable_chest_xray", "ct_chest", "echocardiogram", "ct_abdomen_pelvis", "repeat_xray"], p=[0.34, 0.22, 0.20, 0.14, 0.10]))

    return str(rng.choice(IMAGING_TYPES[1:]))


def _derive_imaging_finding(row, rng, hidden_instability, deceptive_stability, has_event):
    lab_pressure = _safe_float(row, "lab_sixth_sense_pressure_score", 0.0)
    diagnosis_pressure = _safe_float(row, "diagnosis_risk_pressure_score", 0.0)
    treatment_masking = _safe_float(row, "treatment_masking_risk", 0.0)
    operational_false_reassurance = _safe_bool(row, "operational_false_reassurance_flag", False)
    handoff_false_reassurance = _safe_bool(row, "false_reassurance_from_handoff_flag", False)

    weights = {
        "normal": 0.25,
        "mild_abnormality": 0.17,
        "critical_finding": 0.07,
        "incidental_finding": 0.09,
        "ambiguous_finding": 0.11,
        "progressive_worsening": 0.08,
        "partial_resolution": 0.05,
        "new_complication": 0.05,
        "chronic_abnormality": 0.05,
        "subtle_early_abnormality": 0.03,
        "possible_occult_process": 0.025,
        "underappreciated_instability": 0.025,
        "possible_false_negative": 0.015,
    }

    if has_event:
        weights["critical_finding"] += 0.12
        weights["progressive_worsening"] += 0.09
        weights["new_complication"] += 0.05
        weights["underappreciated_instability"] += 0.05

    if hidden_instability >= 0.45:
        weights["possible_false_negative"] += 0.08
        weights["subtle_early_abnormality"] += 0.06
        weights["possible_occult_process"] += 0.07
        weights["underappreciated_instability"] += 0.06

    if deceptive_stability >= 0.45 or treatment_masking >= 0.45:
        weights["normal"] += 0.07
        weights["possible_false_negative"] += 0.09
        weights["partial_resolution"] += 0.04

    if lab_pressure >= 0.45 or diagnosis_pressure >= 0.45:
        weights["ambiguous_finding"] += 0.05
        weights["progressive_worsening"] += 0.05
        weights["underappreciated_instability"] += 0.05

    if operational_false_reassurance or handoff_false_reassurance:
        weights["possible_false_negative"] += 0.05
        weights["subtle_early_abnormality"] += 0.04

    return _weighted_choice(rng, weights)


def _derive_interpretation_state(rng, hidden_instability, overload_pressure, discordant_read_flag, poor_image_quality_flag, final_read):
    weights = {
        "clear_read": 0.43,
        "equivocal_read": 0.18,
        "limited_interpretation": 0.08,
        "subtle_finding_buried": 0.07,
        "possible_false_reassurance": 0.08,
        "radiology_undercalls_severity": 0.07,
        "radiology_overcalls_severity": 0.04,
        "discordant_preliminary_and_final_read": 0.05,
    }

    if hidden_instability >= 0.45:
        weights["possible_false_reassurance"] += 0.09
        weights["radiology_undercalls_severity"] += 0.09
        weights["subtle_finding_buried"] += 0.05

    if overload_pressure >= 0.45:
        weights["subtle_finding_buried"] += 0.07
        weights["equivocal_read"] += 0.04
        weights["limited_interpretation"] += 0.04

    if poor_image_quality_flag:
        weights["limited_interpretation"] += 0.12
        weights["equivocal_read"] += 0.06

    if final_read in ["possible_false_negative", "underappreciated_instability", "subtle_early_abnormality"]:
        weights["possible_false_reassurance"] += 0.08
        weights["radiology_undercalls_severity"] += 0.08

    if discordant_read_flag:
        weights["discordant_preliminary_and_final_read"] += 0.30

    return _weighted_choice(rng, weights)


def _derive_trajectory_state(row, rng, hidden_instability, deceptive_stability, final_read):
    treatment_masking = _safe_float(row, "treatment_masking_risk", 0.0)
    rebound = _safe_float(row, "rebound_deterioration_risk", 0.0)
    recovery_state = str(row.get("recovery_authenticity_state", "PARTIAL_RECOVERY_UNCERTAIN"))
    delayed_reassessment = _safe_bool(row, "delayed_reassessment_flag", False)

    weights = {
        "stable_imaging": 0.32,
        "slow_progression": 0.17,
        "rapid_progression": 0.08,
        "occult_worsening_before_visibility": 0.11,
        "imaging_lagging_clinical_decline": 0.12,
        "false_imaging_reassurance": 0.10,
        "post_treatment_apparent_improvement": 0.05,
        "worsening_despite_reassuring_scan": 0.05,
    }

    if hidden_instability >= 0.45:
        weights["occult_worsening_before_visibility"] += 0.11
        weights["imaging_lagging_clinical_decline"] += 0.11
        weights["false_imaging_reassurance"] += 0.08

    if deceptive_stability >= 0.45 or final_read in ["normal", "mild_abnormality", "partial_resolution"]:
        weights["worsening_despite_reassuring_scan"] += 0.08
        weights["false_imaging_reassurance"] += 0.07

    if treatment_masking >= 0.45 or recovery_state in ["SUPPORT_DEPENDENT_STABILITY", "ARTIFICIAL_RECOVERY_PATTERN", "FALSE_RECOVERY_PATTERN"]:
        weights["post_treatment_apparent_improvement"] += 0.09
        weights["false_imaging_reassurance"] += 0.07

    if rebound >= 0.45 or delayed_reassessment:
        weights["slow_progression"] += 0.06
        weights["imaging_lagging_clinical_decline"] += 0.06

    if final_read in ["critical_finding", "new_complication", "progressive_worsening"]:
        weights["rapid_progression"] += 0.08
        weights["slow_progression"] += 0.06

    return _weighted_choice(rng, weights)


def generate_imaging_evidence(df, random_seed=None):
    """
    Generate synthetic imaging evidence with operational delay, buried findings,
    false reassurance, imaging/clinical disagreement, and outcome-facing pressure.
    """

    if random_seed is None:
        random_seed = SYNTHETIC_ECOSYSTEM_CONFIG["random_seed"]

    rng = np.random.default_rng(random_seed)
    imaging_rows = []

    for row in df.to_dict("records"):
        care_mode = str(row.get("care_mode", row.get("highest_acuity_mode", "OUTPATIENT")))
        has_event = _safe_bool(row, "has_event", False)
        fragility = _safe_float(row, "fragility_score", 0.0)
        deterioration = _safe_float(row, "deterioration_tendency", 0.0)
        hidden_instability = max(
            _safe_float(row, "hidden_instability_score", 0.0),
            _safe_float(row, "silent_collapse_pressure", 0.0),
        )
        deceptive_stability = max(
            _safe_float(row, "deceptive_stability_index", 0.0),
            _safe_float(row, "normal_range_deception_score", 0.0),
        )
        overload_pressure = max(
            _safe_float(row, "hospital_overload_pressure", 0.0),
            _safe_float(row, "hospital_system_failure_pressure", 0.0),
            _safe_float(row, "operational_instability_index", 0.0),
        )
        event_wave_count = _safe_float(row, "event_wave_count", 0.0)
        evidence_pressure = _safe_float(row, "bire_context_pressure_score", 0.0)
        lab_pressure = _safe_float(row, "lab_sixth_sense_pressure_score", 0.0)
        diagnosis_pressure = _safe_float(row, "diagnosis_risk_pressure_score", 0.0)
        monitoring_blind_spot = _safe_float(row, "monitoring_blind_spot_score", 0.0)
        delayed_escalation = _safe_bool(row, "delayed_escalation_flag", False)
        delayed_reassessment = _safe_bool(row, "delayed_reassessment_flag", False)
        handoff_false_reassurance = _safe_bool(row, "false_reassurance_from_handoff_flag", False)
        operational_false_reassurance = _safe_bool(row, "operational_false_reassurance_flag", False)
        treatment_masking = _safe_float(row, "treatment_masking_risk", 0.0)
        rebound = _safe_float(row, "rebound_deterioration_risk", 0.0)

        imaging_probability = _clip(
            0.08
            + float("ER" in care_mode) * 0.22
            + float(care_mode == "INPATIENT") * 0.34
            + float(care_mode == "ICU") * 0.52
            + fragility * 0.20
            + hidden_instability * 0.16
            + deterioration * 0.10
            + float(has_event) * 0.14
            + event_wave_count * 0.035
            + lab_pressure * 0.08
            + diagnosis_pressure * 0.08,
            0.02,
            0.98,
        )

        imaging_not_ordered_despite_risk_flag = bool(
            rng.random()
            < _clip(
                0.01
                + hidden_instability * 0.08
                + monitoring_blind_spot * 0.08
                + overload_pressure * 0.05
                + float(delayed_reassessment) * 0.05,
                0,
                0.45,
            )
        )

        imaging_ordered = bool((rng.random() < imaging_probability) and not imaging_not_ordered_despite_risk_flag)
        imaging_type = _choose_imaging_type(row, rng) if imaging_ordered else "none"

        if imaging_type == "none":
            imaging_result_category = "none"
            preliminary_read = "none"
            final_read = "none"
            discordant_read_flag = False
        else:
            imaging_result_category = _derive_imaging_finding(row, rng, hidden_instability, deceptive_stability, has_event)
            preliminary_read = imaging_result_category
            final_read = imaging_result_category

            discordant_probability = _clip(
                0.03
                + hidden_instability * 0.08
                + overload_pressure * 0.06
                + evidence_pressure * 0.05
                + lab_pressure * 0.04
                + diagnosis_pressure * 0.04,
                0,
                0.55,
            )

            discordant_read_flag = bool(rng.random() < discordant_probability)

            if discordant_read_flag:
                if imaging_result_category in ["normal", "mild_abnormality", "partial_resolution"]:
                    final_read = str(rng.choice(["critical_finding", "progressive_worsening", "underappreciated_instability", "possible_occult_process"]))
                else:
                    preliminary_read = str(rng.choice(["normal", "mild_abnormality", "ambiguous_finding"]))

        limited_study_flag = bool(rng.random() < _clip(0.04 + fragility * 0.05 + overload_pressure * 0.03, 0, 0.40))
        motion_artifact_flag = bool(rng.random() < _clip(0.03 + fragility * 0.05, 0, 0.35))
        poor_positioning_flag = bool(rng.random() < _clip(0.02 + overload_pressure * 0.06, 0, 0.35))
        incomplete_protocol_flag = bool(rng.random() < _clip(0.02 + overload_pressure * 0.05 + imaging_not_ordered_despite_risk_flag * 0.06, 0, 0.35))

        poor_image_quality_flag = bool(
            limited_study_flag
            or motion_artifact_flag
            or poor_positioning_flag
            or incomplete_protocol_flag
        )

        radiology_interpretation_state = _derive_interpretation_state(
            rng=rng,
            hidden_instability=hidden_instability,
            overload_pressure=overload_pressure,
            discordant_read_flag=discordant_read_flag,
            poor_image_quality_flag=poor_image_quality_flag,
            final_read=final_read,
        )

        result_delay_minutes = 0
        if imaging_ordered:
            if care_mode == "ICU":
                delay_pool = [15, 30, 45, 60, 90, 120]
            elif "ER" in care_mode:
                delay_pool = [30, 60, 90, 120, 180, 240, 360]
            else:
                delay_pool = [60, 120, 240, 480, 720, 1440, 2160]

            result_delay_minutes = int(rng.choice(delay_pool))
            result_delay_minutes += int(overload_pressure * 260)
            result_delay_minutes += int(float(delayed_reassessment or delayed_escalation) * rng.choice([30, 60, 120]))

        acknowledgment_states = {
            "acknowledged_on_time": 0.70,
            "acknowledged_late": 0.14,
            "buried_in_chart": 0.06,
            "handoff_missed_result": 0.05,
            "not_acknowledged": 0.05,
        }

        if overload_pressure >= 0.45:
            acknowledgment_states["buried_in_chart"] += 0.08
            acknowledgment_states["handoff_missed_result"] += 0.06
            acknowledgment_states["acknowledged_late"] += 0.04

        if delayed_reassessment or delayed_escalation:
            acknowledgment_states["acknowledged_late"] += 0.05
            acknowledgment_states["not_acknowledged"] += 0.03

        result_acknowledgment_state = _weighted_choice(rng, acknowledgment_states) if imaging_ordered else "no_imaging_to_acknowledge"

        repeat_imaging_flag = False
        if final_read in ["critical_finding", "progressive_worsening", "new_complication", "underappreciated_instability", "possible_false_negative"]:
            repeat_imaging_flag = bool(rng.random() < _clip(0.45 + hidden_instability * 0.16 + diagnosis_pressure * 0.10, 0, 0.85))

        imaging_trajectory_state = _derive_trajectory_state(row, rng, hidden_instability, deceptive_stability, final_read)

        instability_map = {
            "none": 0,
            "normal": 0,
            "mild_abnormality": 1,
            "incidental_finding": 1,
            "ambiguous_finding": 2,
            "partial_resolution": 2,
            "chronic_abnormality": 2,
            "subtle_early_abnormality": 3,
            "possible_occult_process": 4,
            "underappreciated_instability": 5,
            "possible_false_negative": 5,
            "progressive_worsening": 6,
            "new_complication": 7,
            "critical_finding": 8,
        }

        imaging_instability_score = int(instability_map.get(final_read, 0))
        imaging_instability_score += int(poor_image_quality_flag) * 2
        imaging_instability_score += int(repeat_imaging_flag)
        imaging_instability_score += int(discordant_read_flag) * 3
        imaging_instability_score += int(imaging_not_ordered_despite_risk_flag) * 4
        imaging_instability_score += int(result_acknowledgment_state in ["buried_in_chart", "handoff_missed_result", "not_acknowledged"]) * 2
        imaging_instability_score = int(np.clip(imaging_instability_score, 0, 12))

        deceptive_trajectory_flag = imaging_trajectory_state in [
            "false_imaging_reassurance",
            "occult_worsening_before_visibility",
            "imaging_lagging_clinical_decline",
            "worsening_despite_reassuring_scan",
        ]

        imaging_false_reassurance_flag = bool(
            (
                final_read in ["normal", "mild_abnormality", "partial_resolution", "possible_false_negative"]
                and hidden_instability >= 0.42
            )
            or radiology_interpretation_state in ["possible_false_reassurance", "radiology_undercalls_severity"]
            or imaging_trajectory_state in ["false_imaging_reassurance", "worsening_despite_reassuring_scan"]
            or (deceptive_stability >= 0.45 and final_read in ["normal", "mild_abnormality", "partial_resolution"])
        )

        imaging_clinical_discordance_score = _clip(
            hidden_instability * 0.20
            + deceptive_stability * 0.16
            + lab_pressure * 0.14
            + diagnosis_pressure * 0.13
            + treatment_masking * 0.10
            + float(final_read in ["normal", "mild_abnormality", "partial_resolution", "possible_false_negative"]) * 0.10
            + float(poor_image_quality_flag) * 0.07
            + float(discordant_read_flag) * 0.10,
            0,
            1,
        )

        imaging_delay_pressure_score = _clip(
            result_delay_minutes / 1440 * 0.34
            + float(result_acknowledgment_state in ["acknowledged_late", "buried_in_chart", "handoff_missed_result", "not_acknowledged"]) * 0.24
            + float(imaging_not_ordered_despite_risk_flag) * 0.20
            + overload_pressure * 0.12
            + float(delayed_reassessment or delayed_escalation) * 0.10,
            0,
            1,
        )

        imaging_deception_pressure_score = _clip(
            hidden_instability * 0.24
            + deceptive_stability * 0.18
            + float(poor_image_quality_flag) * 0.10
            + float(discordant_read_flag) * 0.14
            + float(deceptive_trajectory_flag) * 0.18
            + float(imaging_false_reassurance_flag) * 0.12
            + imaging_delay_pressure_score * 0.04,
            0,
            1,
        )

        occult_imaging_progression_score = _clip(
            hidden_instability * 0.22
            + rebound * 0.14
            + deterioration * 0.14
            + float(imaging_trajectory_state in ["occult_worsening_before_visibility", "imaging_lagging_clinical_decline"]) * 0.20
            + imaging_clinical_discordance_score * 0.16
            + diagnosis_pressure * 0.08
            + lab_pressure * 0.06,
            0,
            1,
        )

        imaging_recovery_contradiction_score = _clip(
            float(final_read in ["partial_resolution", "normal", "mild_abnormality"]) * 0.14
            + hidden_instability * 0.18
            + rebound * 0.16
            + treatment_masking * 0.16
            + float(imaging_false_reassurance_flag) * 0.18
            + float(imaging_trajectory_state in ["post_treatment_apparent_improvement", "worsening_despite_reassuring_scan"]) * 0.18,
            0,
            1,
        )

        imaging_systemic_miss_support_flag = bool(
            (imaging_instability_score >= 4 and delayed_escalation)
            or (imaging_false_reassurance_flag and hidden_instability >= 0.45)
            or (imaging_delay_pressure_score >= 0.45 and diagnosis_pressure >= 0.35)
            or (imaging_not_ordered_despite_risk_flag and hidden_instability >= 0.45)
        )

        imaging_requires_bire_attention_flag = bool(
            imaging_instability_score >= 6
            or imaging_deception_pressure_score >= 0.45
            or imaging_false_reassurance_flag
            or imaging_systemic_miss_support_flag
            or occult_imaging_progression_score >= 0.50
        )

        imaging_rows.append(
            {
                **row,
                "imaging_ordered": bool(imaging_ordered),
                "imaging_not_ordered_despite_risk_flag": bool(imaging_not_ordered_despite_risk_flag),
                "imaging_type": imaging_type,
                "preliminary_read": preliminary_read,
                "final_read": final_read,
                "imaging_result_category": final_read,
                "discordant_read_flag": bool(discordant_read_flag),
                "radiology_interpretation_state": radiology_interpretation_state,
                "result_delay_minutes": int(result_delay_minutes),
                "result_acknowledgment_state": result_acknowledgment_state,
                "limited_study_flag": bool(limited_study_flag),
                "motion_artifact_flag": bool(motion_artifact_flag),
                "poor_positioning_flag": bool(poor_positioning_flag),
                "incomplete_protocol_flag": bool(incomplete_protocol_flag),
                "poor_image_quality_flag": bool(poor_image_quality_flag),
                "repeat_imaging_flag": bool(repeat_imaging_flag),
                "imaging_trajectory_state": imaging_trajectory_state,
                "imaging_instability_score": int(imaging_instability_score),
                "imaging_false_reassurance_flag": bool(imaging_false_reassurance_flag),
                "imaging_clinical_discordance_score": round(imaging_clinical_discordance_score, 4),
                "imaging_delay_pressure_score": round(imaging_delay_pressure_score, 4),
                "imaging_deception_pressure_score": round(imaging_deception_pressure_score, 4),
                "occult_imaging_progression_score": round(occult_imaging_progression_score, 4),
                "imaging_recovery_contradiction_score": round(imaging_recovery_contradiction_score, 4),
                "imaging_systemic_miss_support_flag": bool(imaging_systemic_miss_support_flag),
                "imaging_requires_bire_attention_flag": bool(imaging_requires_bire_attention_flag),
            }
        )

    return pd.DataFrame(imaging_rows)
