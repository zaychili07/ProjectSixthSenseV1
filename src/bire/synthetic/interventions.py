"""
BIRE OS Synthetic Intervention Meta-Intelligence Engine

Chapter 53 doctrine:
No more babying BIRE OS.

Purpose:
Simulate intervention effects on physiology, uncertainty, confidence,
visibility, recovery authenticity, treatment dependency, masking,
rebound deterioration, failed stabilization, and BIRE OS therapeutic trust.

This module does not recommend treatment.
It creates synthetic intervention pressure so BIRE OS can learn:

- true recovery vs artificial recovery
- support-dependent stability
- treatment-masked deterioration
- rebound after temporary improvement
- failed stabilization
- confidence distortion after treatment
- evidence disagreement after intervention
- intervention-driven uncertainty
- therapeutic system pressure
- treatment dependency visibility loss
- recovery authenticity
- reassessment pressure

Doctrine:
We Detect What Others Miss.
"""

from __future__ import annotations

import ast
import numpy as np
import pandas as pd

from bire.synthetic.config import SYNTHETIC_ECOSYSTEM_CONFIG


INTERVENTION_TARGETS = {
    "oxygen": "oxygenation",
    "high_flow_oxygen": "advanced_oxygenation",
    "bipap": "ventilatory_support",
    "iv_fluids": "hemodynamics",
    "fluid_bolus": "hemodynamics",
    "insulin": "metabolic_control",
    "dextrose": "metabolic_rescue",
    "antibiotics": "infection_control",
    "broad_spectrum_antibiotics": "critical_infection_control",
    "vasopressors": "blood_pressure_support",
    "morphine": "pain_control",
    "dilaudid": "pain_control",
    "bronchodilator": "airway_resistance",
    "steroids": "inflammation_control",
    "anticoagulant": "thrombotic_risk",
    "antiarrhythmic": "rhythm_control",
    "sedative": "sedation_control",
    "diuretic": "fluid_management",
    "antipyretic": "temperature_control",
    "blood_transfusion": "oxygen_carrying_capacity",
}


EXPECTED_RESPONSE_WINDOWS_MIN = {
    "oxygen": 15,
    "high_flow_oxygen": 15,
    "bipap": 20,
    "iv_fluids": 30,
    "fluid_bolus": 20,
    "insulin": 60,
    "dextrose": 15,
    "antibiotics": 240,
    "broad_spectrum_antibiotics": 180,
    "vasopressors": 15,
    "morphine": 30,
    "dilaudid": 30,
    "bronchodilator": 30,
    "steroids": 180,
    "anticoagulant": 360,
    "antiarrhythmic": 60,
    "sedative": 30,
    "diuretic": 90,
    "antipyretic": 60,
    "blood_transfusion": 120,
}


def _clip(value, low=0.0, high=1.0):
    return float(np.clip(value, low, high))


def _safe_get(row, col, default=0.0):
    value = row.get(col, default)
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    return value


def _safe_float(row, col, default=0.0):
    value = _safe_get(row, col, default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                text = text.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
                return [x.strip() for x in text.split(",") if x.strip()]
        if "," in text:
            return [x.strip() for x in text.split(",") if x.strip()]
        return [text]
    return []


def _join(items):
    if not items:
        return "none"
    return " | ".join(sorted(set([str(x) for x in items if str(x)])))


def _targets_from_meds(meds):
    return sorted({INTERVENTION_TARGETS.get(med, "unknown") for med in meds})


def _expected_response_window(meds):
    windows = [
        EXPECTED_RESPONSE_WINDOWS_MIN.get(med)
        for med in meds
        if med in EXPECTED_RESPONSE_WINDOWS_MIN
    ]
    return int(min(windows)) if windows else 0


def _support_intensity_score(meds):
    weights = {
        "oxygen": 0.10,
        "high_flow_oxygen": 0.22,
        "bipap": 0.26,
        "vasopressors": 0.32,
        "sedative": 0.16,
        "iv_fluids": 0.08,
        "fluid_bolus": 0.13,
        "blood_transfusion": 0.16,
        "dextrose": 0.08,
        "insulin": 0.07,
        "antiarrhythmic": 0.10,
    }
    return _clip(sum(weights.get(med, 0.0) for med in meds))


def _pre_intervention_snapshot(row):
    cols = [
        "heart_rate", "resp_rate", "spo2", "temperature",
        "sbp", "dbp", "glucose", "lactate", "wbc",
        "creatinine", "bicarbonate", "ph", "pco2",
        "troponin", "hemoglobin",
    ]
    return {f"pre_intervention_{col}": row[col] for col in cols if col in row}


def _calculate_intervention_response(row, meds, support_intensity, rng):
    fragility = _safe_float(row, "fragility_score", 0.0)
    deterioration = _safe_float(row, "deterioration_tendency", 0.0)
    recovery = _safe_float(row, "recovery_resilience", 0.5)
    hidden_instability = _safe_float(row, "hidden_instability_score", 0.0)
    silent_pressure = _safe_float(row, "silent_collapse_pressure", 0.0)
    critical_labs = _safe_float(row, "critical_lab_burden", 0.0)
    lab_burden = _safe_float(row, "lab_abnormality_burden", 0.0)
    imaging_instability = _safe_float(row, "imaging_instability_score", 0.0)
    diagnosis_pressure = _safe_float(row, "diagnosis_risk_pressure_score", 0.0)
    medication_burden = _safe_float(row, "medication_burden_score", 0.0)
    data_pressure = _safe_float(row, "data_trust_pressure_score", 0.0)

    high_acuity_support = any(
        med in meds
        for med in ["high_flow_oxygen", "bipap", "vasopressors", "blood_transfusion"]
    )

    response = (
        recovery * 0.58
        - deterioration * 0.24
        - fragility * 0.18
        - hidden_instability * 0.18
        - silent_pressure * 0.10
        - min(critical_labs, 5) * 0.040
        - min(lab_burden, 10) * 0.010
        - min(imaging_instability, 10) * 0.012
        - diagnosis_pressure * 0.12
        - min(medication_burden, 18) * 0.010
        - data_pressure * 0.08
        + support_intensity * 0.08
        + float(high_acuity_support) * 0.03
    )

    response += rng.normal(0, 0.10)
    return _clip(response, -1.0, 1.0)


def _classify_response(score):
    if score >= 0.42:
        return "strong_response"
    if score >= 0.15:
        return "partial_response"
    if score >= -0.12:
        return "minimal_response"
    return "failed_response"


def _apply_physiologic_effects(row, meds, response_multiplier, rng):
    notes = []
    flags = {
        "treatment_dependent_stability_flag": False,
        "masked_deterioration_flag": False,
        "iatrogenic_complication_flag": False,
        "overcorrection_flag": False,
        "undertreatment_flag": False,
        "requires_reassessment_flag": False,
        "failed_stabilization_flag": False,
        "temporary_improvement_flag": False,
        "treatment_dependency_visibility_loss": False,
    }

    treatment_failure_reason = "none"

    if any(m in meds for m in ["oxygen", "high_flow_oxygen", "bipap"]) and "spo2" in row:
        boost = 2.8
        if "high_flow_oxygen" in meds:
            boost += 2.2
        if "bipap" in meds:
            boost += 2.6

        row["spo2"] = round(_clip(row["spo2"] + rng.normal(boost, 1.1) * response_multiplier, 40, 100), 2)
        flags["treatment_dependent_stability_flag"] = True
        notes.append("respiratory_support_modified_oxygenation")

        if _safe_float(row, "resp_rate", 0) >= 24 and _safe_float(row, "spo2", 0) >= 94:
            flags["masked_deterioration_flag"] = True
            flags["requires_reassessment_flag"] = True
            flags["treatment_dependency_visibility_loss"] = True
            notes.append("spo2_improved_but_work_of_breathing_remains_high")

    if any(m in meds for m in ["iv_fluids", "fluid_bolus"]) and "sbp" in row:
        boost = rng.normal(8, 3.5)
        if "fluid_bolus" in meds:
            boost += rng.normal(5, 2)

        row["sbp"] = round(_clip(row["sbp"] + boost * response_multiplier, 40, 260), 2)

        if "dbp" in row:
            row["dbp"] = round(_clip(row["dbp"] + rng.normal(3, 2) * response_multiplier, 20, 160), 2)

        flags["treatment_dependent_stability_flag"] = True
        notes.append("fluid_support_modified_blood_pressure")

        if response_multiplier <= 0.35:
            flags["undertreatment_flag"] = True
            treatment_failure_reason = "limited_response_to_fluids"

        if rng.random() < 0.10 and "resp_rate" in row:
            row["resp_rate"] = round(_clip(row["resp_rate"] + rng.normal(2, 1), 4, 60), 2)
            notes.append("fluid_support_possible_respiratory_strain")

    if "vasopressors" in meds and "sbp" in row:
        row["sbp"] = round(_clip(row["sbp"] + rng.normal(16, 5) * max(response_multiplier, 0.50), 50, 280), 2)

        if "dbp" in row:
            row["dbp"] = round(_clip(row["dbp"] + rng.normal(6, 3), 25, 170), 2)

        row["intervention_hemodynamic_support"] = True
        flags["treatment_dependent_stability_flag"] = True
        flags["requires_reassessment_flag"] = True
        flags["treatment_dependency_visibility_loss"] = True
        notes.append("vasopressor_supported_pressure_not_intrinsic_recovery")

        if row["sbp"] >= 90:
            flags["masked_deterioration_flag"] = True

        if "lactate" in row and rng.random() < 0.20:
            row["lactate"] = round(_clip(row["lactate"] + rng.normal(0.35, 0.2), 0.2, 15), 2)
            notes.append("perfusion_mismatch_possible_despite_bp_support")
    else:
        row["intervention_hemodynamic_support"] = False

    if "bronchodilator" in meds and "resp_rate" in row:
        row["resp_rate"] = round(_clip(row["resp_rate"] - rng.normal(3.0, 1.2) * response_multiplier, 4, 60), 2)
        notes.append("bronchodilator_modified_respiratory_rate")

    if "insulin" in meds and "glucose" in row:
        row["glucose"] = round(_clip(row["glucose"] - rng.normal(45, 18) * response_multiplier, 35, 700), 2)
        notes.append("insulin_modified_glucose")
        if row["glucose"] < 70:
            flags["overcorrection_flag"] = True
            flags["iatrogenic_complication_flag"] = True
            notes.append("hypoglycemic_overcorrection_possible")

    if "dextrose" in meds and "glucose" in row:
        row["glucose"] = round(_clip(row["glucose"] + rng.normal(35, 12) * response_multiplier, 35, 700), 2)
        notes.append("dextrose_modified_glucose")

    if any(m in meds for m in ["antibiotics", "broad_spectrum_antibiotics"]):
        if "wbc" in row:
            row["wbc"] = round(_clip(row["wbc"] - rng.normal(1.1, 0.7) * response_multiplier, 0.2, 60), 2)
        if "lactate" in row:
            row["lactate"] = round(_clip(row["lactate"] - rng.normal(0.30, 0.20) * response_multiplier, 0.2, 15), 2)

        notes.append("antibiotic_effect_may_be_delayed_or_partial")

        if response_multiplier <= 0.35:
            flags["undertreatment_flag"] = True
            treatment_failure_reason = "limited_response_to_antibiotics"

    opioid_present = any(m in meds for m in ["morphine", "dilaudid"])

    if opioid_present:
        if "resp_rate" in row:
            row["resp_rate"] = round(_clip(row["resp_rate"] - rng.normal(2.2, 1.0), 3, 60), 2)

        row["opioid_respiratory_suppression_effect"] = True
        flags["treatment_dependency_visibility_loss"] = True
        notes.append("opioid_may_suppress_respiratory_signal")

        if _safe_float(row, "resp_rate", 20) < 10:
            flags["iatrogenic_complication_flag"] = True
            flags["masked_deterioration_flag"] = True
            notes.append("opioid_associated_low_respiratory_rate")
    else:
        row["opioid_respiratory_suppression_effect"] = False

    if "sedative" in meds:
        if "heart_rate" in row:
            row["heart_rate"] = round(_clip(row["heart_rate"] - rng.normal(6, 3), 25, 240), 2)
        if "resp_rate" in row:
            row["resp_rate"] = round(_clip(row["resp_rate"] - rng.normal(1.5, 0.8), 3, 60), 2)

        flags["treatment_dependency_visibility_loss"] = True
        notes.append("sedation_may_mask_instability")

        if opioid_present:
            flags["iatrogenic_complication_flag"] = True
            flags["masked_deterioration_flag"] = True
            notes.append("opioid_sedative_combination_increases_respiratory_risk")

    if "steroids" in meds:
        if "glucose" in row:
            row["glucose"] = round(_clip(row["glucose"] + rng.normal(20, 8), 35, 750), 2)
        if "wbc" in row:
            row["wbc"] = round(_clip(row["wbc"] + rng.normal(1.0, 0.6), 0.2, 60), 2)
        notes.append("steroids_may_distort_glucose_or_wbc")

    if "antiarrhythmic" in meds and "heart_rate" in row:
        row["heart_rate"] = round(_clip(row["heart_rate"] - rng.normal(8, 4) * response_multiplier, 25, 240), 2)
        notes.append("antiarrhythmic_modified_heart_rate")
        if row["heart_rate"] < 50:
            flags["overcorrection_flag"] = True
            notes.append("bradycardic_overcorrection_possible")

    if "diuretic" in meds:
        if "sbp" in row:
            row["sbp"] = round(_clip(row["sbp"] - rng.normal(4, 2), 40, 260), 2)
        if "creatinine" in row and rng.random() < 0.22:
            row["creatinine"] = round(_clip(row["creatinine"] + rng.normal(0.25, 0.12), 0.2, 10), 2)
        notes.append("diuretic_may_improve_congestion_but_stress_renal_function")

    if "anticoagulant" in meds:
        notes.append("anticoagulant_context_requires_bleeding_risk_monitoring")
        fragility = _safe_float(row, "fragility_score", 0.0)
        if rng.random() < (0.02 + fragility * 0.05):
            flags["iatrogenic_complication_flag"] = True
            notes.append("bleeding_risk_complication_possible")

    if "blood_transfusion" in meds:
        if "hemoglobin" in row:
            row["hemoglobin"] = round(_clip(row["hemoglobin"] + rng.normal(1.0, 0.4), 5, 19), 2)
        flags["treatment_dependent_stability_flag"] = True
        notes.append("transfusion_modified_oxygen_carrying_capacity")

    return row, flags, treatment_failure_reason, notes


def _calculate_post_intervention_signal_change(row):
    signal_pairs = [
        ("heart_rate", "lower"),
        ("resp_rate", "lower"),
        ("spo2", "higher"),
        ("sbp", "higher"),
        ("glucose", "normalize"),
        ("lactate", "lower"),
        ("wbc", "lower"),
        ("creatinine", "lower"),
    ]

    improved = 0
    worsened = 0
    unchanged = 0

    for col, direction in signal_pairs:
        pre_col = f"pre_intervention_{col}"
        if col not in row or pre_col not in row:
            continue

        pre = row.get(pre_col)
        post = row.get(col)

        if pre is None or post is None:
            continue

        try:
            if pd.isna(pre) or pd.isna(post):
                continue
        except Exception:
            continue

        delta = float(post) - float(pre)

        if direction == "lower":
            if delta <= -1:
                improved += 1
            elif delta >= 1:
                worsened += 1
            else:
                unchanged += 1

        elif direction == "higher":
            if delta >= 1:
                improved += 1
            elif delta <= -1:
                worsened += 1
            else:
                unchanged += 1

        elif direction == "normalize":
            target = 110 if col == "glucose" else pre
            if abs(post - target) < abs(pre - target):
                improved += 1
            elif abs(post - target) > abs(pre - target):
                worsened += 1
            else:
                unchanged += 1

    total = max(improved + worsened + unchanged, 1)
    return improved, worsened, unchanged, round((improved - worsened) / total, 4)


def _derive_recovery_authenticity(
    masked_deterioration,
    treatment_dependent,
    failed_stabilization,
    temporary_improvement,
    iatrogenic,
    support_intensity,
    response_state,
    rebound_risk,
    stabilization_durability,
    treatment_masking_risk,
):
    if failed_stabilization or iatrogenic:
        return "RECOVERY_NOT_TRUSTWORTHY"

    if masked_deterioration and treatment_dependent:
        return "ARTIFICIAL_RECOVERY_PATTERN"

    if treatment_masking_risk >= 0.60 and stabilization_durability <= 0.45:
        return "FALSE_RECOVERY_PATTERN"

    if temporary_improvement and rebound_risk >= 0.40:
        return "FALSE_RECOVERY_PATTERN"

    if treatment_dependent or support_intensity >= 0.35:
        return "SUPPORT_DEPENDENT_STABILITY"

    if response_state == "strong_response" and rebound_risk < 0.25 and stabilization_durability >= 0.65:
        return "TRUE_RECOVERY_LIKELY"

    return "PARTIAL_RECOVERY_UNCERTAIN"


def _derive_confidence_trajectory(
    recovery_authenticity_state,
    masked_deterioration,
    treatment_dependent,
    failed_stabilization,
    temporary_improvement,
    signal_conflict,
    rebound_risk,
    treatment_masking_risk,
):
    if failed_stabilization and rebound_risk >= 0.50:
        return "CONFIDENCE_COLLAPSE_AFTER_REBOUND"

    if signal_conflict >= 0.55:
        return "MULTI_SIGNAL_DISAGREEMENT"

    if treatment_masking_risk >= 0.60:
        return "INTERVENTION_DISTORTED_VISIBILITY"

    if recovery_authenticity_state == "TRUE_RECOVERY_LIKELY":
        return "HIGH_CONFIDENCE_TRUE_STABILIZATION"

    if recovery_authenticity_state == "SUPPORT_DEPENDENT_STABILITY":
        return "FALSE_CONFIDENCE_AFTER_SUPPORT"

    if masked_deterioration:
        return "INTERVENTION_DISTORTED_VISIBILITY"

    if temporary_improvement:
        return "MASKED_RECOVERY_UNCERTAINTY"

    if treatment_dependent:
        return "UNSTABLE_RESPONSE_CONFIDENCE"

    return "MODERATE_CONFIDENCE_PARTIAL_RESPONSE"


def apply_intervention_effects(df, random_seed=None):
    """
    Apply synthetic intervention effects and therapeutic meta-intelligence.

    Input
    -----
    DataFrame from medication / diagnosis / imaging / labs / context layers.

    Output
    ------
    DataFrame with intervention-derived pressure fields used by outcomes.py.
    """

    if random_seed is None:
        random_seed = SYNTHETIC_ECOSYSTEM_CONFIG["random_seed"]

    rng = np.random.default_rng(random_seed)
    updated_rows = []

    for row in df.to_dict("records"):
        meds = _as_list(row.get("medications_administered", []))

        for key, value in _pre_intervention_snapshot(row).items():
            row[key] = value

        post_event_state = row.get("post_event_state", "NO_EVENT")

        fragility = _safe_float(row, "fragility_score", 0.0)
        deterioration = _safe_float(row, "deterioration_tendency", 0.0)
        recovery = _safe_float(row, "recovery_resilience", 0.5)

        hidden_instability = _safe_float(row, "hidden_instability_score", 0.0)
        silent_pressure = _safe_float(row, "silent_collapse_pressure", 0.0)
        masked_vitals = bool(_safe_get(row, "masked_instability_signal", False))
        physiologic_deception = bool(_safe_get(row, "physiologic_deception_flag", False))

        lab_pressure = _safe_float(row, "lab_sixth_sense_pressure_score", 0.0)
        critical_labs = _safe_float(row, "critical_lab_burden", 0.0)
        imaging_pressure = _safe_float(row, "imaging_deception_pressure_score", 0.0)
        imaging_instability = _safe_float(row, "imaging_instability_score", 0.0)
        diagnosis_pressure = _safe_float(row, "diagnosis_risk_pressure_score", 0.0)
        data_pressure = _safe_float(row, "data_trust_pressure_score", 0.0)

        intervention_targets = _targets_from_meds(meds)
        expected_response_window_minutes = _expected_response_window(meds)
        support_intensity = _support_intensity_score(meds)

        intervention_response_score = _calculate_intervention_response(
            row=row,
            meds=meds,
            support_intensity=support_intensity,
            rng=rng,
        )
        response_state = _classify_response(intervention_response_score)

        response_multiplier = {
            "strong_response": 1.00,
            "partial_response": 0.65,
            "minimal_response": 0.35,
            "failed_response": 0.15,
        }[response_state]

        row, flags, treatment_failure_reason, notes = _apply_physiologic_effects(
            row=row,
            meds=meds,
            response_multiplier=response_multiplier,
            rng=rng,
        )

        improved, worsened, unchanged, signal_balance = _calculate_post_intervention_signal_change(row)

        high_acuity_support = any(
            med in meds
            for med in ["high_flow_oxygen", "bipap", "vasopressors", "blood_transfusion"]
        )

        evidence_conflict_after_intervention = _clip(
            hidden_instability * 0.16
            + silent_pressure * 0.10
            + lab_pressure * 0.14
            + min(critical_labs, 5) * 0.035
            + imaging_pressure * 0.12
            + min(imaging_instability, 10) * 0.012
            + diagnosis_pressure * 0.14
            + data_pressure * 0.10
            + float(worsened > 0) * 0.07
            + float(masked_vitals) * 0.07
            + float(physiologic_deception) * 0.06
        )

        clinical_signal_suppression_score = _clip(
            support_intensity * 0.30
            + float(flags["masked_deterioration_flag"]) * 0.24
            + float(flags["treatment_dependent_stability_flag"]) * 0.16
            + float(flags["treatment_dependency_visibility_loss"]) * 0.18
            + float("sedative" in meds) * 0.10
            + float(("morphine" in meds) or ("dilaudid" in meds)) * 0.10
            + float("vasopressors" in meds) * 0.08
        )

        treatment_masking_risk = _clip(
            clinical_signal_suppression_score * 0.30
            + evidence_conflict_after_intervention * 0.20
            + hidden_instability * 0.18
            + silent_pressure * 0.10
            + support_intensity * 0.12
            + float(high_acuity_support) * 0.08
            + float(masked_vitals or physiologic_deception) * 0.12
        )

        therapeutic_system_pressure_score = _clip(
            support_intensity * 0.20
            + treatment_masking_risk * 0.22
            + evidence_conflict_after_intervention * 0.18
            + deterioration * 0.12
            + fragility * 0.10
            + diagnosis_pressure * 0.08
            + data_pressure * 0.05
            + float(expected_response_window_minutes >= 180) * 0.05
        )

        if response_state == "failed_response":
            flags["requires_reassessment_flag"] = True
            if post_event_state in ["DECLINING_MONITOR", "VOLATILE_MONITOR", "RECOVERY_PENDING", "TEMPORARY_STABILIZATION"]:
                flags["failed_stabilization_flag"] = True
            if deterioration >= 0.55 or hidden_instability >= 0.50 or therapeutic_system_pressure_score >= 0.55:
                flags["failed_stabilization_flag"] = True
            if treatment_failure_reason == "none":
                treatment_failure_reason = "global_failed_response"

        if response_state == "partial_response" and rng.random() < (0.25 + treatment_masking_risk * 0.25):
            flags["temporary_improvement_flag"] = True
            flags["requires_reassessment_flag"] = True

        stabilization_durability_score = _clip(
            recovery * 0.32
            + max(intervention_response_score, 0) * 0.18
            + signal_balance * 0.12
            - treatment_masking_risk * 0.20
            - therapeutic_system_pressure_score * 0.16
            - hidden_instability * 0.12
            - deterioration * 0.10
            - float(flags["failed_stabilization_flag"]) * 0.18
            - float(flags["iatrogenic_complication_flag"]) * 0.16
        )

        rebound_deterioration_risk = _clip(
            float(flags["temporary_improvement_flag"]) * 0.26
            + float(flags["failed_stabilization_flag"]) * 0.36
            + float(flags["treatment_dependent_stability_flag"]) * 0.14
            + float(flags["masked_deterioration_flag"]) * 0.18
            + treatment_masking_risk * 0.18
            + (1 - stabilization_durability_score) * 0.16
            + fragility * 0.14
            + deterioration * 0.12
            + hidden_instability * 0.12
            + therapeutic_system_pressure_score * 0.10
        )

        intervention_uncertainty_score = _clip(
            treatment_masking_risk * 0.20
            + evidence_conflict_after_intervention * 0.18
            + therapeutic_system_pressure_score * 0.16
            + float(flags["masked_deterioration_flag"]) * 0.14
            + float(flags["treatment_dependent_stability_flag"]) * 0.10
            + float(flags["failed_stabilization_flag"]) * 0.16
            + float(flags["iatrogenic_complication_flag"]) * 0.14
            + float(flags["temporary_improvement_flag"]) * 0.08
            + float(expected_response_window_minutes >= 180) * 0.04
        )

        treatment_response_reliability = _clip(
            0.72
            + intervention_response_score * 0.18
            + signal_balance * 0.12
            + stabilization_durability_score * 0.12
            - treatment_masking_risk * 0.20
            - intervention_uncertainty_score * 0.18
            - therapeutic_system_pressure_score * 0.12
            - data_pressure * 0.08
        )

        intervention_visibility_distortion = _clip(
            clinical_signal_suppression_score * 0.32
            + treatment_masking_risk * 0.28
            + evidence_conflict_after_intervention * 0.18
            + hidden_instability * 0.12
            + data_pressure * 0.06
            + imaging_pressure * 0.04
        )

        recovery_authenticity_state = _derive_recovery_authenticity(
            masked_deterioration=flags["masked_deterioration_flag"],
            treatment_dependent=flags["treatment_dependent_stability_flag"],
            failed_stabilization=flags["failed_stabilization_flag"],
            temporary_improvement=flags["temporary_improvement_flag"],
            iatrogenic=flags["iatrogenic_complication_flag"],
            support_intensity=support_intensity,
            response_state=response_state,
            rebound_risk=rebound_deterioration_risk,
            stabilization_durability=stabilization_durability_score,
            treatment_masking_risk=treatment_masking_risk,
        )

        confidence_trajectory_state = _derive_confidence_trajectory(
            recovery_authenticity_state=recovery_authenticity_state,
            masked_deterioration=flags["masked_deterioration_flag"],
            treatment_dependent=flags["treatment_dependent_stability_flag"],
            failed_stabilization=flags["failed_stabilization_flag"],
            temporary_improvement=flags["temporary_improvement_flag"],
            signal_conflict=evidence_conflict_after_intervention,
            rebound_risk=rebound_deterioration_risk,
            treatment_masking_risk=treatment_masking_risk,
        )

        intervention_confidence_shift = _clip(
            treatment_response_reliability
            - intervention_uncertainty_score
            - clinical_signal_suppression_score * 0.22,
            -1,
            1,
        )

        therapeutic_confidence_instability = _clip(
            intervention_uncertainty_score * 0.26
            + evidence_conflict_after_intervention * 0.22
            + treatment_masking_risk * 0.20
            + rebound_deterioration_risk * 0.16
            + float(recovery_authenticity_state != "TRUE_RECOVERY_LIKELY") * 0.10
            + float(flags["treatment_dependency_visibility_loss"]) * 0.06
        )

        post_intervention_uncertainty_pressure = _clip(
            therapeutic_confidence_instability * 0.32
            + intervention_visibility_distortion * 0.24
            + therapeutic_system_pressure_score * 0.20
            + data_pressure * 0.10
            + diagnosis_pressure * 0.08
            + imaging_pressure * 0.06
        )

        treatment_dependency_visibility_loss = bool(
            flags["treatment_dependency_visibility_loss"]
            or (
                support_intensity >= 0.35
                and treatment_masking_risk >= 0.45
            )
            or recovery_authenticity_state in [
                "SUPPORT_DEPENDENT_STABILITY",
                "ARTIFICIAL_RECOVERY_PATTERN",
                "FALSE_RECOVERY_PATTERN",
            ]
        )

        confidence_destabilization_flag = bool(
            therapeutic_confidence_instability >= 0.45
            or confidence_trajectory_state in [
                "FALSE_CONFIDENCE_AFTER_SUPPORT",
                "MASKED_RECOVERY_UNCERTAINTY",
                "INTERVENTION_DISTORTED_VISIBILITY",
                "CONFIDENCE_COLLAPSE_AFTER_REBOUND",
                "MULTI_SIGNAL_DISAGREEMENT",
            ]
        )

        intervention_meta_attention_required_flag = bool(
            confidence_destabilization_flag
            or flags["requires_reassessment_flag"]
            or flags["iatrogenic_complication_flag"]
            or rebound_deterioration_risk >= 0.45
            or intervention_uncertainty_score >= 0.45
            or recovery_authenticity_state in [
                "ARTIFICIAL_RECOVERY_PATTERN",
                "FALSE_RECOVERY_PATTERN",
                "RECOVERY_NOT_TRUSTWORTHY",
            ]
        )

        if flags["failed_stabilization_flag"]:
            intervention_trust_state = "unstable_failed_response"
        elif treatment_dependency_visibility_loss:
            intervention_trust_state = "treatment_supported_stability"
        elif flags["temporary_improvement_flag"]:
            intervention_trust_state = "temporary_improvement_monitor_closely"
        elif treatment_masking_risk >= 0.50:
            intervention_trust_state = "response_visibility_distorted"
        elif response_state == "strong_response" and stabilization_durability_score >= 0.65:
            intervention_trust_state = "response_appears_trustworthy"
        else:
            intervention_trust_state = "response_uncertain"

        therapeutic_failure_modes = []

        if flags["failed_stabilization_flag"]:
            therapeutic_failure_modes.append("failed_stabilization")
        if flags["masked_deterioration_flag"]:
            therapeutic_failure_modes.append("masked_deterioration")
        if treatment_dependency_visibility_loss:
            therapeutic_failure_modes.append("treatment_dependency_visibility_loss")
        if flags["iatrogenic_complication_flag"]:
            therapeutic_failure_modes.append("iatrogenic_complication")
        if flags["temporary_improvement_flag"]:
            therapeutic_failure_modes.append("temporary_improvement")
        if treatment_masking_risk >= 0.50:
            therapeutic_failure_modes.append("treatment_masking")
        if rebound_deterioration_risk >= 0.50:
            therapeutic_failure_modes.append("rebound_risk")
        if not therapeutic_failure_modes:
            therapeutic_failure_modes.append("none")

        row["intervention_targets"] = intervention_targets
        row["expected_response_window_minutes"] = expected_response_window_minutes
        row["support_intensity_score"] = round(support_intensity, 4)

        row["intervention_response_score"] = round(intervention_response_score, 4)
        row["intervention_response_state"] = response_state
        row["intervention_trust_state"] = intervention_trust_state

        row["treatment_dependent_stability_flag"] = bool(flags["treatment_dependent_stability_flag"])
        row["masked_deterioration_flag"] = bool(flags["masked_deterioration_flag"])
        row["iatrogenic_complication_flag"] = bool(flags["iatrogenic_complication_flag"])
        row["overcorrection_flag"] = bool(flags["overcorrection_flag"])
        row["undertreatment_flag"] = bool(flags["undertreatment_flag"])
        row["requires_reassessment_flag"] = bool(flags["requires_reassessment_flag"])
        row["failed_stabilization_flag"] = bool(flags["failed_stabilization_flag"])
        row["temporary_improvement_flag"] = bool(flags["temporary_improvement_flag"])
        row["treatment_dependency_visibility_loss"] = bool(treatment_dependency_visibility_loss)

        row["improved_signals_after_intervention"] = int(improved)
        row["worsened_signals_after_intervention"] = int(worsened)
        row["unchanged_signals_after_intervention"] = int(unchanged)
        row["post_intervention_signal_balance"] = round(signal_balance, 4)

        row["treatment_masking_risk"] = round(treatment_masking_risk, 4)
        row["stabilization_durability_score"] = round(stabilization_durability_score, 4)
        row["rebound_deterioration_risk"] = round(rebound_deterioration_risk, 4)
        row["intervention_uncertainty_score"] = round(intervention_uncertainty_score, 4)
        row["therapeutic_system_pressure_score"] = round(therapeutic_system_pressure_score, 4)

        row["treatment_failure_reason"] = treatment_failure_reason
        row["therapeutic_failure_mode"] = _join(therapeutic_failure_modes)
        row["intervention_effect_notes"] = notes

        row["evidence_conflict_after_intervention"] = round(evidence_conflict_after_intervention, 4)
        row["treatment_response_reliability"] = round(treatment_response_reliability, 4)
        row["intervention_visibility_distortion"] = round(intervention_visibility_distortion, 4)
        row["clinical_signal_suppression_score"] = round(clinical_signal_suppression_score, 4)
        row["intervention_confidence_shift"] = round(intervention_confidence_shift, 4)
        row["therapeutic_confidence_instability"] = round(therapeutic_confidence_instability, 4)
        row["post_intervention_uncertainty_pressure"] = round(post_intervention_uncertainty_pressure, 4)

        row["recovery_authenticity_state"] = recovery_authenticity_state
        row["confidence_trajectory_state"] = confidence_trajectory_state
        row["confidence_destabilization_flag"] = bool(confidence_destabilization_flag)
        row["intervention_meta_attention_required_flag"] = bool(intervention_meta_attention_required_flag)

        updated_rows.append(row)

    return pd.DataFrame(updated_rows)