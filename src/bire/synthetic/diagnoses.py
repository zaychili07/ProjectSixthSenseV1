"""
BIRE OS Synthetic Diagnostic Intelligence Engine

Chapter 53 doctrine:
No more babying BIRE OS.

Purpose:
Generate diagnostic reality under hospital chaos:
- simple diagnoses that evolve into critical diagnoses
- hidden diagnoses
- slow-starting diagnoses
- missed diagnoses
- unavailable / delayed diagnoses
- under-the-radar diagnoses
- medication-induced secondary diagnoses
- premature diagnostic closure
- diagnosis false reassurance
- diagnosis confidence without truth
- diagnostic delay consequence pressure
- imaging/lab/vital diagnostic disagreement
- diagnostic revision
- late truth recognition
- lethal diagnostic pathways

This is not clinical truth.
This is documented diagnostic reality under pressure.

Doctrine:
We Detect What Others Miss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bire.synthetic.config import SYNTHETIC_ECOSYSTEM_CONFIG


CONDITION_TO_BASE_DIAGNOSES = {
    "baseline_stable": ["general_evaluation"],
    "copd_respiratory_vulnerability": ["copd", "chronic_respiratory_disease"],
    "chf_fluid_pressure_instability": ["congestive_heart_failure", "fluid_overload_risk"],
    "ckd_lab_instability": ["chronic_kidney_disease", "renal_insufficiency"],
    "diabetes_metabolic_instability": ["diabetes_mellitus", "hyperglycemia_risk"],
    "sepsis_recovery_risk": ["history_of_sepsis", "infection_risk"],
    "arrhythmia_hemodynamic_instability": ["arrhythmia", "hemodynamic_instability_risk"],
    "post_surgical_recovery": ["postoperative_recovery"],
    "chronic_multimorbidity": ["multimorbidity", "complex_chronic_disease"],
    "recurrent_deterioration_pattern": ["recurrent_clinical_deterioration"],
    "complex_pancreatitis_icu_complication": ["acute_pancreatitis", "icu_complication_risk"],
    "contrast_allergy_imaging_complication": ["contrast_allergy", "imaging_complication_risk"],
    "medication_restriction_complexity": ["medication_restriction", "polypharmacy_risk"],
    "new_onset_diabetes_during_admission": ["new_onset_diabetes_risk"],
    "multi_event_complex_hospitalization": ["complex_hospitalization", "multi_event_course"],
}


EVENT_TO_DIAGNOSES = {
    "respiratory_decline": ["acute_respiratory_decline", "hypoxemia"],
    "hemodynamic_instability": ["hemodynamic_instability", "hypotension"],
    "sepsis_like_deterioration": ["suspected_infection", "sepsis_like_syndrome"],
    "metabolic_instability": ["metabolic_derangement", "glucose_instability"],
    "arrhythmia_instability": ["arrhythmia_event", "cardiac_instability"],
    "renal_lab_instability": ["renal_deterioration", "acute_kidney_injury_risk"],
    "post_procedure_complication": ["post_procedure_complication"],
    "unknown_deterioration": ["undifferentiated_deterioration"],
    "NONE": [],
}


SIMPLE_TO_CRITICAL_DIAGNOSIS_PATHWAYS = {
    "general_evaluation": [
        "occult_sepsis",
        "silent_hypoperfusion",
        "missed_respiratory_failure",
        "hidden_cardiac_instability",
    ],
    "copd": [
        "acute_on_chronic_respiratory_failure",
        "hypercapnic_respiratory_failure",
        "oxygen_masked_respiratory_decline",
    ],
    "congestive_heart_failure": [
        "acute_decompensated_heart_failure",
        "pulmonary_edema",
        "cardiorenal_syndrome",
    ],
    "diabetes_mellitus": [
        "diabetic_ketoacidosis",
        "hyperosmolar_metabolic_crisis",
        "steroid_induced_hyperglycemic_instability",
    ],
    "chronic_kidney_disease": [
        "acute_on_chronic_kidney_failure",
        "hyperkalemic_instability",
        "uremic_instability",
    ],
    "arrhythmia": [
        "unstable_arrhythmia",
        "demand_ischemia",
        "hemodynamic_collapse_from_arrhythmia",
    ],
    "postoperative_recovery": [
        "postoperative_bleeding",
        "postoperative_sepsis",
        "postoperative_respiratory_failure",
    ],
}


HIDDEN_DIAGNOSES = [
    "occult_sepsis",
    "silent_hypoperfusion",
    "early_respiratory_failure",
    "occult_bleeding",
    "early_pulmonary_embolism",
    "subclinical_acute_kidney_injury",
    "silent_myocardial_injury",
    "hidden_metabolic_acidosis",
    "early_multi_organ_dysfunction",
    "underrecognized_delirium",
]


SLOW_BURN_DIAGNOSES = [
    "slow_evolving_sepsis",
    "progressive_respiratory_fatigue",
    "creeping_renal_failure",
    "gradual_perfusion_failure",
    "progressive_fluid_overload",
    "delayed_postoperative_complication",
    "slow_metabolic_decompensation",
    "progressive_treatment_failure",
]


MEDICATION_INDUCED_DIAGNOSES = {
    "opioid_respiratory_decline": "opioid_induced_respiratory_suppression",
    "vasopressor_perfusion_mismatch": "vasopressor_associated_perfusion_mismatch",
    "fluid_overload_respiratory_failure": "iatrogenic_fluid_overload",
    "steroid_hyperglycemic_instability": "steroid_induced_hyperglycemia",
    "anticoagulant_bleeding_instability": "anticoagulant_associated_bleeding",
    "sedation_masked_decline": "sedation_masked_neurologic_or_respiratory_decline",
    "renal_dosing_failure": "medication_related_renal_instability",
    "electrolyte_destabilization": "medication_associated_electrolyte_disturbance",
    "therapy_rebound_instability": "rebound_instability_after_support_change",
}


LETHAL_DIAGNOSIS_PATHWAYS = [
    "septic_shock",
    "refractory_shock",
    "acute_respiratory_failure",
    "multi_organ_failure",
    "massive_bleeding_instability",
    "malignant_arrhythmia",
    "terminal_decompensation",
    "cardiopulmonary_collapse",
]


DIAGNOSTIC_VISIBILITY_STATES = [
    "VISIBLE_DIAGNOSIS",
    "PARTIALLY_VISIBLE_DIAGNOSIS",
    "HIDDEN_DIAGNOSIS",
    "DELAYED_RECOGNITION_DIAGNOSIS",
    "MISLEADING_PRESENTATION_DIAGNOSIS",
    "UNDER_THE_RADAR_DIAGNOSIS",
]


DIAGNOSTIC_TRAJECTORY_STATES = [
    "STABLE_DIAGNOSTIC_CONTEXT",
    "SIMPLE_TO_COMPLEX_EVOLUTION",
    "SLOW_BURN_DIAGNOSTIC_EVOLUTION",
    "HIDDEN_CRITICAL_DIAGNOSIS",
    "MEDICATION_INDUCED_SECONDARY_DIAGNOSIS",
    "MISSED_DIAGNOSIS_UNTIL_DETERIORATION",
    "DIAGNOSIS_REVISED_AFTER_EVIDENCE",
    "LETHAL_DIAGNOSTIC_PATHWAY",
    "PREMATURE_DIAGNOSTIC_CLOSURE",
    "DIAGNOSTIC_FALSE_REASSURANCE",
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


def _unique(items):
    return list(dict.fromkeys([x for x in items if x and x != "none"]))


def _weighted_choice(rng, weights):
    keys = list(weights.keys())
    probs = np.array([max(weights[k], 0.001) for k in keys], dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(keys, p=probs))


def _select_simple_to_critical_pathway(base_diagnoses, rng):
    for dx in base_diagnoses:
        if dx in SIMPLE_TO_CRITICAL_DIAGNOSIS_PATHWAYS:
            return str(rng.choice(SIMPLE_TO_CRITICAL_DIAGNOSIS_PATHWAYS[dx]))
    return str(rng.choice(HIDDEN_DIAGNOSES))


def _derive_diagnostic_visibility(row, diagnostic_pressure, rng):
    hidden = _safe_float(row, "silent_collapse_pressure", 0.0)
    hidden_instability = _safe_float(row, "hidden_instability_score", 0.0)
    deception = _safe_float(row, "deceptive_stability_index", 0.0)
    evidence_trust = str(row.get("evidence_trust_state", "HIGH_TRUST"))
    imaging_traj = str(row.get("imaging_trajectory_state", "stable_imaging"))
    surface_reassurance = _safe_bool(row, "surface_level_reassurance_risk_flag", False)
    handoff_false_reassurance = _safe_bool(row, "false_reassurance_from_handoff_flag", False)
    operational_false_reassurance = _safe_bool(row, "operational_false_reassurance_flag", False)

    weights = {
        "VISIBLE_DIAGNOSIS": 0.34,
        "PARTIALLY_VISIBLE_DIAGNOSIS": 0.24,
        "HIDDEN_DIAGNOSIS": 0.10,
        "DELAYED_RECOGNITION_DIAGNOSIS": 0.12,
        "MISLEADING_PRESENTATION_DIAGNOSIS": 0.11,
        "UNDER_THE_RADAR_DIAGNOSIS": 0.09,
    }

    if hidden >= 0.45 or hidden_instability >= 0.45:
        weights["HIDDEN_DIAGNOSIS"] += 0.16
        weights["UNDER_THE_RADAR_DIAGNOSIS"] += 0.10

    if deception >= 0.45 or surface_reassurance or handoff_false_reassurance or operational_false_reassurance:
        weights["MISLEADING_PRESENTATION_DIAGNOSIS"] += 0.17
        weights["UNDER_THE_RADAR_DIAGNOSIS"] += 0.08

    if evidence_trust == "LOW_TRUST":
        weights["DELAYED_RECOGNITION_DIAGNOSIS"] += 0.12

    if imaging_traj in [
        "imaging_lagging_clinical_decline",
        "false_imaging_reassurance",
        "occult_worsening_before_visibility",
    ]:
        weights["DELAYED_RECOGNITION_DIAGNOSIS"] += 0.15
        weights["MISLEADING_PRESENTATION_DIAGNOSIS"] += 0.08

    if diagnostic_pressure >= 0.55:
        weights["PARTIALLY_VISIBLE_DIAGNOSIS"] += 0.10
        weights["DELAYED_RECOGNITION_DIAGNOSIS"] += 0.10

    return _weighted_choice(rng, weights)


def generate_diagnosis_timeline(df, random_seed=None):
    """
    Generate synthetic diagnostic timeline intelligence.
    """

    if random_seed is None:
        random_seed = SYNTHETIC_ECOSYSTEM_CONFIG["random_seed"]

    rng = np.random.default_rng(random_seed)
    diagnosis_rows = []

    patient_memory = {}

    sort_cols = ["patient_id"]
    if "encounter_start" in df.columns:
        sort_cols.append("encounter_start")

    sorted_df = df.sort_values(sort_cols).reset_index(drop=True)

    for row in sorted_df.to_dict("records"):
        patient_id = row.get("patient_id", "UNKNOWN_PATIENT")

        memory = patient_memory.get(
            patient_id,
            {
                "prior_hidden_dx": False,
                "prior_missed_dx": False,
                "prior_lethal_pathway": False,
                "prior_medication_induced_dx": False,
                "prior_false_reassurance_dx": False,
                "prior_premature_closure": False,
                "diagnostic_revision_count": 0,
            },
        )

        condition_profile = row.get("condition_profile", "baseline_stable")
        event_type = row.get("event_type", "NONE")
        has_event = _safe_bool(row, "has_event", False)
        care_mode = str(row.get("care_mode", "OUTPATIENT"))
        post_event_state = row.get("post_event_state", "NO_EVENT")

        lab_burden = _safe_int(row, "lab_abnormality_burden", 0)
        critical_lab_burden = _safe_int(row, "critical_lab_burden", 0)
        imaging_result = row.get("imaging_result_category", "none")
        medication_burden = _safe_int(row, "medication_burden_score", 0)
        secondary_deterioration_type = row.get("secondary_deterioration_type", "none")

        fragility = _safe_float(row, "fragility_score", 0.0)
        deterioration = _safe_float(row, "deterioration_tendency", 0.0)
        hidden_instability = _safe_float(row, "hidden_instability_score", 0.0)
        hidden_pressure = _safe_float(row, "silent_collapse_pressure", 0.0)
        deceptive_stability = _safe_float(row, "deceptive_stability_index", 0.0)

        lab_pressure = _safe_float(row, "lab_sixth_sense_pressure_score", 0.0)
        imaging_pressure = _safe_float(row, "imaging_deception_pressure_score", 0.0)
        imaging_instability = _safe_float(row, "imaging_instability_score", 0.0)
        therapeutic_pressure = _safe_float(row, "therapeutic_system_pressure_score", 0.0)
        treatment_masking_risk = _safe_float(row, "treatment_masking_risk", 0.0)
        intervention_uncertainty = _safe_float(row, "intervention_uncertainty_score", 0.0)
        rebound_risk = _safe_float(row, "rebound_deterioration_risk", 0.0)

        data_pressure = _safe_float(row, "data_trust_pressure_score", 0.0)
        context_pressure = _safe_float(row, "bire_context_pressure_score", 0.0)
        terminal_pressure = _safe_float(row, "terminal_decline_pressure", 0.0)

        operational_instability = _safe_float(row, "operational_instability_index", 0.0)
        hospital_failure_pressure = _safe_float(row, "hospital_system_failure_pressure", 0.0)
        resource_dependent_risk = _safe_float(row, "resource_dependent_deterioration_risk", 0.0)
        monitoring_blind_spot = _safe_float(row, "monitoring_blind_spot_score", 0.0)
        handoff_uncertainty = _safe_float(row, "handoff_uncertainty_score", 0.0)
        handoff_deception = _safe_float(row, "handoff_deception_pressure_score", 0.0)
        memory_decay = _safe_float(row, "operational_memory_decay_score", 0.0)

        delayed_escalation = _safe_bool(row, "delayed_escalation_flag", False)
        delayed_reassessment = _safe_bool(row, "delayed_reassessment_flag", False)
        operational_false_reassurance = _safe_bool(row, "operational_false_reassurance_flag", False)
        handoff_false_reassurance = _safe_bool(row, "false_reassurance_from_handoff_flag", False)
        hidden_trend_loss = _safe_bool(row, "hidden_trend_loss_flag", False)
        failed_stabilization = _safe_bool(row, "failed_stabilization_flag", False)
        masked_deterioration = _safe_bool(row, "masked_deterioration_flag", False)

        recovery_authenticity_state = row.get(
            "recovery_authenticity_state",
            "PARTIAL_RECOVERY_UNCERTAIN",
        )

        base_diagnoses = CONDITION_TO_BASE_DIAGNOSES.get(
            condition_profile,
            ["unspecified_clinical_context"],
        ).copy()

        event_diagnoses = EVENT_TO_DIAGNOSES.get(event_type, []).copy()
        secondary_diagnoses = []

        diagnostic_pressure_score = _clip(
            hidden_pressure * 0.14
            + hidden_instability * 0.16
            + deceptive_stability * 0.10
            + lab_pressure * 0.14
            + imaging_pressure * 0.12
            + therapeutic_pressure * 0.10
            + treatment_masking_risk * 0.08
            + operational_instability * 0.08
            + data_pressure * 0.08
        )

        physiologic_diagnostic_disagreement_score = _clip(
            hidden_instability * 0.18
            + deceptive_stability * 0.14
            + lab_pressure * 0.14
            + imaging_pressure * 0.12
            + treatment_masking_risk * 0.12
            + float(masked_deterioration) * 0.12
            + monitoring_blind_spot * 0.10
            + data_pressure * 0.08
        )

        diagnostic_operational_fragmentation_score = _clip(
            handoff_uncertainty * 0.16
            + handoff_deception * 0.14
            + memory_decay * 0.14
            + operational_instability * 0.14
            + hospital_failure_pressure * 0.12
            + monitoring_blind_spot * 0.12
            + float(delayed_reassessment) * 0.10
            + float(delayed_escalation) * 0.08
        )

        diagnosis_delay_consequence_pressure = _clip(
            delayed_reassessment * 0.16
            + delayed_escalation * 0.14
            + monitoring_blind_spot * 0.14
            + handoff_uncertainty * 0.12
            + data_pressure * 0.10
            + imaging_pressure * 0.10
            + lab_pressure * 0.10
            + operational_instability * 0.08
            + resource_dependent_risk * 0.06
        )

        if lab_burden >= 4:
            secondary_diagnoses.append("multi_lab_abnormality")

        if critical_lab_burden >= 1:
            secondary_diagnoses.append("critical_lab_abnormality")

        if imaging_result in ["critical_finding", "new_complication", "progressive_worsening"]:
            secondary_diagnoses.append("imaging_supported_complication")

        if imaging_result in ["possible_false_negative", "underappreciated_instability"]:
            secondary_diagnoses.append("imaging_underrecognized_instability")

        if medication_burden >= 12:
            secondary_diagnoses.append("high_therapeutic_burden")

        if care_mode == "ICU":
            secondary_diagnoses.append("critical_care_context")

        if post_event_state in ["VOLATILE_MONITOR", "DECLINING_MONITOR", "TEMPORARY_STABILIZATION"]:
            secondary_diagnoses.append("unstable_recovery_context")

        medication_induced_diagnosis = "none"
        medication_induced_diagnosis_flag = False

        if secondary_deterioration_type in MEDICATION_INDUCED_DIAGNOSES:
            medication_induced_diagnosis = MEDICATION_INDUCED_DIAGNOSES[secondary_deterioration_type]
            medication_induced_diagnosis_flag = True
            secondary_diagnoses.append(medication_induced_diagnosis)

        hidden_diagnosis_flag = False
        hidden_diagnosis_name = "none"

        hidden_probability = _clip(
            0.03
            + hidden_pressure * 0.18
            + hidden_instability * 0.16
            + deceptive_stability * 0.10
            + imaging_pressure * 0.08
            + lab_pressure * 0.08
            + treatment_masking_risk * 0.08
            + float(hidden_trend_loss) * 0.08
            + float(memory["prior_hidden_dx"]) * 0.06
        )

        if rng.random() < hidden_probability:
            hidden_diagnosis_flag = True
            hidden_diagnosis_name = str(rng.choice(HIDDEN_DIAGNOSES))
            secondary_diagnoses.append(hidden_diagnosis_name)

        slow_burn_diagnosis_flag = False
        slow_burn_diagnosis_name = "none"

        slow_burn_probability = _clip(
            0.04
            + _safe_float(row, "ten_year_instability_burden", 0.0) * 0.14
            + _safe_float(row, "recurrence_risk_score", 0.0) * 0.12
            + rebound_risk * 0.08
            + deterioration * 0.08
            + float(memory["prior_missed_dx"]) * 0.08
        )

        if rng.random() < slow_burn_probability:
            slow_burn_diagnosis_flag = True
            slow_burn_diagnosis_name = str(rng.choice(SLOW_BURN_DIAGNOSES))
            secondary_diagnoses.append(slow_burn_diagnosis_name)

        simple_to_critical_flag = False
        evolved_critical_diagnosis = "none"

        simple_to_critical_probability = _clip(
            0.025
            + diagnostic_pressure_score * 0.16
            + critical_lab_burden * 0.030
            + float(has_event) * 0.07
            + hidden_instability * 0.08
            + diagnosis_delay_consequence_pressure * 0.07
        )

        if rng.random() < simple_to_critical_probability:
            simple_to_critical_flag = True
            evolved_critical_diagnosis = _select_simple_to_critical_pathway(base_diagnoses, rng)
            secondary_diagnoses.append(evolved_critical_diagnosis)

        lethal_pathway_flag = False
        lethal_pathway_diagnosis = "none"

        lethal_probability = _clip(
            0.01
            + terminal_pressure * 0.20
            + critical_lab_burden * 0.035
            + _safe_int(row, "critical_vital_count", 0) * 0.030
            + float(_safe_bool(row, "critical_data_quality_failure_flag", False)) * 0.035
            + hospital_failure_pressure * 0.06
            + failed_stabilization * 0.08
            + float(memory["prior_lethal_pathway"]) * 0.06
        )

        if rng.random() < lethal_probability:
            lethal_pathway_flag = True
            lethal_pathway_diagnosis = str(rng.choice(LETHAL_DIAGNOSIS_PATHWAYS))
            secondary_diagnoses.append(lethal_pathway_diagnosis)

        diagnosis_revision_flag = False
        revised_from_diagnosis = "none"
        revised_to_diagnosis = "none"

        revision_probability = _clip(
            0.04
            + data_pressure * 0.10
            + imaging_pressure * 0.09
            + lab_pressure * 0.08
            + therapeutic_pressure * 0.07
            + physiologic_diagnostic_disagreement_score * 0.10
            + diagnostic_operational_fragmentation_score * 0.08
        )

        if rng.random() < revision_probability:
            diagnosis_revision_flag = True
            memory["diagnostic_revision_count"] += 1
            revised_from_diagnosis = str(rng.choice(base_diagnoses))
            revised_to_diagnosis = str(
                rng.choice(
                    _unique(
                        [hidden_diagnosis_name, slow_burn_diagnosis_name, evolved_critical_diagnosis]
                        + HIDDEN_DIAGNOSES
                    )
                )
            )
            secondary_diagnoses.append(revised_to_diagnosis)

        missed_diagnosis_flag = False

        missed_probability = _clip(
            0.02
            + hidden_pressure * 0.12
            + hidden_instability * 0.14
            + deceptive_stability * 0.10
            + data_pressure * 0.08
            + context_pressure * 0.06
            + hospital_failure_pressure * 0.08
            + monitoring_blind_spot * 0.08
            + handoff_uncertainty * 0.06
            + treatment_masking_risk * 0.06
            + float(memory["prior_missed_dx"]) * 0.05
        )

        if rng.random() < missed_probability:
            missed_diagnosis_flag = True

        diagnosis_delayed_flag = False
        diagnosis_chart_delay_minutes = 0

        delay_probability = _clip(
            0.05
            + float(care_mode in ["INPATIENT", "ICU"]) * 0.08
            + data_pressure * 0.08
            + context_pressure * 0.06
            + hospital_failure_pressure * 0.08
            + operational_instability * 0.06
            + delayed_reassessment * 0.08
            + missed_diagnosis_flag * 0.12
        )

        if rng.random() < delay_probability:
            diagnosis_delayed_flag = True
            diagnosis_chart_delay_minutes = int(
                rng.choice([60, 120, 240, 480, 720, 1440, 2880])
            )

        diagnostic_visibility_state = _derive_diagnostic_visibility(
            row,
            diagnostic_pressure_score,
            rng,
        )

        diagnosis_false_reassurance_flag = bool(
            (
                diagnostic_visibility_state in [
                    "MISLEADING_PRESENTATION_DIAGNOSIS",
                    "UNDER_THE_RADAR_DIAGNOSIS",
                    "DELAYED_RECOGNITION_DIAGNOSIS",
                ]
                and hidden_instability >= 0.42
            )
            or (
                handoff_false_reassurance
                and diagnostic_pressure_score >= 0.35
            )
            or (
                operational_false_reassurance
                and physiologic_diagnostic_disagreement_score >= 0.40
            )
            or (
                treatment_masking_risk >= 0.50
                and diagnostic_pressure_score >= 0.35
            )
        )

        premature_diagnostic_closure_flag = bool(
            (
                diagnostic_visibility_state == "VISIBLE_DIAGNOSIS"
                and diagnostic_pressure_score >= 0.45
                and physiologic_diagnostic_disagreement_score >= 0.40
            )
            or (
                recovery_authenticity_state in [
                    "SUPPORT_DEPENDENT_STABILITY",
                    "ARTIFICIAL_RECOVERY_PATTERN",
                    "FALSE_RECOVERY_PATTERN",
                ]
                and hidden_instability >= 0.40
            )
            or (
                diagnosis_false_reassurance_flag
                and rebound_risk >= 0.40
            )
        )

        diagnosis_confidence_without_truth = bool(
            (
                diagnostic_visibility_state in ["VISIBLE_DIAGNOSIS", "PARTIALLY_VISIBLE_DIAGNOSIS"]
                and (
                    hidden_diagnosis_flag
                    or missed_diagnosis_flag
                    or diagnosis_false_reassurance_flag
                    or premature_diagnostic_closure_flag
                )
            )
        )

        diagnosis_hidden_progression_flag = bool(
            hidden_diagnosis_flag
            or slow_burn_diagnosis_flag
            or (
                hidden_instability >= 0.50
                and diagnosis_delayed_flag
            )
            or (
                hidden_trend_loss
                and diagnostic_pressure_score >= 0.35
            )
        )

        diagnosis_underestimated_severity_flag = bool(
            (
                diagnostic_visibility_state in [
                    "VISIBLE_DIAGNOSIS",
                    "PARTIALLY_VISIBLE_DIAGNOSIS",
                    "MISLEADING_PRESENTATION_DIAGNOSIS",
                ]
                and (
                    simple_to_critical_flag
                    or lethal_pathway_flag
                    or critical_lab_burden >= 1
                    or hidden_instability >= 0.55
                )
            )
        )

        diagnosis_rebound_after_resolution_flag = bool(
            rebound_risk >= 0.45
            and (
                diagnosis_false_reassurance_flag
                or premature_diagnostic_closure_flag
                or recovery_authenticity_state in [
                    "SUPPORT_DEPENDENT_STABILITY",
                    "ARTIFICIAL_RECOVERY_PATTERN",
                    "FALSE_RECOVERY_PATTERN",
                ]
            )
        )

        diagnosis_operational_miss_probability = _clip(
            missed_probability * 0.28
            + delay_probability * 0.20
            + diagnostic_operational_fragmentation_score * 0.22
            + monitoring_blind_spot * 0.12
            + hospital_failure_pressure * 0.10
            + float(handoff_false_reassurance) * 0.08
        )

        diagnostic_late_truth_pressure_score = _clip(
            diagnosis_delay_consequence_pressure * 0.25
            + diagnosis_operational_miss_probability * 0.20
            + float(diagnosis_revision_flag) * 0.12
            + float(missed_diagnosis_flag) * 0.12
            + float(diagnosis_false_reassurance_flag) * 0.12
            + hidden_instability * 0.10
            + handoff_uncertainty * 0.09
        )

        diagnostic_trajectory_state = "STABLE_DIAGNOSTIC_CONTEXT"

        if lethal_pathway_flag:
            diagnostic_trajectory_state = "LETHAL_DIAGNOSTIC_PATHWAY"
        elif premature_diagnostic_closure_flag:
            diagnostic_trajectory_state = "PREMATURE_DIAGNOSTIC_CLOSURE"
        elif diagnosis_false_reassurance_flag:
            diagnostic_trajectory_state = "DIAGNOSTIC_FALSE_REASSURANCE"
        elif medication_induced_diagnosis_flag:
            diagnostic_trajectory_state = "MEDICATION_INDUCED_SECONDARY_DIAGNOSIS"
        elif simple_to_critical_flag:
            diagnostic_trajectory_state = "SIMPLE_TO_COMPLEX_EVOLUTION"
        elif hidden_diagnosis_flag:
            diagnostic_trajectory_state = "HIDDEN_CRITICAL_DIAGNOSIS"
        elif slow_burn_diagnosis_flag:
            diagnostic_trajectory_state = "SLOW_BURN_DIAGNOSTIC_EVOLUTION"
        elif missed_diagnosis_flag:
            diagnostic_trajectory_state = "MISSED_DIAGNOSIS_UNTIL_DETERIORATION"
        elif diagnosis_revision_flag:
            diagnostic_trajectory_state = "DIAGNOSIS_REVISED_AFTER_EVIDENCE"

        diagnosis_uncertainty_score = _clip(
            diagnostic_pressure_score * 0.25
            + missed_diagnosis_flag * 0.12
            + diagnosis_delayed_flag * 0.10
            + diagnosis_revision_flag * 0.10
            + hidden_diagnosis_flag * 0.10
            + float(diagnostic_visibility_state != "VISIBLE_DIAGNOSIS") * 0.10
            + diagnosis_false_reassurance_flag * 0.12
            + premature_diagnostic_closure_flag * 0.08
            + diagnostic_operational_fragmentation_score * 0.08
            + diagnosis_confidence_without_truth * 0.05
        )

        diagnosis_risk_pressure_score = _clip(
            diagnosis_uncertainty_score * 0.24
            + simple_to_critical_flag * 0.14
            + lethal_pathway_flag * 0.18
            + medication_induced_diagnosis_flag * 0.12
            + slow_burn_diagnosis_flag * 0.09
            + hidden_diagnosis_flag * 0.11
            + diagnosis_false_reassurance_flag * 0.10
            + premature_diagnostic_closure_flag * 0.08
            + diagnosis_underestimated_severity_flag * 0.08
            + diagnostic_late_truth_pressure_score * 0.08
        )

        systemically_missed_decline_diagnostic_support = bool(
            (
                diagnosis_risk_pressure_score >= 0.45
                and (
                    delayed_escalation
                    or delayed_reassessment
                    or monitoring_blind_spot >= 0.45
                    or handoff_uncertainty >= 0.45
                )
            )
            or (
                diagnosis_false_reassurance_flag
                and hidden_instability >= 0.45
            )
            or (
                missed_diagnosis_flag
                and diagnostic_operational_fragmentation_score >= 0.40
            )
        )

        all_documented_diagnoses = _unique(
            base_diagnoses + event_diagnoses + secondary_diagnoses
        )

        primary_diagnosis = all_documented_diagnoses[0]

        diagnosis_requires_bire_attention_flag = bool(
            diagnosis_risk_pressure_score >= 0.35
            or missed_diagnosis_flag
            or hidden_diagnosis_flag
            or lethal_pathway_flag
            or simple_to_critical_flag
            or medication_induced_diagnosis_flag
            or diagnosis_false_reassurance_flag
            or premature_diagnostic_closure_flag
            or diagnosis_confidence_without_truth
        )

        diagnosis_rows.append(
            {
                **row,

                "primary_diagnosis": primary_diagnosis,
                "secondary_diagnoses": secondary_diagnoses,
                "all_documented_diagnoses": all_documented_diagnoses,
                "diagnosis_count": len(all_documented_diagnoses),

                "hidden_diagnosis_flag": bool(hidden_diagnosis_flag),
                "hidden_diagnosis_name": hidden_diagnosis_name,

                "slow_burn_diagnosis_flag": bool(slow_burn_diagnosis_flag),
                "slow_burn_diagnosis_name": slow_burn_diagnosis_name,

                "simple_to_critical_flag": bool(simple_to_critical_flag),
                "evolved_critical_diagnosis": evolved_critical_diagnosis,

                "lethal_pathway_flag": bool(lethal_pathway_flag),
                "lethal_pathway_diagnosis": lethal_pathway_diagnosis,

                "medication_induced_diagnosis_flag": bool(medication_induced_diagnosis_flag),
                "medication_induced_diagnosis": medication_induced_diagnosis,

                "missed_diagnosis_flag": bool(missed_diagnosis_flag),
                "diagnosis_revision_flag": bool(diagnosis_revision_flag),
                "revised_from_diagnosis": revised_from_diagnosis,
                "revised_to_diagnosis": revised_to_diagnosis,

                "diagnosis_delayed_flag": bool(diagnosis_delayed_flag),
                "diagnosis_chart_delay_minutes": int(diagnosis_chart_delay_minutes),

                "diagnostic_visibility_state": diagnostic_visibility_state,
                "diagnostic_trajectory_state": diagnostic_trajectory_state,

                "diagnostic_pressure_score": round(diagnostic_pressure_score, 4),
                "physiologic_diagnostic_disagreement_score": round(physiologic_diagnostic_disagreement_score, 4),
                "diagnostic_operational_fragmentation_score": round(diagnostic_operational_fragmentation_score, 4),
                "diagnosis_delay_consequence_pressure": round(diagnosis_delay_consequence_pressure, 4),

                "diagnosis_false_reassurance_flag": bool(diagnosis_false_reassurance_flag),
                "diagnosis_confidence_without_truth": bool(diagnosis_confidence_without_truth),
                "premature_diagnostic_closure_flag": bool(premature_diagnostic_closure_flag),
                "diagnosis_hidden_progression_flag": bool(diagnosis_hidden_progression_flag),
                "diagnosis_underestimated_severity_flag": bool(diagnosis_underestimated_severity_flag),
                "diagnosis_rebound_after_resolution_flag": bool(diagnosis_rebound_after_resolution_flag),
                "diagnosis_operational_miss_probability": round(diagnosis_operational_miss_probability, 4),
                "diagnostic_late_truth_pressure_score": round(diagnostic_late_truth_pressure_score, 4),
                "systemically_missed_decline_diagnostic_support": bool(systemically_missed_decline_diagnostic_support),

                "diagnosis_uncertainty_score": round(diagnosis_uncertainty_score, 4),
                "diagnosis_risk_pressure_score": round(diagnosis_risk_pressure_score, 4),

                "diagnosis_requires_bire_attention_flag": bool(diagnosis_requires_bire_attention_flag),
                "diagnostic_revision_count_so_far": memory["diagnostic_revision_count"],
            }
        )

        if hidden_diagnosis_flag:
            memory["prior_hidden_dx"] = True

        if missed_diagnosis_flag:
            memory["prior_missed_dx"] = True

        if lethal_pathway_flag:
            memory["prior_lethal_pathway"] = True

        if medication_induced_diagnosis_flag:
            memory["prior_medication_induced_dx"] = True

        if diagnosis_false_reassurance_flag:
            memory["prior_false_reassurance_dx"] = True

        if premature_diagnostic_closure_flag:
            memory["prior_premature_closure"] = True

        patient_memory[patient_id] = memory

    return pd.DataFrame(diagnosis_rows)