"""
BIRE OS Synthetic Context Generator

Chapter 53 doctrine:
No more babying BIRE OS.

Purpose:
Generate contextual healthcare evidence around each encounter:
- labs
- imaging
- medications
- allergies
- documentation context
- ordering failures
- contradictions
- trust states
- evidence completeness
- treatment masking risk
- handoff risk
- hospital overload
- delayed result acknowledgment
- surface-level deception
- BIRE OS context pressure

This is the messy evidence layer around the patient.

Doctrine:
We Detect What Others Miss.
"""

import numpy as np
import pandas as pd

from bire.synthetic.config import SYNTHETIC_ECOSYSTEM_CONFIG


LAB_PROFILES = [
    "normal_labs",
    "mild_nonspecific_abnormality",
    "inflammatory_response",
    "infection_pattern",
    "sepsis_like_pattern",
    "renal_instability",
    "acute_kidney_injury_pattern",
    "chronic_kidney_disease_pattern",
    "metabolic_instability",
    "hyperglycemia_pattern",
    "hypoglycemia_pattern",
    "electrolyte_derangement",
    "hyperkalemia_pattern",
    "hypokalemia_pattern",
    "hyponatremia_pattern",
    "anion_gap_metabolic_acidosis",
    "respiratory_acidosis_pattern",
    "respiratory_alkalosis_pattern",
    "lactic_acidosis_pattern",
    "cardiac_marker_elevation",
    "troponin_leak_pattern",
    "heart_failure_congestion_pattern",
    "liver_injury_pattern",
    "pancreatitis_pattern",
    "coagulation_abnormality",
    "bleeding_anemia_pattern",
    "thrombocytopenia_pattern",
    "mixed_instability",
    "multi_organ_stress_pattern",
    "normal_but_trending_wrong",
    "delayed_abnormal_labs",
    "contradictory_labs",
    "hemolyzed_sample",
    "contaminated_sample_possible",
    "critical_labs_pending",
    "critical_result_delayed",
    "repeat_labs_ordered",
    "labs_not_ordered_despite_risk",
]


IMAGING_TYPES = [
    "none",
    "chest_xray",
    "portable_chest_xray",
    "ct_head",
    "ct_chest",
    "ct_abdomen_pelvis",
    "ct_angio_chest",
    "ct_angio_head_neck",
    "mri_brain",
    "mri_spine",
    "ultrasound_abdomen",
    "renal_ultrasound",
    "vascular_ultrasound",
    "echo",
    "ekg",
    "repeat_ct",
    "repeat_chest_xray",
    "delayed_imaging_needed",
    "imaging_not_ordered_despite_risk",
]


MEDICATIONS = [
    "none",
    "oxygen",
    "high_flow_oxygen",
    "bipap",
    "intubation_sedation",
    "iv_fluids",
    "fluid_bolus",
    "maintenance_fluids",
    "insulin",
    "dextrose",
    "antibiotics",
    "broad_spectrum_antibiotics",
    "vasopressors",
    "diuretic",
    "bronchodilator",
    "steroids",
    "morphine",
    "dilaudid",
    "sedative",
    "antiarrhythmic",
    "anticoagulant",
    "antiplatelet",
    "beta_blocker",
    "antihypertensive",
    "electrolyte_repletion",
    "potassium_repletion",
    "magnesium_repletion",
    "blood_transfusion",
    "nausea_medication",
    "pain_control",
    "nebulizer_treatment",
    "contrast_premedication",
    "medication_held_due_to_bp",
    "medication_held_due_to_renal_function",
    "home_meds_continued",
    "home_meds_missing",
]


ALLERGY_TYPES = [
    "none",
    "contrast_dye",
    "penicillin",
    "cephalosporin",
    "sulfa",
    "opioid_sensitivity",
    "morphine",
    "dilaudid",
    "latex",
    "iodine_listed",
    "shellfish_reported",
    "unknown_allergy_discovered_late",
    "allergy_documented_in_old_chart_only",
    "allergy_conflicting_between_notes",
]


DOCUMENTATION_CONTEXTS = [
    "clean_documentation",
    "delayed_charting",
    "duplicate_documentation",
    "copy_forward_note",
    "conflicting_provider_notes",
    "missing_reassessment_note",
    "pending_result_not_acknowledged",
    "handoff_summary_incomplete",
    "discharge_summary_incomplete",
    "med_rec_incomplete",
    "allergy_not_carried_forward",
    "outside_records_missing",
    "wrong_timestamp_possible",
    "triage_note_understates_severity",
]


ORDERING_FAILURE_TYPES = [
    "none",
    "labs_delayed",
    "imaging_delayed",
    "medication_delayed",
    "repeat_labs_not_ordered",
    "critical_result_not_acknowledged",
    "consult_delayed",
    "antibiotics_delayed",
    "fluids_delayed",
    "oxygen_escalation_delayed",
    "icu_transfer_delayed",
    "discharge_before_result_finalized",
]


CONTEXT_CONTRADICTION_TYPES = [
    "none",
    "vitals_worse_than_labs_suggest",
    "labs_worse_than_vitals_suggest",
    "imaging_worse_than_vitals_suggest",
    "medication_masks_vital_instability",
    "documentation_says_stable_but_vitals_worsen",
    "discharge_plan_conflicts_with_risk",
    "allergy_conflicts_with_order",
    "care_mode_conflicts_with_treatment_intensity",
    "normal_labs_but_hidden_vitals_unstable",
    "delayed_results_change_interpretation",
]


HOSPITAL_CONTEXT_STATES = [
    "normal_operations",
    "er_crowding",
    "icu_saturation",
    "staffing_shortage",
    "night_shift_pressure",
    "weekend_coverage_gap",
    "holiday_resource_constraint",
    "mass_casualty_spillover",
    "lab_backlog",
    "imaging_backlog",
    "multi_unit_bottleneck",
]


RESULT_ACKNOWLEDGMENT_STATES = [
    "acknowledged_on_time",
    "acknowledged_late",
    "not_acknowledged",
    "acknowledged_but_not_acted_on",
    "buried_in_chart",
    "handoff_missed_result",
]


EVIDENCE_SURFACE_DECEPTION_TYPES = [
    "none",
    "labs_look_reassuring_but_patient_worsening",
    "imaging_pending_but_vitals_compensating",
    "medication_temporarily_normalizes_vitals",
    "documentation_says_stable_but_trajectory_worse",
    "normal_initial_workup_but_hidden_instability",
    "delayed_result_confirms_bire_suspicion",
    "surface_level_low_acuity_but_high_hidden_risk",
]


def _clip(value, low=0.0, high=1.0):
    return float(np.clip(value, low, high))


def _weighted_choice(rng, weights):
    keys = list(weights.keys())
    probs = np.array([max(weights[k], 0.001) for k in keys], dtype=float)
    probs = probs / probs.sum()
    return rng.choice(keys, p=probs)


def _normalized_choice(rng, keys, probs):
    probs = np.array(probs, dtype=float)
    probs = probs / probs.sum()
    return rng.choice(keys, p=probs)


def _choose_hospital_context_state(row, rng):
    chaos = row.get("encounter_chaos_score", 0.0)
    seasonal = row.get("seasonal_pressure", 0.0)
    acuity_gap = row.get("hidden_acuity_gap_score", 0.0)
    care_mode = str(row.get("care_mode", ""))

    weights = {state: 0.03 for state in HOSPITAL_CONTEXT_STATES}
    weights["normal_operations"] = 0.42

    weights["er_crowding"] += chaos * 0.16 + seasonal * 0.10
    weights["icu_saturation"] += (care_mode == "ICU") * 0.14 + acuity_gap * 0.12
    weights["staffing_shortage"] += seasonal * 0.12 + chaos * 0.10
    weights["night_shift_pressure"] += 0.08
    weights["weekend_coverage_gap"] += 0.06
    weights["holiday_resource_constraint"] += seasonal * 0.08
    weights["lab_backlog"] += chaos * 0.10
    weights["imaging_backlog"] += chaos * 0.10
    weights["multi_unit_bottleneck"] += chaos * 0.12 + acuity_gap * 0.08
    weights["mass_casualty_spillover"] += chaos * 0.04 + seasonal * 0.04

    return _weighted_choice(rng, weights)


def _choose_lab_profile(row, rng):
    has_event = bool(row.get("has_event", False))
    event_type = str(row.get("event_type", "NONE"))
    condition = str(row.get("condition_profile", ""))
    presentation = str(row.get("presentation_profile", ""))

    hidden_vitals = row.get("silent_collapse_pressure", 0.0)
    lifecycle_uncertainty = row.get("lifecycle_uncertainty_score", 0.0)
    masking = row.get("normal_range_deception_score", 0.0)
    terminal_pressure = row.get("terminal_decline_pressure", 0.0)
    wave_count = row.get("event_wave_count", 0)

    weights = {profile: 0.04 for profile in LAB_PROFILES}
    weights["normal_labs"] = 0.24
    weights["mild_nonspecific_abnormality"] = 0.10
    weights["normal_but_trending_wrong"] = 0.08

    if has_event:
        weights["normal_labs"] -= 0.14
        weights["mixed_instability"] += 0.10
        weights["multi_organ_stress_pattern"] += 0.08
        weights["critical_labs_pending"] += 0.07
        weights["critical_result_delayed"] += 0.06

    if wave_count >= 2:
        weights["mixed_instability"] += 0.08
        weights["multi_organ_stress_pattern"] += 0.08
        weights["critical_result_delayed"] += 0.05

    if terminal_pressure >= 0.55:
        weights["multi_organ_stress_pattern"] += 0.12
        weights["lactic_acidosis_pattern"] += 0.10
        weights["critical_labs_pending"] += 0.08

    if hidden_vitals >= 0.45 or masking >= 0.45:
        weights["normal_but_trending_wrong"] += 0.12
        weights["contradictory_labs"] += 0.08
        weights["delayed_abnormal_labs"] += 0.08
        weights["labs_not_ordered_despite_risk"] += 0.04

    if lifecycle_uncertainty >= 0.35:
        weights["contradictory_labs"] += 0.08
        weights["repeat_labs_ordered"] += 0.05

    if "sepsis" in event_type or "sepsis" in condition or "infection" in presentation or "fever" in presentation:
        weights["infection_pattern"] += 0.16
        weights["sepsis_like_pattern"] += 0.14
        weights["inflammatory_response"] += 0.10
        weights["lactic_acidosis_pattern"] += 0.08

    if "renal" in event_type or "ckd" in condition:
        weights["renal_instability"] += 0.16
        weights["acute_kidney_injury_pattern"] += 0.14
        weights["chronic_kidney_disease_pattern"] += 0.10
        weights["hyperkalemia_pattern"] += 0.06

    if "diabetes" in condition or "metabolic" in event_type or "hyperglycemia" in presentation:
        weights["metabolic_instability"] += 0.14
        weights["hyperglycemia_pattern"] += 0.12
        weights["anion_gap_metabolic_acidosis"] += 0.08
        weights["electrolyte_derangement"] += 0.06

    if "respiratory" in event_type or "copd" in condition or "shortness_of_breath" in presentation:
        weights["respiratory_acidosis_pattern"] += 0.16
        weights["respiratory_alkalosis_pattern"] += 0.06

    if "arrhythmia" in event_type or "chest_pain" in presentation:
        weights["cardiac_marker_elevation"] += 0.12
        weights["troponin_leak_pattern"] += 0.10

    if "chf" in condition or "hemodynamic" in event_type:
        weights["heart_failure_congestion_pattern"] += 0.14
        weights["cardiac_marker_elevation"] += 0.07

    if "pancreatitis" in condition or "abdominal_pain" in presentation:
        weights["pancreatitis_pattern"] += 0.18
        weights["liver_injury_pattern"] += 0.06

    return _weighted_choice(rng, weights)


def _choose_imaging_type(row, rng):
    care_mode = str(row.get("care_mode", row.get("highest_acuity_mode", "OUTPATIENT")))
    event_type = str(row.get("event_type", "NONE"))
    condition = str(row.get("condition_profile", ""))
    presentation = str(row.get("presentation_profile", ""))

    probability = (
        0.08
        + row.get("fragility_score", 0.0) * 0.12
        + row.get("hidden_acuity_gap_score", 0.0) * 0.12
        + row.get("monitoring_intensity_need", 0.0) * 0.12
        + row.get("terminal_decline_pressure", 0.0) * 0.08
        + row.get("event_wave_count", 0) * 0.03
    )

    if care_mode in ["ER_ESI_1", "ER_ESI_2"]:
        probability += 0.34
    elif care_mode == "ER_ESI_3":
        probability += 0.22
    elif care_mode in ["INPATIENT", "ICU"]:
        probability += 0.38

    if row.get("has_event", False):
        probability += 0.14

    if rng.random() > _clip(probability, 0.02, 0.96):
        if row.get("silent_collapse_pressure", 0.0) >= 0.50 and rng.random() < 0.20:
            return "imaging_not_ordered_despite_risk"
        return "none"

    if "respiratory" in event_type or "copd" in condition or "shortness_of_breath" in presentation:
        return rng.choice(
            ["chest_xray", "portable_chest_xray", "ct_chest", "ct_angio_chest", "repeat_chest_xray"],
            p=[0.35, 0.22, 0.22, 0.10, 0.11],
        )

    if "chest_pain" in presentation or "arrhythmia" in event_type:
        return rng.choice(
            ["ekg", "echo", "chest_xray", "ct_angio_chest"],
            p=[0.36, 0.26, 0.22, 0.16],
        )

    if "hemodynamic" in event_type or "chf" in condition:
        return rng.choice(
            ["echo", "chest_xray", "portable_chest_xray", "ct_chest"],
            p=[0.38, 0.24, 0.22, 0.16],
        )

    if "pancreatitis" in condition or "abdominal_pain" in presentation:
        return rng.choice(
            ["ct_abdomen_pelvis", "ultrasound_abdomen", "repeat_ct"],
            p=[0.52, 0.30, 0.18],
        )

    if "altered_mental_status" in presentation or "dizziness" in presentation:
        return rng.choice(
            ["ct_head", "mri_brain", "ct_angio_head_neck"],
            p=[0.48, 0.32, 0.20],
        )

    if "renal" in event_type or "ckd" in condition:
        return rng.choice(
            ["renal_ultrasound", "ultrasound_abdomen", "ct_abdomen_pelvis"],
            p=[0.52, 0.28, 0.20],
        )

    return rng.choice(IMAGING_TYPES[1:])


def _choose_medications(row, rng):
    care_mode = str(row.get("care_mode", row.get("highest_acuity_mode", "OUTPATIENT")))
    event_type = str(row.get("event_type", "NONE"))
    condition = str(row.get("condition_profile", ""))
    presentation = str(row.get("presentation_profile", ""))

    if care_mode == "OUTPATIENT":
        target_count = rng.integers(0, 4)
    elif "ER" in care_mode:
        target_count = rng.integers(1, 6)
    elif care_mode == "INPATIENT":
        target_count = rng.integers(2, 8)
    elif care_mode == "ICU":
        target_count = rng.integers(4, 12)
    else:
        target_count = rng.integers(1, 5)

    if row.get("event_wave_count", 0) >= 2:
        target_count += 1

    if row.get("terminal_decline_pressure", 0.0) >= 0.55:
        target_count += 1

    med_pool = []

    if row.get("has_event", False):
        med_pool.extend(["oxygen", "iv_fluids", "antibiotics"])

    if "respiratory" in event_type or "copd" in condition:
        med_pool.extend(["oxygen", "high_flow_oxygen", "bipap", "bronchodilator", "steroids", "nebulizer_treatment"])

    if "hemodynamic" in event_type or care_mode == "ICU":
        med_pool.extend(["iv_fluids", "fluid_bolus", "vasopressors", "blood_transfusion"])

    if "metabolic" in event_type or "diabetes" in condition:
        med_pool.extend(["insulin", "dextrose", "iv_fluids", "electrolyte_repletion"])

    if "sepsis" in event_type or "sepsis" in condition:
        med_pool.extend(["broad_spectrum_antibiotics", "iv_fluids", "fluid_bolus", "vasopressors"])

    if "arrhythmia" in event_type or "arrhythmia" in condition or "chest_pain" in presentation:
        med_pool.extend(["antiarrhythmic", "anticoagulant", "antiplatelet", "beta_blocker"])

    if "chf" in condition:
        med_pool.extend(["diuretic", "oxygen", "antihypertensive"])

    if "renal" in event_type or "ckd" in condition:
        med_pool.extend(["medication_held_due_to_renal_function", "electrolyte_repletion"])

    if "pain" in presentation or "post_surgical" in condition or "pancreatitis" in condition:
        med_pool.extend(["morphine", "dilaudid", "pain_control", "nausea_medication"])

    if row.get("normal_range_deception_score", 0.0) >= 0.45:
        med_pool.extend(["oxygen", "iv_fluids", "home_meds_continued"])

    if target_count <= 0:
        return []

    med_pool.extend([m for m in MEDICATIONS if m != "none"])
    unique_pool = sorted(set(med_pool))

    return list(
        rng.choice(
            unique_pool,
            size=min(target_count, len(unique_pool)),
            replace=False,
        )
    )


def _choose_documentation_context(row, rng, hospital_overload_pressure=0.0):
    chaos = row.get("encounter_chaos_score", 0.0)
    uncertainty = row.get("lifecycle_uncertainty_score", 0.0)
    contradiction = row.get("contextual_contradiction_score", 0.0)

    weights = {d: 0.03 for d in DOCUMENTATION_CONTEXTS}
    weights["clean_documentation"] = 0.38

    weights["delayed_charting"] += chaos * 0.20 + uncertainty * 0.12 + hospital_overload_pressure * 0.14
    weights["copy_forward_note"] += chaos * 0.10 + hospital_overload_pressure * 0.06
    weights["conflicting_provider_notes"] += uncertainty * 0.14
    weights["missing_reassessment_note"] += row.get("monitoring_intensity_need", 0.0) * 0.15
    weights["pending_result_not_acknowledged"] += row.get("has_event", False) * 0.10 + hospital_overload_pressure * 0.10
    weights["handoff_summary_incomplete"] += row.get("handoff_risk", 0.0) * 0.12 + hospital_overload_pressure * 0.08
    weights["triage_note_understates_severity"] += row.get("normal_range_deception_score", 0.0) * 0.16
    weights["outside_records_missing"] += row.get("care_fragmentation_risk", 0.0) * 0.14
    weights["wrong_timestamp_possible"] += chaos * 0.10
    weights["allergy_not_carried_forward"] += 0.04
    weights["med_rec_incomplete"] += 0.05
    weights["discharge_summary_incomplete"] += row.get("failed_discharge_flag", False) * 0.14
    weights["duplicate_documentation"] += chaos * 0.08
    weights["conflicting_provider_notes"] += contradiction * 0.20

    return _weighted_choice(rng, weights)


def _choose_ordering_failure(row, lab_profile, imaging_type, medications, rng, hospital_overload_pressure=0.0):
    weights = {failure: 0.02 for failure in ORDERING_FAILURE_TYPES}
    weights["none"] = 0.44

    chaos = row.get("encounter_chaos_score", 0.0)
    uncertainty = row.get("lifecycle_uncertainty_score", 0.0)
    masking = row.get("silent_collapse_pressure", 0.0)

    weights["labs_delayed"] += chaos * 0.14 + hospital_overload_pressure * 0.10
    weights["imaging_delayed"] += chaos * 0.12 + hospital_overload_pressure * 0.12
    weights["medication_delayed"] += chaos * 0.10 + hospital_overload_pressure * 0.08
    weights["critical_result_not_acknowledged"] += uncertainty * 0.10 + hospital_overload_pressure * 0.08
    weights["consult_delayed"] += row.get("monitoring_intensity_need", 0.0) * 0.10 + hospital_overload_pressure * 0.07
    weights["icu_transfer_delayed"] += row.get("hidden_acuity_gap_score", 0.0) * 0.12 + hospital_overload_pressure * 0.05
    weights["discharge_before_result_finalized"] += row.get("failed_discharge_flag", False) * 0.15
    weights["repeat_labs_not_ordered"] += lab_profile in ["normal_but_trending_wrong", "contradictory_labs", "delayed_abnormal_labs"]
    weights["oxygen_escalation_delayed"] += "respiratory" in str(row.get("event_type", ""))
    weights["antibiotics_delayed"] += "sepsis" in str(row.get("event_type", "")) or "infection" in lab_profile
    weights["fluids_delayed"] += "hemodynamic" in str(row.get("event_type", ""))
    weights["labs_delayed"] += masking * 0.08

    return _weighted_choice(rng, weights)


def _choose_context_contradiction(row, lab_profile, imaging_type, medications, documentation_context, rng):
    weights = {contradiction: 0.02 for contradiction in CONTEXT_CONTRADICTION_TYPES}
    weights["none"] = 0.40

    silent_collapse = row.get("silent_collapse_pressure", 0.0)
    normal_deception = row.get("normal_range_deception_score", 0.0)

    if silent_collapse >= 0.45:
        weights["normal_labs_but_hidden_vitals_unstable"] += 0.18
        weights["vitals_worse_than_labs_suggest"] += 0.12

    if lab_profile in ["critical_labs_pending", "critical_result_delayed", "delayed_abnormal_labs"]:
        weights["delayed_results_change_interpretation"] += 0.16

    if lab_profile in ["contradictory_labs", "normal_but_trending_wrong"]:
        weights["labs_worse_than_vitals_suggest"] += 0.12
        weights["vitals_worse_than_labs_suggest"] += 0.12

    if imaging_type not in ["none", "imaging_not_ordered_despite_risk"] and row.get("has_event", False):
        weights["imaging_worse_than_vitals_suggest"] += 0.08

    if any(
        medication in medications
        for medication in ["oxygen", "high_flow_oxygen", "vasopressors", "iv_fluids", "sedative", "morphine", "dilaudid"]
    ):
        weights["medication_masks_vital_instability"] += 0.16

    if documentation_context in ["conflicting_provider_notes", "copy_forward_note", "triage_note_understates_severity"]:
        weights["documentation_says_stable_but_vitals_worsen"] += 0.16

    if row.get("failed_discharge_flag", False) or row.get("bounceback_flag", False):
        weights["discharge_plan_conflicts_with_risk"] += 0.14

    if normal_deception >= 0.50:
        weights["care_mode_conflicts_with_treatment_intensity"] += 0.08

    return _weighted_choice(rng, weights)


def _choose_result_acknowledgment_state(rng, hospital_overload_pressure):
    if hospital_overload_pressure >= 0.45:
        probs = [0.28, 0.24, 0.14, 0.14, 0.10, 0.10]
    else:
        probs = [0.50, 0.18, 0.08, 0.09, 0.08, 0.07]

    return _normalized_choice(
        rng,
        RESULT_ACKNOWLEDGMENT_STATES,
        probs,
    )


def _choose_surface_deception_type(
    row,
    lab_profile,
    imaging_context_state,
    documentation_context,
    context_contradiction_type,
    treatment_masking_risk,
    care_mode,
):
    if (
        row.get("physiologic_deception_flag", 0) == 1
        and lab_profile in ["normal_labs", "mild_nonspecific_abnormality", "normal_but_trending_wrong"]
    ):
        return "labs_look_reassuring_but_patient_worsening"

    if imaging_context_state == "RESULT_DELAYED" and row.get("silent_collapse_pressure", 0.0) >= 0.45:
        return "imaging_pending_but_vitals_compensating"

    if treatment_masking_risk >= 0.35:
        return "medication_temporarily_normalizes_vitals"

    if documentation_context in ["clean_documentation", "copy_forward_note"] and row.get("trajectory_stress_score", 0.0) >= 0.45:
        return "documentation_says_stable_but_trajectory_worse"

    if context_contradiction_type == "normal_labs_but_hidden_vitals_unstable":
        return "normal_initial_workup_but_hidden_instability"

    if lab_profile in ["critical_result_delayed", "delayed_abnormal_labs"]:
        return "delayed_result_confirms_bire_suspicion"

    if row.get("hidden_acuity_gap_score", 0.0) >= 0.50 and care_mode in ["OUTPATIENT", "ER_ESI_5", "ER_ESI_4"]:
        return "surface_level_low_acuity_but_high_hidden_risk"

    return "none"


def generate_contextual_operational_data(lifecycle_df, random_seed=None):
    """
    Generate synthetic contextual healthcare evidence.
    """

    if random_seed is None:
        random_seed = SYNTHETIC_ECOSYSTEM_CONFIG["random_seed"]

    rng = np.random.default_rng(random_seed)
    context_rows = []

    for _, row in lifecycle_df.iterrows():
        row_dict = row.to_dict()

        fragility = row.get("fragility_score", 0.0)
        care_mode = str(row.get("care_mode", row.get("highest_acuity_mode", "OUTPATIENT")))
        has_event = bool(row.get("has_event", False))
        chaos = row.get("encounter_chaos_score", 0.0)
        lifecycle_uncertainty = row.get("lifecycle_uncertainty_score", 0.0)
        monitoring_need = row.get("monitoring_intensity_need", 0.0)

        hospital_context_state = _choose_hospital_context_state(row, rng)

        hospital_overload_pressure = _clip(
            chaos * 0.24
            + lifecycle_uncertainty * 0.18
            + row.get("seasonal_pressure", 0.0) * 0.16
            + row.get("hidden_acuity_gap_score", 0.0) * 0.14
            + (hospital_context_state != "normal_operations") * 0.18
            + (hospital_context_state in ["mass_casualty_spillover", "multi_unit_bottleneck"]) * 0.10
        )

        lab_profile = _choose_lab_profile(row, rng)
        imaging_type = _choose_imaging_type(row, rng)
        medications_administered = _choose_medications(row, rng)

        lab_ordered = bool(
            lab_profile != "labs_not_ordered_despite_risk"
            and rng.random()
            < _clip(
                0.30
                + fragility * 0.18
                + has_event * 0.20
                + monitoring_need * 0.18
                + hospital_overload_pressure * 0.08,
                0.05,
                0.97,
            )
        )

        lab_delay_minutes = 0
        if lab_ordered:
            lab_delay_minutes = int(
                np.clip(
                    rng.normal(
                        35
                        + chaos * 100
                        + lifecycle_uncertainty * 75
                        + hospital_overload_pressure * 90,
                        30,
                    ),
                    5,
                    720,
                )
            )

        critical_lab_pending_flag = bool(
            lab_profile in ["critical_labs_pending", "critical_result_delayed"]
            or (lab_ordered and lab_delay_minutes >= 150 and has_event and rng.random() < 0.35)
        )

        lab_context_trust_state = (
            "LOW_TRUST"
            if lab_profile in [
                "contradictory_labs",
                "critical_labs_pending",
                "critical_result_delayed",
                "labs_not_ordered_despite_risk",
            ]
            else "PARTIAL_TRUST"
            if lab_delay_minutes >= 90
            or lab_profile in [
                "delayed_abnormal_labs",
                "hemolyzed_sample",
                "contaminated_sample_possible",
                "normal_but_trending_wrong",
            ]
            else "HIGH_TRUST"
        )

        imaging_ordered = bool(imaging_type not in ["none", "imaging_not_ordered_despite_risk"])

        imaging_delay_minutes = 0
        if imaging_ordered:
            imaging_delay_minutes = int(
                np.clip(
                    rng.normal(
                        55
                        + chaos * 160
                        + lifecycle_uncertainty * 95
                        + hospital_overload_pressure * 140,
                        50,
                    ),
                    10,
                    1200,
                )
            )

        imaging_result_available = bool(
            imaging_ordered
            and rng.random() > _clip(imaging_delay_minutes / 1200, 0.05, 0.90)
        )

        imaging_context_state = (
            "NO_IMAGING"
            if imaging_type == "none"
            else "IMAGING_NOT_ORDERED_DESPITE_RISK"
            if imaging_type == "imaging_not_ordered_despite_risk"
            else "RESULT_DELAYED"
            if not imaging_result_available
            else "RESULT_AVAILABLE"
        )

        allergy_weights = np.array(
            [
                0.62,
                0.07,
                0.06,
                0.04,
                0.035,
                0.035,
                0.02,
                0.02,
                0.025,
                0.015,
                0.015,
                0.025,
                0.02,
                0.02,
            ],
            dtype=float,
        )
        allergy_weights = allergy_weights / allergy_weights.sum()

        allergy_type = rng.choice(
            ALLERGY_TYPES,
            p=allergy_weights,
        )

        contrast_allergy_conflict = bool(
            allergy_type in ["contrast_dye", "iodine_listed", "shellfish_reported"]
            and imaging_type in ["ct_abdomen_pelvis", "ct_chest", "ct_angio_chest", "ct_angio_head_neck", "repeat_ct"]
            and rng.random() < 0.42
        )

        antibiotic_allergy_conflict = bool(
            allergy_type in ["penicillin", "cephalosporin", "sulfa"]
            and any(med in medications_administered for med in ["antibiotics", "broad_spectrum_antibiotics"])
            and rng.random() < 0.32
        )

        opioid_allergy_conflict = bool(
            allergy_type in ["opioid_sensitivity", "morphine", "dilaudid"]
            and any(med in medications_administered for med in ["morphine", "dilaudid", "pain_control"])
            and rng.random() < 0.36
        )

        medication_context_conflict_flag = bool(
            contrast_allergy_conflict
            or antibiotic_allergy_conflict
            or opioid_allergy_conflict
        )

        care_mode_medication_mismatch_flag = bool(
            care_mode in ["OUTPATIENT", "ER_ESI_5", "ER_ESI_4"]
            and any(
                med in medications_administered
                for med in ["vasopressors", "intubation_sedation", "bipap", "high_flow_oxygen"]
            )
        )

        treatment_masking_risk = _clip(
            ("oxygen" in medications_administered) * 0.10
            + ("high_flow_oxygen" in medications_administered) * 0.18
            + ("bipap" in medications_administered) * 0.18
            + ("iv_fluids" in medications_administered) * 0.12
            + ("fluid_bolus" in medications_administered) * 0.16
            + ("vasopressors" in medications_administered) * 0.28
            + ("morphine" in medications_administered) * 0.14
            + ("dilaudid" in medications_administered) * 0.16
            + ("sedative" in medications_administered) * 0.18
            + ("steroids" in medications_administered) * 0.10
            + ("beta_blocker" in medications_administered) * 0.12
            + ("antihypertensive" in medications_administered) * 0.08
        )

        documentation_context = _choose_documentation_context(
            row,
            rng,
            hospital_overload_pressure=hospital_overload_pressure,
        )

        delayed_charting_flag = documentation_context in [
            "delayed_charting",
            "missing_reassessment_note",
            "pending_result_not_acknowledged",
            "wrong_timestamp_possible",
        ]

        duplicate_documentation_flag = documentation_context in [
            "duplicate_documentation",
            "copy_forward_note",
        ]

        documentation_conflict_flag = documentation_context in [
            "conflicting_provider_notes",
            "copy_forward_note",
            "triage_note_understates_severity",
            "wrong_timestamp_possible",
        ]

        missing_context_flag = documentation_context in [
            "outside_records_missing",
            "handoff_summary_incomplete",
            "discharge_summary_incomplete",
            "med_rec_incomplete",
            "allergy_not_carried_forward",
        ]

        ordering_failure_type = _choose_ordering_failure(
            row,
            lab_profile,
            imaging_type,
            medications_administered,
            rng,
            hospital_overload_pressure=hospital_overload_pressure,
        )

        context_contradiction_type = _choose_context_contradiction(
            row,
            lab_profile,
            imaging_type,
            medications_administered,
            documentation_context,
            rng,
        )

        result_acknowledgment_state = _choose_result_acknowledgment_state(
            rng,
            hospital_overload_pressure,
        )

        contextual_contradiction_score = _clip(
            medication_context_conflict_flag * 0.14
            + care_mode_medication_mismatch_flag * 0.14
            + documentation_conflict_flag * 0.12
            + missing_context_flag * 0.10
            + critical_lab_pending_flag * 0.14
            + (lab_context_trust_state == "LOW_TRUST") * 0.12
            + (imaging_context_state in ["RESULT_DELAYED", "IMAGING_NOT_ORDERED_DESPITE_RISK"]) * 0.10
            + (ordering_failure_type != "none") * 0.10
            + (context_contradiction_type != "none") * 0.12
            + (result_acknowledgment_state != "acknowledged_on_time") * 0.10
        )

        evidence_completeness_score = _clip(
            1
            - (
                missing_context_flag * 0.16
                + delayed_charting_flag * 0.10
                + duplicate_documentation_flag * 0.07
                + documentation_conflict_flag * 0.11
                + critical_lab_pending_flag * 0.13
                + (lab_profile == "labs_not_ordered_despite_risk") * 0.12
                + (imaging_context_state == "RESULT_DELAYED") * 0.10
                + (imaging_context_state == "IMAGING_NOT_ORDERED_DESPITE_RISK") * 0.14
                + medication_context_conflict_flag * 0.09
                + (result_acknowledgment_state in ["not_acknowledged", "handoff_missed_result"]) * 0.10
                + hospital_overload_pressure * 0.06
            )
        )

        evidence_trust_state = (
            "LOW_TRUST"
            if evidence_completeness_score < 0.55 or contextual_contradiction_score >= 0.45
            else "PARTIAL_TRUST"
            if evidence_completeness_score < 0.80 or contextual_contradiction_score >= 0.22
            else "HIGH_TRUST"
        )

        evidence_surface_deception_type = _choose_surface_deception_type(
            row=row,
            lab_profile=lab_profile,
            imaging_context_state=imaging_context_state,
            documentation_context=documentation_context,
            context_contradiction_type=context_contradiction_type,
            treatment_masking_risk=treatment_masking_risk,
            care_mode=care_mode,
        )

        context_operational_confusion_score = _clip(
            hospital_overload_pressure * 0.24
            + contextual_contradiction_score * 0.22
            + (result_acknowledgment_state != "acknowledged_on_time") * 0.16
            + (evidence_surface_deception_type != "none") * 0.16
            + missing_context_flag * 0.10
            + documentation_conflict_flag * 0.08
            + critical_lab_pending_flag * 0.04
        )

        bire_context_pressure_score = _clip(
            row.get("silent_collapse_pressure", 0.0) * 0.18
            + row.get("deceptive_stability_index", 0.0) * 0.16
            + row.get("signal_conflict_burden", 0.0) * 0.14
            + context_operational_confusion_score * 0.18
            + hospital_overload_pressure * 0.14
            + treatment_masking_risk * 0.10
            + (evidence_trust_state == "LOW_TRUST") * 0.10
        )

        context_requires_bire_skepticism_flag = bool(
            bire_context_pressure_score >= 0.45
            or evidence_trust_state == "LOW_TRUST"
            or contextual_contradiction_score >= 0.40
            or evidence_surface_deception_type != "none"
        )

        delayed_confirmation_possible_flag = bool(
            lab_profile in ["delayed_abnormal_labs", "critical_result_delayed"]
            or imaging_context_state == "RESULT_DELAYED"
            or result_acknowledgment_state in ["acknowledged_late", "buried_in_chart", "handoff_missed_result"]
        )

        surface_level_reassurance_risk_flag = bool(
            evidence_surface_deception_type != "none"
            or (
                lab_profile in ["normal_labs", "mild_nonspecific_abnormality", "normal_but_trending_wrong"]
                and row.get("silent_collapse_pressure", 0.0) >= 0.40
            )
        )

        # -------------------------------------------------
        # Outcome-facing context hardening
        # -------------------------------------------------

        lab_abnormality_burden = int(
            0
            + (lab_profile in [
                "mild_nonspecific_abnormality",
                "inflammatory_response",
                "infection_pattern",
                "renal_instability",
                "metabolic_instability",
                "hyperglycemia_pattern",
                "hypoglycemia_pattern",
                "electrolyte_derangement",
                "cardiac_marker_elevation",
                "troponin_leak_pattern",
                "heart_failure_congestion_pattern",
                "coagulation_abnormality",
            ]) * 1
            + (lab_profile in [
                "sepsis_like_pattern",
                "acute_kidney_injury_pattern",
                "hyperkalemia_pattern",
                "hypokalemia_pattern",
                "hyponatremia_pattern",
                "anion_gap_metabolic_acidosis",
                "respiratory_acidosis_pattern",
                "lactic_acidosis_pattern",
                "bleeding_anemia_pattern",
                "thrombocytopenia_pattern",
                "mixed_instability",
                "multi_organ_stress_pattern",
            ]) * 3
            + (lab_profile in [
                "critical_labs_pending",
                "critical_result_delayed",
                "contradictory_labs",
                "delayed_abnormal_labs",
                "labs_not_ordered_despite_risk",
            ]) * 2
            + critical_lab_pending_flag * 2
            + (lab_context_trust_state == "LOW_TRUST") * 1
        )

        critical_lab_burden = int(
            0
            + (lab_profile in [
                "sepsis_like_pattern",
                "anion_gap_metabolic_acidosis",
                "lactic_acidosis_pattern",
                "hyperkalemia_pattern",
                "critical_labs_pending",
                "critical_result_delayed",
                "multi_organ_stress_pattern",
            ]) * 1
            + (critical_lab_pending_flag) * 1
            + (lab_profile == "labs_not_ordered_despite_risk" and row.get("silent_collapse_pressure", 0.0) >= 0.45) * 1
        )

        imaging_instability_score = _clip(
            (imaging_type in [
                "ct_angio_chest",
                "ct_angio_head_neck",
                "echo",
                "repeat_ct",
                "repeat_chest_xray",
                "delayed_imaging_needed",
                "imaging_not_ordered_despite_risk",
            ]) * 0.20
            + (imaging_context_state == "RESULT_DELAYED") * 0.20
            + (imaging_context_state == "IMAGING_NOT_ORDERED_DESPITE_RISK") * 0.28
            + hospital_overload_pressure * 0.14
            + row.get("silent_collapse_pressure", 0.0) * 0.12
            + contextual_contradiction_score * 0.14
            + row.get("hidden_acuity_gap_score", 0.0) * 0.12
        )

        delayed_result_risk_score = _clip(
            lab_delay_minutes / 720 * 0.18
            + imaging_delay_minutes / 1200 * 0.18
            + (result_acknowledgment_state != "acknowledged_on_time") * 0.18
            + critical_lab_pending_flag * 0.16
            + (ordering_failure_type != "none") * 0.12
            + hospital_overload_pressure * 0.10
            + contextual_contradiction_score * 0.08
        )

        data_trust_pressure_score = _clip(
            (1 - evidence_completeness_score) * 0.32
            + contextual_contradiction_score * 0.22
            + delayed_result_risk_score * 0.18
            + missing_context_flag * 0.08
            + documentation_conflict_flag * 0.08
            + duplicate_documentation_flag * 0.04
            + hospital_overload_pressure * 0.08
        )

        context_hidden_instability_pressure = _clip(
            row.get("silent_collapse_pressure", 0.0) * 0.22
            + row.get("normal_range_deception_score", 0.0) * 0.16
            + row.get("deceptive_stability_index", 0.0) * 0.14
            + contextual_contradiction_score * 0.14
            + treatment_masking_risk * 0.10
            + delayed_result_risk_score * 0.10
            + (evidence_trust_state == "LOW_TRUST") * 0.08
            + surface_level_reassurance_risk_flag * 0.06
        )

        context_false_reassurance_pressure = _clip(
            surface_level_reassurance_risk_flag * 0.20
            + (evidence_surface_deception_type != "none") * 0.18
            + (context_contradiction_type != "none") * 0.14
            + treatment_masking_risk * 0.12
            + (documentation_context in ["clean_documentation", "copy_forward_note", "triage_note_understates_severity"]) * 0.10
            + row.get("normal_range_deception_score", 0.0) * 0.10
            + delayed_result_risk_score * 0.08
            + (lab_context_trust_state != "HIGH_TRUST") * 0.08
        )

        context_masked_deterioration_flag = bool(
            (
                context_hidden_instability_pressure >= 0.45
                and surface_level_reassurance_risk_flag
            )
            or (
                treatment_masking_risk >= 0.45
                and row.get("silent_collapse_pressure", 0.0) >= 0.35
            )
            or context_contradiction_type in [
                "medication_masks_vital_instability",
                "normal_labs_but_hidden_vitals_unstable",
                "documentation_says_stable_but_vitals_worsen",
            ]
        )

        context_false_reassurance_flag = bool(
            context_false_reassurance_pressure >= 0.45
            or evidence_surface_deception_type in [
                "labs_look_reassuring_but_patient_worsening",
                "documentation_says_stable_but_trajectory_worse",
                "normal_initial_workup_but_hidden_instability",
                "surface_level_low_acuity_but_high_hidden_risk",
            ]
        )

        context_deterioration_pressure = _clip(
            row.get("deterioration_tendency", 0.0) * 0.45
            + context_hidden_instability_pressure * 0.18
            + lab_abnormality_burden / 10 * 0.10
            + critical_lab_burden / 5 * 0.12
            + imaging_instability_score * 0.08
            + delayed_result_risk_score * 0.07
        )

        context_recovery_resilience_pressure = _clip(
            treatment_masking_risk * 0.16
            + data_trust_pressure_score * 0.16
            + context_false_reassurance_pressure * 0.14
            + hospital_overload_pressure * 0.12
            + contextual_contradiction_score * 0.12
            + delayed_result_risk_score * 0.10
            + critical_lab_burden / 5 * 0.10
            + context_hidden_instability_pressure * 0.10
        )

        deterioration_tendency = _clip(
            row.get("deterioration_tendency", 0.0)
            + context_deterioration_pressure * 0.12
            + context_false_reassurance_pressure * 0.05
        )

        recovery_resilience = _clip(
            row.get("recovery_resilience", 0.5)
            - context_recovery_resilience_pressure * 0.18
            + (evidence_trust_state == "HIGH_TRUST") * 0.04
            - context_masked_deterioration_flag * 0.04
        )

        hidden_instability_score = _clip(
            max(
                row.get("hidden_instability_score", 0.0),
                context_hidden_instability_pressure,
            )
        )

        masked_deterioration_flag = bool(
            row.get("masked_deterioration_flag", False)
            or context_masked_deterioration_flag
        )

        handoff_risk = _clip(
            fragility * 0.24
            + chaos * 0.20
            + lifecycle_uncertainty * 0.16
            + hospital_overload_pressure * 0.14
            + (care_mode in ["INPATIENT", "ICU"]) * 0.12
            + delayed_charting_flag * 0.08
            + missing_context_flag * 0.06
        )

        treatment_complexity_score = int(
            len(medications_administered)
            + int(lab_ordered)
            + int(imaging_ordered)
            + int(has_event)
            + int(delayed_charting_flag)
            + int(critical_lab_pending_flag)
            + int(medication_context_conflict_flag)
            + int(documentation_conflict_flag)
            + int(ordering_failure_type != "none")
            + int(context_contradiction_type != "none")
            + int(result_acknowledgment_state != "acknowledged_on_time")
            + int(hospital_context_state != "normal_operations")
            + int(evidence_surface_deception_type != "none")
        )

        context_rows.append(
            {
                **row_dict,

                "hospital_context_state": hospital_context_state,
                "hospital_overload_pressure": round(float(hospital_overload_pressure), 4),

                "lab_profile": lab_profile,
                "lab_ordered": bool(lab_ordered),
                "lab_delay_minutes": int(lab_delay_minutes),
                "critical_lab_pending_flag": bool(critical_lab_pending_flag),
                "lab_context_trust_state": lab_context_trust_state,

                "imaging_type": imaging_type,
                "imaging_ordered": bool(imaging_ordered),
                "imaging_delay_minutes": int(imaging_delay_minutes),
                "imaging_result_available": bool(imaging_result_available),
                "imaging_context_state": imaging_context_state,

                "medications_administered": medications_administered,
                "medication_count": int(len(medications_administered)),
                "allergy_type": allergy_type,
                "contrast_allergy_conflict": bool(contrast_allergy_conflict),
                "antibiotic_allergy_conflict": bool(antibiotic_allergy_conflict),
                "opioid_allergy_conflict": bool(opioid_allergy_conflict),
                "medication_context_conflict_flag": bool(medication_context_conflict_flag),
                "care_mode_medication_mismatch_flag": bool(care_mode_medication_mismatch_flag),
                "treatment_masking_risk": round(float(treatment_masking_risk), 4),

                "documentation_context": documentation_context,
                "delayed_charting_flag": bool(delayed_charting_flag),
                "duplicate_documentation_flag": bool(duplicate_documentation_flag),
                "documentation_conflict_flag": bool(documentation_conflict_flag),
                "missing_context_flag": bool(missing_context_flag),

                "ordering_failure_type": ordering_failure_type,
                "context_contradiction_type": context_contradiction_type,
                "result_acknowledgment_state": result_acknowledgment_state,
                "evidence_surface_deception_type": evidence_surface_deception_type,

                "contextual_contradiction_score": round(float(contextual_contradiction_score), 4),
                "evidence_completeness_score": round(float(evidence_completeness_score), 4),
                "evidence_trust_state": evidence_trust_state,

                "context_operational_confusion_score": round(float(context_operational_confusion_score), 4),
                "bire_context_pressure_score": round(float(bire_context_pressure_score), 4),
                "context_requires_bire_skepticism_flag": bool(context_requires_bire_skepticism_flag),
                "delayed_confirmation_possible_flag": bool(delayed_confirmation_possible_flag),
                "surface_level_reassurance_risk_flag": bool(surface_level_reassurance_risk_flag),

                "lab_abnormality_burden": int(lab_abnormality_burden),
                "critical_lab_burden": int(critical_lab_burden),
                "imaging_instability_score": round(float(imaging_instability_score), 4),

                "delayed_result_risk_score": round(float(delayed_result_risk_score), 4),
                "data_trust_pressure_score": round(float(data_trust_pressure_score), 4),
                "context_hidden_instability_pressure": round(float(context_hidden_instability_pressure), 4),
                "context_false_reassurance_pressure": round(float(context_false_reassurance_pressure), 4),
                "context_deterioration_pressure": round(float(context_deterioration_pressure), 4),
                "context_recovery_resilience_pressure": round(float(context_recovery_resilience_pressure), 4),

                "context_masked_deterioration_flag": bool(context_masked_deterioration_flag),
                "context_false_reassurance_flag": bool(context_false_reassurance_flag),
                "hidden_instability_score": round(float(hidden_instability_score), 4),
                "masked_deterioration_flag": bool(masked_deterioration_flag),
                "deterioration_tendency": round(float(deterioration_tendency), 4),
                "recovery_resilience": round(float(recovery_resilience), 4),

                "handoff_risk": round(float(handoff_risk), 4),
                "treatment_complexity_score": treatment_complexity_score,
            }
        )

    return pd.DataFrame(context_rows)