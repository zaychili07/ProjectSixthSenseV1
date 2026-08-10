"""
BIRE OS Synthetic Patient Profile Generator

Chapter 53 doctrine:
No more babying BIRE OS.

53.2 is the identity layer of the synthetic healthcare world.

Target:
- 30,000 patients
- 10-year longitudinal ecosystem
- average patients through extremely complex patients
- hidden instability
- false reassurance
- longitudinal memory need
- care fragmentation
- readmission pressure
- mortality vulnerability
- operational dependency
- overconfidence traps

Doctrine:
We Detect What Others Miss.
"""

import numpy as np
import pandas as pd

from bire.synthetic.config import (
    SYNTHETIC_ECOSYSTEM_CONFIG,
    CONDITION_PROFILES,
)


# =========================================================
# Condition Profiles
# =========================================================

CONDITION_WEIGHTS = {
    "baseline_stable": 0.145,
    "copd_respiratory_vulnerability": 0.095,
    "chf_fluid_pressure_instability": 0.095,
    "ckd_lab_instability": 0.085,
    "diabetes_metabolic_instability": 0.085,
    "sepsis_recovery_risk": 0.075,
    "arrhythmia_hemodynamic_instability": 0.075,
    "post_surgical_recovery": 0.065,
    "chronic_multimorbidity": 0.075,
    "recurrent_deterioration_pattern": 0.065,
    "complex_pancreatitis_icu_complication": 0.035,
    "contrast_allergy_imaging_complication": 0.030,
    "medication_restriction_complexity": 0.035,
    "new_onset_diabetes_during_admission": 0.030,
    "multi_event_complex_hospitalization": 0.045,
}


CONDITION_RISK_MAP = {
    "baseline_stable": 0.02,
    "copd_respiratory_vulnerability": 0.18,
    "chf_fluid_pressure_instability": 0.21,
    "ckd_lab_instability": 0.19,
    "diabetes_metabolic_instability": 0.15,
    "sepsis_recovery_risk": 0.24,
    "arrhythmia_hemodynamic_instability": 0.20,
    "post_surgical_recovery": 0.17,
    "chronic_multimorbidity": 0.30,
    "recurrent_deterioration_pattern": 0.33,
    "complex_pancreatitis_icu_complication": 0.37,
    "contrast_allergy_imaging_complication": 0.25,
    "medication_restriction_complexity": 0.25,
    "new_onset_diabetes_during_admission": 0.27,
    "multi_event_complex_hospitalization": 0.40,
}


CONDITION_COMPLEXITY_MAP = {
    "baseline_stable": 0.05,
    "copd_respiratory_vulnerability": 0.36,
    "chf_fluid_pressure_instability": 0.42,
    "ckd_lab_instability": 0.40,
    "diabetes_metabolic_instability": 0.34,
    "sepsis_recovery_risk": 0.48,
    "arrhythmia_hemodynamic_instability": 0.44,
    "post_surgical_recovery": 0.36,
    "chronic_multimorbidity": 0.72,
    "recurrent_deterioration_pattern": 0.72,
    "complex_pancreatitis_icu_complication": 0.82,
    "contrast_allergy_imaging_complication": 0.56,
    "medication_restriction_complexity": 0.62,
    "new_onset_diabetes_during_admission": 0.58,
    "multi_event_complex_hospitalization": 0.88,
}


# =========================================================
# Presentation Profiles
# =========================================================

PRESENTATION_PROFILES = [
    "minor_headache_migraine",
    "abdominal_pain_unspecified",
    "chest_pain_low_risk",
    "chest_pain_hidden_cardiac_risk",
    "shortness_of_breath_mild",
    "shortness_of_breath_hidden_respiratory_decline",
    "dizziness_near_syncope",
    "fatigue_weakness",
    "nausea_vomiting_dehydration",
    "minor_injury",
    "post_op_pain",
    "fever_unspecified",
    "infection_concern",
    "hyperglycemia_symptoms",
    "palpitations",
    "altered_mental_status",
    "fluid_retention_swelling",
    "back_pain",
    "normal_appearing_complex_case",
    "silent_instability_case",
    "multi_system_complex_case",
    "vague_complaint_high_risk",
    "weakness_in_elderly_hidden_sepsis",
    "falls_with_hidden_instability",
    "discharge_return_worse",
    "medication_side_effect_confusion",
]


PRESENTATION_WEIGHTS = {
    "minor_headache_migraine": 0.080,
    "abdominal_pain_unspecified": 0.085,
    "chest_pain_low_risk": 0.070,
    "chest_pain_hidden_cardiac_risk": 0.045,
    "shortness_of_breath_mild": 0.070,
    "shortness_of_breath_hidden_respiratory_decline": 0.045,
    "dizziness_near_syncope": 0.055,
    "fatigue_weakness": 0.070,
    "nausea_vomiting_dehydration": 0.065,
    "minor_injury": 0.065,
    "post_op_pain": 0.040,
    "fever_unspecified": 0.060,
    "infection_concern": 0.055,
    "hyperglycemia_symptoms": 0.040,
    "palpitations": 0.040,
    "altered_mental_status": 0.030,
    "fluid_retention_swelling": 0.035,
    "back_pain": 0.050,
    "normal_appearing_complex_case": 0.035,
    "silent_instability_case": 0.030,
    "multi_system_complex_case": 0.030,
    "vague_complaint_high_risk": 0.030,
    "weakness_in_elderly_hidden_sepsis": 0.025,
    "falls_with_hidden_instability": 0.025,
    "discharge_return_worse": 0.025,
    "medication_side_effect_confusion": 0.025,
}


PRESENTATION_RISK_MAP = {
    "minor_headache_migraine": 0.03,
    "abdominal_pain_unspecified": 0.10,
    "chest_pain_low_risk": 0.08,
    "chest_pain_hidden_cardiac_risk": 0.25,
    "shortness_of_breath_mild": 0.12,
    "shortness_of_breath_hidden_respiratory_decline": 0.28,
    "dizziness_near_syncope": 0.15,
    "fatigue_weakness": 0.11,
    "nausea_vomiting_dehydration": 0.11,
    "minor_injury": 0.02,
    "post_op_pain": 0.17,
    "fever_unspecified": 0.14,
    "infection_concern": 0.21,
    "hyperglycemia_symptoms": 0.19,
    "palpitations": 0.17,
    "altered_mental_status": 0.32,
    "fluid_retention_swelling": 0.19,
    "back_pain": 0.05,
    "normal_appearing_complex_case": 0.32,
    "silent_instability_case": 0.36,
    "multi_system_complex_case": 0.40,
    "vague_complaint_high_risk": 0.30,
    "weakness_in_elderly_hidden_sepsis": 0.38,
    "falls_with_hidden_instability": 0.26,
    "discharge_return_worse": 0.34,
    "medication_side_effect_confusion": 0.24,
}


# =========================================================
# Longitudinal Archetypes
# =========================================================

LONGITUDINAL_ARCHETYPES = [
    "average_low_touch_patient",
    "routine_preventive_care_patient",
    "episodic_er_patient",
    "poor_followup_patient",
    "chronic_disease_progressor",
    "readmission_prone_patient",
    "silent_deteriorator",
    "false_stability_patient",
    "fragmented_care_patient",
    "complex_multisystem_patient",
    "frequent_high_utilizer",
    "end_of_life_decline_patient",
    "resilient_recovery_patient",
    "operationally_lost_patient",
]


LONGITUDINAL_ARCHETYPE_WEIGHTS = {
    "average_low_touch_patient": 0.18,
    "routine_preventive_care_patient": 0.12,
    "episodic_er_patient": 0.12,
    "poor_followup_patient": 0.08,
    "chronic_disease_progressor": 0.11,
    "readmission_prone_patient": 0.09,
    "silent_deteriorator": 0.06,
    "false_stability_patient": 0.06,
    "fragmented_care_patient": 0.06,
    "complex_multisystem_patient": 0.05,
    "frequent_high_utilizer": 0.035,
    "end_of_life_decline_patient": 0.025,
    "resilient_recovery_patient": 0.035,
    "operationally_lost_patient": 0.035,
}


# =========================================================
# Helpers
# =========================================================

def _clip(values, low=0.0, high=1.0):
    return np.clip(values, low, high)


def _normalized_probabilities(weight_map, keys):
    probs = np.array([weight_map.get(k, 0.01) for k in keys], dtype=float)
    return probs / probs.sum()


def _normalized_condition_probabilities():
    return _normalized_probabilities(CONDITION_WEIGHTS, CONDITION_PROFILES)


def _normalized_presentation_probabilities():
    return _normalized_probabilities(PRESENTATION_WEIGHTS, PRESENTATION_PROFILES)


def _normalized_longitudinal_archetype_probabilities():
    return _normalized_probabilities(
        LONGITUDINAL_ARCHETYPE_WEIGHTS,
        LONGITUDINAL_ARCHETYPES,
    )


def _assign_age_band(ages):
    return pd.cut(
        ages,
        bins=[17, 39, 64, 79, 120],
        labels=[
            "young_adult",
            "middle_adult",
            "older_adult",
            "elderly",
        ],
    ).astype(str)


def _build_baseline_vital_profile(rng, ages, condition_profile, presentation_profile):
    n = len(ages)

    baseline_hr = rng.normal(78, 10, size=n)
    baseline_rr = rng.normal(17, 3, size=n)
    baseline_spo2 = rng.normal(97, 1.5, size=n)
    baseline_temp_f = rng.normal(98.6, 0.6, size=n)
    baseline_sbp = rng.normal(122, 14, size=n)
    baseline_dbp = rng.normal(76, 9, size=n)

    age_push = _clip((ages - 65) / 60, 0, 0.35)
    baseline_hr += age_push * 5
    baseline_rr += age_push * 1.5
    baseline_spo2 -= age_push * 1.2

    for i, profile in enumerate(condition_profile):

        if "copd" in profile:
            baseline_rr[i] += rng.normal(2.5, 1.0)
            baseline_spo2[i] -= rng.normal(2.5, 1.0)

        if "chf" in profile:
            baseline_sbp[i] += rng.normal(8, 4)
            baseline_rr[i] += rng.normal(1.5, 0.8)

        if "ckd" in profile:
            baseline_sbp[i] += rng.normal(6, 4)

        if "diabetes" in profile:
            baseline_hr[i] += rng.normal(3, 2)

        if "arrhythmia" in profile:
            baseline_hr[i] += rng.normal(10, 5)

        if "sepsis" in profile:
            baseline_hr[i] += rng.normal(6, 3)
            baseline_temp_f[i] += rng.normal(0.7, 0.4)

        if "chronic_multimorbidity" in profile:
            baseline_hr[i] += rng.normal(5, 3)
            baseline_rr[i] += rng.normal(1.5, 0.8)
            baseline_sbp[i] += rng.normal(5, 4)

        if "recurrent_deterioration" in profile:
            baseline_hr[i] += rng.normal(8, 4)
            baseline_rr[i] += rng.normal(2, 1)

        if "multi_event_complex" in profile:
            baseline_hr[i] += rng.normal(8, 4)
            baseline_rr[i] += rng.normal(2.5, 1.0)
            baseline_spo2[i] -= rng.normal(1.5, 0.8)

    for i, presentation in enumerate(presentation_profile):

        if presentation == "chest_pain_hidden_cardiac_risk":
            baseline_hr[i] += rng.normal(4, 3)

        if presentation == "shortness_of_breath_hidden_respiratory_decline":
            baseline_rr[i] += rng.normal(2.0, 1.0)
            baseline_spo2[i] -= rng.normal(1.5, 0.8)

        if presentation == "silent_instability_case":
            baseline_hr[i] += rng.normal(2, 2)
            baseline_rr[i] += rng.normal(1, 1)

        if presentation == "infection_concern":
            baseline_temp_f[i] += rng.normal(0.6, 0.4)

        if presentation == "weakness_in_elderly_hidden_sepsis":
            baseline_hr[i] += rng.normal(4, 2)
            baseline_temp_f[i] += rng.normal(0.4, 0.3)

        if presentation == "nausea_vomiting_dehydration":
            baseline_hr[i] += rng.normal(5, 3)
            baseline_sbp[i] -= rng.normal(5, 3)

    return {
        "baseline_hr": np.round(np.clip(baseline_hr, 45, 135), 2),
        "baseline_rr": np.round(np.clip(baseline_rr, 8, 34), 2),
        "baseline_spo2": np.round(np.clip(baseline_spo2, 84, 100), 2),
        "baseline_temp_f": np.round(np.clip(baseline_temp_f, 95, 103.5), 2),
        "baseline_sbp": np.round(np.clip(baseline_sbp, 80, 195), 2),
        "baseline_dbp": np.round(np.clip(baseline_dbp, 40, 120), 2),
    }


def _choose_archetype_adjustments(archetypes):
    adjustments = {
        "readmission_boost": np.zeros(len(archetypes)),
        "fragmentation_boost": np.zeros(len(archetypes)),
        "hidden_signal_boost": np.zeros(len(archetypes)),
        "false_stability_boost": np.zeros(len(archetypes)),
        "progression_boost": np.zeros(len(archetypes)),
        "mortality_boost": np.zeros(len(archetypes)),
        "resilience_boost": np.zeros(len(archetypes)),
        "utilization_boost": np.zeros(len(archetypes)),
    }

    for i, archetype in enumerate(archetypes):
        if archetype == "poor_followup_patient":
            adjustments["fragmentation_boost"][i] += 0.18
            adjustments["readmission_boost"][i] += 0.08

        elif archetype == "chronic_disease_progressor":
            adjustments["progression_boost"][i] += 0.20
            adjustments["readmission_boost"][i] += 0.08

        elif archetype == "readmission_prone_patient":
            adjustments["readmission_boost"][i] += 0.25
            adjustments["utilization_boost"][i] += 0.12

        elif archetype == "silent_deteriorator":
            adjustments["hidden_signal_boost"][i] += 0.25
            adjustments["false_stability_boost"][i] += 0.12

        elif archetype == "false_stability_patient":
            adjustments["false_stability_boost"][i] += 0.25
            adjustments["hidden_signal_boost"][i] += 0.10

        elif archetype == "fragmented_care_patient":
            adjustments["fragmentation_boost"][i] += 0.25
            adjustments["utilization_boost"][i] += 0.06

        elif archetype == "complex_multisystem_patient":
            adjustments["progression_boost"][i] += 0.18
            adjustments["readmission_boost"][i] += 0.14
            adjustments["mortality_boost"][i] += 0.10

        elif archetype == "frequent_high_utilizer":
            adjustments["utilization_boost"][i] += 0.25
            adjustments["readmission_boost"][i] += 0.12

        elif archetype == "end_of_life_decline_patient":
            adjustments["progression_boost"][i] += 0.22
            adjustments["mortality_boost"][i] += 0.22
            adjustments["readmission_boost"][i] += 0.10

        elif archetype == "resilient_recovery_patient":
            adjustments["resilience_boost"][i] += 0.22
            adjustments["mortality_boost"][i] -= 0.05

        elif archetype == "operationally_lost_patient":
            adjustments["fragmentation_boost"][i] += 0.22
            adjustments["hidden_signal_boost"][i] += 0.10
            adjustments["utilization_boost"][i] += 0.08

    return adjustments


# =========================================================
# Main Generator
# =========================================================

def generate_patient_master_table(
    n_patients=None,
    random_seed=None,
):
    """
    Generate synthetic patient-level profiles.

    These profiles provide the longitudinal identity layer for a
    30,000-patient, 10-year synthetic healthcare ecosystem.
    """

    if n_patients is None:
        n_patients = SYNTHETIC_ECOSYSTEM_CONFIG.get(
            "n_patients",
            SYNTHETIC_ECOSYSTEM_CONFIG.get("total_patients", 30000),
        )

    if random_seed is None:
        random_seed = SYNTHETIC_ECOSYSTEM_CONFIG["random_seed"]

    rng = np.random.default_rng(random_seed)

    patient_ids = [
        f"P{str(i).zfill(5)}"
        for i in range(1, n_patients + 1)
    ]

    ages = rng.integers(18, 96, size=n_patients)
    age_band = _assign_age_band(ages)

    sex = rng.choice(
        ["F", "M"],
        size=n_patients,
        p=[0.52, 0.48],
    )

    condition_profile = rng.choice(
        CONDITION_PROFILES,
        size=n_patients,
        p=_normalized_condition_probabilities(),
    )

    presentation_profile = rng.choice(
        PRESENTATION_PROFILES,
        size=n_patients,
        p=_normalized_presentation_probabilities(),
    )

    longitudinal_archetype = rng.choice(
        LONGITUDINAL_ARCHETYPES,
        size=n_patients,
        p=_normalized_longitudinal_archetype_probabilities(),
    )

    adjustments = _choose_archetype_adjustments(longitudinal_archetype)

    baseline_risk = rng.beta(2, 6, size=n_patients)

    age_risk_adjustment = np.clip(
        (ages - 50) / 100,
        0,
        0.38,
    )

    condition_adjustment = np.array(
        [CONDITION_RISK_MAP.get(c, 0.18) for c in condition_profile]
    )

    presentation_adjustment = np.array(
        [PRESENTATION_RISK_MAP.get(p, 0.10) for p in presentation_profile]
    )

    chronic_complexity_score = np.array(
        [CONDITION_COMPLEXITY_MAP.get(c, 0.30) for c in condition_profile]
    )

    multimorbidity_count = rng.poisson(
        lam=1 + chronic_complexity_score * 3.5 + age_risk_adjustment * 2,
        size=n_patients,
    )

    multimorbidity_count = np.clip(multimorbidity_count, 0, 10)

    frailty_modifier = np.clip((ages - 65) / 75, 0, 0.40)

    frailty_score = np.clip(
        chronic_complexity_score * 0.42
        + frailty_modifier
        + adjustments["progression_boost"] * 0.25
        + rng.normal(0, 0.06, size=n_patients),
        0,
        1,
    )

    fragility_score = np.clip(
        baseline_risk
        + age_risk_adjustment
        + condition_adjustment
        + presentation_adjustment * 0.35
        + frailty_score * 0.22,
        0,
        1,
    )

    chronic_instability_score = np.clip(
        chronic_complexity_score * 0.48
        + presentation_adjustment * 0.20
        + adjustments["progression_boost"] * 0.22
        + rng.normal(0, 0.08, size=n_patients),
        0,
        1,
    )

    ten_year_progression_pressure = np.clip(
        chronic_complexity_score * 0.30
        + frailty_score * 0.24
        + adjustments["progression_boost"]
        + age_risk_adjustment * 0.18
        + rng.normal(0, 0.07, size=n_patients),
        0,
        1,
    )

    readmission_tendency = np.clip(
        fragility_score * 0.52
        + chronic_instability_score * 0.26
        + adjustments["readmission_boost"]
        + rng.normal(0, 0.08, size=n_patients),
        0,
        1,
    )

    deterioration_tendency = np.clip(
        fragility_score * 0.58
        + chronic_instability_score * 0.30
        + presentation_adjustment * 0.10
        + ten_year_progression_pressure * 0.16
        + rng.normal(0, 0.10, size=n_patients),
        0,
        1,
    )

    recovery_resilience = np.clip(
        1
        - fragility_score * 0.55
        - chronic_instability_score * 0.18
        - ten_year_progression_pressure * 0.10
        + adjustments["resilience_boost"]
        + rng.normal(0, 0.10, size=n_patients),
        0,
        1,
    )

    data_messiness_tendency = np.clip(
        rng.beta(2, 5, size=n_patients)
        + fragility_score * 0.18
        + chronic_complexity_score * 0.10
        + adjustments["fragmentation_boost"] * 0.25,
        0,
        1,
    )

    care_access_friction = np.clip(
        rng.beta(2, 4, size=n_patients)
        + chronic_complexity_score * 0.08
        + adjustments["fragmentation_boost"] * 0.30,
        0,
        1,
    )

    followup_reliability = np.clip(
        1
        - care_access_friction * 0.48
        - data_messiness_tendency * 0.20
        + adjustments["resilience_boost"] * 0.18
        + rng.normal(0, 0.08, size=n_patients),
        0,
        1,
    )

    health_literacy_complexity = np.clip(
        rng.beta(2, 3, size=n_patients)
        + care_access_friction * 0.15
        + data_messiness_tendency * 0.10,
        0,
        1,
    )

    care_fragmentation_risk = np.clip(
        data_messiness_tendency * 0.32
        + care_access_friction * 0.34
        + chronic_complexity_score * 0.22
        + adjustments["fragmentation_boost"]
        + rng.normal(0, 0.06, size=n_patients),
        0,
        1,
    )

    baseline_signal_noise = np.clip(
        rng.beta(2, 6, size=n_patients)
        + chronic_instability_score * 0.18
        + adjustments["hidden_signal_boost"] * 0.15,
        0,
        1,
    )

    weak_signal_likelihood = np.clip(
        deterioration_tendency * 0.32
        + baseline_signal_noise * 0.34
        + chronic_instability_score * 0.24
        + adjustments["hidden_signal_boost"]
        + rng.normal(0, 0.05, size=n_patients),
        0,
        1,
    )

    false_stability_risk = np.clip(
        recovery_resilience * 0.18
        + chronic_instability_score * 0.33
        + weak_signal_likelihood * 0.28
        + adjustments["false_stability_boost"]
        + rng.normal(0, 0.06, size=n_patients),
        0,
        1,
    )

    overconfidence_trap_risk = np.clip(
        false_stability_risk * 0.36
        + baseline_signal_noise * 0.24
        + data_messiness_tendency * 0.18
        + weak_signal_likelihood * 0.16
        + rng.normal(0, 0.05, size=n_patients),
        0,
        1,
    )

    longitudinal_memory_need = np.clip(
        readmission_tendency * 0.32
        + chronic_instability_score * 0.32
        + care_fragmentation_risk * 0.20
        + ten_year_progression_pressure * 0.16
        + rng.normal(0, 0.05, size=n_patients),
        0,
        1,
    )

    misleading_presentation_risk = np.clip(
        (
            np.isin(
                presentation_profile,
                [
                    "chest_pain_hidden_cardiac_risk",
                    "shortness_of_breath_hidden_respiratory_decline",
                    "normal_appearing_complex_case",
                    "silent_instability_case",
                    "vague_complaint_high_risk",
                    "weakness_in_elderly_hidden_sepsis",
                    "falls_with_hidden_instability",
                    "discharge_return_worse",
                ],
            ).astype(float)
            * 0.42
        )
        + weak_signal_likelihood * 0.24
        + false_stability_risk * 0.20
        + rng.normal(0, 0.05, size=n_patients),
        0,
        1,
    )

    mortality_vulnerability = np.clip(
        fragility_score * 0.28
        + frailty_score * 0.22
        + chronic_instability_score * 0.18
        + ten_year_progression_pressure * 0.18
        + adjustments["mortality_boost"]
        + rng.normal(0, 0.06, size=n_patients),
        0,
        1,
    )

    utilization_intensity = np.clip(
        readmission_tendency * 0.32
        + care_access_friction * 0.16
        + chronic_complexity_score * 0.20
        + adjustments["utilization_boost"]
        + rng.normal(0, 0.06, size=n_patients),
        0,
        1,
    )

    operational_dependency_score = np.clip(
        care_fragmentation_risk * 0.28
        + longitudinal_memory_need * 0.26
        + utilization_intensity * 0.22
        + data_messiness_tendency * 0.14
        + rng.normal(0, 0.05, size=n_patients),
        0,
        1,
    )

    hospital_memory_fragility = np.clip(
        care_fragmentation_risk * 0.35
        + longitudinal_memory_need * 0.25
        + data_messiness_tendency * 0.20
        + utilization_intensity * 0.10
        + rng.normal(0, 0.05, size=n_patients),
        0,
        1,
    )

    ten_year_instability_burden = np.clip(
        deterioration_tendency * 0.30
        + readmission_tendency * 0.22
        + ten_year_progression_pressure * 0.24
        + mortality_vulnerability * 0.14
        + weak_signal_likelihood * 0.10,
        0,
        1,
    )

    vital_baselines = _build_baseline_vital_profile(
        rng,
        ages,
        condition_profile,
        presentation_profile,
    )

    patient_df = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "age": ages,
            "age_band": age_band,
            "sex": sex,
            "condition_profile": condition_profile,
            "presentation_profile": presentation_profile,
            "longitudinal_archetype": longitudinal_archetype,

            "baseline_risk": baseline_risk.round(4),
            "presentation_risk_adjustment": presentation_adjustment.round(4),
            "chronic_complexity_score": chronic_complexity_score.round(4),
            "multimorbidity_count": multimorbidity_count,
            "frailty_score": frailty_score.round(4),
            "fragility_score": fragility_score.round(4),
            "chronic_instability_score": chronic_instability_score.round(4),

            "ten_year_progression_pressure": ten_year_progression_pressure.round(4),
            "ten_year_instability_burden": ten_year_instability_burden.round(4),
            "mortality_vulnerability": mortality_vulnerability.round(4),

            "readmission_tendency": readmission_tendency.round(4),
            "deterioration_tendency": deterioration_tendency.round(4),
            "recovery_resilience": recovery_resilience.round(4),
            "utilization_intensity": utilization_intensity.round(4),

            "data_messiness_tendency": data_messiness_tendency.round(4),
            "care_access_friction": care_access_friction.round(4),
            "followup_reliability": followup_reliability.round(4),
            "health_literacy_complexity": health_literacy_complexity.round(4),
            "care_fragmentation_risk": care_fragmentation_risk.round(4),

            "baseline_signal_noise": baseline_signal_noise.round(4),
            "weak_signal_likelihood": weak_signal_likelihood.round(4),
            "false_stability_risk": false_stability_risk.round(4),
            "overconfidence_trap_risk": overconfidence_trap_risk.round(4),
            "longitudinal_memory_need": longitudinal_memory_need.round(4),
            "misleading_presentation_risk": misleading_presentation_risk.round(4),

            "operational_dependency_score": operational_dependency_score.round(4),
            "hospital_memory_fragility": hospital_memory_fragility.round(4),

            **vital_baselines,
        }
    )

    patient_df["patient_complexity_tier"] = pd.cut(
        patient_df["chronic_complexity_score"],
        bins=[-0.01, 0.25, 0.50, 0.75, 1.01],
        labels=[
            "LOW_COMPLEXITY",
            "MODERATE_COMPLEXITY",
            "HIGH_COMPLEXITY",
            "EXTREME_COMPLEXITY",
        ],
    ).astype(str)

    patient_df["patient_risk_tier"] = pd.cut(
        patient_df["fragility_score"],
        bins=[-0.01, 0.30, 0.55, 0.75, 1.01],
        labels=[
            "LOW_RISK",
            "MODERATE_RISK",
            "HIGH_RISK",
            "CRITICAL_RISK",
        ],
    ).astype(str)

    patient_df["ten_year_burden_tier"] = pd.cut(
        patient_df["ten_year_instability_burden"],
        bins=[-0.01, 0.25, 0.50, 0.75, 1.01],
        labels=[
            "LOW_10YR_BURDEN",
            "MODERATE_10YR_BURDEN",
            "HIGH_10YR_BURDEN",
            "EXTREME_10YR_BURDEN",
        ],
    ).astype(str)

    patient_df["synthetic_patient_archetype"] = np.select(
        [
            patient_df["mortality_vulnerability"] >= 0.75,
            patient_df["ten_year_instability_burden"] >= 0.75,
            patient_df["misleading_presentation_risk"] >= 0.65,
            patient_df["weak_signal_likelihood"] >= 0.65,
            patient_df["false_stability_risk"] >= 0.65,
            patient_df["care_fragmentation_risk"] >= 0.65,
            patient_df["hospital_memory_fragility"] >= 0.65,
            patient_df["chronic_instability_score"] >= 0.65,
        ],
        [
            "mortality_vulnerable_patient",
            "extreme_ten_year_instability_patient",
            "misleading_normal_appearing_patient",
            "weak_signal_patient",
            "false_stability_patient",
            "fragmented_care_patient",
            "hospital_memory_fragile_patient",
            "chronically_unstable_patient",
        ],
        default="standard_longitudinal_patient",
    )

    return patient_df