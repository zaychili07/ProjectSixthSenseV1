"""
BIRE OS Synthetic Ecosystem Configuration
"""


# Synthetic Measurement Unit Doctrine
UNIT_SYSTEM = "US_HOSPITAL_CONVENTIONAL"

VITAL_UNITS = {
    "heart_rate": "beats_per_minute",
    "resp_rate": "breaths_per_minute",
    "spo2": "percent",
    "temperature": "fahrenheit",
    "sbp": "mmHg",
    "dbp": "mmHg",
}

LAB_UNITS = {
    "glucose": "mg/dL",
    "lactate": "mmol/L",
    "wbc": "K/uL",
    "creatinine": "mg/dL",
    "bun": "mg/dL",
    "potassium": "mEq/L",
    "sodium": "mEq/L",
    "bicarbonate": "mEq/L",
    "ph": "unitless",
    "pco2": "mmHg",
    "troponin": "ng/mL",
    "hemoglobin": "g/dL",
    "platelets": "K/uL",
    "ast": "U/L",
    "alt": "U/L",
    "bilirubin": "mg/dL",
    "lipase": "U/L",
}

TEMPERATURE_UNIT = "fahrenheit"
NORMAL_TEMP_F = 98.6
FEVER_TEMP_F = 100.4
HYPOTHERMIA_TEMP_F = 95.0
CRITICAL_FEVER_TEMP_F = 102.2

SYNTHETIC_ECOSYSTEM_CONFIG = {
    "dataset_name": "synthetic_messy_healthcare_ecosystem",
    "version": "v0_53_1_design",
    "n_patients": 500,
    "start_date": "2024-01-01",
    "end_date": "2025-12-31",
    "time_granularity": "5min",
    "purpose": "longitudinal BIRE OS stress testing",
    "messiness_level": "high",
    "random_seed": 42,
}


CARE_MODES = [
    "OUTPATIENT",
    "ER_ESI_5",
    "ER_ESI_4",
    "ER_ESI_3",
    "ER_ESI_2",
    "ER_ESI_1",
    "INPATIENT",
    "ICU",
    "DISCHARGED",
]


CONDITION_PROFILES = [
    "baseline_stable",
    "copd_respiratory_vulnerability",
    "chf_fluid_pressure_instability",
    "ckd_lab_instability",
    "diabetes_metabolic_instability",
    "sepsis_recovery_risk",
    "arrhythmia_hemodynamic_instability",
    "post_surgical_recovery",
    "chronic_multimorbidity",
    "recurrent_deterioration_pattern",
    "complex_pancreatitis_icu_complication",
    "contrast_allergy_imaging_complication",
    "medication_restriction_complexity",
    "new_onset_diabetes_during_admission",
    "multi_event_complex_hospitalization",
]


OPERATIONAL_PHASES = [
    "PRE_EVENT_SURVEILLANCE",
    "EVENT_TRANSITION",
    "POST_EVENT_MONITOR",
    "RECOVERY_EVALUATION",
    "RE_ESCALATION",
    "CRITICAL_POST_EVENT",
    "REINTEGRATION",
    "DISCHARGED_SURVEILLANCE_GAP",
    "READMISSION_CONTEXT",
]


SYNTHETIC_MESSINESS_TYPES = [
    "missing_vitals",
    "delayed_labs",
    "duplicate_timestamp",
    "out_of_order_timestamp",
    "mixed_units",
    "typo_care_mode",
    "typo_condition",
    "device_dropout",
    "copy_forward_values",
    "impossible_value",
    "contradictory_medication_flag",
    "conflicting_event_label",
    "uncertain_event_timing",
    "overlapping_encounter",
    "discharge_without_clean_closure",
]


JOURNEY_ARCHETYPES = [
    "simple_low_acuity_visit",
    "minor_complaint_discharge",
    "small_finding_followup_needed",
    "small_finding_complex_escalation",
    "er_to_inpatient_progression",
    "inpatient_to_icu_escalation",
    "multi_level_care_complex_case",
    "complex_icu_complication_case",
    "readmission_after_discharge",
    "chronic_recurrent_instability",
]