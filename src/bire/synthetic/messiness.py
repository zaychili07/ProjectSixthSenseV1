"""
BIRE OS Synthetic Messiness Engine

Chapter 53 doctrine:
The Siege of BIRE OS

Purpose:
Inject controlled healthcare data corruption into the synthetic ecosystem.

This module creates structured data chaos:
- missing values
- impossible vitals
- mixed units
- categorical typos
- duplicate rows
- stale / copy-forward values
- out-of-order timestamps
- delayed charting
- conflicting event labels
- context corruption
- medication list corruption
- imaging/lab contradiction
- identifier corruption
- hospital-overload-driven data decay
- BIRE OS data skepticism flags

This is structured chaos, not random nonsense.

Doctrine:
We Detect What Others Miss.
"""

import numpy as np
import pandas as pd

from bire.synthetic.config import SYNTHETIC_ECOSYSTEM_CONFIG


CARE_MODE_TYPOS = {
    "OUTPATIENT": ["OUTPATENT", "OUT_PT", "outpatient", "OUTPATIENTT"],
    "ER_ESI_5": ["ER_ESI5", "ER-ESI-5", "ESI_5", "ER_ESI_55"],
    "ER_ESI_4": ["ER_ESI4", "ER-ESI-4", "ESI_4", "ER_ESI_44"],
    "ER_ESI_3": ["ER_ESI3", "ER-ESI-3", "ESI_3", "ER_ESI_33"],
    "ER_ESI_2": ["ER_ESI2", "ER-ESI-2", "ESI_2", "ER_ESI_22"],
    "ER_ESI_1": ["ER_ESI1", "ER-ESI-1", "ESI_1", "ER_ESI_11"],
    "INPATIENT": ["IN_PT", "INPATENT", "inpatient", "INPATIENTT"],
    "ICU": ["I.C.U.", "icu", "ICUU", "INTENSIVE_CARE"],
}


CONDITION_TYPOS = {
    "baseline_stable": ["baseline stable", "base_stable", "baseline_stabl"],
    "copd_respiratory_vulnerability": ["COPD_resp_vuln", "copd respiratory", "copd_resp_vulnerability"],
    "chf_fluid_pressure_instability": ["CHF_fluid_instability", "chf pressure", "chf_fluid_pressure"],
    "ckd_lab_instability": ["CKD_lab_instability", "ckd labs unstable", "ckd_lab_instabilty"],
    "diabetes_metabolic_instability": ["DM_metabolic", "diabetes metabolic", "diabetes_metabolic_instabilty"],
    "sepsis_recovery_risk": ["sepsis recovery", "sepsis_risk", "sepsis_recovery_isk"],
    "arrhythmia_hemodynamic_instability": ["arrhythmia_hemo", "arrythmia_hemodynamic", "arrhythmia_instability"],
    "post_surgical_recovery": ["post surgical", "post_op_recovery", "post_surg_recovery"],
    "chronic_multimorbidity": ["multi_morbidity", "chronic multimorbidity", "chronic_multimorbity"],
    "recurrent_deterioration_pattern": ["recurrent_deterioration", "recurrent decline", "recurrent_deterioration_pattrn"],
}


LAB_PROFILE_TYPOS = {
    "normal_labs": ["normal labs", "norm_labs", "normal_lab"],
    "critical_labs_pending": ["critical pending", "crit_labs_pending", "critical_lab_pendng"],
    "critical_result_delayed": ["critical delayed", "crit_result_delay", "critical_result_delayd"],
    "contradictory_labs": ["contradictory labs", "conflicting_labs", "contradict_labs"],
    "normal_but_trending_wrong": ["normal trending wrong", "normal_wrong_trend", "normal_but_trending_wron"],
}


IMAGING_TYPOS = {
    "chest_xray": ["chest xray", "CXR", "chest-x-ray"],
    "portable_chest_xray": ["portable CXR", "port_chest_xray", "portable chest"],
    "ct_chest": ["CT chest", "ct thorax", "ct_chst"],
    "ct_abdomen_pelvis": ["CT AP", "ct_abd_pelv", "ct abdomen pelvis"],
    "echo": ["ECHO", "echocardiogram", "eccho"],
    "ekg": ["ECG", "EKG ", "electrocardiogram"],
    "none": ["None", "NONE", ""],
}


def _append_flag(existing, new_flag):
    if pd.isna(existing) or existing == "":
        return new_flag
    return f"{existing};{new_flag}"


def _choose_indices(rng, df, probability):
    if len(df) == 0:
        return []
    mask = rng.random(len(df)) < probability
    return df.index[mask].tolist()


def initialize_messiness_columns(df):
    df = df.copy()

    if "messiness_flags" not in df.columns:
        df["messiness_flags"] = ""

    if "data_quality_score" not in df.columns:
        df["data_quality_score"] = 1.0

    return df


def _degrade(df, idx, amount, flag):
    for i in idx:
        df.at[i, "messiness_flags"] = _append_flag(df.at[i, "messiness_flags"], flag)
    df.loc[idx, "data_quality_score"] -= amount
    df["data_quality_score"] = df["data_quality_score"].clip(0, 1)
    return df


def _effective_probability(base_probability, multiplier=1.0, df=None, overload_col="hospital_overload_pressure"):
    probability = base_probability * multiplier
    if df is not None and overload_col in df.columns:
        overload = df[overload_col].fillna(0).clip(0, 1)
        return probability * (1 + overload)
    return probability


def inject_missing_values(df, columns, probability=0.02, random_seed=None):
    rng = np.random.default_rng(random_seed)
    df = initialize_messiness_columns(df)

    for col in columns:
        if col not in df.columns:
            continue

        idx = _choose_indices(rng, df, probability)
        df.loc[idx, col] = np.nan
        df = _degrade(df, idx, 0.08, f"missing_{col}")

    return df


def inject_burst_missingness(
    df,
    columns,
    probability=0.006,
    random_seed=None,
):
    """
    Simulate short bursts where multiple fields are missing together.
    Example: monitor downtime, charting backlog, interface failure.
    """
    rng = np.random.default_rng(random_seed)
    df = initialize_messiness_columns(df)

    existing_cols = [col for col in columns if col in df.columns]
    if not existing_cols:
        return df

    idx = _choose_indices(rng, df, probability)

    for i in idx:
        n_cols = int(rng.integers(2, min(len(existing_cols), 6) + 1))
        selected_cols = rng.choice(existing_cols, size=n_cols, replace=False)

        for col in selected_cols:
            df.at[i, col] = np.nan

        df.at[i, "messiness_flags"] = _append_flag(
            df.at[i, "messiness_flags"],
            "burst_missingness_multiple_fields",
        )
        df.at[i, "data_quality_score"] -= 0.16

    df["data_quality_score"] = df["data_quality_score"].clip(0, 1)
    return df


def inject_categorical_typos(df, column, typo_map, probability=0.03, random_seed=None):
    rng = np.random.default_rng(random_seed)
    df = initialize_messiness_columns(df)

    if column not in df.columns:
        return df

    idx = _choose_indices(rng, df, probability)

    for i in idx:
        original_value = df.at[i, column]
        if original_value in typo_map:
            df.at[i, column] = rng.choice(typo_map[original_value])
            df.at[i, "messiness_flags"] = _append_flag(df.at[i, "messiness_flags"], f"typo_{column}")
            df.at[i, "data_quality_score"] -= 0.05

    df["data_quality_score"] = df["data_quality_score"].clip(0, 1)
    return df


def inject_impossible_values(df, random_seed=None, probability=0.01):
    rng = np.random.default_rng(random_seed)
    df = initialize_messiness_columns(df)

    impossible_value_map = {
        "heart_rate": [-10, 0, 320, 999],
        "resp_rate": [-3, 0, 90, 120],
        "spo2": [-5, 0, 150, 999],
        "temperature": [70.0, 75.0, 115.0, 120.0],
        "sbp": [-20, 0, 300, 999],
        "dbp": [-10, 0, 220, 999],
        "lab_delay_minutes": [-20, 9999],
        "imaging_delay_minutes": [-30, 9999],
    }

    for col, bad_values in impossible_value_map.items():
        if col not in df.columns:
            continue

        idx = _choose_indices(rng, df, probability)

        for i in idx:
            df.at[i, col] = rng.choice(bad_values)
            df.at[i, "messiness_flags"] = _append_flag(df.at[i, "messiness_flags"], f"impossible_{col}")
            df.at[i, "data_quality_score"] -= 0.15

    df["data_quality_score"] = df["data_quality_score"].clip(0, 1)
    return df


def inject_plausible_but_wrong_vitals(df, random_seed=None, probability=0.012):
    """
    Inject values that are not impossible but clinically suspicious.
    These are more dangerous than impossible values because they may pass validation.
    """
    rng = np.random.default_rng(random_seed)
    df = initialize_messiness_columns(df)

    vital_cols = ["heart_rate", "resp_rate", "spo2", "temperature", "sbp", "dbp"]
    if not all(col in df.columns for col in vital_cols):
        return df

    idx = _choose_indices(rng, df, probability)

    for i in idx:
        pattern = rng.choice(
            [
                "reassuring_spo2_despite_respiratory_strain",
                "normal_bp_despite_high_shock_index",
                "rounded_all_vitals",
                "swapped_hr_rr",
                "stale_normal_vitals",
            ]
        )

        if pattern == "reassuring_spo2_despite_respiratory_strain":
            df.at[i, "spo2"] = rng.choice([96.0, 97.0, 98.0])
            df.at[i, "resp_rate"] = rng.choice([26.0, 28.0, 30.0])

        elif pattern == "normal_bp_despite_high_shock_index":
            df.at[i, "sbp"] = rng.choice([108.0, 112.0, 118.0])
            df.at[i, "heart_rate"] = rng.choice([112.0, 118.0, 124.0])

        elif pattern == "rounded_all_vitals":
            for col in vital_cols:
                if pd.notna(df.at[i, col]):
                    df.at[i, col] = round(float(df.at[i, col]) / 5) * 5

        elif pattern == "swapped_hr_rr":
            hr = df.at[i, "heart_rate"]
            rr = df.at[i, "resp_rate"]
            df.at[i, "heart_rate"] = rr
            df.at[i, "resp_rate"] = hr

        elif pattern == "stale_normal_vitals":
            df.at[i, "heart_rate"] = 80.0
            df.at[i, "resp_rate"] = 18.0
            df.at[i, "spo2"] = 98.0
            df.at[i, "temperature"] = 98.6
            df.at[i, "sbp"] = 120.0
            df.at[i, "dbp"] = 80.0

        df.at[i, "messiness_flags"] = _append_flag(df.at[i, "messiness_flags"], f"plausible_wrong_vitals_{pattern}")
        df.at[i, "data_quality_score"] -= 0.13

    df["data_quality_score"] = df["data_quality_score"].clip(0, 1)
    return df


def inject_mixed_units(df, random_seed=None, probability=0.01):
    rng = np.random.default_rng(random_seed)
    df = initialize_messiness_columns(df)

    if "temperature" not in df.columns:
        return df

    idx = _choose_indices(rng, df, probability)

    for i in idx:
        temp_f = df.at[i, "temperature"]
        if pd.isna(temp_f):
            continue

        temp_c = (temp_f - 32) * 5 / 9
        df.at[i, "temperature"] = round(float(temp_c), 1)
        df.at[i, "messiness_flags"] = _append_flag(
            df.at[i, "messiness_flags"],
            "mixed_unit_temperature_celsius_in_fahrenheit_column",
        )
        df.at[i, "data_quality_score"] -= 0.12

    df["data_quality_score"] = df["data_quality_score"].clip(0, 1)
    return df


def inject_duplicate_rows(df, probability=0.01, random_seed=None):
    rng = np.random.default_rng(random_seed)
    df = initialize_messiness_columns(df)

    idx = _choose_indices(rng, df, probability)
    duplicate_df = df.loc[idx].copy()

    if len(duplicate_df) == 0:
        return df

    duplicate_df["messiness_flags"] = duplicate_df["messiness_flags"].apply(
        lambda x: _append_flag(x, "duplicate_row")
    )
    duplicate_df["data_quality_score"] = (duplicate_df["data_quality_score"] - 0.10).clip(0, 1)

    out_df = pd.concat([df, duplicate_df], ignore_index=True)
    return out_df


def inject_near_duplicate_rows(df, probability=0.006, random_seed=None):
    """
    Duplicate rows but slightly alter chart time or one value.
    Harder than exact duplicate detection.
    """
    rng = np.random.default_rng(random_seed)
    df = initialize_messiness_columns(df)

    idx = _choose_indices(rng, df, probability)
    duplicate_df = df.loc[idx].copy()

    if len(duplicate_df) == 0:
        return df

    if "chart_time" in duplicate_df.columns:
        duplicate_df["chart_time"] = pd.to_datetime(duplicate_df["chart_time"], errors="coerce") + pd.to_timedelta(
            rng.integers(1, 20, size=len(duplicate_df)),
            unit="m",
        )

    for col in ["heart_rate", "resp_rate", "spo2", "sbp", "dbp"]:
        if col in duplicate_df.columns:
            mask = rng.random(len(duplicate_df)) < 0.5
            duplicate_df.loc[mask, col] = duplicate_df.loc[mask, col] + rng.normal(0, 1, mask.sum())

    duplicate_df["messiness_flags"] = duplicate_df["messiness_flags"].apply(
        lambda x: _append_flag(x, "near_duplicate_row_slightly_modified")
    )
    duplicate_df["data_quality_score"] = (duplicate_df["data_quality_score"] - 0.12).clip(0, 1)

    return pd.concat([df, duplicate_df], ignore_index=True)


def inject_out_of_order_timestamps(
    df,
    patient_col="patient_id",
    time_col="encounter_start",
    probability=0.01,
    random_seed=None,
):
    rng = np.random.default_rng(random_seed)
    df = initialize_messiness_columns(df)

    if time_col not in df.columns:
        return df

    idx = _choose_indices(rng, df, probability)

    for i in idx:
        if pd.isna(df.at[i, time_col]):
            continue

        df.at[i, time_col] = (
            pd.Timestamp(df.at[i, time_col])
            - pd.Timedelta(minutes=int(rng.choice([5, 10, 15, 30, 60, 240, 720])))
        )

        df.at[i, "messiness_flags"] = _append_flag(df.at[i, "messiness_flags"], "out_of_order_timestamp")
        df.at[i, "data_quality_score"] -= 0.08

    df["data_quality_score"] = df["data_quality_score"].clip(0, 1)
    return df


def inject_delayed_charting(df, time_col="encounter_start", random_seed=None, probability=0.03):
    rng = np.random.default_rng(random_seed)
    df = initialize_messiness_columns(df)

    if time_col not in df.columns:
        return df

    df["chart_time"] = pd.to_datetime(df[time_col], errors="coerce")

    idx = _choose_indices(rng, df, probability)

    for i in idx:
        delay_minutes = int(rng.choice([15, 30, 60, 120, 240, 480, 720, 1440]))

        df.at[i, "chart_time"] = (
            pd.Timestamp(df.at[i, time_col])
            + pd.Timedelta(minutes=delay_minutes)
        )

        df.at[i, "messiness_flags"] = _append_flag(df.at[i, "messiness_flags"], "delayed_charting")
        df.at[i, "data_quality_score"] -= 0.06

    df["data_quality_score"] = df["data_quality_score"].clip(0, 1)
    return df


def inject_future_charting_errors(df, time_col="encounter_start", random_seed=None, probability=0.004):
    rng = np.random.default_rng(random_seed)
    df = initialize_messiness_columns(df)

    if time_col not in df.columns:
        return df

    if "chart_time" not in df.columns:
        df["chart_time"] = pd.to_datetime(df[time_col], errors="coerce")

    idx = _choose_indices(rng, df, probability)

    for i in idx:
        df.at[i, "chart_time"] = (
            pd.Timestamp(df.at[i, time_col])
            + pd.Timedelta(days=int(rng.choice([2, 3, 7, 14])))
        )

        df.at[i, "messiness_flags"] = _append_flag(df.at[i, "messiness_flags"], "future_charting_timestamp_error")
        df.at[i, "data_quality_score"] -= 0.12

    df["data_quality_score"] = df["data_quality_score"].clip(0, 1)
    return df


def inject_conflicting_event_labels(df, probability=0.01, random_seed=None):
    rng = np.random.default_rng(random_seed)
    df = initialize_messiness_columns(df)

    if "has_event" not in df.columns:
        return df

    idx = _choose_indices(rng, df, probability)

    for i in idx:
        original_value = bool(df.at[i, "has_event"])
        df.at[i, "has_event"] = not original_value
        df.at[i, "messiness_flags"] = _append_flag(df.at[i, "messiness_flags"], "conflicting_event_label")
        df.at[i, "data_quality_score"] -= 0.14

    df["data_quality_score"] = df["data_quality_score"].clip(0, 1)
    return df


def inject_conflicting_lifecycle_states(df, probability=0.008, random_seed=None):
    rng = np.random.default_rng(random_seed)
    df = initialize_messiness_columns(df)

    if "current_lifecycle_state" not in df.columns:
        return df

    conflicting_states = [
        "PRE_EVENT_LOW_RISK_SURVEILLANCE",
        "PRE_EVENT_FALSE_REASSURANCE",
        "POST_EVENT_MONITOR",
        "FALSE_RECOVERY",
        "RE_ESCALATION",
        "RECOVERY_CONFIRMED",
    ]

    idx = _choose_indices(rng, df, probability)

    for i in idx:
        df.at[i, "current_lifecycle_state"] = rng.choice(conflicting_states)
        df.at[i, "messiness_flags"] = _append_flag(df.at[i, "messiness_flags"], "conflicting_lifecycle_state")
        df.at[i, "data_quality_score"] -= 0.13

    df["data_quality_score"] = df["data_quality_score"].clip(0, 1)
    return df


def inject_copy_forward_values(df, patient_col="patient_id", probability=0.02, random_seed=None):
    rng = np.random.default_rng(random_seed)
    df = initialize_messiness_columns(df)

    vital_cols = ["heart_rate", "resp_rate", "spo2", "temperature", "sbp", "dbp"]
    available_vitals = [col for col in vital_cols if col in df.columns]

    if not available_vitals or patient_col not in df.columns:
        return df

    sort_cols = [patient_col]
    if "encounter_start" in df.columns:
        sort_cols.append("encounter_start")
    elif "timestamp" in df.columns:
        sort_cols.append("timestamp")

    df = df.sort_values(sort_cols).reset_index(drop=True)
    idx = _choose_indices(rng, df, probability)

    for i in idx:
        if i == 0:
            continue
        if df.at[i, patient_col] != df.at[i - 1, patient_col]:
            continue

        for col in available_vitals:
            df.at[i, col] = df.at[i - 1, col]

        df.at[i, "messiness_flags"] = _append_flag(df.at[i, "messiness_flags"], "copy_forward_vitals")
        df.at[i, "data_quality_score"] -= 0.09

    df["data_quality_score"] = df["data_quality_score"].clip(0, 1)
    return df


def inject_stale_context_values(df, patient_col="patient_id", probability=0.014, random_seed=None):
    """
    Copy prior contextual evidence forward.
    Example: old lab/imaging/medication context persists into a new encounter.
    """
    rng = np.random.default_rng(random_seed)
    df = initialize_messiness_columns(df)

    context_cols = [
        "lab_profile",
        "imaging_type",
        "imaging_context_state",
        "medications_administered",
        "documentation_context",
        "evidence_trust_state",
        "result_acknowledgment_state",
    ]
    context_cols = [col for col in context_cols if col in df.columns]

    if not context_cols or patient_col not in df.columns:
        return df

    sort_cols = [patient_col]
    if "encounter_start" in df.columns:
        sort_cols.append("encounter_start")

    df = df.sort_values(sort_cols).reset_index(drop=True)
    idx = _choose_indices(rng, df, probability)

    for i in idx:
        if i == 0:
            continue
        if df.at[i, patient_col] != df.at[i - 1, patient_col]:
            continue

        selected_cols = rng.choice(context_cols, size=int(rng.integers(1, len(context_cols) + 1)), replace=False)

        for col in selected_cols:
            df.at[i, col] = df.at[i - 1, col]

        df.at[i, "messiness_flags"] = _append_flag(df.at[i, "messiness_flags"], "stale_context_copied_forward")
        df.at[i, "data_quality_score"] -= 0.11

    df["data_quality_score"] = df["data_quality_score"].clip(0, 1)
    return df


def inject_context_contradictions(df, probability=0.012, random_seed=None):
    """
    Inject explicit contradictions between context fields.
    """
    rng = np.random.default_rng(random_seed)
    df = initialize_messiness_columns(df)

    idx = _choose_indices(rng, df, probability)

    for i in idx:
        contradiction = rng.choice(
            [
                "result_available_with_no_imaging",
                "critical_lab_pending_but_high_trust",
                "no_event_but_event_type_present",
                "event_present_but_event_type_none",
                "normal_operations_with_mass_casualty_flag",
                "result_acknowledged_but_missing_result",
            ]
        )

        if contradiction == "result_available_with_no_imaging":
            if "imaging_ordered" in df.columns:
                df.at[i, "imaging_ordered"] = False
            if "imaging_context_state" in df.columns:
                df.at[i, "imaging_context_state"] = "RESULT_AVAILABLE"

        elif contradiction == "critical_lab_pending_but_high_trust":
            if "critical_lab_pending_flag" in df.columns:
                df.at[i, "critical_lab_pending_flag"] = True
            if "lab_context_trust_state" in df.columns:
                df.at[i, "lab_context_trust_state"] = "HIGH_TRUST"

        elif contradiction == "no_event_but_event_type_present":
            if "has_event" in df.columns:
                df.at[i, "has_event"] = False
            if "event_type" in df.columns:
                df.at[i, "event_type"] = rng.choice(["respiratory_decline", "hemodynamic_instability"])

        elif contradiction == "event_present_but_event_type_none":
            if "has_event" in df.columns:
                df.at[i, "has_event"] = True
            if "event_type" in df.columns:
                df.at[i, "event_type"] = "NONE"

        elif contradiction == "normal_operations_with_mass_casualty_flag":
            if "hospital_context_state" in df.columns:
                df.at[i, "hospital_context_state"] = "normal_operations"
            if "hospital_overload_pressure" in df.columns:
                df.at[i, "hospital_overload_pressure"] = 0.85

        elif contradiction == "result_acknowledged_but_missing_result":
            if "result_acknowledgment_state" in df.columns:
                df.at[i, "result_acknowledgment_state"] = "acknowledged_on_time"
            if "imaging_result_available" in df.columns:
                df.at[i, "imaging_result_available"] = False

        df.at[i, "messiness_flags"] = _append_flag(df.at[i, "messiness_flags"], f"context_contradiction_{contradiction}")
        df.at[i, "data_quality_score"] -= 0.16

    df["data_quality_score"] = df["data_quality_score"].clip(0, 1)
    return df


def inject_medication_list_corruption(df, probability=0.012, random_seed=None):
    rng = np.random.default_rng(random_seed)
    df = initialize_messiness_columns(df)

    if "medications_administered" not in df.columns:
        return df

    idx = _choose_indices(rng, df, probability)

    for i in idx:
        corruption = rng.choice(["stringified_list", "empty_list_despite_count", "duplicated_med", "unknown_medication"])

        meds = df.at[i, "medications_administered"]

        if corruption == "stringified_list":
            df.at[i, "medications_administered"] = str(meds)

        elif corruption == "empty_list_despite_count":
            df.at[i, "medications_administered"] = []
            if "medication_count" in df.columns:
                df.at[i, "medication_count"] = int(rng.integers(2, 8))

        elif corruption == "duplicated_med":
            if isinstance(meds, list) and len(meds) > 0:
                df.at[i, "medications_administered"] = meds + [meds[0]]

        elif corruption == "unknown_medication":
            if isinstance(meds, list):
                df.at[i, "medications_administered"] = meds + ["UNKNOWN_MED_???"]
            else:
                df.at[i, "medications_administered"] = ["UNKNOWN_MED_???"]

        df.at[i, "messiness_flags"] = _append_flag(df.at[i, "messiness_flags"], f"medication_list_corruption_{corruption}")
        df.at[i, "data_quality_score"] -= 0.11

    df["data_quality_score"] = df["data_quality_score"].clip(0, 1)
    return df


def inject_identifier_corruption(df, probability=0.004, random_seed=None):
    rng = np.random.default_rng(random_seed)
    df = initialize_messiness_columns(df)

    possible_cols = [col for col in ["patient_id", "encounter_id", "event_episode_id"] if col in df.columns]

    if not possible_cols:
        return df

    idx = _choose_indices(rng, df, probability)

    for i in idx:
        col = rng.choice(possible_cols)
        original = df.at[i, col]

        if pd.isna(original):
            continue

        corruption = rng.choice(["lowercase", "extra_space", "truncated", "suffix_noise"])

        if corruption == "lowercase":
            df.at[i, col] = str(original).lower()
        elif corruption == "extra_space":
            df.at[i, col] = f" {original} "
        elif corruption == "truncated":
            df.at[i, col] = str(original)[:-1]
        elif corruption == "suffix_noise":
            df.at[i, col] = f"{original}_X"

        df.at[i, "messiness_flags"] = _append_flag(df.at[i, "messiness_flags"], f"identifier_corruption_{col}_{corruption}")
        df.at[i, "data_quality_score"] -= 0.18

    df["data_quality_score"] = df["data_quality_score"].clip(0, 1)
    return df


def inject_overload_amplified_decay(df, probability=0.015, random_seed=None):
    """
    When the hospital is overloaded, data quality should degrade more often.
    """
    rng = np.random.default_rng(random_seed)
    df = initialize_messiness_columns(df)

    if "hospital_overload_pressure" not in df.columns:
        return df

    overload = df["hospital_overload_pressure"].fillna(0).clip(0, 1)
    mask = rng.random(len(df)) < (probability * (1 + overload * 2))
    idx = df.index[mask].tolist()

    for i in idx:
        decay_type = rng.choice(
            [
                "overload_chart_delay",
                "overload_missing_context",
                "overload_result_delay",
                "overload_triage_understatement",
                "overload_unreconciled_context",
            ]
        )

        if decay_type == "overload_chart_delay" and "documentation_context" in df.columns:
            df.at[i, "documentation_context"] = "delayed_charting"

        elif decay_type == "overload_missing_context" and "missing_context_flag" in df.columns:
            df.at[i, "missing_context_flag"] = True

        elif decay_type == "overload_result_delay" and "result_acknowledgment_state" in df.columns:
            df.at[i, "result_acknowledgment_state"] = rng.choice(["acknowledged_late", "buried_in_chart", "handoff_missed_result"])

        elif decay_type == "overload_triage_understatement" and "documentation_context" in df.columns:
            df.at[i, "documentation_context"] = "triage_note_understates_severity"

        elif decay_type == "overload_unreconciled_context" and "context_contradiction_type" in df.columns:
            df.at[i, "context_contradiction_type"] = "delayed_results_change_interpretation"

        df.at[i, "messiness_flags"] = _append_flag(df.at[i, "messiness_flags"], decay_type)
        df.at[i, "data_quality_score"] -= 0.10

    df["data_quality_score"] = df["data_quality_score"].clip(0, 1)
    return df


def finalize_messiness_metrics(df):
    df = initialize_messiness_columns(df)

    df["has_messiness"] = df["messiness_flags"].fillna("").ne("")

    df["messiness_burden"] = df["messiness_flags"].fillna("").apply(
        lambda x: 0 if x == "" else len(str(x).split(";"))
    )

    flag_text = df["messiness_flags"].fillna("")

    df["missingness_burden_flag"] = flag_text.str.contains("missing", regex=False)
    df["temporal_disorder_flag"] = (
        flag_text.str.contains("timestamp", regex=False)
        | flag_text.str.contains("delayed_charting", regex=False)
        | flag_text.str.contains("future_charting", regex=False)
    )
    df["context_corruption_flag"] = (
        flag_text.str.contains("context", regex=False)
        | flag_text.str.contains("stale_context", regex=False)
        | flag_text.str.contains("result", regex=False)
        | flag_text.str.contains("documentation", regex=False)
    )
    df["vital_corruption_flag"] = (
        flag_text.str.contains("heart_rate", regex=False)
        | flag_text.str.contains("resp_rate", regex=False)
        | flag_text.str.contains("spo2", regex=False)
        | flag_text.str.contains("temperature", regex=False)
        | flag_text.str.contains("sbp", regex=False)
        | flag_text.str.contains("dbp", regex=False)
        | flag_text.str.contains("vitals", regex=False)
    )
    df["label_conflict_flag"] = (
        flag_text.str.contains("conflicting_event_label", regex=False)
        | flag_text.str.contains("conflicting_lifecycle_state", regex=False)
    )
    df["identifier_corruption_flag"] = flag_text.str.contains("identifier_corruption", regex=False)

    df["critical_data_quality_failure_flag"] = (
        (df["data_quality_score"] <= 0.60)
        | (df["messiness_burden"] >= 3)
        | df["identifier_corruption_flag"]
        | df["label_conflict_flag"]
        | flag_text.str.contains("impossible_", regex=False)
    )

    df["data_quality_instability_score"] = np.clip(
        (1 - df["data_quality_score"].fillna(1.0)) * 0.45
        + df["messiness_burden"].fillna(0).clip(0, 6) / 6 * 0.25
        + df["critical_data_quality_failure_flag"].astype(int) * 0.20
        + df["temporal_disorder_flag"].astype(int) * 0.05
        + df["context_corruption_flag"].astype(int) * 0.05,
        0,
        1,
    )

    if "bire_context_pressure_score" in df.columns:
        context_pressure = df["bire_context_pressure_score"].fillna(0)
    else:
        context_pressure = 0

    if "lifecycle_uncertainty_score" in df.columns:
        lifecycle_uncertainty = df["lifecycle_uncertainty_score"].fillna(0)
    else:
        lifecycle_uncertainty = 0

    df["data_trust_pressure_score"] = np.clip(
        df["data_quality_instability_score"].fillna(0) * 0.45
        + context_pressure * 0.25
        + lifecycle_uncertainty * 0.15
        + df["critical_data_quality_failure_flag"].astype(int) * 0.15,
        0,
        1,
    )

    df["bire_data_skepticism_required_flag"] = (
        (df["data_trust_pressure_score"] >= 0.42)
        | df["critical_data_quality_failure_flag"]
        | (df["data_quality_score"] <= 0.72)
    )

    df["data_quality_tier"] = pd.cut(
        df["data_quality_score"],
        bins=[-0.01, 0.45, 0.70, 0.88, 1.01],
        labels=[
            "CRITICAL_DATA_FAILURE",
            "LOW_TRUST_DATA",
            "PARTIAL_TRUST_DATA",
            "HIGH_TRUST_DATA",
        ],
    ).astype(str)

    return df


def apply_synthetic_messiness(df, random_seed=None, level="high"):
    """
    Apply full structured messiness pipeline.

    Levels:
    - low
    - medium
    - high
    - extreme
    - hospital_hell
    """

    if random_seed is None:
        random_seed = SYNTHETIC_ECOSYSTEM_CONFIG["random_seed"]

    if level == "low":
        multiplier = 0.5
    elif level == "medium":
        multiplier = 1.0
    elif level == "high":
        multiplier = 1.5
    elif level == "extreme":
        multiplier = 2.5
    elif level == "hospital_hell":
        multiplier = 3.25
    else:
        multiplier = 1.0

    df = initialize_messiness_columns(df)

    df = inject_missing_values(
        df,
        columns=[
            "heart_rate",
            "resp_rate",
            "spo2",
            "temperature",
            "sbp",
            "dbp",
            "lab_profile",
            "imaging_type",
            "imaging_context_state",
            "medications_administered",
            "documentation_context",
            "result_acknowledgment_state",
            "evidence_surface_deception_type",
        ],
        probability=0.018 * multiplier,
        random_seed=random_seed + 1,
    )

    df = inject_burst_missingness(
        df,
        columns=[
            "heart_rate",
            "resp_rate",
            "spo2",
            "sbp",
            "dbp",
            "lab_profile",
            "imaging_type",
            "evidence_trust_state",
            "context_contradiction_type",
        ],
        probability=0.004 * multiplier,
        random_seed=random_seed + 2,
    )

    df = inject_categorical_typos(
        df,
        column="care_mode",
        typo_map=CARE_MODE_TYPOS,
        probability=0.020 * multiplier,
        random_seed=random_seed + 3,
    )

    df = inject_categorical_typos(
        df,
        column="condition_profile",
        typo_map=CONDITION_TYPOS,
        probability=0.016 * multiplier,
        random_seed=random_seed + 4,
    )

    df = inject_categorical_typos(
        df,
        column="lab_profile",
        typo_map=LAB_PROFILE_TYPOS,
        probability=0.012 * multiplier,
        random_seed=random_seed + 5,
    )

    df = inject_categorical_typos(
        df,
        column="imaging_type",
        typo_map=IMAGING_TYPOS,
        probability=0.012 * multiplier,
        random_seed=random_seed + 6,
    )

    df = inject_impossible_values(
        df,
        probability=0.006 * multiplier,
        random_seed=random_seed + 7,
    )

    df = inject_plausible_but_wrong_vitals(
        df,
        probability=0.010 * multiplier,
        random_seed=random_seed + 8,
    )

    df = inject_mixed_units(
        df,
        probability=0.008 * multiplier,
        random_seed=random_seed + 9,
    )

    df = inject_duplicate_rows(
        df,
        probability=0.006 * multiplier,
        random_seed=random_seed + 10,
    )

    df = inject_near_duplicate_rows(
        df,
        probability=0.004 * multiplier,
        random_seed=random_seed + 11,
    )

    df = inject_out_of_order_timestamps(
        df,
        time_col="encounter_start" if "encounter_start" in df.columns else "timestamp",
        probability=0.008 * multiplier,
        random_seed=random_seed + 12,
    )

    df = inject_delayed_charting(
        df,
        time_col="encounter_start" if "encounter_start" in df.columns else "timestamp",
        probability=0.022 * multiplier,
        random_seed=random_seed + 13,
    )

    df = inject_future_charting_errors(
        df,
        time_col="encounter_start" if "encounter_start" in df.columns else "timestamp",
        probability=0.003 * multiplier,
        random_seed=random_seed + 14,
    )

    df = inject_conflicting_event_labels(
        df,
        probability=0.006 * multiplier,
        random_seed=random_seed + 15,
    )

    df = inject_conflicting_lifecycle_states(
        df,
        probability=0.006 * multiplier,
        random_seed=random_seed + 16,
    )

    df = inject_copy_forward_values(
        df,
        probability=0.012 * multiplier,
        random_seed=random_seed + 17,
    )

    df = inject_stale_context_values(
        df,
        probability=0.010 * multiplier,
        random_seed=random_seed + 18,
    )

    df = inject_context_contradictions(
        df,
        probability=0.010 * multiplier,
        random_seed=random_seed + 19,
    )

    df = inject_medication_list_corruption(
        df,
        probability=0.010 * multiplier,
        random_seed=random_seed + 20,
    )

    df = inject_identifier_corruption(
        df,
        probability=0.003 * multiplier,
        random_seed=random_seed + 21,
    )

    df = inject_overload_amplified_decay(
        df,
        probability=0.012 * multiplier,
        random_seed=random_seed + 22,
    )

    df = finalize_messiness_metrics(df)

    return df