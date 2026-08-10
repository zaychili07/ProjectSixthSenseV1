"""
BIRE OS Synthetic Vitals Utilities

Chapter 53 doctrine:
No more babying BIRE OS.

Purpose:
Challenge BIRE OS to earn the Project Sixth Sense name.

This module supports:
- derived vital signal generation
- hidden physiologic instability detection
- compensated shock detection
- normal-range deception detection
- pre-threshold deterioration detection
- masked deterioration typing
- false vital recovery detection
- temporal vital trajectory intelligence

Doctrine:
We Detect What Others Miss.
"""

import numpy as np
import pandas as pd


# =========================================================
# Safe Math Helpers
# =========================================================

def _safe_divide(numerator, denominator):
    result = numerator / denominator

    if isinstance(result, pd.Series):
        result = result.replace([np.inf, -np.inf], np.nan)

    return result


def _clip(value, low=0.0, high=1.0):
    return np.clip(value, low, high)


# =========================================================
# Vital Burden Counters
# =========================================================

def _count_abnormal_vitals(df):
    abnormal_flags = pd.DataFrame(index=df.index)

    abnormal_flags["hr_abnormal"] = (
        (df["heart_rate"] < 50)
        | (df["heart_rate"] > 120)
    )

    abnormal_flags["rr_abnormal"] = (
        (df["resp_rate"] < 10)
        | (df["resp_rate"] > 24)
    )

    abnormal_flags["spo2_abnormal"] = df["spo2"] < 92

    abnormal_flags["temp_abnormal"] = (
        (df["temperature"] < 95.0)
        | (df["temperature"] > 100.4)
    )

    abnormal_flags["sbp_abnormal"] = (
        (df["sbp"] < 90)
        | (df["sbp"] > 180)
    )

    abnormal_flags["dbp_abnormal"] = (
        (df["dbp"] < 50)
        | (df["dbp"] > 110)
    )

    return abnormal_flags.sum(axis=1)


def _count_critical_vitals(df):
    critical_flags = pd.DataFrame(index=df.index)

    critical_flags["hr_critical"] = (
        (df["heart_rate"] < 40)
        | (df["heart_rate"] > 140)
    )

    critical_flags["rr_critical"] = (
        (df["resp_rate"] < 8)
        | (df["resp_rate"] > 30)
    )

    critical_flags["spo2_critical"] = df["spo2"] < 88

    critical_flags["temp_critical"] = (
        (df["temperature"] < 95.0)
        | (df["temperature"] > 102.2)
    )

    critical_flags["sbp_critical"] = df["sbp"] < 80

    critical_flags["dbp_critical"] = df["dbp"] < 40

    return critical_flags.sum(axis=1)


# =========================================================
# Raw Lifecycle-Aware Synthetic Vitals Generator
# =========================================================

def generate_raw_vitals_from_lifecycle(
    df,
    random_seed=42,
):
    """
    Generate raw synthetic vitals using lifecycle state, care mode,
    event type, event waves, cascade chains, terminal decline,
    deceptive stability, failed reintegration, recognition delay,
    and masking pressure.
    """

    rng = np.random.default_rng(random_seed)
    vital_df = df.copy()
    n = len(vital_df)

    vital_df["heart_rate"] = rng.normal(84, 14, n)
    vital_df["resp_rate"] = rng.normal(18, 4, n)
    vital_df["spo2"] = rng.normal(96, 2.5, n)
    vital_df["temperature"] = rng.normal(98.6, 1.0, n)
    vital_df["sbp"] = rng.normal(124, 18, n)
    vital_df["dbp"] = rng.normal(74, 10, n)
    vital_df["pulse_mask_hint"] = 0

    for idx, row in vital_df.iterrows():

        state = str(row.get("current_lifecycle_state", ""))
        event_type = str(row.get("event_type", "NONE"))
        care_mode = str(row.get("highest_acuity_mode", row.get("care_mode", "")))

        fragility = row.get("fragility_score", 0.0)
        hidden_pressure = row.get("hidden_instability_pressure", 0.0)
        false_reassurance = row.get("false_reassurance_pressure", 0.0)
        volatility = row.get("post_event_volatility_score", 0.0)
        uncertainty = row.get("lifecycle_uncertainty_score", 0.0)
        visibility = row.get("deterioration_visibility_score", 1.0)

        event_wave_count = row.get("event_wave_count", 0)
        cascade_chain = str(row.get("event_cascade_chain", "NONE"))
        multi_crash = row.get("multi_crash_encounter_flag", False)
        terminal_pressure = row.get("terminal_decline_pressure", 0.0)
        failed_reintegration = row.get("failed_reintegration_risk_flag", False)
        recognition_delay = row.get("recognition_delay_hours", 0)
        deceptive_stability = row.get("deceptive_stability_index", 0.0)
        signal_conflict = row.get("signal_conflict_burden", 0.0)
        warning_window = row.get("pre_event_warning_window", 0)

        acuity_push = 0.0

        if care_mode == "ICU":
            acuity_push = 1.00
        elif care_mode == "ER_ESI_1":
            acuity_push = 0.85
        elif care_mode in ["ER_ESI_2", "INPATIENT"]:
            acuity_push = 0.60
        elif care_mode == "ER_ESI_3":
            acuity_push = 0.35
        elif care_mode in ["ER_ESI_4", "ER_ESI_5"]:
            acuity_push = 0.15

        instability_push = (
            fragility * 0.20
            + hidden_pressure * 0.24
            + volatility * 0.18
            + uncertainty * 0.12
            + acuity_push * 0.08
            + event_wave_count * 0.04
            + terminal_pressure * 0.08
            + signal_conflict * 0.06
        )

        # =====================================================
        # Obvious event physiology
        # =====================================================

        if row.get("has_event", False):

            vital_df.at[idx, "heart_rate"] += 18 * instability_push + rng.normal(8, 6)
            vital_df.at[idx, "resp_rate"] += 8 * instability_push + rng.normal(3, 3)
            vital_df.at[idx, "spo2"] -= 6 * instability_push + rng.normal(1, 2)
            vital_df.at[idx, "sbp"] -= 18 * instability_push + rng.normal(2, 8)
            vital_df.at[idx, "dbp"] -= 8 * instability_push + rng.normal(1, 5)

            if event_type == "respiratory_decline":
                vital_df.at[idx, "resp_rate"] += rng.normal(6, 3)
                vital_df.at[idx, "spo2"] -= rng.normal(5, 2)

            elif event_type == "hemodynamic_instability":
                vital_df.at[idx, "heart_rate"] += rng.normal(12, 5)
                vital_df.at[idx, "sbp"] -= rng.normal(18, 8)
                vital_df.at[idx, "dbp"] -= rng.normal(8, 4)

            elif event_type == "sepsis_like_deterioration":
                vital_df.at[idx, "heart_rate"] += rng.normal(14, 6)
                vital_df.at[idx, "resp_rate"] += rng.normal(4, 2)
                vital_df.at[idx, "temperature"] += rng.normal(2.2, 1.0)
                vital_df.at[idx, "sbp"] -= rng.normal(10, 6)

            elif event_type == "metabolic_instability":
                vital_df.at[idx, "heart_rate"] += rng.normal(8, 5)
                vital_df.at[idx, "resp_rate"] += rng.normal(5, 3)

            elif event_type == "arrhythmia_instability":
                vital_df.at[idx, "heart_rate"] += rng.normal(18, 10)
                vital_df.at[idx, "sbp"] -= rng.normal(8, 6)

            elif event_type == "renal_lab_instability":
                vital_df.at[idx, "sbp"] += rng.normal(8, 8)

            elif event_type == "post_procedure_complication":
                vital_df.at[idx, "heart_rate"] += rng.normal(10, 5)
                vital_df.at[idx, "sbp"] -= rng.normal(10, 6)

        # =====================================================
        # Multi-wave deterioration pressure
        # =====================================================

        if event_wave_count >= 2 or multi_crash:
            vital_df.at[idx, "heart_rate"] += rng.normal(8 + event_wave_count * 3, 5)
            vital_df.at[idx, "resp_rate"] += rng.normal(3 + event_wave_count, 2)
            vital_df.at[idx, "spo2"] -= rng.normal(2 + event_wave_count, 1.5)
            vital_df.at[idx, "sbp"] -= rng.normal(6 + event_wave_count * 2, 5)
            vital_df.at[idx, "dbp"] -= rng.normal(3 + event_wave_count, 3)

        # =====================================================
        # Cascade-chain physiology
        # =====================================================

        if "respiratory_decline" in cascade_chain:
            vital_df.at[idx, "resp_rate"] += rng.normal(3, 2)
            vital_df.at[idx, "spo2"] -= rng.normal(3, 1.5)

        if "hemodynamic_instability" in cascade_chain:
            vital_df.at[idx, "heart_rate"] += rng.normal(6, 3)
            vital_df.at[idx, "sbp"] -= rng.normal(8, 4)

        if "renal_lab_instability" in cascade_chain:
            vital_df.at[idx, "sbp"] += rng.normal(4, 4)

        if "arrhythmia_instability" in cascade_chain:
            vital_df.at[idx, "heart_rate"] += rng.normal(8, 6)

        if "metabolic_instability" in cascade_chain:
            vital_df.at[idx, "resp_rate"] += rng.normal(2, 2)
            vital_df.at[idx, "heart_rate"] += rng.normal(3, 2)

        # =====================================================
        # Hidden / masked physiology
        # =====================================================

        if (
            state in [
                "PRE_EVENT_HIDDEN_INSTABILITY",
                "PRE_EVENT_FALSE_REASSURANCE",
                "PRE_EVENT_WORSENING_WATCH",
            ]
            or false_reassurance >= 0.45
            or visibility <= 0.65
        ):
            vital_df.at[idx, "heart_rate"] += rng.normal(6, 4)
            vital_df.at[idx, "resp_rate"] += rng.normal(2.5, 2)
            vital_df.at[idx, "spo2"] -= rng.normal(1.5, 1)
            vital_df.at[idx, "sbp"] -= rng.normal(4, 4)

            if rng.random() < 0.65:
                vital_df.at[idx, "heart_rate"] = np.clip(vital_df.at[idx, "heart_rate"], 82, 112)
                vital_df.at[idx, "resp_rate"] = np.clip(vital_df.at[idx, "resp_rate"], 18, 24)
                vital_df.at[idx, "spo2"] = np.clip(vital_df.at[idx, "spo2"], 92, 97)
                vital_df.at[idx, "sbp"] = np.clip(vital_df.at[idx, "sbp"], 96, 128)
                vital_df.at[idx, "pulse_mask_hint"] = 1

        # =====================================================
        # False recovery / de-escalation masking
        # =====================================================

        if state in ["FALSE_RECOVERY", "DE_ESCALATION_MONITOR", "REINTEGRATION"]:

            vital_df.at[idx, "heart_rate"] -= rng.normal(4, 3)
            vital_df.at[idx, "resp_rate"] -= rng.normal(2, 2)
            vital_df.at[idx, "spo2"] += rng.normal(1, 1)
            vital_df.at[idx, "sbp"] += rng.normal(4, 4)

            if false_reassurance >= 0.35 or volatility >= 0.30:
                vital_df.at[idx, "pulse_mask_hint"] = 1
                vital_df.at[idx, "heart_rate"] = np.clip(vital_df.at[idx, "heart_rate"], 78, 108)
                vital_df.at[idx, "resp_rate"] = np.clip(vital_df.at[idx, "resp_rate"], 17, 24)
                vital_df.at[idx, "spo2"] = np.clip(vital_df.at[idx, "spo2"], 93, 97)
                vital_df.at[idx, "sbp"] = np.clip(vital_df.at[idx, "sbp"], 94, 122)

        # =====================================================
        # Volatile / recurrent instability
        # =====================================================

        if state in [
            "POST_EVENT_VOLATILE_MONITOR",
            "RE_ESCALATION",
            "CRITICAL_POST_EVENT",
            "RECURRENT_INSTABILITY",
        ]:
            vital_df.at[idx, "heart_rate"] += rng.normal(14, 8)
            vital_df.at[idx, "resp_rate"] += rng.normal(5, 3)
            vital_df.at[idx, "spo2"] -= rng.normal(4, 2)
            vital_df.at[idx, "sbp"] -= rng.normal(14, 8)
            vital_df.at[idx, "dbp"] -= rng.normal(6, 4)

        # =====================================================
        # Terminal decline physiology
        # =====================================================

        if state == "TERMINAL_DECLINE_PHASE" or terminal_pressure >= 0.65:
            vital_df.at[idx, "heart_rate"] += rng.normal(10, 8)
            vital_df.at[idx, "resp_rate"] += rng.normal(5, 4)
            vital_df.at[idx, "spo2"] -= rng.normal(5, 3)
            vital_df.at[idx, "sbp"] -= rng.normal(14, 8)
            vital_df.at[idx, "dbp"] -= rng.normal(6, 4)
            vital_df.at[idx, "temperature"] += rng.normal(0.8, 0.8)

        # =====================================================
        # Failed reintegration physiology
        # =====================================================

        if state == "FAILED_REINTEGRATION" or failed_reintegration:
            vital_df.at[idx, "heart_rate"] += rng.normal(7, 4)
            vital_df.at[idx, "resp_rate"] += rng.normal(3, 2)
            vital_df.at[idx, "spo2"] -= rng.normal(2, 1)
            vital_df.at[idx, "sbp"] -= rng.normal(6, 4)

        # =====================================================
        # Recognition-delay / warning-window physiology
        # =====================================================

        if recognition_delay >= 24 or warning_window >= 24:
            vital_df.at[idx, "heart_rate"] += rng.normal(3, 2)
            vital_df.at[idx, "resp_rate"] += rng.normal(1.5, 1)
            vital_df.at[idx, "spo2"] -= rng.normal(1.0, 0.8)

        # =====================================================
        # Deceptive stability masking
        # =====================================================

        if deceptive_stability >= 0.45:
            vital_df.at[idx, "heart_rate"] = np.clip(vital_df.at[idx, "heart_rate"], 78, 112)
            vital_df.at[idx, "resp_rate"] = np.clip(vital_df.at[idx, "resp_rate"], 17, 25)
            vital_df.at[idx, "spo2"] = np.clip(vital_df.at[idx, "spo2"], 92, 97)
            vital_df.at[idx, "sbp"] = np.clip(vital_df.at[idx, "sbp"], 92, 130)
            vital_df.at[idx, "pulse_mask_hint"] = 1

        # Controlled measurement noise
        vital_df.at[idx, "heart_rate"] += rng.normal(0, 3)
        vital_df.at[idx, "resp_rate"] += rng.normal(0, 1.5)
        vital_df.at[idx, "spo2"] += rng.normal(0, 0.8)
        vital_df.at[idx, "temperature"] += rng.normal(0, 0.3)
        vital_df.at[idx, "sbp"] += rng.normal(0, 4)
        vital_df.at[idx, "dbp"] += rng.normal(0, 3)

    vital_df["heart_rate"] = vital_df["heart_rate"].clip(38, 175).round(1)
    vital_df["resp_rate"] = vital_df["resp_rate"].clip(7, 42).round(1)
    vital_df["spo2"] = vital_df["spo2"].clip(76, 100).round(1)
    vital_df["temperature"] = vital_df["temperature"].clip(94, 106).round(1)
    vital_df["sbp"] = vital_df["sbp"].clip(65, 205).round(1)
    vital_df["dbp"] = vital_df["dbp"].clip(35, 125).round(1)

    too_narrow = (vital_df["sbp"] - vital_df["dbp"]) < 12

    vital_df.loc[too_narrow, "dbp"] = (
        vital_df.loc[too_narrow, "sbp"]
        - rng.normal(22, 6, too_narrow.sum())
    ).clip(35, 125)

    vital_df["dbp"] = vital_df["dbp"].round(1)
    vital_df["pulse_mask_hint"] = vital_df["pulse_mask_hint"].fillna(0).astype(int)

    return vital_df

# =========================================================
# Basic Derived Vital Signals
# =========================================================

def add_basic_derived_vital_signals(df):
    """
    Add basic derived vital signals.

    Expected columns:
    - heart_rate
    - resp_rate
    - spo2
    - temperature
    - sbp
    - dbp

    Temperature is Fahrenheit.
    """

    df = df.copy()

    df["shock_index"] = _safe_divide(
        df["heart_rate"],
        df["sbp"],
    )

    df["pulse_pressure"] = df["sbp"] - df["dbp"]

    df["map_estimate"] = (
        df["sbp"] + (2 * df["dbp"])
    ) / 3

    df["hr_rr_ratio"] = _safe_divide(
        df["heart_rate"],
        df["resp_rate"],
    )

    df["resp_spo2_stress"] = _safe_divide(
        df["resp_rate"],
        df["spo2"],
    )

    df["cardiorespiratory_stress"] = _safe_divide(
        df["heart_rate"] * df["resp_rate"],
        df["spo2"],
    )

    df["pressure_rate_product"] = (
        df["heart_rate"] * df["sbp"]
    )

    df["temperature_deviation"] = (
        df["temperature"] - 98.6
    ).abs()

    df["fever_stress"] = np.maximum(
        df["temperature"] - 100.4,
        0,
    )

    df["hypothermia_stress"] = np.maximum(
        95.0 - df["temperature"],
        0,
    )

    df["oxygen_gap"] = 100 - df["spo2"]

    df["bp_narrowing_index"] = _safe_divide(
        df["pulse_pressure"],
        df["sbp"],
    )

    df["diastolic_pressure_ratio"] = _safe_divide(
        df["dbp"],
        df["sbp"],
    )

    df["hemodynamic_ratio"] = _safe_divide(
        df["heart_rate"],
        df["map_estimate"],
    )

    df["respiratory_load_index"] = (
        df["resp_rate"] * df["oxygen_gap"]
    )

    df["oxygenation_pressure_stress"] = _safe_divide(
        df["oxygen_gap"],
        df["sbp"],
    )

    df["vital_instability_count"] = _count_abnormal_vitals(df)

    df["critical_vital_count"] = _count_critical_vitals(df)

    df["vital_burden_score"] = (
        df["vital_instability_count"]
        + (2 * df["critical_vital_count"])
    )

    return df


# =========================================================
# Hidden Physiologic Signals
# =========================================================

def add_hidden_physiologic_signals(df):
    """
    Add hidden physiologic instability signals.

    These are designed to detect situations where simple vitals
    may still look acceptable, but relationships between vitals
    suggest hidden deterioration.
    """

    df = df.copy()

    if "shock_index" not in df.columns:
        df = add_basic_derived_vital_signals(df)

    df["shock_resp_combo"] = (
        df["shock_index"] * df["resp_rate"]
    )

    df["shock_oxygen_combo"] = (
        df["shock_index"] * df["oxygen_gap"]
    )

    df["shock_temp_combo"] = (
        df["shock_index"] * df["temperature_deviation"]
    )

    df["cardio_pressure_stress"] = (
        df["shock_index"] * df["pressure_rate_product"]
    )

    df["resp_oxygen_temp_stress"] = (
        df["resp_spo2_stress"] * df["temperature_deviation"]
    )

    df["hemodynamic_oxygen_stress"] = (
        df["hemodynamic_ratio"] * df["oxygen_gap"]
    )

    df["narrow_pressure_shock"] = _safe_divide(
        df["shock_index"],
        df["pulse_pressure"],
    )

    df["map_oxygen_stress"] = _safe_divide(
        df["oxygen_gap"],
        df["map_estimate"],
    )

    df["cardiorespiratory_hemodynamic_stress"] = (
        df["cardiorespiratory_stress"] * df["shock_index"]
    )

    df["hidden_instability_score"] = (
        df["shock_index"].fillna(0)
        + df["oxygen_gap"].fillna(0) / 10
        + df["resp_rate"].fillna(0) / 30
        + _safe_divide(40, df["pulse_pressure"]).fillna(0)
        + df["temperature_deviation"].fillna(0) / 5
    )

    df["compensated_shock_signal"] = (
        (df["shock_index"] >= 0.90)
        & (df["sbp"] >= 90)
    ).astype(int)

    df["silent_respiratory_strain"] = (
        (df["resp_rate"] >= 22)
        & (df["spo2"] >= 92)
    ).astype(int)

    df["masked_instability_signal"] = (
        (df["vital_instability_count"] <= 1)
        & (
            (df["shock_index"] >= 0.85)
            | (df["resp_spo2_stress"] >= 0.24)
            | (df["hemodynamic_ratio"] >= 1.15)
            | (df["pulse_pressure"] <= 38)
        )
    ).astype(int)

    df["physiologic_discordance_score"] = (
        ((df["heart_rate"] >= 100) & (df["sbp"] >= 100)).astype(int)
        + ((df["resp_rate"] >= 22) & (df["spo2"] >= 92)).astype(int)
        + ((df["pulse_pressure"] <= 38) & (df["sbp"] >= 90)).astype(int)
        + ((df["oxygen_gap"] >= 5) & (df["resp_rate"] < 24)).astype(int)
        + ((df["shock_index"] >= 0.85) & (df["vital_instability_count"] <= 1)).astype(int)
    )

    df["compensation_reserve_score"] = _clip(
        1
        - (
            df["shock_index"].fillna(0) * 0.25
            + df["oxygen_gap"].fillna(0) / 20 * 0.20
            + df["resp_rate"].fillna(0) / 40 * 0.20
            + _safe_divide(40, df["pulse_pressure"]).fillna(0) * 0.20
            + df["vital_burden_score"].fillna(0) / 8 * 0.15
        )
    )

    df["small_signal_instability_score"] = _clip(
        (
            (df["shock_index"] - 0.75).clip(lower=0) * 0.35
            + ((40 - df["pulse_pressure"]).clip(lower=0) / 20) * 0.25
            + ((df["resp_rate"] - 18).clip(lower=0) / 12) * 0.20
            + ((96 - df["spo2"]).clip(lower=0) / 8) * 0.20
        )
    )

    df["normal_range_deception_score"] = _clip(
        (
            (df["vital_instability_count"] == 0).astype(int) * 0.30
            + (df["shock_index"] >= 0.85).astype(int) * 0.20
            + (df["pulse_pressure"] <= 38).astype(int) * 0.15
            + (df["resp_rate"].between(20, 24)).astype(int) * 0.15
            + (df["spo2"].between(92, 95)).astype(int) * 0.10
            + (df["oxygen_gap"] >= 5).astype(int) * 0.10
        )
    )

    df["silent_collapse_pressure"] = _clip(
        df["hidden_instability_score"].fillna(0) / 6 * 0.30
        + df["small_signal_instability_score"].fillna(0) * 0.25
        + df["normal_range_deception_score"].fillna(0) * 0.20
        + df["physiologic_discordance_score"].fillna(0) / 5 * 0.15
        + (1 - df["compensation_reserve_score"].fillna(1)) * 0.10
    )

    df["pre_threshold_deterioration_flag"] = (
        (df["vital_instability_count"] <= 1)
        & (
            (df["small_signal_instability_score"] >= 0.35)
            | (df["normal_range_deception_score"] >= 0.45)
            | (df["silent_collapse_pressure"] >= 0.45)
        )
    ).astype(int)

    df["physiologic_deception_flag"] = (
        (df["vital_instability_count"] <= 1)
        & (
            (df["silent_collapse_pressure"] >= 0.40)
            | (df["normal_range_deception_score"] >= 0.45)
            | (df["physiologic_discordance_score"] >= 2)
            | (df["compensated_shock_signal"] == 1)
        )
    ).astype(int)

    df["compensation_failure_risk"] = _clip(
        (1 - df["compensation_reserve_score"].fillna(1)) * 0.35
        + df["silent_collapse_pressure"].fillna(0) * 0.30
        + df["small_signal_instability_score"].fillna(0) * 0.20
        + df["physiologic_discordance_score"].fillna(0) / 5 * 0.15
    )

    conditions = [
        (
            (df["vital_instability_count"] == 0)
            & (df["shock_index"] >= 0.90)
        ),
        (
            (df["vital_instability_count"] == 0)
            & (df["pulse_pressure"] <= 35)
        ),
        (
            (df["spo2"] >= 92)
            & (df["resp_rate"] >= 22)
        ),
        (
            (df["sbp"] >= 90)
            & (df["shock_index"] >= 0.90)
        ),
        (
            (df["vital_instability_count"] <= 1)
            & (df["normal_range_deception_score"] >= 0.55)
        ),
        (
            (df["vital_instability_count"] <= 1)
            & (df["silent_collapse_pressure"] >= 0.50)
        ),
    ]

    choices = [
        "normal_vitals_high_shock_index",
        "normal_vitals_narrowing_pulse_pressure",
        "normal_spo2_high_respiratory_work",
        "compensated_hemodynamic_instability",
        "normal_range_deception",
        "silent_collapse_pressure",
    ]

    df["masked_deterioration_type"] = np.select(
        conditions,
        choices,
        default="none",
    )

    df["vital_masking_pattern"] = np.select(
        [
            df["masked_deterioration_type"] != "none",
            df["physiologic_deception_flag"] == 1,
            df["pre_threshold_deterioration_flag"] == 1,
        ],
        [
            "EXPLICIT_MASKED_DETERIORATION",
            "PHYSIOLOGIC_DECEPTION",
            "PRE_THRESHOLD_DETERIORATION",
        ],
        default="NO_MASKING_PATTERN",
    )

    return df


# =========================================================
# Temporal Vital Trajectory Intelligence
# =========================================================

def add_temporal_hidden_vital_signals(
    df,
    patient_col="patient_id",
    time_col="timestamp",
):
    """
    Add temporal hidden vital trajectory signals.

    These features describe:
    - deterioration momentum
    - recovery momentum
    - physiologic volatility
    - oscillatory instability
    - false recovery
    - exhaustion
    - re-escalation pressure
    """

    df = df.copy()

    if patient_col not in df.columns or time_col not in df.columns:
        return df

    df = df.sort_values(
        [patient_col, time_col]
    ).reset_index(drop=True)

    if "hidden_instability_score" not in df.columns:
        df = add_hidden_physiologic_signals(df)

    grouped = df.groupby(patient_col)

    tracked_cols = [
        "hidden_instability_score",
        "shock_index",
        "oxygen_gap",
        "resp_rate",
        "pulse_pressure",
        "map_estimate",
        "silent_collapse_pressure",
        "compensation_reserve_score",
        "vital_burden_score",
        "small_signal_instability_score",
        "normal_range_deception_score",
    ]

    for col in tracked_cols:
        if col in df.columns:
            df[f"{col}_velocity"] = grouped[col].diff()
            df[f"{col}_acceleration"] = grouped[f"{col}_velocity"].diff()

            df[f"{col}_rolling_mean_6"] = grouped[col].transform(
                lambda s: s.shift(1).rolling(
                    6,
                    min_periods=2,
                ).mean()
            )

            df[f"{col}_rolling_std_6"] = grouped[col].transform(
                lambda s: s.shift(1).rolling(
                    6,
                    min_periods=2,
                ).std()
            )

    df["deterioration_momentum"] = _clip(
        df["hidden_instability_score_velocity"].fillna(0) * 0.30
        + df["shock_index_velocity"].fillna(0) * 0.20
        + df["oxygen_gap_velocity"].fillna(0) / 5 * 0.20
        + df["resp_rate_velocity"].fillna(0) / 10 * 0.15
        - df["compensation_reserve_score_velocity"].fillna(0) * 0.15
    )

    df["recovery_momentum"] = _clip(
        -df["hidden_instability_score_velocity"].fillna(0) * 0.30
        -df["shock_index_velocity"].fillna(0) * 0.18
        -df["oxygen_gap_velocity"].fillna(0) / 5 * 0.18
        -df["resp_rate_velocity"].fillna(0) / 10 * 0.14
        + df["compensation_reserve_score_velocity"].fillna(0) * 0.20
    )

    df["physiologic_volatility_burden"] = _clip(
        df["hidden_instability_score_rolling_std_6"].fillna(0) / 2 * 0.25
        + df["shock_index_rolling_std_6"].fillna(0) * 0.20
        + df["oxygen_gap_rolling_std_6"].fillna(0) / 5 * 0.20
        + df["resp_rate_rolling_std_6"].fillna(0) / 5 * 0.20
        + df["small_signal_instability_score_rolling_std_6"].fillna(0) * 0.15
    )

    df["oscillatory_instability_score"] = grouped[
        "hidden_instability_score_velocity"
    ].transform(
        lambda s: (
            np.sign(s)
            .diff()
            .abs()
            .fillna(0)
            .rolling(6, min_periods=2)
            .sum()
            / 6
        )
    )

    df["trajectory_stress_score"] = _clip(
        df["hidden_instability_score"].fillna(0) / 6 * 0.25
        + df["deterioration_momentum"].fillna(0) * 0.20
        + df["silent_collapse_pressure"].fillna(0) * 0.25
        + df["physiologic_volatility_burden"].fillna(0) * 0.15
        + df["normal_range_deception_score"].fillna(0) * 0.15
    )

    df["recovery_instability_score"] = _clip(
        df["recovery_momentum"].fillna(0) * 0.20
        + df["silent_collapse_pressure"].fillna(0) * 0.25
        + df["physiologic_volatility_burden"].fillna(0) * 0.20
        + (1 - df["compensation_reserve_score"].fillna(1)) * 0.20
        + df["normal_range_deception_score"].fillna(0) * 0.15
    )

    if "post_event_volatility_score" in df.columns:
        lifecycle_volatility = df["post_event_volatility_score"].fillna(0)
    else:
        lifecycle_volatility = 0

    df["re_escalation_pressure_score"] = _clip(
        df["deterioration_momentum"].fillna(0) * 0.25
        + lifecycle_volatility * 0.20
        + df["silent_collapse_pressure"].fillna(0) * 0.25
        + df["physiologic_volatility_burden"].fillna(0) * 0.15
        + df["compensation_failure_risk"].fillna(0) * 0.15
    )

    df["false_vital_recovery_signal"] = (
        (df["recovery_momentum"] >= 0.18)
        & (
            (df["silent_collapse_pressure"] >= 0.35)
            | (df["physiologic_volatility_burden"] >= 0.28)
            | (df["normal_range_deception_score"] >= 0.45)
            | (
                df["hidden_instability_score"]
                >= df["hidden_instability_score_rolling_mean_6"].fillna(
                    df["hidden_instability_score"]
                )
            )
        )
    ).astype(int)

    df["physiologic_exhaustion_signal"] = (
        (df["compensation_reserve_score"] <= 0.30)
        & (
            (df["deterioration_momentum"] >= 0.18)
            | (df["silent_collapse_pressure"] >= 0.48)
            | (df["vital_burden_score"] >= 3)
            | (df["compensation_failure_risk"] >= 0.50)
        )
    ).astype(int)

    df["masked_reescalation_warning"] = (
        (df["vital_instability_count"] <= 1)
        & (
            (df["re_escalation_pressure_score"] >= 0.45)
            | (df["false_vital_recovery_signal"] == 1)
            | (df["physiologic_exhaustion_signal"] == 1)
        )
    ).astype(int)

    return df




# =========================================================
# Outcomes-Aligned HVI Pressure Layer
# =========================================================

def add_outcomes_aligned_vital_signals(df):
    """
    Add outcome-facing hidden vital intelligence signals.

    This layer intentionally aligns vitals.py with outcomes.py by producing:
    - hidden_instability_score        0-1 normalized HVI pressure
    - masked_deterioration_flag      bool/int hidden deterioration marker
    - deterioration_tendency         0-1 forward deterioration pressure
    - recovery_resilience            0-1 recovery durability/reserve signal

    It also preserves the original raw hidden score as:
    - raw_hidden_instability_score
    """

    df = df.copy()

    if "hidden_instability_score" not in df.columns:
        df = add_hidden_physiologic_signals(df)

    raw_hidden = df["hidden_instability_score"].fillna(0)

    if "raw_hidden_instability_score" not in df.columns:
        df["raw_hidden_instability_score"] = raw_hidden

    # Original hidden score was intentionally expressive, but outcomes.py expects
    # pressure-style 0-1 behavior around thresholds like 0.45 and 0.50.
    raw_hidden_norm = _clip(raw_hidden / 6.0, 0, 1)

    silent_collapse = df.get("silent_collapse_pressure", pd.Series(0, index=df.index)).fillna(0)
    normal_deception = df.get("normal_range_deception_score", pd.Series(0, index=df.index)).fillna(0)
    compensation_failure = df.get("compensation_failure_risk", pd.Series(0, index=df.index)).fillna(0)
    small_signal = df.get("small_signal_instability_score", pd.Series(0, index=df.index)).fillna(0)
    physiologic_discordance = df.get("physiologic_discordance_score", pd.Series(0, index=df.index)).fillna(0)
    vital_burden = df.get("vital_burden_score", pd.Series(0, index=df.index)).fillna(0)
    pulse_mask_hint = df.get("pulse_mask_hint", pd.Series(0, index=df.index)).fillna(0)

    trajectory_stress = df.get("trajectory_stress_score", pd.Series(0, index=df.index)).fillna(0)
    deterioration_momentum = df.get("deterioration_momentum", pd.Series(0, index=df.index)).fillna(0)
    recovery_momentum = df.get("recovery_momentum", pd.Series(0, index=df.index)).fillna(0)
    volatility = df.get("physiologic_volatility_burden", pd.Series(0, index=df.index)).fillna(0)
    reescalation_pressure = df.get("re_escalation_pressure_score", pd.Series(0, index=df.index)).fillna(0)

    false_vital_recovery = df.get("false_vital_recovery_signal", pd.Series(0, index=df.index)).fillna(0)
    exhaustion = df.get("physiologic_exhaustion_signal", pd.Series(0, index=df.index)).fillna(0)
    masked_reescalation = df.get("masked_reescalation_warning", pd.Series(0, index=df.index)).fillna(0)

    compensated_shock = df.get("compensated_shock_signal", pd.Series(0, index=df.index)).fillna(0)
    silent_resp = df.get("silent_respiratory_strain", pd.Series(0, index=df.index)).fillna(0)
    masked_instability = df.get("masked_instability_signal", pd.Series(0, index=df.index)).fillna(0)
    pre_threshold = df.get("pre_threshold_deterioration_flag", pd.Series(0, index=df.index)).fillna(0)
    physiologic_deception = df.get("physiologic_deception_flag", pd.Series(0, index=df.index)).fillna(0)

    df["hidden_instability_score"] = _clip(
        raw_hidden_norm * 0.24
        + silent_collapse * 0.22
        + normal_deception * 0.14
        + compensation_failure * 0.14
        + small_signal * 0.10
        + (physiologic_discordance / 5.0) * 0.08
        + compensated_shock * 0.03
        + silent_resp * 0.02
        + pulse_mask_hint * 0.03,
        0,
        1,
    )

    df["vital_hidden_signal_burden"] = _clip(
        compensated_shock * 0.18
        + silent_resp * 0.14
        + masked_instability * 0.18
        + pre_threshold * 0.16
        + physiologic_deception * 0.18
        + false_vital_recovery * 0.10
        + masked_reescalation * 0.06,
        0,
        1,
    )

    df["hvi_signal_strength_score"] = _clip(
        df["hidden_instability_score"].fillna(0) * 0.34
        + df["vital_hidden_signal_burden"].fillna(0) * 0.24
        + trajectory_stress * 0.16
        + reescalation_pressure * 0.12
        + volatility * 0.08
        + exhaustion * 0.06,
        0,
        1,
    )

    df["masked_deterioration_flag"] = (
        (df.get("masked_deterioration_type", pd.Series("none", index=df.index)).astype(str) != "none")
        | (masked_instability >= 1)
        | (physiologic_deception >= 1)
        | (pre_threshold >= 1)
        | (pulse_mask_hint >= 1)
        | (df["hvi_signal_strength_score"] >= 0.48)
        | ((df["hidden_instability_score"] >= 0.45) & (vital_burden <= 1))
    ).astype(bool)

    df["vital_false_reassurance_pressure"] = _clip(
        normal_deception * 0.22
        + pulse_mask_hint * 0.16
        + false_vital_recovery * 0.18
        + df["masked_deterioration_flag"].astype(int) * 0.16
        + df["hidden_instability_score"] * 0.14
        + (vital_burden <= 1).astype(int) * 0.08
        + masked_reescalation * 0.06,
        0,
        1,
    )

    prior_deterioration = df.get("deterioration_tendency", pd.Series(0, index=df.index)).fillna(0)
    has_event = df.get("has_event", pd.Series(False, index=df.index)).fillna(False).astype(int)
    terminal_pressure = df.get("terminal_decline_pressure", pd.Series(0, index=df.index)).fillna(0)
    failed_reintegration = df.get("failed_reintegration_risk_flag", pd.Series(False, index=df.index)).fillna(False).astype(int)

    df["deterioration_tendency"] = _clip(
        prior_deterioration * 0.12
        + df["hidden_instability_score"] * 0.22
        + trajectory_stress * 0.18
        + deterioration_momentum * 0.16
        + compensation_failure * 0.12
        + reescalation_pressure * 0.10
        + (vital_burden / 8.0) * 0.06
        + terminal_pressure * 0.03
        + failed_reintegration * 0.01,
        0,
        1,
    )

    prior_resilience = df.get("recovery_resilience", pd.Series(0.5, index=df.index)).fillna(0.5)
    compensation_reserve = df.get("compensation_reserve_score", pd.Series(0.5, index=df.index)).fillna(0.5)

    df["recovery_resilience"] = _clip(
        prior_resilience * 0.18
        + compensation_reserve * 0.28
        + recovery_momentum * 0.18
        - df["hidden_instability_score"] * 0.16
        - df["vital_false_reassurance_pressure"] * 0.12
        - volatility * 0.08
        - exhaustion * 0.06
        - df["masked_deterioration_flag"].astype(int) * 0.04,
        0,
        1,
    )

    df["vital_recovery_authenticity_warning"] = (
        (df["recovery_resilience"] >= 0.35)
        & (
            (df["hidden_instability_score"] >= 0.45)
            | (df["vital_false_reassurance_pressure"] >= 0.45)
            | (false_vital_recovery >= 1)
            | (masked_reescalation >= 1)
        )
    ).astype(bool)

    df["hvi_outcomes_alignment_score"] = _clip(
        df["hidden_instability_score"] * 0.28
        + df["deterioration_tendency"] * 0.22
        + df["vital_false_reassurance_pressure"] * 0.18
        + df["masked_deterioration_flag"].astype(int) * 0.14
        + (1 - df["recovery_resilience"]) * 0.12
        + df["vital_recovery_authenticity_warning"].astype(int) * 0.06,
        0,
        1,
    )

    return df


# =========================================================
# Full Derived Vital Signal Pipeline
# =========================================================

def add_all_derived_vital_signals(
    df,
    patient_col="patient_id",
    time_col="timestamp",
    include_temporal=True,
):
    """
    Add all derived, hidden, masking, and temporal vital signals.
    """

    df = add_basic_derived_vital_signals(df)

    df = add_hidden_physiologic_signals(df)

    # Normalize and align HVI pressure before temporal features so velocity and
    # rolling signals use outcome-style 0-1 hidden instability pressure.
    df = add_outcomes_aligned_vital_signals(df)

    if include_temporal:
        df = add_temporal_hidden_vital_signals(
            df,
            patient_col=patient_col,
            time_col=time_col,
        )

        # Reconcile temporal signals back into the outcome-facing HVI layer.
        df = add_outcomes_aligned_vital_signals(df)

    return df


def generate_vitals_with_hidden_signals(
    df,
    patient_col="patient_id",
    time_col="encounter_start",
    random_seed=42,
    include_temporal=True,
):
    """
    Generate raw lifecycle-aware vitals and add all hidden vital signals.
    """

    vital_df = generate_raw_vitals_from_lifecycle(
        df=df,
        random_seed=random_seed,
    )

    vital_df = add_all_derived_vital_signals(
        df=vital_df,
        patient_col=patient_col,
        time_col=time_col,
        include_temporal=include_temporal,
    )

    return vital_df