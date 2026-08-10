"""
BIRE OS Synthetic Numeric Labs Generator

Chapter 53 doctrine:
No more babying BIRE OS.

Purpose:
Generate numeric laboratory values that challenge BIRE OS with:
- abnormal labs
- delayed abnormality
- deceptive normal labs
- lab/vital contradiction
- critical pending labs
- treatment-masked labs
- organ stress
- lab trajectory pressure
- false recovery chemistry
- hidden deterioration chemistry
- multi-organ instability

This module is for synthetic stress testing only.
It is not clinically validated and must not be used for patient care.

Doctrine:
We Detect What Others Miss.
"""

import numpy as np
import pandas as pd

from bire.synthetic.config import SYNTHETIC_ECOSYSTEM_CONFIG


LAB_COLUMNS = [
    "glucose",
    "lactate",
    "wbc",
    "creatinine",
    "bun",
    "potassium",
    "sodium",
    "bicarbonate",
    "ph",
    "pco2",
    "troponin",
    "hemoglobin",
    "platelets",
    "ast",
    "alt",
    "bilirubin",
    "lipase",
]


def _clip(value, low, high):
    return float(np.clip(value, low, high))


def _normal_lab_baseline(rng):
    return {
        "glucose": rng.normal(105, 18),
        "lactate": rng.normal(1.3, 0.4),
        "wbc": rng.normal(7.5, 2.0),
        "creatinine": rng.normal(1.0, 0.25),
        "bun": rng.normal(16, 5),
        "potassium": rng.normal(4.1, 0.35),
        "sodium": rng.normal(138, 3),
        "bicarbonate": rng.normal(24, 3),
        "ph": rng.normal(7.40, 0.04),
        "pco2": rng.normal(40, 5),
        "troponin": abs(rng.normal(0.01, 0.015)),
        "hemoglobin": rng.normal(13.2, 1.5),
        "platelets": rng.normal(250, 60),
        "ast": rng.normal(28, 12),
        "alt": rng.normal(30, 14),
        "bilirubin": rng.normal(0.8, 0.35),
        "lipase": rng.normal(45, 25),
    }


def _apply_lab_profile_effects(labs, lab_profile, rng):
    labs = labs.copy()

    if lab_profile in ["inflammatory_response", "infection_pattern"]:
        labs["wbc"] += rng.normal(5, 2)
        labs["lactate"] += rng.normal(0.8, 0.4)

    elif lab_profile == "sepsis_like_pattern":
        labs["wbc"] += rng.normal(8, 3)
        labs["lactate"] += rng.normal(1.8, 0.7)
        labs["platelets"] -= rng.normal(40, 25)
        labs["bicarbonate"] -= rng.normal(2.5, 1.2)
        labs["ph"] -= rng.normal(0.035, 0.02)

    elif lab_profile in ["renal_instability", "acute_kidney_injury_pattern"]:
        labs["creatinine"] += rng.normal(1.4, 0.6)
        labs["bun"] += rng.normal(28, 10)
        labs["potassium"] += rng.normal(0.55, 0.35)
        labs["bicarbonate"] -= rng.normal(3.5, 1.5)

    elif lab_profile == "chronic_kidney_disease_pattern":
        labs["creatinine"] += rng.normal(1.1, 0.4)
        labs["bun"] += rng.normal(24, 8)
        labs["hemoglobin"] -= rng.normal(1.2, 0.5)
        labs["potassium"] += rng.normal(0.3, 0.2)

    elif lab_profile in ["metabolic_instability", "hyperglycemia_pattern"]:
        labs["glucose"] += rng.normal(100, 45)
        labs["bicarbonate"] -= rng.normal(5, 2)
        labs["ph"] -= rng.normal(0.055, 0.03)
        labs["potassium"] += rng.normal(0.35, 0.25)

    elif lab_profile == "hypoglycemia_pattern":
        labs["glucose"] -= rng.normal(55, 18)

    elif lab_profile in ["electrolyte_derangement", "hyperkalemia_pattern"]:
        labs["potassium"] += rng.normal(1.2, 0.4)

    elif lab_profile == "hypokalemia_pattern":
        labs["potassium"] -= rng.normal(1.0, 0.3)

    elif lab_profile == "hyponatremia_pattern":
        labs["sodium"] -= rng.normal(10, 4)

    elif lab_profile == "anion_gap_metabolic_acidosis":
        labs["bicarbonate"] -= rng.normal(8, 3)
        labs["ph"] -= rng.normal(0.08, 0.03)
        labs["lactate"] += rng.normal(1.2, 0.6)

    elif lab_profile == "respiratory_acidosis_pattern":
        labs["pco2"] += rng.normal(18, 7)
        labs["ph"] -= rng.normal(0.075, 0.03)
        labs["bicarbonate"] += rng.normal(2, 1.5)

    elif lab_profile == "respiratory_alkalosis_pattern":
        labs["pco2"] -= rng.normal(10, 4)
        labs["ph"] += rng.normal(0.045, 0.02)

    elif lab_profile == "lactic_acidosis_pattern":
        labs["lactate"] += rng.normal(3.0, 1.0)
        labs["bicarbonate"] -= rng.normal(5, 2)
        labs["ph"] -= rng.normal(0.06, 0.03)

    elif lab_profile in ["cardiac_marker_elevation", "troponin_leak_pattern"]:
        labs["troponin"] += abs(rng.normal(0.35, 0.35))

    elif lab_profile == "heart_failure_congestion_pattern":
        labs["sodium"] -= rng.normal(3, 1.5)
        labs["creatinine"] += rng.normal(0.35, 0.2)
        labs["bun"] += rng.normal(10, 4)
        labs["troponin"] += abs(rng.normal(0.05, 0.05))

    elif lab_profile == "liver_injury_pattern":
        labs["ast"] += rng.normal(180, 90)
        labs["alt"] += rng.normal(160, 80)
        labs["bilirubin"] += rng.normal(2.0, 1.0)

    elif lab_profile == "pancreatitis_pattern":
        labs["lipase"] += rng.normal(1200, 500)
        labs["wbc"] += rng.normal(4, 2)
        labs["glucose"] += rng.normal(45, 25)

    elif lab_profile == "coagulation_abnormality":
        labs["platelets"] -= rng.normal(70, 35)
        labs["hemoglobin"] -= rng.normal(0.8, 0.5)

    elif lab_profile == "bleeding_anemia_pattern":
        labs["hemoglobin"] -= rng.normal(3.5, 1.0)
        labs["bun"] += rng.normal(8, 4)

    elif lab_profile == "thrombocytopenia_pattern":
        labs["platelets"] -= rng.normal(130, 55)

    elif lab_profile in ["mixed_instability", "multi_organ_stress_pattern"]:
        labs["wbc"] += rng.normal(7, 3)
        labs["lactate"] += rng.normal(2.0, 0.9)
        labs["creatinine"] += rng.normal(1.0, 0.5)
        labs["bun"] += rng.normal(22, 9)
        labs["bicarbonate"] -= rng.normal(5, 2)
        labs["ph"] -= rng.normal(0.05, 0.03)
        labs["ast"] += rng.normal(90, 50)
        labs["alt"] += rng.normal(75, 45)
        labs["platelets"] -= rng.normal(45, 30)

    elif lab_profile == "normal_but_trending_wrong":
        labs["lactate"] += rng.normal(0.45, 0.2)
        labs["creatinine"] += rng.normal(0.25, 0.12)
        labs["bicarbonate"] -= rng.normal(1.5, 0.8)
        labs["troponin"] += abs(rng.normal(0.03, 0.03))

    elif lab_profile == "delayed_abnormal_labs":
        labs["lactate"] += rng.normal(1.0, 0.6)
        labs["wbc"] += rng.normal(3, 2)
        labs["creatinine"] += rng.normal(0.4, 0.25)

    elif lab_profile == "contradictory_labs":
        if rng.random() < 0.5:
            labs["lactate"] += rng.normal(2.5, 0.8)
            labs["wbc"] -= rng.normal(2.0, 1.0)
        else:
            labs["wbc"] += rng.normal(8, 3)
            labs["lactate"] -= rng.normal(0.3, 0.2)

    elif lab_profile == "hemolyzed_sample":
        labs["potassium"] += rng.normal(1.6, 0.5)
        labs["ast"] += rng.normal(80, 40)

    elif lab_profile == "contaminated_sample_possible":
        labs["wbc"] += rng.normal(4, 3)
        labs["potassium"] += rng.normal(0.4, 0.4)

    elif lab_profile in ["critical_labs_pending", "critical_result_delayed"]:
        labs["lactate"] += rng.normal(2.8, 1.2)
        labs["potassium"] += rng.normal(0.9, 0.5)
        labs["creatinine"] += rng.normal(1.2, 0.6)
        labs["bicarbonate"] -= rng.normal(5, 2)
        labs["ph"] -= rng.normal(0.06, 0.03)

    return labs


def _apply_condition_effects(labs, condition_profile, rng):
    labs = labs.copy()

    if condition_profile == "diabetes_metabolic_instability":
        labs["glucose"] += rng.normal(80, 35)

    elif condition_profile == "ckd_lab_instability":
        labs["creatinine"] += rng.normal(1.0, 0.4)
        labs["bun"] += rng.normal(22, 8)
        labs["potassium"] += rng.normal(0.35, 0.2)

    elif condition_profile == "sepsis_recovery_risk":
        labs["wbc"] += rng.normal(4, 2)
        labs["lactate"] += rng.normal(0.8, 0.4)
        labs["platelets"] -= rng.normal(25, 20)

    elif condition_profile == "chf_fluid_pressure_instability":
        labs["sodium"] -= rng.normal(2, 1.5)
        labs["creatinine"] += rng.normal(0.25, 0.15)
        labs["bun"] += rng.normal(8, 3)

    elif condition_profile == "copd_respiratory_vulnerability":
        labs["pco2"] += rng.normal(6, 3)
        labs["bicarbonate"] += rng.normal(2, 1.5)

    elif condition_profile == "arrhythmia_hemodynamic_instability":
        labs["troponin"] += abs(rng.normal(0.04, 0.05))
        labs["potassium"] += rng.normal(0.15, 0.25)

    elif condition_profile == "post_surgical_recovery":
        labs["hemoglobin"] -= rng.normal(1.0, 0.6)
        labs["wbc"] += rng.normal(2.5, 1.5)

    elif condition_profile == "chronic_multimorbidity":
        labs["creatinine"] += rng.normal(0.6, 0.3)
        labs["bun"] += rng.normal(14, 6)
        labs["hemoglobin"] -= rng.normal(1.2, 0.6)
        labs["wbc"] += rng.normal(2, 1.5)

    elif condition_profile == "recurrent_deterioration_pattern":
        labs["lactate"] += rng.normal(0.8, 0.4)
        labs["wbc"] += rng.normal(3, 2)

    elif condition_profile == "complex_pancreatitis_icu_complication":
        labs["lipase"] += rng.normal(900, 350)
        labs["glucose"] += rng.normal(85, 35)
        labs["wbc"] += rng.normal(6, 3)
        labs["lactate"] += rng.normal(0.8, 0.5)
        labs["ast"] += rng.normal(45, 25)
        labs["alt"] += rng.normal(40, 25)

    elif condition_profile == "new_onset_diabetes_during_admission":
        labs["glucose"] += rng.normal(120, 45)
        labs["bicarbonate"] -= rng.normal(3, 2)

    elif condition_profile == "multi_event_complex_hospitalization":
        labs["lactate"] += rng.normal(1.4, 0.8)
        labs["creatinine"] += rng.normal(0.8, 0.4)
        labs["wbc"] += rng.normal(5, 2.5)
        labs["bicarbonate"] -= rng.normal(3, 1.5)

    return labs


def _apply_event_effects(labs, row, rng):
    labs = labs.copy()

    has_event = bool(row.get("has_event", False))
    event_type = str(row.get("event_type", "NONE"))
    post_event_state = str(row.get("post_event_state", "NO_EVENT"))
    care_mode = str(row.get("care_mode", "OUTPATIENT"))
    event_wave_count = float(row.get("event_wave_count", 0))
    terminal_pressure = float(row.get("terminal_decline_pressure", 0.0))
    volatility = float(row.get("post_event_volatility_score", 0.0))
    deceptive_stability = float(row.get("deceptive_stability_index", 0.0))

    if has_event:
        labs["lactate"] += rng.normal(1.0, 0.6)
        labs["wbc"] += rng.normal(3.0, 2.0)

    if event_type == "respiratory_decline":
        labs["pco2"] += rng.normal(8, 4)
        labs["ph"] -= rng.normal(0.035, 0.02)

    elif event_type == "hemodynamic_instability":
        labs["lactate"] += rng.normal(1.5, 0.7)
        labs["creatinine"] += rng.normal(0.35, 0.2)

    elif event_type == "sepsis_like_deterioration":
        labs["lactate"] += rng.normal(2.0, 0.9)
        labs["wbc"] += rng.normal(7, 3)
        labs["platelets"] -= rng.normal(35, 25)
        labs["bicarbonate"] -= rng.normal(3, 1.5)

    elif event_type == "metabolic_instability":
        labs["glucose"] += rng.normal(90, 45)
        labs["bicarbonate"] -= rng.normal(5, 2)
        labs["ph"] -= rng.normal(0.05, 0.025)

    elif event_type == "arrhythmia_instability":
        labs["troponin"] += abs(rng.normal(0.15, 0.20))
        labs["potassium"] += rng.normal(0.3, 0.3)

    elif event_type == "renal_lab_instability":
        labs["creatinine"] += rng.normal(1.2, 0.5)
        labs["bun"] += rng.normal(25, 8)
        labs["potassium"] += rng.normal(0.45, 0.3)

    elif event_type == "post_procedure_complication":
        labs["wbc"] += rng.normal(4, 2)
        labs["hemoglobin"] -= rng.normal(1.2, 0.7)
        labs["lactate"] += rng.normal(0.7, 0.4)

    if event_wave_count >= 2:
        labs["lactate"] += event_wave_count * rng.normal(0.55, 0.20)
        labs["creatinine"] += event_wave_count * rng.normal(0.18, 0.08)
        labs["bicarbonate"] -= event_wave_count * rng.normal(0.9, 0.35)
        labs["wbc"] += event_wave_count * rng.normal(1.2, 0.6)

    if post_event_state in ["VOLATILE_MONITOR", "DECLINING_MONITOR", "TEMPORARY_STABILIZATION"]:
        labs["lactate"] += rng.normal(1.0, 0.5)
        labs["bicarbonate"] -= rng.normal(2.0, 1.0)

    elif post_event_state == "RECOVERY_PENDING":
        labs["lactate"] += rng.normal(0.4, 0.3)

    elif post_event_state == "RECOVERY_CONFIRMED":
        labs["lactate"] -= rng.normal(0.3, 0.2)
        labs["wbc"] -= rng.normal(1.5, 1.0)

    if care_mode == "ICU":
        labs["lactate"] += rng.normal(0.5, 0.3)
        labs["creatinine"] += rng.normal(0.2, 0.15)

    if terminal_pressure >= 0.60:
        labs["lactate"] += rng.normal(2.5, 1.0)
        labs["creatinine"] += rng.normal(1.4, 0.7)
        labs["bun"] += rng.normal(30, 10)
        labs["bicarbonate"] -= rng.normal(5, 2)
        labs["ph"] -= rng.normal(0.06, 0.03)
        labs["platelets"] -= rng.normal(55, 35)
        labs["ast"] += rng.normal(180, 90)
        labs["alt"] += rng.normal(130, 75)

    if deceptive_stability >= 0.45 and volatility >= 0.35:
        labs["lactate"] += rng.normal(0.9, 0.4)
        labs["bicarbonate"] -= rng.normal(1.8, 0.8)
        labs["creatinine"] += rng.normal(0.25, 0.12)

    return labs


def _apply_context_effects(labs, row, rng):
    labs = labs.copy()

    treatment_masking = float(row.get("treatment_masking_risk", 0.0))
    overload = float(row.get("hospital_overload_pressure", 0.0))
    delayed_confirmation = bool(row.get("delayed_confirmation_possible_flag", False))
    surface_reassurance = bool(row.get("surface_level_reassurance_risk_flag", False))
    result_state = str(row.get("result_acknowledgment_state", "acknowledged_on_time"))
    evidence_deception = str(row.get("evidence_surface_deception_type", "none"))

    if treatment_masking >= 0.45:
        labs["lactate"] -= rng.normal(0.35, 0.2)
        labs["wbc"] -= rng.normal(1.0, 0.5)
        labs["ph"] += rng.normal(0.015, 0.01)

    if overload >= 0.50:
        labs["lactate"] += rng.normal(0.5, 0.25)
        labs["creatinine"] += rng.normal(0.15, 0.08)

    if delayed_confirmation:
        labs["lactate"] += rng.normal(0.6, 0.3)
        labs["creatinine"] += rng.normal(0.2, 0.1)

    if surface_reassurance or evidence_deception in [
        "labs_look_reassuring_but_patient_worsening",
        "normal_initial_workup_but_hidden_instability",
    ]:
        labs["lactate"] = min(labs["lactate"], rng.normal(2.1, 0.35))
        labs["wbc"] = min(labs["wbc"], rng.normal(11.5, 1.5))
        labs["creatinine"] = min(labs["creatinine"], rng.normal(1.45, 0.25))

    if result_state in ["not_acknowledged", "handoff_missed_result", "buried_in_chart"]:
        labs["lactate"] += rng.normal(0.45, 0.25)

    return labs


def _finalize_lab_bounds(labs):
    bounds = {
        "glucose": (40, 650),
        "lactate": (0.4, 12),
        "wbc": (0.5, 45),
        "creatinine": (0.2, 10),
        "bun": (3, 140),
        "potassium": (2.0, 7.5),
        "sodium": (115, 160),
        "bicarbonate": (5, 45),
        "ph": (6.8, 7.8),
        "pco2": (15, 95),
        "troponin": (0.0, 8.0),
        "hemoglobin": (5.0, 19.0),
        "platelets": (10, 800),
        "ast": (5, 1200),
        "alt": (5, 1200),
        "bilirubin": (0.1, 25),
        "lipase": (5, 5000),
    }

    return {
        col: round(_clip(value, bounds[col][0], bounds[col][1]), 3)
        for col, value in labs.items()
    }


def _derive_lab_pattern(row):
    profile = str(row.get("lab_profile", "normal_labs"))
    has_event = bool(row.get("has_event", False))
    hidden = float(row.get("silent_collapse_pressure", 0.0))
    deception = float(row.get("deceptive_stability_index", 0.0))
    lactate = float(row.get("lactate", 0.0))
    creatinine = float(row.get("creatinine", 0.0))
    troponin = float(row.get("troponin", 0.0))
    burden = int(row.get("lab_abnormality_burden", 0))
    critical = int(row.get("critical_lab_burden", 0))

    if profile in ["normal_labs", "mild_nonspecific_abnormality", "normal_but_trending_wrong"] and hidden >= 0.45:
        return "normal_labs_but_hidden_instability"

    if has_event and burden <= 2 and deception >= 0.40:
        return "event_with_reassuring_labs"

    if lactate >= 4 or critical >= 2:
        return "critical_chemistry_instability"

    if creatinine >= 2.5:
        return "renal_pressure_pattern"

    if troponin >= 0.1:
        return "cardiac_marker_pressure"

    if burden >= 6:
        return "multi_system_lab_stress"

    return "standard_lab_pattern"


def _derive_lab_trajectory(row):
    delayed = bool(row.get("delayed_confirmation_possible_flag", False))
    false_recovery = str(row.get("current_lifecycle_state", "")) in ["FALSE_RECOVERY", "DE_ESCALATION_MONITOR", "REINTEGRATION"]
    volatile = float(row.get("post_event_volatility_score", 0.0))
    lactate = float(row.get("lactate", 0.0))
    burden = int(row.get("lab_abnormality_burden", 0))

    if delayed and lactate >= 2.2:
        return "delayed_worsening_confirmation"

    if false_recovery and burden >= 3:
        return "lab_disagrees_with_recovery"

    if volatile >= 0.55 and burden >= 4:
        return "unstable_lab_trajectory"

    if burden <= 1:
        return "reassuring_snapshot"

    return "abnormal_but_noncritical_snapshot"


def generate_numeric_labs(df, random_seed=None):
    """
    Generate numeric synthetic labs for each row in a contextual dataframe.
    """

    if random_seed is None:
        random_seed = SYNTHETIC_ECOSYSTEM_CONFIG["random_seed"]

    rng = np.random.default_rng(random_seed)
    lab_rows = []

    for _, row in df.iterrows():
        lab_profile = row.get("lab_profile", "normal_labs")
        condition_profile = row.get("condition_profile", "baseline_stable")
        fragility = float(row.get("fragility_score", 0.0))
        hidden = float(row.get("silent_collapse_pressure", 0.0))
        uncertainty = float(row.get("lifecycle_uncertainty_score", 0.0))
        context_pressure = float(row.get("bire_context_pressure_score", 0.0))
        data_trust = float(row.get("data_trust_pressure_score", 0.0))

        labs = _normal_lab_baseline(rng)
        labs = _apply_lab_profile_effects(labs, lab_profile, rng)
        labs = _apply_condition_effects(labs, condition_profile, rng)
        labs = _apply_event_effects(labs, row, rng)
        labs = _apply_context_effects(labs, row, rng)

        labs["lactate"] += fragility * rng.normal(0.8, 0.3)
        labs["creatinine"] += fragility * rng.normal(0.4, 0.15)
        labs["bun"] += fragility * rng.normal(8, 3)
        labs["wbc"] += fragility * rng.normal(2, 1)

        labs["lactate"] += hidden * rng.normal(0.8, 0.35)
        labs["bicarbonate"] -= uncertainty * rng.normal(2.0, 0.8)
        labs["creatinine"] += context_pressure * rng.normal(0.25, 0.12)

        if data_trust >= 0.45 and rng.random() < 0.35:
            labs["lactate"] += rng.normal(0.4, 0.25)

        finalized_labs = _finalize_lab_bounds(labs)
        lab_rows.append(finalized_labs)

    lab_df = pd.DataFrame(lab_rows)

    out_df = pd.concat(
        [df.reset_index(drop=True), lab_df.reset_index(drop=True)],
        axis=1,
    )

    out_df["lab_abnormality_burden"] = (
        (out_df["glucose"].lt(70) | out_df["glucose"].gt(180)).astype(int)
        + out_df["lactate"].gt(2.2).astype(int)
        + out_df["wbc"].gt(12).astype(int)
        + out_df["creatinine"].gt(1.5).astype(int)
        + out_df["bun"].gt(25).astype(int)
        + (out_df["potassium"].lt(3.2) | out_df["potassium"].gt(5.3)).astype(int)
        + (out_df["sodium"].lt(130) | out_df["sodium"].gt(150)).astype(int)
        + out_df["bicarbonate"].lt(20).astype(int)
        + (out_df["ph"].lt(7.32) | out_df["ph"].gt(7.48)).astype(int)
        + out_df["pco2"].gt(50).astype(int)
        + out_df["troponin"].gt(0.04).astype(int)
        + out_df["hemoglobin"].lt(9).astype(int)
        + out_df["platelets"].lt(100).astype(int)
        + out_df["ast"].gt(80).astype(int)
        + out_df["alt"].gt(80).astype(int)
        + out_df["bilirubin"].gt(2.0).astype(int)
        + out_df["lipase"].gt(300).astype(int)
    )

    out_df["critical_lab_burden"] = (
        out_df["glucose"].gt(350).astype(int)
        + out_df["lactate"].gt(4.0).astype(int)
        + out_df["wbc"].gt(20).astype(int)
        + out_df["creatinine"].gt(3.0).astype(int)
        + out_df["potassium"].gt(6.0).astype(int)
        + out_df["sodium"].lt(125).astype(int)
        + out_df["ph"].lt(7.20).astype(int)
        + out_df["troponin"].gt(0.5).astype(int)
        + out_df["hemoglobin"].lt(7.0).astype(int)
        + out_df["platelets"].lt(50).astype(int)
        + out_df["lipase"].gt(1000).astype(int)
    )

    def _s(col, default=0.0):
        if col in out_df.columns:
            return out_df[col].fillna(default)
        return pd.Series(default, index=out_df.index)

    def _b(col, default=False):
        if col in out_df.columns:
            return out_df[col].fillna(default).astype(bool).astype(int)
        return pd.Series(int(default), index=out_df.index)

    out_df["lab_pattern_type"] = out_df.apply(_derive_lab_pattern, axis=1)
    out_df["lab_trajectory_state"] = out_df.apply(_derive_lab_trajectory, axis=1)

    # =====================================================
    # Organ-system burden domains
    # =====================================================

    out_df["metabolic_lab_burden"] = (
        (out_df["glucose"].lt(70) | out_df["glucose"].gt(180)).astype(int)
        + (out_df["bicarbonate"].lt(20)).astype(int)
        + (out_df["ph"].lt(7.32) | out_df["ph"].gt(7.48)).astype(int)
        + (out_df["lactate"].gt(2.2)).astype(int)
    )

    out_df["renal_lab_burden"] = (
        out_df["creatinine"].gt(1.5).astype(int)
        + out_df["bun"].gt(25).astype(int)
        + out_df["potassium"].gt(5.3).astype(int)
        + out_df["bicarbonate"].lt(20).astype(int)
    )

    out_df["respiratory_lab_burden"] = (
        out_df["pco2"].gt(50).astype(int)
        + out_df["ph"].lt(7.32).astype(int)
        + out_df["bicarbonate"].lt(20).astype(int)
    )

    out_df["hematologic_lab_burden"] = (
        out_df["hemoglobin"].lt(9).astype(int)
        + out_df["platelets"].lt(100).astype(int)
        + out_df["wbc"].gt(12).astype(int)
    )

    out_df["hepatic_pancreatic_lab_burden"] = (
        out_df["ast"].gt(80).astype(int)
        + out_df["alt"].gt(80).astype(int)
        + out_df["bilirubin"].gt(2.0).astype(int)
        + out_df["lipase"].gt(300).astype(int)
    )

    out_df["multi_organ_lab_pressure_score"] = np.clip(
        out_df["metabolic_lab_burden"].clip(0, 4) / 4 * 0.20
        + out_df["renal_lab_burden"].clip(0, 4) / 4 * 0.20
        + out_df["respiratory_lab_burden"].clip(0, 3) / 3 * 0.15
        + out_df["hematologic_lab_burden"].clip(0, 3) / 3 * 0.15
        + out_df["hepatic_pancreatic_lab_burden"].clip(0, 4) / 4 * 0.15
        + out_df["critical_lab_burden"].clip(0, 4) / 4 * 0.15,
        0,
        1,
    ).round(4)

    # =====================================================
    # Hidden chemistry / contradiction / false reassurance
    # =====================================================

    out_df["lab_vital_disagreement_score"] = np.clip(
        _s("silent_collapse_pressure") * 0.22
        + _b("physiologic_deception_flag") * 0.18
        + _b("surface_level_reassurance_risk_flag") * 0.14
        + out_df["lab_pattern_type"].isin([
            "normal_labs_but_hidden_instability",
            "event_with_reassuring_labs",
        ]).astype(int) * 0.20
        + out_df["lab_trajectory_state"].eq("lab_disagrees_with_recovery").astype(int) * 0.14
        + _s("hidden_instability_score") * 0.12,
        0,
        1,
    ).round(4)

    out_df["lab_delay_pressure_score"] = np.clip(
        _s("lab_delay_minutes") / 720 * 0.30
        + _b("critical_lab_pending_flag") * 0.22
        + _b("delayed_confirmation_possible_flag") * 0.18
        + (_s("result_acknowledgment_state", "acknowledged_on_time") != "acknowledged_on_time").astype(int) * 0.16
        + _s("lab_processing_backlog_minutes") / 240 * 0.14,
        0,
        1,
    ).round(4)

    out_df["lab_false_reassurance_score"] = np.clip(
        out_df["lab_pattern_type"].isin([
            "normal_labs_but_hidden_instability",
            "event_with_reassuring_labs",
        ]).astype(int) * 0.24
        + out_df["lab_trajectory_state"].isin([
            "reassuring_snapshot",
            "lab_disagrees_with_recovery",
        ]).astype(int) * 0.14
        + _s("normal_range_deception_score") * 0.16
        + _s("treatment_masking_risk") * 0.16
        + _s("silent_collapse_pressure") * 0.16
        + _b("surface_level_reassurance_risk_flag") * 0.14,
        0,
        1,
    ).round(4)

    out_df["hidden_chemistry_instability_score"] = np.clip(
        ((out_df["lactate"] - 1.8).clip(lower=0) / 4.0) * 0.22
        + ((1.2 - out_df["bicarbonate"].sub(20).clip(lower=0) / 10).clip(0, 1) * (out_df["bicarbonate"] < 22).astype(int)) * 0.12
        + ((out_df["creatinine"] - 1.2).clip(lower=0) / 3.0) * 0.16
        + ((7.38 - out_df["ph"]).clip(lower=0) / 0.20) * 0.16
        + ((out_df["troponin"] - 0.03).clip(lower=0) / 0.5) * 0.10
        + out_df["lab_vital_disagreement_score"] * 0.14
        + out_df["lab_false_reassurance_score"] * 0.10,
        0,
        1,
    ).round(4)

    out_df["lab_recovery_contradiction_score"] = np.clip(
        out_df["lab_trajectory_state"].eq("lab_disagrees_with_recovery").astype(int) * 0.22
        + _s("recovery_momentum") * 0.10
        + _s("recovery_resilience") * 0.06
        + out_df["hidden_chemistry_instability_score"] * 0.22
        + out_df["multi_organ_lab_pressure_score"] * 0.18
        + _s("rebound_deterioration_risk") * 0.12
        + _s("treatment_masking_risk") * 0.10,
        0,
        1,
    ).round(4)

    out_df["lab_abnormality_acceleration_pressure"] = np.clip(
        out_df["lab_abnormality_burden"].clip(0, 8) / 8 * 0.25
        + out_df["critical_lab_burden"].clip(0, 4) / 4 * 0.25
        + out_df["hidden_chemistry_instability_score"] * 0.20
        + _s("deterioration_momentum") * 0.15
        + _s("trajectory_stress_score") * 0.15,
        0,
        1,
    ).round(4)

    out_df["lab_masked_deterioration_flag"] = (
        (out_df["lab_false_reassurance_score"] >= 0.45)
        & (
            (out_df["hidden_chemistry_instability_score"] >= 0.35)
            | (_s("hidden_instability_score") >= 0.45)
            | (_s("silent_collapse_pressure") >= 0.45)
        )
    )

    out_df["lab_delayed_truth_flag"] = (
        (out_df["lab_delay_pressure_score"] >= 0.45)
        & (
            (out_df["critical_lab_burden"] >= 1)
            | (out_df["hidden_chemistry_instability_score"] >= 0.45)
            | (out_df["multi_organ_lab_pressure_score"] >= 0.45)
        )
    )

    out_df["critical_lab_consequence_pressure"] = np.clip(
        out_df["critical_lab_burden"].clip(0, 5) / 5 * 0.35
        + out_df["multi_organ_lab_pressure_score"] * 0.22
        + out_df["hidden_chemistry_instability_score"] * 0.18
        + out_df["lab_delay_pressure_score"] * 0.15
        + _s("hospital_system_failure_pressure") * 0.10,
        0,
        1,
    ).round(4)

    # outcomes.py consumes lab_abnormality_burden, critical_lab_burden,
    # lab_sixth_sense_pressure_score, and indirectly physiologic consequence pressure.
    out_df["lab_sixth_sense_pressure_score"] = np.clip(
        out_df["lab_abnormality_burden"].clip(0, 8) / 8 * 0.15
        + out_df["critical_lab_burden"].clip(0, 4) / 4 * 0.18
        + out_df["lab_vital_disagreement_score"] * 0.17
        + out_df["lab_delay_pressure_score"] * 0.14
        + out_df["hidden_chemistry_instability_score"] * 0.16
        + out_df["multi_organ_lab_pressure_score"] * 0.12
        + _s("bire_context_pressure_score") * 0.08,
        0,
        1,
    ).round(4)

    out_df["lab_outcome_alignment_score"] = np.clip(
        out_df["critical_lab_consequence_pressure"] * 0.22
        + out_df["lab_sixth_sense_pressure_score"] * 0.22
        + out_df["lab_recovery_contradiction_score"] * 0.18
        + out_df["lab_false_reassurance_score"] * 0.16
        + out_df["lab_delay_pressure_score"] * 0.12
        + out_df["lab_abnormality_acceleration_pressure"] * 0.10,
        0,
        1,
    ).round(4)

    out_df["labs_require_bire_attention_flag"] = (
        (out_df["critical_lab_burden"] >= 1)
        | (out_df["lab_sixth_sense_pressure_score"] >= 0.45)
        | (out_df["lab_pattern_type"].isin([
            "normal_labs_but_hidden_instability",
            "event_with_reassuring_labs",
            "critical_chemistry_instability",
            "multi_system_lab_stress",
        ]))
        | (out_df["lab_trajectory_state"].isin([
            "delayed_worsening_confirmation",
            "lab_disagrees_with_recovery",
            "unstable_lab_trajectory",
        ]))
        | (out_df["lab_masked_deterioration_flag"])
        | (out_df["lab_delayed_truth_flag"])
    )

    return out_df
