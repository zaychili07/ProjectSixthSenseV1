"""
BIRE OS Synthetic Operational Lifecycle State Machine

Chapter 53 doctrine:
No more babying BIRE OS.

Purpose:
Challenge BIRE OS to earn the Project Sixth Sense name.

This module does not simply label encounters as event/no-event.
It creates longitudinal lifecycle transitions across patient journeys.

BIRE OS must learn:
- hidden pre-event instability
- false reassurance
- delayed recognition
- event onset
- multi-wave deterioration
- post-event volatility
- temporary stabilization
- false recovery
- de-escalation ambiguity
- re-escalation
- reintegration failure
- recurrence risk
- death trajectory pressure
- historical pattern loss
- lifecycle uncertainty

Doctrine:
We Detect What Others Miss.
"""

import numpy as np
import pandas as pd

from bire.synthetic.config import SYNTHETIC_ECOSYSTEM_CONFIG


EVENT_TYPES = [
    "respiratory_decline",
    "hemodynamic_instability",
    "sepsis_like_deterioration",
    "metabolic_instability",
    "arrhythmia_instability",
    "renal_lab_instability",
    "post_procedure_complication",
    "unknown_deterioration",
]


HIGH_RISK_STATES = [
    "PRE_EVENT_HIDDEN_INSTABILITY",
    "PRE_EVENT_FALSE_REASSURANCE",
    "PRE_EVENT_WORSENING_WATCH",
    "EVENT_TRANSITION",
    "POST_EVENT_VOLATILE_MONITOR",
    "FALSE_RECOVERY",
    "RE_ESCALATION",
    "CRITICAL_POST_EVENT",
    "RECURRENT_INSTABILITY",
    "TERMINAL_DECLINE_PHASE",
    "FAILED_REINTEGRATION",
]


def _clip(value, low=0.0, high=1.0):
    return float(np.clip(value, low, high))


def _event_type_weights(row):
    condition = str(row.get("condition_profile", ""))
    presentation = str(row.get("presentation_profile", ""))

    weights = {
        "respiratory_decline": 0.14,
        "hemodynamic_instability": 0.15,
        "sepsis_like_deterioration": 0.13,
        "metabolic_instability": 0.12,
        "arrhythmia_instability": 0.10,
        "renal_lab_instability": 0.10,
        "post_procedure_complication": 0.08,
        "unknown_deterioration": 0.18,
    }

    if "copd" in condition or "respiratory" in presentation or "shortness_of_breath" in presentation:
        weights["respiratory_decline"] += 0.22

    if "chf" in condition or "hemodynamic" in condition:
        weights["hemodynamic_instability"] += 0.20

    if "sepsis" in condition or "infection" in presentation or "fever" in presentation:
        weights["sepsis_like_deterioration"] += 0.20

    if "diabetes" in condition or "hyperglycemia" in presentation:
        weights["metabolic_instability"] += 0.20

    if "arrhythmia" in condition or "palpitations" in presentation or "chest_pain" in presentation:
        weights["arrhythmia_instability"] += 0.20

    if "ckd" in condition or "renal" in condition:
        weights["renal_lab_instability"] += 0.20

    if "post_surgical" in condition:
        weights["post_procedure_complication"] += 0.20

    probs = np.array([weights[event] for event in EVENT_TYPES], dtype=float)
    return probs / probs.sum()


def _base_event_probability(row):
    care_mode = row.get("care_mode", row.get("highest_acuity_mode", "OUTPATIENT"))

    event_probability = (
        0.025
        + row.get("fragility_score", 0.0) * 0.20
        + row.get("weak_signal_likelihood", 0.0) * 0.11
        + row.get("false_stability_risk", 0.0) * 0.12
        + row.get("misleading_presentation_risk", 0.0) * 0.10
        + row.get("encounter_continuity_risk", 0.0) * 0.11
        + row.get("encounter_chaos_score", 0.0) * 0.13
        + row.get("hidden_acuity_gap_score", 0.0) * 0.15
        + row.get("ten_year_instability_burden", 0.0) * 0.10
        + row.get("hospital_memory_fragility", 0.0) * 0.06
    )

    if care_mode == "ICU":
        event_probability += 0.22
    elif care_mode == "ER_ESI_1":
        event_probability += 0.20
    elif care_mode in ["ER_ESI_2", "INPATIENT"]:
        event_probability += 0.13
    elif care_mode == "ER_ESI_3":
        event_probability += 0.07
    elif care_mode in ["OUTPATIENT", "ER_ESI_5", "ER_ESI_4"]:
        event_probability -= 0.03

    if row.get("care_escalation_occurred", False):
        event_probability += 0.09

    if row.get("failed_discharge_flag", False):
        event_probability += 0.10

    if row.get("bounceback_flag", False):
        event_probability += 0.08

    if row.get("unexpected_escalation_flag", False):
        event_probability += 0.10

    if row.get("stepdown_rebound_flag", False):
        event_probability += 0.12

    if row.get("long_gap_return_flag", False):
        event_probability += 0.06

    return _clip(event_probability, 0.01, 0.96)


def _pressure_scores(row):
    hidden_instability_pressure = _clip(
        row.get("weak_signal_likelihood", 0.0) * 0.24
        + row.get("hidden_acuity_gap_score", 0.0) * 0.22
        + row.get("misleading_presentation_risk", 0.0) * 0.18
        + row.get("false_stability_risk", 0.0) * 0.14
        + row.get("ten_year_instability_burden", 0.0) * 0.12
        + (1 - row.get("deterioration_visibility_score", 1.0)) * 0.10
    )

    false_reassurance_pressure = _clip(
        row.get("false_stability_risk", 0.0) * 0.26
        + row.get("misleading_presentation_risk", 0.0) * 0.20
        + row.get("failed_discharge_flag", False) * 0.16
        + row.get("bounceback_flag", False) * 0.14
        + row.get("hospital_memory_fragility", 0.0) * 0.12
        + (1 - row.get("deterioration_visibility_score", 1.0)) * 0.12
    )

    operational_friction_pressure = _clip(
        row.get("encounter_chaos_score", 0.0) * 0.30
        + row.get("encounter_continuity_risk", 0.0) * 0.26
        + row.get("operational_dependency_score", 0.0) * 0.14
        + row.get("hospital_memory_fragility", 0.0) * 0.12
        + row.get("seasonal_pressure", 0.0) * 0.08
        + row.get("escalation_delay_hours", 0.0) / 96 * 0.10
    )

    return hidden_instability_pressure, false_reassurance_pressure, operational_friction_pressure


def _event_timing(row, rng):
    los_hours = max(float(row.get("length_of_stay_hours", 1)), 1.0)
    escalation_delay = float(row.get("escalation_delay_hours", 0))

    earliest = max(1, int(los_hours * 0.08))
    latest = max(earliest + 1, int(los_hours * 0.90))

    if escalation_delay > 0:
        center = min(los_hours * 0.78, los_hours * 0.30 + escalation_delay)
    else:
        center = los_hours * rng.choice([0.22, 0.45, 0.68], p=[0.25, 0.50, 0.25])

    event_hour = int(
        np.clip(
            rng.normal(center, max(los_hours * 0.12, 2)),
            earliest,
            latest,
        )
    )

    if event_hour <= los_hours * 0.25:
        timing = "EARLY_ENCOUNTER_EVENT"
    elif event_hour <= los_hours * 0.65:
        timing = "MID_ENCOUNTER_EVENT"
    else:
        timing = "LATE_ENCOUNTER_EVENT"

    return event_hour, timing


def _choose_event_wave_count(row, has_event, hidden_pressure, false_reassurance, operational_friction, rng):
    if not has_event:
        return 0

    wave_pressure = _clip(
        row.get("fragility_score", 0.0) * 0.20
        + row.get("ten_year_instability_burden", 0.0) * 0.18
        + hidden_pressure * 0.18
        + false_reassurance * 0.16
        + operational_friction * 0.14
        + row.get("post_event_volatility_score", 0.0) * 0.08
        + row.get("stepdown_rebound_flag", False) * 0.10
        + row.get("failed_discharge_flag", False) * 0.08
    )

    if wave_pressure >= 0.70:
        return int(rng.choice([2, 3, 4], p=[0.48, 0.36, 0.16]))

    if wave_pressure >= 0.48:
        return int(rng.choice([1, 2, 3], p=[0.58, 0.32, 0.10]))

    return int(rng.choice([1, 2], p=[0.84, 0.16]))


def _cascade_chain(primary_event_type, event_wave_count, rng):
    if event_wave_count <= 1:
        return primary_event_type

    cascade_options = {
        "respiratory_decline": ["respiratory_decline", "hemodynamic_instability", "arrhythmia_instability"],
        "hemodynamic_instability": ["hemodynamic_instability", "renal_lab_instability", "metabolic_instability"],
        "sepsis_like_deterioration": ["sepsis_like_deterioration", "hemodynamic_instability", "renal_lab_instability"],
        "metabolic_instability": ["metabolic_instability", "arrhythmia_instability", "renal_lab_instability"],
        "arrhythmia_instability": ["arrhythmia_instability", "hemodynamic_instability", "respiratory_decline"],
        "renal_lab_instability": ["renal_lab_instability", "metabolic_instability", "hemodynamic_instability"],
        "post_procedure_complication": ["post_procedure_complication", "infection_pattern", "hemodynamic_instability"],
        "unknown_deterioration": ["unknown_deterioration", "respiratory_decline", "hemodynamic_instability"],
    }

    chain = cascade_options.get(primary_event_type, [primary_event_type])

    selected = []
    for i in range(event_wave_count):
        selected.append(chain[min(i, len(chain) - 1)])

    return " -> ".join(selected)


def _choose_pre_event_state(event_probability, hidden_pressure, false_reassurance, prior_state):
    if hidden_pressure >= 0.62 and event_probability >= 0.50:
        return "PRE_EVENT_HIDDEN_INSTABILITY"

    if false_reassurance >= 0.60:
        return "PRE_EVENT_FALSE_REASSURANCE"

    if prior_state in ["PRE_EVENT_HIDDEN_INSTABILITY", "PRE_EVENT_FALSE_REASSURANCE"] and event_probability >= 0.35:
        return "PRE_EVENT_WORSENING_WATCH"

    if event_probability >= 0.55:
        return "PRE_EVENT_HIGH_RISK_SURVEILLANCE"

    if event_probability >= 0.30:
        return "PRE_EVENT_WATCHFUL_SURVEILLANCE"

    return "PRE_EVENT_LOW_RISK_SURVEILLANCE"


def _choose_post_event_state(
    row,
    hidden_pressure,
    false_reassurance,
    operational_friction,
    prior_state,
    prior_false_recovery,
    prior_reintegration,
    event_wave_count,
    rng,
):
    visibility = row.get("deterioration_visibility_score", 1.0)
    false_stability = row.get("false_stability_risk", 0.0)
    mortality = row.get("mortality_vulnerability", 0.0)

    recovery_legitimacy_score = _clip(
        0.68
        + visibility * 0.14
        - hidden_pressure * 0.18
        - false_reassurance * 0.22
        - operational_friction * 0.18
        - false_stability * 0.16
        - event_wave_count * 0.04
        - mortality * 0.08
    )

    volatility_score = _clip(
        hidden_pressure * 0.22
        + false_reassurance * 0.22
        + operational_friction * 0.22
        + false_stability * 0.16
        + event_wave_count * 0.05
        + prior_false_recovery * 0.08
        + row.get("ten_year_instability_burden", 0.0) * 0.05
    )

    deescalation_safety_score = _clip(
        recovery_legitimacy_score * 0.50
        + visibility * 0.18
        + (1 - volatility_score) * 0.22
        - prior_false_recovery * 0.06
        - prior_reintegration * 0.02
    )

    terminal_pressure = _clip(
        mortality * 0.35
        + row.get("fragility_score", 0.0) * 0.22
        + row.get("ten_year_instability_burden", 0.0) * 0.18
        + volatility_score * 0.18
        + (row.get("discharge_disposition", "") == "expired") * 0.30
    )

    roll = rng.random()

    if terminal_pressure >= 0.72 and roll < 0.55:
        return "TERMINAL_DECLINE_PHASE", "DECLINING_MONITOR", recovery_legitimacy_score, volatility_score, deescalation_safety_score, terminal_pressure

    if prior_state == "REINTEGRATION" and volatility_score >= 0.42 and roll < 0.42:
        return "FAILED_REINTEGRATION", "VOLATILE_MONITOR", recovery_legitimacy_score, volatility_score, deescalation_safety_score, terminal_pressure

    if prior_state == "FALSE_RECOVERY" and volatility_score >= 0.45:
        return "RE_ESCALATION", "VOLATILE_MONITOR", recovery_legitimacy_score, volatility_score, deescalation_safety_score, terminal_pressure

    if volatility_score >= 0.78 and roll < 0.70:
        return "CRITICAL_POST_EVENT", "DECLINING_MONITOR", recovery_legitimacy_score, volatility_score, deescalation_safety_score, terminal_pressure

    if volatility_score >= 0.62 and roll < 0.60:
        return "POST_EVENT_VOLATILE_MONITOR", "VOLATILE_MONITOR", recovery_legitimacy_score, volatility_score, deescalation_safety_score, terminal_pressure

    if false_reassurance >= 0.58 and roll < 0.45:
        return "FALSE_RECOVERY", "TEMPORARY_STABILIZATION", recovery_legitimacy_score, volatility_score, deescalation_safety_score, terminal_pressure

    if deescalation_safety_score >= 0.66 and roll < 0.40:
        return "REINTEGRATION", "RECOVERY_CONFIRMED", recovery_legitimacy_score, volatility_score, deescalation_safety_score, terminal_pressure

    if deescalation_safety_score >= 0.52 and roll < 0.62:
        return "DE_ESCALATION_MONITOR", "IMPROVING_MONITOR", recovery_legitimacy_score, volatility_score, deescalation_safety_score, terminal_pressure

    if roll < 0.78:
        return "POST_EVENT_MONITOR", "STABLE_MONITOR", recovery_legitimacy_score, volatility_score, deescalation_safety_score, terminal_pressure

    return "RECOVERY_EVALUATION", "RECOVERY_PENDING", recovery_legitimacy_score, volatility_score, deescalation_safety_score, terminal_pressure


def _transition_reason(row, previous_state, current_state, hidden_pressure, false_reassurance, volatility_score):
    reasons = []

    if previous_state != current_state:
        reasons.append(f"{previous_state}_TO_{current_state}")

    if hidden_pressure >= 0.60:
        reasons.append("hidden_instability_pressure_high")

    if false_reassurance >= 0.58:
        reasons.append("false_reassurance_pressure_high")

    if volatility_score >= 0.62:
        reasons.append("post_event_volatility_high")

    if row.get("care_escalation_occurred", False):
        reasons.append("care_escalation_occurred")

    if row.get("failed_discharge_flag", False):
        reasons.append("failed_discharge_context")

    if row.get("bounceback_flag", False):
        reasons.append("bounceback_context")

    if row.get("stepdown_rebound_flag", False):
        reasons.append("stepdown_rebound_context")

    if row.get("unexpected_escalation_flag", False):
        reasons.append("unexpected_escalation_context")

    if row.get("long_gap_return_flag", False):
        reasons.append("long_gap_return_context")

    if row.get("seasonal_pressure", 0.0) >= 0.10:
        reasons.append("seasonal_operational_pressure")

    if not reasons:
        reasons.append("stable_or_low_signal_transition")

    return " | ".join(reasons)


def assign_operational_phases(encounter_df, random_seed=None):
    """
    Assign longitudinal lifecycle state-machine phases to synthetic encounters.
    """

    if random_seed is None:
        random_seed = SYNTHETIC_ECOSYSTEM_CONFIG["random_seed"]

    rng = np.random.default_rng(random_seed)

    rows = []
    patient_memory = {}

    sorted_df = encounter_df.sort_values(
        ["patient_id", "encounter_start"]
    ).reset_index(drop=True)

    for _, row in sorted_df.iterrows():
        patient_id = row["patient_id"]

        memory = patient_memory.get(
            patient_id,
            {
                "previous_state": "NEW_PATIENT",
                "event_episode_count": 0,
                "had_event": False,
                "had_false_recovery": False,
                "had_reintegration": False,
                "had_terminal_decline": False,
                "had_failed_reintegration": False,
                "last_event_type": "NONE",
                "lifecycle_stage_index": 0,
            },
        )

        previous_state = memory["previous_state"]
        lifecycle_stage_index = memory["lifecycle_stage_index"] + 1

        base_event_probability = _base_event_probability(row)

        hidden_pressure, false_reassurance, operational_friction = _pressure_scores(row)

        historical_pattern_missed_score = _clip(
            memory["had_event"] * 0.16
            + memory["had_false_recovery"] * 0.16
            + memory["had_failed_reintegration"] * 0.14
            + row.get("hospital_memory_fragility", 0.0) * 0.22
            + row.get("longitudinal_memory_need", 0.0) * 0.18
            + row.get("care_fragmentation_risk", 0.0) * 0.14
        )

        recurrence_risk_score = _clip(
            row.get("fragility_score", 0.0) * 0.20
            + row.get("weak_signal_likelihood", 0.0) * 0.16
            + row.get("false_stability_risk", 0.0) * 0.14
            + row.get("encounter_continuity_risk", 0.0) * 0.14
            + row.get("encounter_chaos_score", 0.0) * 0.14
            + row.get("ten_year_instability_burden", 0.0) * 0.12
            + memory["had_event"] * 0.07
            + memory["had_false_recovery"] * 0.05
            + memory["had_failed_reintegration"] * 0.04
        )

        adjusted_event_probability = _clip(
            base_event_probability
            + recurrence_risk_score * 0.10
            + historical_pattern_missed_score * 0.08
            + memory["had_false_recovery"] * 0.06
            - memory["had_reintegration"] * 0.03,
            0.01,
            0.97,
        )

        has_event = rng.random() < adjusted_event_probability

        event_type = "NONE"
        event_hour = None
        event_timing_category = "NO_EVENT"
        event_episode_id = None
        episode_position = "NO_EVENT"
        event_wave_count = 0
        event_cascade_chain = "NONE"
        multi_crash_encounter_flag = False

        recovery_legitimacy_score = 0.0
        volatility_score = recurrence_risk_score
        deescalation_safety_score = 0.0
        terminal_decline_pressure = _clip(row.get("mortality_vulnerability", 0.0) * 0.30)

        if not has_event:
            current_state = _choose_pre_event_state(
                adjusted_event_probability,
                hidden_pressure,
                false_reassurance,
                previous_state,
            )

            post_event_state = "NO_EVENT"

        else:
            event_type = rng.choice(
                EVENT_TYPES,
                p=_event_type_weights(row),
            )

            event_hour, event_timing_category = _event_timing(row, rng)

            event_wave_count = _choose_event_wave_count(
                row,
                has_event,
                hidden_pressure,
                false_reassurance,
                operational_friction,
                rng,
            )

            event_cascade_chain = _cascade_chain(
                event_type,
                event_wave_count,
                rng,
            )

            multi_crash_encounter_flag = event_wave_count >= 2

            if previous_state not in [
                "EVENT_TRANSITION",
                "POST_EVENT_MONITOR",
                "POST_EVENT_VOLATILE_MONITOR",
                "FALSE_RECOVERY",
                "RE_ESCALATION",
                "CRITICAL_POST_EVENT",
                "DE_ESCALATION_MONITOR",
                "RECOVERY_EVALUATION",
                "FAILED_REINTEGRATION",
                "TERMINAL_DECLINE_PHASE",
            ]:
                memory["event_episode_count"] += 1
                episode_position = "NEW_EVENT_EPISODE"
            else:
                episode_position = "CONTINUING_EVENT_EPISODE"

            event_episode_id = f"{patient_id}_EP{str(memory['event_episode_count']).zfill(3)}"

            if previous_state in [
                "PRE_EVENT_HIDDEN_INSTABILITY",
                "PRE_EVENT_FALSE_REASSURANCE",
                "PRE_EVENT_WORSENING_WATCH",
            ]:
                current_state = "EVENT_TRANSITION"
                post_event_state = "EARLY_EVENT"
                recovery_legitimacy_score = 0.0
                volatility_score = _clip(
                    hidden_pressure * 0.42
                    + false_reassurance * 0.32
                    + operational_friction * 0.18
                    + event_wave_count * 0.04
                )
                deescalation_safety_score = 0.0
                terminal_decline_pressure = _clip(
                    row.get("mortality_vulnerability", 0.0) * 0.30
                    + volatility_score * 0.20
                )
            else:
                (
                    current_state,
                    post_event_state,
                    recovery_legitimacy_score,
                    volatility_score,
                    deescalation_safety_score,
                    terminal_decline_pressure,
                ) = _choose_post_event_state(
                    row,
                    hidden_pressure,
                    false_reassurance,
                    operational_friction,
                    previous_state,
                    memory["had_false_recovery"],
                    memory["had_reintegration"],
                    event_wave_count,
                    rng,
                )

        lifecycle_transition = f"{previous_state} -> {current_state}"

        signal_present_before_recognition_hours = int(
            np.clip(
                row.get("escalation_delay_hours", 0)
                + hidden_pressure * 24
                + false_reassurance * 18
                + operational_friction * 16
                + historical_pattern_missed_score * 18,
                0,
                168,
            )
        )

        recognition_delay_hours = int(
            np.clip(
                row.get("escalation_delay_hours", 0)
                + operational_friction * 24
                + historical_pattern_missed_score * 18
                + row.get("seasonal_pressure", 0.0) * 24,
                0,
                168,
            )
        )

        pre_event_warning_window = int(
            np.clip(
                signal_present_before_recognition_hours - recognition_delay_hours * 0.25,
                0,
                168,
            )
        )

        deceptive_stability_index = _clip(
            false_reassurance * 0.34
            + hidden_pressure * 0.24
            + row.get("false_stability_risk", 0.0) * 0.20
            + row.get("misleading_presentation_risk", 0.0) * 0.12
            + historical_pattern_missed_score * 0.10
        )

        signal_conflict_burden = _clip(
            hidden_pressure * 0.24
            + false_reassurance * 0.20
            + operational_friction * 0.18
            + row.get("hidden_acuity_gap_score", 0.0) * 0.16
            + row.get("care_mode_mismatch_flag", False) * 0.12
            + row.get("unexpected_escalation_flag", False) * 0.10
        )

        lifecycle_uncertainty_score = _clip(
            hidden_pressure * 0.22
            + false_reassurance * 0.22
            + volatility_score * 0.20
            + operational_friction * 0.16
            + recurrence_risk_score * 0.10
            + historical_pattern_missed_score * 0.10
        )

        monitoring_intensity_need = _clip(
            lifecycle_uncertainty_score * 0.30
            + adjusted_event_probability * 0.22
            + volatility_score * 0.18
            + recurrence_risk_score * 0.16
            + terminal_decline_pressure * 0.14
        )

        lifecycle_trust_state = (
            "LOW_TRUST"
            if lifecycle_uncertainty_score >= 0.50
            else "PARTIAL_TRUST"
            if lifecycle_uncertainty_score >= 0.28
            else "HIGH_TRUST"
        )

        transition_reason = _transition_reason(
            row,
            previous_state,
            current_state,
            hidden_pressure,
            false_reassurance,
            volatility_score,
        )

        requires_close_monitoring = bool(
            monitoring_intensity_need >= 0.48
            or current_state in HIGH_RISK_STATES
        )

        re_escalation_risk_flag = bool(
            volatility_score >= 0.60
            or current_state in ["FALSE_RECOVERY", "RE_ESCALATION", "POST_EVENT_VOLATILE_MONITOR", "FAILED_REINTEGRATION"]
        )

        false_recovery_risk_flag = bool(
            false_reassurance >= 0.56
            or current_state == "FALSE_RECOVERY"
        )

        de_escalation_candidate_flag = bool(
            current_state in ["DE_ESCALATION_MONITOR", "REINTEGRATION"]
            and deescalation_safety_score >= 0.50
            and lifecycle_uncertainty_score < 0.58
        )

        failed_reintegration_risk_flag = bool(
            current_state == "FAILED_REINTEGRATION"
            or (
                memory["had_reintegration"]
                and volatility_score >= 0.45
                and recurrence_risk_score >= 0.45
            )
        )

        terminal_decline_risk_flag = bool(
            current_state == "TERMINAL_DECLINE_PHASE"
            or terminal_decline_pressure >= 0.65
        )

        rows.append(
            {
                **row.to_dict(),

                "previous_lifecycle_state": previous_state,
                "current_lifecycle_state": current_state,
                "operational_phase": current_state,
                "post_event_state": post_event_state,
                "lifecycle_transition": lifecycle_transition,
                "state_transition_reason": transition_reason,
                "lifecycle_stage_index": lifecycle_stage_index,

                "has_event": bool(has_event),
                "event_type": event_type,
                "event_hour": event_hour,
                "event_timing_category": event_timing_category,
                "event_episode_id": event_episode_id,
                "episode_position": episode_position,

                "event_wave_count": int(event_wave_count),
                "event_cascade_chain": event_cascade_chain,
                "multi_crash_encounter_flag": bool(multi_crash_encounter_flag),

                "base_event_probability": round(float(base_event_probability), 4),
                "adjusted_event_probability": round(float(adjusted_event_probability), 4),
                "event_probability": round(float(adjusted_event_probability), 4),

                "hidden_instability_pressure": round(float(hidden_pressure), 4),
                "false_reassurance_pressure": round(float(false_reassurance), 4),
                "operational_friction_pressure": round(float(operational_friction), 4),
                "recurrence_risk_score": round(float(recurrence_risk_score), 4),
                "historical_pattern_missed_score": round(float(historical_pattern_missed_score), 4),

                "recovery_legitimacy_score": round(float(recovery_legitimacy_score), 4),
                "post_event_volatility_score": round(float(volatility_score), 4),
                "deescalation_safety_score": round(float(deescalation_safety_score), 4),
                "terminal_decline_pressure": round(float(terminal_decline_pressure), 4),

                "signal_present_before_recognition_hours": signal_present_before_recognition_hours,
                "recognition_delay_hours": recognition_delay_hours,
                "pre_event_warning_window": pre_event_warning_window,
                "deceptive_stability_index": round(float(deceptive_stability_index), 4),
                "signal_conflict_burden": round(float(signal_conflict_burden), 4),

                "lifecycle_uncertainty_score": round(float(lifecycle_uncertainty_score), 4),
                "monitoring_intensity_need": round(float(monitoring_intensity_need), 4),
                "lifecycle_trust_state": lifecycle_trust_state,

                "prior_event_history": bool(memory["had_event"]),
                "prior_false_recovery": bool(memory["had_false_recovery"]),
                "prior_reintegration": bool(memory["had_reintegration"]),
                "prior_terminal_decline": bool(memory["had_terminal_decline"]),
                "prior_failed_reintegration": bool(memory["had_failed_reintegration"]),
                "prior_event_type": memory["last_event_type"],

                "requires_close_monitoring": requires_close_monitoring,
                "re_escalation_risk_flag": re_escalation_risk_flag,
                "false_recovery_risk_flag": false_recovery_risk_flag,
                "de_escalation_candidate_flag": de_escalation_candidate_flag,
                "failed_reintegration_risk_flag": failed_reintegration_risk_flag,
                "terminal_decline_risk_flag": terminal_decline_risk_flag,
            }
        )

        memory["previous_state"] = current_state
        memory["lifecycle_stage_index"] = lifecycle_stage_index

        if has_event:
            memory["had_event"] = True
            memory["last_event_type"] = event_type

        if current_state == "FALSE_RECOVERY":
            memory["had_false_recovery"] = True

        if current_state == "REINTEGRATION":
            memory["had_reintegration"] = True

        if current_state == "TERMINAL_DECLINE_PHASE":
            memory["had_terminal_decline"] = True

        if current_state == "FAILED_REINTEGRATION":
            memory["had_failed_reintegration"] = True

        patient_memory[patient_id] = memory

    return pd.DataFrame(rows)