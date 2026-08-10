"""
BIRE OS Synthetic Encounter + Care Movement Chaos Generator

Chapter 53 doctrine:
No more babying BIRE OS.

This module generates decade-scale longitudinal healthcare encounters:
- initial / highest / final care modes
- transfer pathways
- escalation chains
- bouncebacks
- failed discharges
- ICU rebounds
- quiet gaps before escalation
- misleading presentations
- care-mode mismatch
- continuity fragmentation
- seasonal pressure
- hospital memory fragility
- operationally chaotic patient journeys

Structured chaos, not random nonsense.

Doctrine:
We Detect What Others Miss.
"""

import numpy as np
import pandas as pd

from bire.synthetic.config import SYNTHETIC_ECOSYSTEM_CONFIG


ACUITY_RANK = {
    "OUTPATIENT": 0,
    "ER_ESI_5": 1,
    "ER_ESI_4": 2,
    "ER_ESI_3": 3,
    "ER_ESI_2": 4,
    "ER_ESI_1": 5,
    "INPATIENT": 6,
    "ICU": 7,
}


JOURNEY_ARCHETYPES = [
    "routine_low_acuity",
    "episodic_er_user",
    "chronic_recurrent_instability",
    "progressive_escalation",
    "readmission_prone",
    "complex_multi_level_care",
    "silent_instability_then_escalation",
    "normal_appearing_then_complex",
    "frequent_high_utilizer",
    "end_of_life_decline",
    "operationally_lost_patient",
    "false_recovery_bounceback",
    "long_gap_then_collapse",
]


JOURNEY_WEIGHTS = {
    "routine_low_acuity": 0.22,
    "episodic_er_user": 0.13,
    "chronic_recurrent_instability": 0.13,
    "progressive_escalation": 0.10,
    "readmission_prone": 0.11,
    "complex_multi_level_care": 0.08,
    "silent_instability_then_escalation": 0.06,
    "normal_appearing_then_complex": 0.06,
    "frequent_high_utilizer": 0.04,
    "end_of_life_decline": 0.025,
    "operationally_lost_patient": 0.035,
    "false_recovery_bounceback": 0.035,
    "long_gap_then_collapse": 0.035,
}


DISCHARGE_DISPOSITIONS = [
    "home",
    "home_with_support",
    "rehab",
    "stepdown",
    "long_term_care",
    "against_medical_advice",
    "expired",
    "left_without_followup",
    "transfer_to_outside_hospital",
]


ARRIVAL_METHODS = [
    "walk_in",
    "private_vehicle",
    "ambulance",
    "air_transport",
    "facility_transfer",
    "clinic_referral",
    "family_brought_in",
]


TRANSFER_SOURCES = [
    "none",
    "outpatient_clinic",
    "urgent_care",
    "outside_hospital",
    "stepdown_unit",
    "rehab_facility",
    "long_term_care",
    "home_health",
]


def _clip(value, low=0.0, high=1.0):
    return float(np.clip(value, low, high))


def _normalized_probs(weight_map, keys):
    probs = np.array([weight_map.get(k, 0.01) for k in keys], dtype=float)
    return probs / probs.sum()


def _acuity_gap(low_mode, high_mode):
    return ACUITY_RANK.get(high_mode, 0) - ACUITY_RANK.get(low_mode, 0)


def _seasonal_pressure(timestamp):
    month = timestamp.month

    if month in [12, 1, 2]:
        return 0.18

    if month in [7, 8]:
        return 0.08

    if month in [3, 4]:
        return 0.06

    return 0.0


def _choose_journey_archetype(patient, rng):
    fragility = patient.get("fragility_score", 0.0)
    complexity = patient.get("chronic_complexity_score", 0.0)
    weak_signal = patient.get("weak_signal_likelihood", 0.0)
    misleading = patient.get("misleading_presentation_risk", 0.0)
    readmission = patient.get("readmission_tendency", 0.0)
    utilization = patient.get("utilization_intensity", 0.0)
    mortality = patient.get("mortality_vulnerability", 0.0)
    memory_fragility = patient.get("hospital_memory_fragility", 0.0)
    false_stability = patient.get("false_stability_risk", 0.0)
    ten_year_burden = patient.get("ten_year_instability_burden", 0.0)
    longitudinal_archetype = str(patient.get("longitudinal_archetype", ""))

    if longitudinal_archetype == "end_of_life_decline_patient" or mortality >= 0.75:
        return "end_of_life_decline"

    if longitudinal_archetype == "frequent_high_utilizer" or utilization >= 0.75:
        return "frequent_high_utilizer"

    if longitudinal_archetype == "operationally_lost_patient" or memory_fragility >= 0.70:
        return "operationally_lost_patient"

    if false_stability >= 0.65:
        return "false_recovery_bounceback"

    if weak_signal >= 0.65 and misleading >= 0.55:
        return "silent_instability_then_escalation"

    if misleading >= 0.62:
        return "normal_appearing_then_complex"

    if readmission >= 0.70:
        return "readmission_prone"

    if ten_year_burden >= 0.70 or fragility >= 0.85 or complexity >= 0.78:
        return rng.choice(
            [
                "chronic_recurrent_instability",
                "progressive_escalation",
                "readmission_prone",
                "complex_multi_level_care",
                "false_recovery_bounceback",
            ]
        )

    if weak_signal >= 0.55:
        return rng.choice(
            ["silent_instability_then_escalation", "long_gap_then_collapse"],
            p=[0.65, 0.35],
        )

    return rng.choice(
        JOURNEY_ARCHETYPES,
        p=_normalized_probs(JOURNEY_WEIGHTS, JOURNEY_ARCHETYPES),
    )


def _encounter_count(journey_archetype, patient, rng, min_encounters, max_encounters):
    utilization = patient.get("utilization_intensity", 0.0)
    readmission = patient.get("readmission_tendency", 0.0)
    burden = patient.get("ten_year_instability_burden", 0.0)

    ranges = {
        "routine_low_acuity": (2, 10),
        "episodic_er_user": (5, 18),
        "chronic_recurrent_instability": (12, 38),
        "progressive_escalation": (8, 30),
        "readmission_prone": (12, 42),
        "complex_multi_level_care": (14, 48),
        "silent_instability_then_escalation": (6, 24),
        "normal_appearing_then_complex": (6, 24),
        "frequent_high_utilizer": (22, 70),
        "end_of_life_decline": (10, 46),
        "operationally_lost_patient": (8, 34),
        "false_recovery_bounceback": (10, 40),
        "long_gap_then_collapse": (3, 18),
    }

    low, high = ranges.get(journey_archetype, (min_encounters, max_encounters + 1))

    extra = int(np.round(utilization * 8 + readmission * 6 + burden * 5))
    low = max(min_encounters, low + int(extra * 0.25))
    high = max(low + 1, min(max(max_encounters, high), high + extra))

    return int(rng.integers(low, high))


def _days_until_next_encounter(patient, journey_archetype, encounter_index, n_encounters, rng):
    readmission = patient.get("readmission_tendency", 0.0)
    utilization = patient.get("utilization_intensity", 0.0)
    progression = patient.get("ten_year_progression_pressure", 0.0)

    progress = encounter_index / max(n_encounters - 1, 1)

    ranges = {
        "routine_low_acuity": (80, 420),
        "episodic_er_user": (25, 180),
        "chronic_recurrent_instability": (7, 80),
        "progressive_escalation": (12, 120),
        "readmission_prone": (1, 45),
        "complex_multi_level_care": (2, 70),
        "silent_instability_then_escalation": (35, 260),
        "normal_appearing_then_complex": (35, 240),
        "frequent_high_utilizer": (1, 35),
        "end_of_life_decline": (3, 90),
        "operationally_lost_patient": (10, 220),
        "false_recovery_bounceback": (1, 60),
        "long_gap_then_collapse": (180, 900),
    }

    low, high = ranges.get(journey_archetype, (20, 120))

    days = rng.integers(low, high)

    compression = (
        readmission * 0.30
        + utilization * 0.28
        + progression * progress * 0.22
    )

    days = days * (1 - min(compression, 0.70))

    return max(1, int(days))


def _choose_initial_care_mode(patient, journey_archetype, encounter_index, n_encounters, rng):
    progress = encounter_index / max(n_encounters - 1, 1)
    misleading = patient.get("misleading_presentation_risk", 0.0)
    weak_signal = patient.get("weak_signal_likelihood", 0.0)
    access_friction = patient.get("care_access_friction", 0.0)

    if journey_archetype == "routine_low_acuity":
        return rng.choice(["OUTPATIENT", "ER_ESI_5", "ER_ESI_4"], p=[0.74, 0.17, 0.09])

    if journey_archetype == "episodic_er_user":
        return rng.choice(["OUTPATIENT", "ER_ESI_5", "ER_ESI_4", "ER_ESI_3"], p=[0.22, 0.30, 0.30, 0.18])

    if journey_archetype == "chronic_recurrent_instability":
        return rng.choice(["ER_ESI_4", "ER_ESI_3", "ER_ESI_2", "INPATIENT"], p=[0.16, 0.38, 0.26, 0.20])

    if journey_archetype == "progressive_escalation":
        if progress < 0.45:
            return rng.choice(["OUTPATIENT", "ER_ESI_4", "ER_ESI_3"], p=[0.42, 0.32, 0.26])
        return rng.choice(["ER_ESI_3", "ER_ESI_2", "ER_ESI_1", "INPATIENT"], p=[0.18, 0.34, 0.18, 0.30])

    if journey_archetype == "readmission_prone":
        return rng.choice(["ER_ESI_4", "ER_ESI_3", "ER_ESI_2", "INPATIENT"], p=[0.12, 0.30, 0.30, 0.28])

    if journey_archetype == "complex_multi_level_care":
        return rng.choice(["ER_ESI_3", "ER_ESI_2", "ER_ESI_1", "INPATIENT"], p=[0.18, 0.32, 0.20, 0.30])

    if journey_archetype == "silent_instability_then_escalation":
        if progress < 0.65:
            return rng.choice(["OUTPATIENT", "ER_ESI_5", "ER_ESI_4"], p=[0.55, 0.27, 0.18])
        return rng.choice(["ER_ESI_4", "ER_ESI_3", "ER_ESI_2"], p=[0.18, 0.42, 0.40])

    if journey_archetype == "normal_appearing_then_complex":
        if progress < 0.55 or misleading > 0.60:
            return rng.choice(["OUTPATIENT", "ER_ESI_5", "ER_ESI_4"], p=[0.60, 0.25, 0.15])
        return rng.choice(["ER_ESI_4", "ER_ESI_3", "ER_ESI_2", "INPATIENT"], p=[0.18, 0.32, 0.25, 0.25])

    if journey_archetype == "frequent_high_utilizer":
        return rng.choice(["ER_ESI_5", "ER_ESI_4", "ER_ESI_3", "ER_ESI_2"], p=[0.18, 0.32, 0.34, 0.16])

    if journey_archetype == "end_of_life_decline":
        return rng.choice(["ER_ESI_3", "ER_ESI_2", "INPATIENT", "ICU"], p=[0.18, 0.28, 0.36, 0.18])

    if journey_archetype == "operationally_lost_patient":
        return rng.choice(["OUTPATIENT", "ER_ESI_5", "ER_ESI_4", "ER_ESI_3"], p=[0.35, 0.26, 0.25, 0.14])

    if journey_archetype == "false_recovery_bounceback":
        return rng.choice(["ER_ESI_4", "ER_ESI_3", "ER_ESI_2", "INPATIENT"], p=[0.22, 0.36, 0.24, 0.18])

    if journey_archetype == "long_gap_then_collapse":
        if progress < 0.70:
            return rng.choice(["OUTPATIENT", "ER_ESI_5", "ER_ESI_4"], p=[0.64, 0.24, 0.12])
        return rng.choice(["ER_ESI_3", "ER_ESI_2", "INPATIENT"], p=[0.35, 0.35, 0.30])

    if weak_signal > 0.65 or access_friction > 0.70:
        return rng.choice(["OUTPATIENT", "ER_ESI_5", "ER_ESI_4"], p=[0.50, 0.30, 0.20])

    return rng.choice(["OUTPATIENT", "ER_ESI_5", "ER_ESI_4", "ER_ESI_3"], p=[0.45, 0.20, 0.20, 0.15])


def _choose_highest_acuity_mode(
    patient,
    journey_archetype,
    initial_care_mode,
    previous_highest_mode,
    encounter_index,
    n_encounters,
    seasonal_pressure,
    rng,
):
    progress = encounter_index / max(n_encounters - 1, 1)

    fragility = patient.get("fragility_score", 0.0)
    weak_signal = patient.get("weak_signal_likelihood", 0.0)
    false_stability = patient.get("false_stability_risk", 0.0)
    misleading = patient.get("misleading_presentation_risk", 0.0)
    burden = patient.get("ten_year_instability_burden", 0.0)
    mortality = patient.get("mortality_vulnerability", 0.0)

    escalation_pressure = (
        fragility * 0.24
        + weak_signal * 0.20
        + false_stability * 0.17
        + misleading * 0.13
        + burden * 0.13
        + mortality * 0.08
        + progress * 0.05
        + seasonal_pressure * 0.08
    )

    candidates = list(ACUITY_RANK.keys())
    initial_rank = ACUITY_RANK[initial_care_mode]

    if escalation_pressure < 0.35:
        possible = [m for m in candidates if ACUITY_RANK[m] <= initial_rank + 1]
    elif escalation_pressure < 0.55:
        possible = [m for m in candidates if initial_rank <= ACUITY_RANK[m] <= min(initial_rank + 3, 7)]
    elif escalation_pressure < 0.75:
        possible = [m for m in candidates if initial_rank <= ACUITY_RANK[m] <= 7]
    else:
        possible = ["ER_ESI_2", "ER_ESI_1", "INPATIENT", "ICU"]

    if journey_archetype in ["progressive_escalation", "complex_multi_level_care", "end_of_life_decline"]:
        if progress > 0.50:
            possible = list(set(possible + ["INPATIENT", "ICU", "ER_ESI_1"]))

    if journey_archetype in ["silent_instability_then_escalation", "normal_appearing_then_complex", "long_gap_then_collapse"]:
        if progress > 0.62:
            possible = list(set(possible + ["ER_ESI_2", "INPATIENT", "ICU"]))

    if previous_highest_mode == "ICU" and rng.random() < 0.38:
        possible = list(set(possible + ["INPATIENT", "ICU"]))

    possible = sorted(possible, key=lambda m: ACUITY_RANK[m])

    weights = np.array([(ACUITY_RANK[m] + 1) for m in possible], dtype=float)
    weights = weights / weights.sum()

    highest = rng.choice(possible, p=weights)

    if ACUITY_RANK[highest] < ACUITY_RANK[initial_care_mode]:
        highest = initial_care_mode

    return highest


def _choose_final_care_mode(highest_acuity_mode, patient, journey_archetype, rng):
    false_stability = patient.get("false_stability_risk", 0.0)
    fragility = patient.get("fragility_score", 0.0)
    mortality = patient.get("mortality_vulnerability", 0.0)

    if highest_acuity_mode == "ICU":
        if journey_archetype == "end_of_life_decline" and rng.random() < mortality * 0.20:
            return "expired"
        return rng.choice(["ICU", "INPATIENT", "stepdown"], p=[0.34, 0.44, 0.22])

    if highest_acuity_mode == "INPATIENT":
        if journey_archetype == "end_of_life_decline" and rng.random() < mortality * 0.08:
            return "expired"
        return rng.choice(["INPATIENT", "stepdown", "home_with_support"], p=[0.44, 0.25, 0.31])

    if highest_acuity_mode in ["ER_ESI_1", "ER_ESI_2"]:
        return rng.choice(["ER_ESI_2", "INPATIENT", "ICU", "home_with_support"], p=[0.16, 0.42, 0.24, 0.18])

    if highest_acuity_mode in ["ER_ESI_3", "ER_ESI_4", "ER_ESI_5"]:
        unstable_discharge_risk = false_stability * 0.35 + fragility * 0.20
        if rng.random() < unstable_discharge_risk:
            return rng.choice(["home", "home_with_support", "INPATIENT"], p=[0.38, 0.36, 0.26])
        return rng.choice(["home", "home_with_support"], p=[0.74, 0.26])

    return "home"


def _build_transfer_pathway(initial_care_mode, highest_acuity_mode, final_care_mode, rng):
    path = [initial_care_mode]

    if ACUITY_RANK.get(highest_acuity_mode, 0) > ACUITY_RANK.get(initial_care_mode, 0):
        if "ER" not in initial_care_mode and highest_acuity_mode in ["INPATIENT", "ICU"]:
            path.append(rng.choice(["ER_ESI_3", "ER_ESI_2"], p=[0.55, 0.45]))

        if highest_acuity_mode == "ICU" and "INPATIENT" not in path and rng.random() < 0.45:
            path.append("INPATIENT")

        path.append(highest_acuity_mode)

    if final_care_mode not in path:
        path.append(final_care_mode)

    cleaned = []
    for item in path:
        if not cleaned or cleaned[-1] != item:
            cleaned.append(item)

    return cleaned


def _length_of_stay(highest_acuity_mode, journey_archetype, patient, rng):
    if highest_acuity_mode == "OUTPATIENT":
        los = rng.integers(1, 8)
    elif "ER" in highest_acuity_mode:
        los = rng.integers(3, 54)
    elif highest_acuity_mode == "INPATIENT":
        los = rng.integers(24, 360)
    elif highest_acuity_mode == "ICU":
        los = rng.integers(48, 840)
    else:
        los = rng.integers(2, 24)

    if journey_archetype in ["complex_multi_level_care", "end_of_life_decline"]:
        los *= rng.choice([1, 2, 3], p=[0.52, 0.31, 0.17])

    if journey_archetype == "frequent_high_utilizer":
        los *= rng.choice([0.75, 1.0, 1.25], p=[0.25, 0.55, 0.20])

    if patient.get("care_fragmentation_risk", 0.0) > 0.65 and rng.random() < 0.28:
        los *= rng.choice([1.25, 1.50, 2.00], p=[0.48, 0.36, 0.16])

    return int(np.clip(los, 1, 1680))


def _choose_arrival_method(initial_care_mode, highest_acuity_mode, patient, rng):
    access_friction = patient.get("care_access_friction", 0.0)

    if highest_acuity_mode == "ICU":
        return rng.choice(
            ["ambulance", "facility_transfer", "air_transport", "private_vehicle", "family_brought_in"],
            p=[0.54, 0.23, 0.06, 0.10, 0.07],
        )

    if highest_acuity_mode in ["ER_ESI_1", "ER_ESI_2"]:
        return rng.choice(
            ["ambulance", "private_vehicle", "facility_transfer", "walk_in", "family_brought_in"],
            p=[0.50, 0.24, 0.12, 0.08, 0.06],
        )

    if "ER" in initial_care_mode:
        if access_friction > 0.65:
            return rng.choice(["walk_in", "private_vehicle", "ambulance", "urgent_care", "family_brought_in"], p=[0.34, 0.32, 0.18, 0.06, 0.10])
        return rng.choice(["walk_in", "private_vehicle", "ambulance", "urgent_care"], p=[0.38, 0.34, 0.23, 0.05])

    return rng.choice(["walk_in", "private_vehicle", "clinic_referral"], p=[0.50, 0.30, 0.20])


def _choose_transfer_source(initial_care_mode, highest_acuity_mode, rng):
    if highest_acuity_mode in ["INPATIENT", "ICU"] and initial_care_mode not in ["INPATIENT", "ICU"]:
        return rng.choice(
            ["none", "outpatient_clinic", "urgent_care", "outside_hospital", "long_term_care", "home_health"],
            p=[0.54, 0.12, 0.10, 0.16, 0.04, 0.04],
        )

    if initial_care_mode in ["INPATIENT", "ICU"]:
        return rng.choice(
            ["outside_hospital", "stepdown_unit", "rehab_facility", "long_term_care", "none"],
            p=[0.22, 0.28, 0.14, 0.10, 0.26],
        )

    return rng.choice(["none", "outpatient_clinic", "urgent_care"], p=[0.74, 0.15, 0.11])


def _choose_discharge_disposition(final_care_mode, failed_discharge_flag, ama_discharge_flag, patient, rng):
    mortality = patient.get("mortality_vulnerability", 0.0)

    if final_care_mode == "expired":
        return "expired"

    if ama_discharge_flag:
        return "against_medical_advice"

    if final_care_mode == "ICU":
        expired_prob = min(0.18, mortality * 0.12)
        if rng.random() < expired_prob:
            return "expired"
        return rng.choice(["stepdown", "long_term_care", "transfer_to_outside_hospital"], p=[0.62, 0.28, 0.10])

    if final_care_mode == "INPATIENT":
        expired_prob = min(0.10, mortality * 0.06)
        if rng.random() < expired_prob:
            return "expired"
        return rng.choice(["home_with_support", "rehab", "stepdown", "long_term_care"], p=[0.42, 0.23, 0.25, 0.10])

    if final_care_mode == "stepdown":
        return rng.choice(["home_with_support", "rehab", "long_term_care"], p=[0.55, 0.30, 0.15])

    if failed_discharge_flag:
        return rng.choice(["home", "home_with_support", "left_without_followup"], p=[0.62, 0.28, 0.10])

    return rng.choice(
        DISCHARGE_DISPOSITIONS,
        p=[0.52, 0.22, 0.06, 0.06, 0.03, 0.04, 0.01, 0.04, 0.02],
    )


def generate_longitudinal_encounters(
    patient_df,
    min_encounters=2,
    max_encounters=70,
    random_seed=None,
):
    """
    Generate chaotic but narratively coherent decade-scale healthcare encounters.
    """

    if random_seed is None:
        random_seed = SYNTHETIC_ECOSYSTEM_CONFIG["random_seed"]

    rng = np.random.default_rng(random_seed)

    start_date = pd.Timestamp(SYNTHETIC_ECOSYSTEM_CONFIG["start_date"])
    end_date = pd.Timestamp(SYNTHETIC_ECOSYSTEM_CONFIG["end_date"])

    encounter_rows = []
    encounter_counter = 1

    for _, patient in patient_df.iterrows():
        patient_id = patient["patient_id"]

        journey_archetype = _choose_journey_archetype(patient, rng)

        n_encounters = _encounter_count(
            journey_archetype,
            patient,
            rng,
            min_encounters,
            max_encounters,
        )

        current_date = start_date + pd.Timedelta(days=int(rng.integers(0, 180)))

        previous_highest_mode = None
        previous_failed_discharge = False
        previous_final_mode = None
        previous_discharge_disposition = None
        previous_encounter_end = None

        for encounter_index in range(n_encounters):
            days_forward = _days_until_next_encounter(
                patient,
                journey_archetype,
                encounter_index,
                n_encounters,
                rng,
            )

            if previous_failed_discharge:
                days_forward = int(rng.integers(1, 8))

            if previous_discharge_disposition in ["against_medical_advice", "left_without_followup"]:
                if rng.random() < patient.get("readmission_tendency", 0.0):
                    days_forward = int(rng.integers(1, 21))

            current_date += pd.Timedelta(days=days_forward)

            if current_date >= end_date:
                break

            encounter_id = f"E{str(encounter_counter).zfill(7)}"
            encounter_counter += 1

            encounter_start = current_date + pd.Timedelta(hours=int(rng.integers(0, 24)))
            seasonal_pressure = _seasonal_pressure(encounter_start)

            years_since_sim_start = round((encounter_start - start_date).days / 365.25, 3)
            encounter_year = int(encounter_start.year)
            patient_age_at_encounter = int(patient.get("age", 18) + years_since_sim_start)

            initial_care_mode = _choose_initial_care_mode(
                patient,
                journey_archetype,
                encounter_index,
                n_encounters,
                rng,
            )

            highest_acuity_mode = _choose_highest_acuity_mode(
                patient,
                journey_archetype,
                initial_care_mode,
                previous_highest_mode,
                encounter_index,
                n_encounters,
                seasonal_pressure,
                rng,
            )

            final_care_mode = _choose_final_care_mode(
                highest_acuity_mode,
                patient,
                journey_archetype,
                rng,
            )

            transfer_pathway = _build_transfer_pathway(
                initial_care_mode,
                highest_acuity_mode,
                final_care_mode,
                rng,
            )

            acuity_escalation_gap = max(
                0,
                _acuity_gap(initial_care_mode, highest_acuity_mode),
            )

            care_escalation_occurred = acuity_escalation_gap >= 2
            care_mode = highest_acuity_mode

            los_hours = _length_of_stay(
                highest_acuity_mode,
                journey_archetype,
                patient,
                rng,
            )

            encounter_end = encounter_start + pd.Timedelta(hours=int(los_hours))

            readmission_flag = rng.random() < patient.get("readmission_tendency", 0.0)

            bounceback_flag = (
                previous_failed_discharge
                or rng.random() < patient.get("care_fragmentation_risk", 0.0) * 0.35
                or previous_discharge_disposition in ["against_medical_advice", "left_without_followup"]
            )

            failed_discharge_flag = rng.random() < (
                patient.get("false_stability_risk", 0.0) * 0.24
                + patient.get("hospital_memory_fragility", 0.0) * 0.08
            )

            left_without_being_seen_flag = (
                "ER" in initial_care_mode
                and highest_acuity_mode not in ["INPATIENT", "ICU"]
                and rng.random() < (0.04 + patient.get("care_access_friction", 0.0) * 0.06)
            )

            ama_discharge_flag = rng.random() < (
                0.025 + patient.get("care_access_friction", 0.0) * 0.025
            )

            boarding_before_admission_flag = (
                highest_acuity_mode in ["INPATIENT", "ICU"]
                and initial_care_mode.startswith("ER")
                and rng.random() < (0.28 + seasonal_pressure * 0.50)
            )

            care_mode_mismatch_flag = (
                rng.random() < patient.get("misleading_presentation_risk", 0.0) * 0.25
                or acuity_escalation_gap >= 3
            )

            visit_cluster_flag = days_forward <= 7

            quiet_gap_before_escalation_flag = (
                days_forward >= 90
                and patient.get("weak_signal_likelihood", 0.0) >= 0.55
                and care_escalation_occurred
            )

            unexpected_escalation_flag = (
                care_escalation_occurred
                and rng.random() < patient.get("weak_signal_likelihood", 0.0) * 0.55
            )

            stepdown_rebound_flag = (
                previous_highest_mode == "ICU"
                and highest_acuity_mode in ["INPATIENT", "ICU"]
                and previous_final_mode in ["stepdown", "INPATIENT", "home_with_support"]
                and rng.random() < 0.30
            )

            long_gap_return_flag = (
                previous_encounter_end is not None
                and (encounter_start - previous_encounter_end).days >= 180
            )

            escalation_delay_hours = 0
            if care_escalation_occurred:
                escalation_delay_hours = int(
                    np.clip(
                        rng.normal(
                            loc=acuity_escalation_gap * 3
                            + patient.get("care_fragmentation_risk", 0.0) * 10
                            + patient.get("misleading_presentation_risk", 0.0) * 8
                            + patient.get("hospital_memory_fragility", 0.0) * 8
                            + seasonal_pressure * 12,
                            scale=4,
                        ),
                        0,
                        96,
                    )
                )

            deterioration_visibility_score = np.clip(
                1
                - (
                    patient.get("weak_signal_likelihood", 0.0) * 0.30
                    + patient.get("misleading_presentation_risk", 0.0) * 0.30
                    + patient.get("false_stability_risk", 0.0) * 0.18
                    + patient.get("hospital_memory_fragility", 0.0) * 0.10
                    + boarding_before_admission_flag * 0.08
                    + seasonal_pressure * 0.08
                ),
                0,
                1,
            )

            hidden_acuity_gap_score = np.clip(
                acuity_escalation_gap / 7,
                0,
                1,
            )

            encounter_continuity_risk = np.clip(
                patient.get("care_fragmentation_risk", 0.0) * 0.24
                + patient.get("longitudinal_memory_need", 0.0) * 0.22
                + patient.get("hospital_memory_fragility", 0.0) * 0.18
                + patient.get("operational_dependency_score", 0.0) * 0.10
                + bounceback_flag * 0.08
                + failed_discharge_flag * 0.12
                + stepdown_rebound_flag * 0.12
                + visit_cluster_flag * 0.05
                + seasonal_pressure * 0.08,
                0,
                1,
            )

            encounter_chaos_score = np.clip(
                encounter_continuity_risk * 0.26
                + boarding_before_admission_flag * 0.10
                + care_mode_mismatch_flag * 0.13
                + unexpected_escalation_flag * 0.13
                + bounceback_flag * 0.08
                + left_without_being_seen_flag * 0.08
                + hidden_acuity_gap_score * 0.10
                + seasonal_pressure * 0.06
                + patient.get("ten_year_instability_burden", 0.0) * 0.06,
                0,
                1,
            )

            arrival_method = _choose_arrival_method(
                initial_care_mode,
                highest_acuity_mode,
                patient,
                rng,
            )

            transfer_source = _choose_transfer_source(
                initial_care_mode,
                highest_acuity_mode,
                rng,
            )

            discharge_disposition = _choose_discharge_disposition(
                final_care_mode,
                failed_discharge_flag,
                ama_discharge_flag,
                patient,
                rng,
            )

            encounter_rows.append(
                {
                    "encounter_id": encounter_id,
                    "patient_id": patient_id,

                    "encounter_start": encounter_start,
                    "encounter_end": encounter_end,
                    "encounter_year": encounter_year,
                    "years_since_sim_start": years_since_sim_start,
                    "patient_age_at_encounter": patient_age_at_encounter,
                    "length_of_stay_hours": int(los_hours),

                    "journey_archetype": journey_archetype,
                    "longitudinal_archetype": patient.get("longitudinal_archetype", None),
                    "encounter_sequence": encounter_index + 1,

                    "initial_care_mode": initial_care_mode,
                    "highest_acuity_mode": highest_acuity_mode,
                    "final_care_mode": final_care_mode,
                    "care_mode": highest_acuity_mode,
                    "transfer_pathway": transfer_pathway,
                    "transfer_pathway_text": " -> ".join(transfer_pathway),

                    "arrival_method": arrival_method,
                    "transfer_source": transfer_source,
                    "discharge_disposition": discharge_disposition,

                    "acuity_escalation_gap": int(acuity_escalation_gap),
                    "care_escalation_occurred": bool(care_escalation_occurred),
                    "escalation_delay_hours": int(escalation_delay_hours),
                    "deterioration_visibility_score": round(float(deterioration_visibility_score), 4),
                    "hidden_acuity_gap_score": round(float(hidden_acuity_gap_score), 4),

                    "readmission_flag": bool(readmission_flag),
                    "bounceback_flag": bool(bounceback_flag),
                    "failed_discharge_flag": bool(failed_discharge_flag),
                    "left_without_being_seen_flag": bool(left_without_being_seen_flag),
                    "ama_discharge_flag": bool(ama_discharge_flag),
                    "boarding_before_admission_flag": bool(boarding_before_admission_flag),
                    "care_mode_mismatch_flag": bool(care_mode_mismatch_flag),
                    "visit_cluster_flag": bool(visit_cluster_flag),
                    "quiet_gap_before_escalation_flag": bool(quiet_gap_before_escalation_flag),
                    "unexpected_escalation_flag": bool(unexpected_escalation_flag),
                    "stepdown_rebound_flag": bool(stepdown_rebound_flag),
                    "long_gap_return_flag": bool(long_gap_return_flag),

                    "seasonal_pressure": round(float(seasonal_pressure), 4),
                    "encounter_continuity_risk": round(float(encounter_continuity_risk), 4),
                    "encounter_chaos_score": round(float(encounter_chaos_score), 4),

                    "condition_profile": patient.get("condition_profile", None),
                    "presentation_profile": patient.get("presentation_profile", None),
                    "fragility_score": patient.get("fragility_score", None),
                    "weak_signal_likelihood": patient.get("weak_signal_likelihood", None),
                    "false_stability_risk": patient.get("false_stability_risk", None),
                    "misleading_presentation_risk": patient.get("misleading_presentation_risk", None),
                    "care_fragmentation_risk": patient.get("care_fragmentation_risk", None),
                    "longitudinal_memory_need": patient.get("longitudinal_memory_need", None),

                    "ten_year_instability_burden": patient.get("ten_year_instability_burden", None),
                    "mortality_vulnerability": patient.get("mortality_vulnerability", None),
                    "utilization_intensity": patient.get("utilization_intensity", None),
                    "hospital_memory_fragility": patient.get("hospital_memory_fragility", None),
                    "operational_dependency_score": patient.get("operational_dependency_score", None),
                }
            )

            previous_highest_mode = highest_acuity_mode
            previous_failed_discharge = failed_discharge_flag
            previous_final_mode = final_care_mode
            previous_discharge_disposition = discharge_disposition
            previous_encounter_end = encounter_end

            current_date = encounter_end.normalize()

    encounter_df = pd.DataFrame(encounter_rows)

    if encounter_df.empty:
        return encounter_df

    encounter_df = encounter_df.sort_values(
        ["patient_id", "encounter_start"]
    ).reset_index(drop=True)

    return encounter_df