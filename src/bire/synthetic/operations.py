"""
BIRE OS Synthetic Hospital Operations Pressure Engine

Chapter 53 doctrine:
No more babying BIRE OS.

Purpose:
Simulate hospital-wide operational pressure, resource strain,
staffing instability, queue saturation, delays, throughput failure,
diagnostic bottlenecks, therapeutic access delay, monitoring blind spots,
delayed escalation, delayed reassessment, wrong-level-of-care exposure,
and operational false reassurance.

Sometimes the patient is unstable.
Sometimes the hospital is unstable around the patient.

We Detect What Others Miss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bire.synthetic.config import SYNTHETIC_ECOSYSTEM_CONFIG


HOSPITAL_STATES = [
    "stable_operations",
    "elevated_pressure",
    "high_strain",
    "critical_overload",
    "system_fragility_event",
]

RESOURCE_STATES = [
    "adequate",
    "strained",
    "limited",
    "critical_shortage",
    "system_unavailable",
]

BED_FLOW_STATES = [
    "normal_bed_flow",
    "delayed_bed_flow",
    "boarding_pressure",
    "hospital_gridlock",
    "unsafe_capacity_mismatch",
]

ESCALATION_QUEUE_STATES = [
    "queue_clear",
    "queue_active",
    "queue_delayed",
    "queue_saturated",
    "queue_failure_risk",
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


def _state_from_score(score, states):
    if score >= 0.82:
        return states[-1]
    if score >= 0.62:
        return states[-2]
    if score >= 0.40:
        return states[-3]
    if score >= 0.20:
        return states[-4]
    return states[0]


def generate_operational_pressure(df, random_seed=None):
    """
    Generate synthetic hospital operations pressure.
    """

    if random_seed is None:
        random_seed = SYNTHETIC_ECOSYSTEM_CONFIG["random_seed"]

    rng = np.random.default_rng(random_seed)
    updated_rows = []

    for row in df.to_dict("records"):
        care_mode = str(row.get("care_mode", "OUTPATIENT"))
        shift_type = str(row.get("shift_type", "day_shift"))

        fragility = _safe_float(row, "fragility_score", 0.0)
        deterioration = _safe_float(row, "deterioration_tendency", 0.0)

        handoff_uncertainty = _safe_float(row, "handoff_uncertainty_score", 0.0)
        continuity_risk = _safe_float(row, "longitudinal_continuity_risk", 0.0)
        memory_decay = _safe_float(row, "operational_memory_decay_score", 0.0)
        handoff_deception = _safe_float(row, "handoff_deception_pressure_score", 0.0)
        handoff_false_reassurance = _safe_bool(row, "false_reassurance_from_handoff_flag", False)
        hidden_trend_loss = _safe_bool(row, "hidden_trend_loss_flag", False)

        upstream_delayed_escalation = _safe_bool(row, "delayed_escalation_flag", False)
        upstream_delayed_reassessment = _safe_bool(row, "delayed_reassessment_flag", False)

        hidden_instability = _safe_float(row, "hidden_instability_score", 0.0)
        rebound_risk = _safe_float(row, "rebound_deterioration_risk", 0.0)

        intervention_uncertainty = _safe_float(row, "intervention_uncertainty_score", 0.0)
        therapeutic_pressure = _safe_float(row, "therapeutic_system_pressure_score", 0.0)
        treatment_masking_risk = _safe_float(row, "treatment_masking_risk", 0.0)
        stabilization_durability = _safe_float(row, "stabilization_durability_score", 0.5)

        diagnosis_pressure = _safe_float(row, "diagnosis_risk_pressure_score", 0.0)
        imaging_pressure = _safe_float(row, "imaging_deception_pressure_score", 0.0)
        imaging_instability = _safe_float(row, "imaging_instability_score", 0.0)
        lab_pressure = _safe_float(row, "lab_sixth_sense_pressure_score", 0.0)
        data_trust_pressure = _safe_float(row, "data_trust_pressure_score", 0.0)

        provider_change_count = _safe_int(row, "provider_change_count", 0)
        consult_service_count = _safe_int(row, "consult_service_count", 0)

        handoff_attention_required = _safe_bool(row, "handoff_attention_required_flag", False)

        hospital_state = rng.choice(
            HOSPITAL_STATES,
            p=[0.36, 0.29, 0.21, 0.10, 0.04],
        )

        hospital_pressure_map = {
            "stable_operations": 0.10,
            "elevated_pressure": 0.35,
            "high_strain": 0.65,
            "critical_overload": 0.88,
            "system_fragility_event": 0.96,
        }

        hospital_pressure_score = hospital_pressure_map[hospital_state]

        if shift_type in ["night_shift", "weekend_shift", "holiday_shift", "overnight_cross_cover"]:
            hospital_pressure_score = _clip(hospital_pressure_score + 0.06)

        if shift_type == "shift_change_window":
            hospital_pressure_score = _clip(hospital_pressure_score + 0.08)

        hospital_pressure_score = _clip(
            hospital_pressure_score
            + continuity_risk * 0.04
            + handoff_uncertainty * 0.03
            + rng.normal(0, 0.025),
            0,
            1,
        )

        icu_capacity_utilization = round(
            float(np.clip(rng.normal(0.76 + hospital_pressure_score * 0.25, 0.09), 0.30, 1.40)),
            4,
        )
        icu_overflow_flag = bool(icu_capacity_utilization >= 1.0)

        er_census_pressure = round(
            float(np.clip(rng.normal(0.50 + hospital_pressure_score * 0.42, 0.12), 0, 1.65)),
            4,
        )
        er_overcrowding_flag = bool(er_census_pressure >= 0.85)

        bed_availability_score = round(
            _clip(
                1.0
                - hospital_pressure_score * 0.75
                - float(er_overcrowding_flag) * 0.08
                - float(icu_overflow_flag) * 0.10
                - continuity_risk * 0.05
                + rng.normal(0, 0.05),
                0,
                1,
            ),
            4,
        )
        bed_shortage_flag = bool(bed_availability_score <= 0.25)

        bed_flow_pressure_score = round(
            _clip(
                hospital_pressure_score * 0.34
                + er_census_pressure * 0.18
                + icu_capacity_utilization * 0.20
                + (1 - bed_availability_score) * 0.22
                + continuity_risk * 0.06,
                0,
                1,
            ),
            4,
        )
        bed_flow_state = _state_from_score(bed_flow_pressure_score, BED_FLOW_STATES)

        wrong_level_of_care_probability = _clip(
            0.02
            + bed_flow_pressure_score * 0.22
            + float(icu_overflow_flag) * 0.18
            + float("ER" in care_mode) * 0.06
            + fragility * 0.05
            + hidden_instability * 0.06
            + rebound_risk * 0.05,
            0,
            0.85,
        )

        icu_level_care_outside_icu_flag = bool(
            care_mode != "ICU"
            and (
                hidden_instability >= 0.55
                or rebound_risk >= 0.55
                or fragility >= 0.70
                or deterioration >= 0.65
            )
            and rng.random() < wrong_level_of_care_probability
        )

        hallway_care_probability = _clip(
            0.01
            + float(er_overcrowding_flag) * 0.20
            + bed_flow_pressure_score * 0.13
            + float("ER" in care_mode) * 0.12,
            0,
            0.80,
        )
        hallway_care_flag = bool(rng.random() < hallway_care_probability)

        if icu_level_care_outside_icu_flag:
            care_location_state = "wrong_level_of_care"
        elif hallway_care_flag:
            care_location_state = "hallway_care"
        elif bed_flow_state in ["hospital_gridlock", "unsafe_capacity_mismatch"]:
            care_location_state = str(rng.choice(["boarding_area", "overflow_unit"], p=[0.55, 0.45]))
        else:
            care_location_state = "standard_care_location"

        boarding_delay_minutes = 0
        boarding_probability = _clip(
            0.04
            + hospital_pressure_score * 0.35
            + bed_flow_pressure_score * 0.25
            + float("ER" in care_mode) * 0.12,
            0,
            0.90,
        )

        if "ER" in care_mode and rng.random() < boarding_probability:
            boarding_delay_minutes = int(rng.choice([60, 120, 240, 480, 720, 960]))

        prolonged_boarding_flag = bool(boarding_delay_minutes >= 240)

        discharge_bottleneck_score = round(
            _clip(
                hospital_pressure_score * 0.55
                + bed_flow_pressure_score * 0.25
                + memory_decay * 0.08
                + rng.normal(0, 0.08),
                0,
                1,
            ),
            4,
        )
        discharge_delay_flag = bool(discharge_bottleneck_score >= 0.65)

        environmental_services_delay_minutes = int(
            max(0, rng.normal(15 + hospital_pressure_score * 95 + discharge_bottleneck_score * 45, 20))
        )
        bed_turnover_delay_flag = bool(environmental_services_delay_minutes >= 75)

        patient_placement_delay_minutes = int(max(0, rng.normal(25 + bed_flow_pressure_score * 180, 35)))
        patient_placement_bottleneck_flag = bool(patient_placement_delay_minutes >= 120)

        nurse_patient_ratio = round(
            float(np.clip(rng.normal(3.5 + hospital_pressure_score * 4.5, 1.3), 1, 15)),
            2,
        )
        unsafe_nursing_ratio_flag = bool(nurse_patient_ratio >= 7)

        charge_nurse_overload_score = round(
            _clip(
                hospital_pressure_score * 0.45
                + bed_flow_pressure_score * 0.20
                + continuity_risk * 0.15
                + rng.normal(0, 0.06),
                0,
                1,
            ),
            4,
        )

        cross_cover_unfamiliarity_score = round(
            _clip(
                float(shift_type in ["night_shift", "overnight_cross_cover"]) * 0.20
                + provider_change_count * 0.035
                + consult_service_count * 0.025
                + memory_decay * 0.20,
                0,
                1,
            ),
            4,
        )

        nurse_break_coverage_gap_flag = bool(
            rng.random()
            < _clip(
                0.03
                + hospital_pressure_score * 0.18
                + float(unsafe_nursing_ratio_flag) * 0.12,
                0,
                0.75,
            )
        )

        operational_fatigue_score = round(
            _clip(
                hospital_pressure_score * 0.42
                + continuity_risk * 0.16
                + handoff_uncertainty * 0.14
                + charge_nurse_overload_score * 0.13
                + cross_cover_unfamiliarity_score * 0.10
                + float(prolonged_boarding_flag) * 0.05,
                0,
                1,
            ),
            4,
        )

        documentation_backlog_score = round(
            _clip(
                hospital_pressure_score * 0.48
                + operational_fatigue_score * 0.25
                + handoff_uncertainty * 0.10
                + data_trust_pressure * 0.08
                + rng.normal(0, 0.05),
                0,
                1,
            ),
            4,
        )
        documentation_delay_flag = bool(documentation_backlog_score >= 0.60)

        imaging_backlog_minutes = int(max(0, rng.normal(25 + hospital_pressure_score * 140 + imaging_pressure * 90, 30)))
        transport_delay_minutes = int(max(0, rng.normal(8 + hospital_pressure_score * 55 + bed_flow_pressure_score * 30, 14)))
        transport_instability_flag = bool(transport_delay_minutes >= 45)
        ct_mri_transport_bottleneck_flag = bool(imaging_backlog_minutes >= 120 or transport_delay_minutes >= 60)

        lab_processing_backlog_minutes = int(
            max(0, rng.normal(12 + hospital_pressure_score * 95 + data_trust_pressure * 60 + lab_pressure * 35, 22))
        )
        critical_lab_delay_flag = bool(lab_processing_backlog_minutes >= 60)

        lab_specimen_lost_flag = bool(
            rng.random()
            < _clip(
                0.01
                + hospital_pressure_score * 0.05
                + documentation_backlog_score * 0.04
                + data_trust_pressure * 0.03,
                0,
                0.25,
            )
        )

        redraw_required_flag = bool(
            lab_specimen_lost_flag
            or rng.random()
            < _clip(
                0.015
                + hospital_pressure_score * 0.04
                + data_trust_pressure * 0.05,
                0,
                0.20,
            )
        )

        redraw_delay_minutes = 0
        if redraw_required_flag:
            redraw_delay_minutes = int(rng.choice([30, 60, 90, 120, 180]))

        diagnostic_bottleneck_score = round(
            _clip(
                imaging_backlog_minutes / 360 * 0.25
                + lab_processing_backlog_minutes / 240 * 0.23
                + transport_delay_minutes / 180 * 0.14
                + redraw_delay_minutes / 240 * 0.14
                + imaging_instability * 0.015
                + imaging_pressure * 0.09
                + lab_pressure * 0.08
                + data_trust_pressure * 0.07,
                0,
                1,
            ),
            4,
        )

        pharmacy_verification_delay_minutes = int(
            max(0, rng.normal(5 + hospital_pressure_score * 50 + therapeutic_pressure * 35, 12))
        )
        delayed_med_verification_flag = bool(pharmacy_verification_delay_minutes >= 30)

        medication_shortage_flag = bool(
            rng.random()
            < _clip(
                0.015
                + hospital_pressure_score * 0.12
                + therapeutic_pressure * 0.06,
                0,
                0.50,
            )
        )

        medication_substitution_delay_minutes = 0
        if medication_shortage_flag:
            medication_substitution_delay_minutes = int(rng.choice([30, 60, 120, 240]))

        operational_medication_delay = bool(
            rng.random()
            < _clip(
                0.05 + hospital_pressure_score * 0.30 + therapeutic_pressure * 0.08,
                0,
                0.80,
            )
        )

        operational_medication_delay_minutes = 0
        if operational_medication_delay:
            operational_medication_delay_minutes = int(rng.choice([15, 30, 60, 120, 240]))

        therapeutic_access_delay_score = round(
            _clip(
                pharmacy_verification_delay_minutes / 180 * 0.24
                + operational_medication_delay_minutes / 240 * 0.24
                + medication_substitution_delay_minutes / 240 * 0.18
                + float(medication_shortage_flag) * 0.14
                + therapeutic_pressure * 0.14
                + treatment_masking_risk * 0.06,
                0,
                1,
            ),
            4,
        )

        respiratory_therapy_availability = round(
            _clip(
                1.0
                - hospital_pressure_score * 0.55
                - hidden_instability * 0.05
                - therapeutic_pressure * 0.04
                + rng.normal(0, 0.08),
                0,
                1,
            ),
            4,
        )
        rt_shortage_flag = bool(respiratory_therapy_availability <= 0.35)

        rapid_response_team_availability = round(
            _clip(
                1.0
                - hospital_pressure_score * 0.65
                - er_census_pressure * 0.08
                - handoff_attention_required * 0.05
                + rng.normal(0, 0.10),
                0,
                1,
            ),
            4,
        )
        rrt_saturation_flag = bool(rapid_response_team_availability <= 0.30)

        consult_service_delay_minutes = int(
            max(0, rng.normal(20 + hospital_pressure_score * 110 + consult_service_count * 12, 30))
        )
        consult_delay_flag = bool(consult_service_delay_minutes >= 90)

        procedure_suite_delay_minutes = int(
            max(0, rng.normal(10 + hospital_pressure_score * 95 + diagnosis_pressure * 60, 25))
        )
        procedure_delay_flag = bool(procedure_suite_delay_minutes >= 90)

        ems_arrival_surge_count = int(max(0, rng.normal(1 + hospital_pressure_score * 8, 2)))

        ambulance_diversion_pressure = round(
            _clip(
                hospital_pressure_score * 0.50
                + er_census_pressure * 0.22
                + bed_flow_pressure_score * 0.18
                + ems_arrival_surge_count * 0.025,
                0,
                1,
            ),
            4,
        )
        ambulance_diversion_flag = bool(ambulance_diversion_pressure >= 0.75)

        concurrent_emergency_load = int(max(0, rng.normal(1 + hospital_pressure_score * 7, 1.8)))
        multi_emergency_strain_flag = bool(concurrent_emergency_load >= 4)

        or_emergency_diversion_pressure = round(
            _clip(
                hospital_pressure_score * 0.35
                + concurrent_emergency_load * 0.04
                + float(procedure_delay_flag) * 0.12,
                0,
                1,
            ),
            4,
        )

        device_shortage_flag = bool(
            rng.random() < _clip(0.02 + hospital_pressure_score * 0.18, 0, 0.60)
        )

        telemetry_monitor_overload_score = round(
            _clip(
                hospital_pressure_score * 0.35
                + er_census_pressure * 0.12
                + float(unsafe_nursing_ratio_flag) * 0.12
                + documentation_backlog_score * 0.10
                + hidden_instability * 0.08,
                0,
                1,
            ),
            4,
        )
        telemetry_overload_flag = bool(telemetry_monitor_overload_score >= 0.65)

        isolation_room_shortage_flag = bool(
            rng.random()
            < _clip(
                0.02 + hospital_pressure_score * 0.12 + float(bed_shortage_flag) * 0.08,
                0,
                0.55,
            )
        )

        monitoring_reliability_score = round(
            _clip(
                1.0
                - hospital_pressure_score * 0.26
                - handoff_uncertainty * 0.18
                - telemetry_monitor_overload_score * 0.16
                - float(unsafe_nursing_ratio_flag) * 0.08
                - float(hallway_care_flag) * 0.08
                - data_trust_pressure * 0.08,
                0,
                1,
            ),
            4,
        )

        if shift_type == "night_shift":
            monitoring_reliability_score = round(_clip(monitoring_reliability_score - 0.08), 4)

        if shift_type == "weekend_shift":
            monitoring_reliability_score = round(_clip(monitoring_reliability_score - 0.05), 4)

        monitoring_blind_spot_score = round(
            _clip(
                (1 - monitoring_reliability_score) * 0.28
                + telemetry_monitor_overload_score * 0.18
                + documentation_backlog_score * 0.14
                + float(hallway_care_flag) * 0.10
                + float(nurse_break_coverage_gap_flag) * 0.08
                + handoff_deception * 0.10
                + float(hidden_trend_loss) * 0.06
                + float(handoff_false_reassurance) * 0.06,
                0,
                1,
            ),
            4,
        )

        escalation_queue_saturation_score = round(
            _clip(
                hospital_pressure_score * 0.28
                + float(rrt_saturation_flag) * 0.18
                + float(multi_emergency_strain_flag) * 0.15
                + float(handoff_attention_required) * 0.10
                + float(telemetry_overload_flag) * 0.10
                + charge_nurse_overload_score * 0.10
                + continuity_risk * 0.09,
                0,
                1,
            ),
            4,
        )

        escalation_queue_state = _state_from_score(
            escalation_queue_saturation_score,
            ESCALATION_QUEUE_STATES,
        )

        escalation_latency_minutes = int(
            max(
                0,
                rng.normal(
                    5
                    + hospital_pressure_score * 50
                    + continuity_risk * 25
                    + escalation_queue_saturation_score * 60,
                    12,
                ),
            )
        )

        delayed_escalation_flag = bool(
            upstream_delayed_escalation
            or escalation_latency_minutes >= 30
            or (
                escalation_queue_saturation_score >= 0.55
                and hidden_instability >= 0.45
            )
        )

        reassessment_delay_minutes = int(
            max(
                0,
                rng.normal(
                    10
                    + hospital_pressure_score * 95
                    + intervention_uncertainty * 45
                    + escalation_queue_saturation_score * 30,
                    18,
                ),
            )
        )

        delayed_reassessment_flag = bool(
            upstream_delayed_reassessment
            or reassessment_delay_minutes >= 45
            or (
                monitoring_blind_spot_score >= 0.45
                and intervention_uncertainty >= 0.40
            )
            or (
                handoff_false_reassurance
                and hidden_instability >= 0.40
            )
        )

        reassessment_after_intervention_gap_flag = bool(
            intervention_uncertainty >= 0.40
            and reassessment_delay_minutes >= 60
        )

        alert_queue_congestion_score = round(
            _clip(
                escalation_queue_saturation_score * 0.35
                + telemetry_monitor_overload_score * 0.20
                + hospital_pressure_score * 0.20
                + float(handoff_attention_required) * 0.10
                + monitoring_blind_spot_score * 0.15,
                0,
                1,
            ),
            4,
        )

        surveillance_fatigue_score = round(
            _clip(
                alert_queue_congestion_score * 0.30
                + operational_fatigue_score * 0.25
                + documentation_backlog_score * 0.20
                + nurse_patient_ratio / 14 * 0.15
                + memory_decay * 0.10,
                0,
                1,
            ),
            4,
        )

        operational_false_reassurance_flag = bool(
            (
                monitoring_blind_spot_score >= 0.45
                and hidden_instability >= 0.45
            )
            or (
                documentation_delay_flag
                and handoff_deception >= 0.45
            )
            or (
                delayed_reassessment_flag
                and intervention_uncertainty >= 0.40
            )
            or (
                delayed_escalation_flag
                and hidden_instability >= 0.45
            )
            or (
                care_location_state in ["hallway_care", "wrong_level_of_care"]
                and fragility >= 0.60
            )
            or (
                handoff_false_reassurance
                and monitoring_blind_spot_score >= 0.35
            )
            or (
                treatment_masking_risk >= 0.50
                and stabilization_durability <= 0.45
            )
        )

        resource_constraint_probs = np.array(
            [
                0.48 - hospital_pressure_score * 0.20,
                0.28,
                0.16 + hospital_pressure_score * 0.10,
                0.06 + hospital_pressure_score * 0.08,
                0.02 + hospital_pressure_score * 0.04,
            ]
        )
        resource_constraint_probs = np.clip(resource_constraint_probs, 0.01, None)
        resource_constraint_probs = resource_constraint_probs / resource_constraint_probs.sum()

        resource_constraint_state = str(rng.choice(RESOURCE_STATES, p=resource_constraint_probs))

        operational_cascade_risk = round(
            _clip(
                hospital_pressure_score * 0.20
                + operational_fatigue_score * 0.13
                + handoff_uncertainty * 0.10
                + documentation_backlog_score * 0.08
                + discharge_bottleneck_score * 0.07
                + escalation_queue_saturation_score * 0.13
                + monitoring_blind_spot_score * 0.10
                + diagnostic_bottleneck_score * 0.08
                + therapeutic_access_delay_score * 0.06
                + float(multi_emergency_strain_flag) * 0.05,
                0,
                1,
            ),
            4,
        )
        operational_cascade_flag = bool(operational_cascade_risk >= 0.70)

        resource_dependent_deterioration_risk = round(
            _clip(
                hidden_instability * 0.17
                + rebound_risk * 0.14
                + therapeutic_access_delay_score * 0.15
                + diagnostic_bottleneck_score * 0.14
                + monitoring_blind_spot_score * 0.14
                + escalation_queue_saturation_score * 0.11
                + float(icu_level_care_outside_icu_flag) * 0.10
                + float(delayed_reassessment_flag) * 0.08
                + float(delayed_escalation_flag) * 0.07,
                0,
                1,
            ),
            4,
        )

        hospital_system_failure_pressure = round(
            _clip(
                operational_cascade_risk * 0.22
                + bed_flow_pressure_score * 0.16
                + escalation_queue_saturation_score * 0.16
                + monitoring_blind_spot_score * 0.14
                + diagnostic_bottleneck_score * 0.10
                + therapeutic_access_delay_score * 0.08
                + float(operational_false_reassurance_flag) * 0.08
                + float(delayed_reassessment_flag) * 0.03
                + float(delayed_escalation_flag) * 0.03,
                0,
                1,
            ),
            4,
        )

        operational_instability_index = round(
            _clip(
                hospital_pressure_score * 0.18
                + continuity_risk * 0.12
                + handoff_uncertainty * 0.10
                + operational_fatigue_score * 0.12
                + operational_cascade_risk * 0.20
                + resource_dependent_deterioration_risk * 0.15
                + hospital_system_failure_pressure * 0.13,
                0,
                1,
            ),
            4,
        )

        if operational_instability_index >= 0.82:
            system_strain_state = "critical_overload"
        elif operational_instability_index >= 0.60:
            system_strain_state = "high_strain"
        elif operational_instability_index >= 0.35:
            system_strain_state = "strained"
        else:
            system_strain_state = "stable"

        bire_operations_attention_required_flag = bool(
            operational_instability_index >= 0.55
            or hospital_system_failure_pressure >= 0.55
            or operational_false_reassurance_flag
            or icu_level_care_outside_icu_flag
            or escalation_queue_saturation_score >= 0.60
        )

        row["hospital_state"] = hospital_state
        row["hospital_pressure_score"] = round(float(hospital_pressure_score), 4)

        row["icu_capacity_utilization"] = icu_capacity_utilization
        row["icu_overflow_flag"] = icu_overflow_flag

        row["er_census_pressure"] = er_census_pressure
        row["er_overcrowding_flag"] = er_overcrowding_flag

        row["bed_availability_score"] = bed_availability_score
        row["bed_shortage_flag"] = bed_shortage_flag
        row["bed_flow_pressure_score"] = bed_flow_pressure_score
        row["bed_flow_state"] = bed_flow_state

        row["care_location_state"] = care_location_state
        row["hallway_care_flag"] = hallway_care_flag
        row["icu_level_care_outside_icu_flag"] = icu_level_care_outside_icu_flag

        row["boarding_delay_minutes"] = boarding_delay_minutes
        row["prolonged_boarding_flag"] = prolonged_boarding_flag
        row["discharge_bottleneck_score"] = discharge_bottleneck_score
        row["discharge_delay_flag"] = discharge_delay_flag
        row["environmental_services_delay_minutes"] = environmental_services_delay_minutes
        row["bed_turnover_delay_flag"] = bed_turnover_delay_flag
        row["patient_placement_delay_minutes"] = patient_placement_delay_minutes
        row["patient_placement_bottleneck_flag"] = patient_placement_bottleneck_flag

        row["nurse_patient_ratio"] = nurse_patient_ratio
        row["unsafe_nursing_ratio_flag"] = unsafe_nursing_ratio_flag
        row["charge_nurse_overload_score"] = charge_nurse_overload_score
        row["cross_cover_unfamiliarity_score"] = cross_cover_unfamiliarity_score
        row["nurse_break_coverage_gap_flag"] = nurse_break_coverage_gap_flag
        row["operational_fatigue_score"] = operational_fatigue_score

        row["imaging_backlog_minutes"] = imaging_backlog_minutes
        row["transport_delay_minutes"] = transport_delay_minutes
        row["transport_instability_flag"] = transport_instability_flag
        row["ct_mri_transport_bottleneck_flag"] = ct_mri_transport_bottleneck_flag
        row["lab_processing_backlog_minutes"] = lab_processing_backlog_minutes
        row["critical_lab_delay_flag"] = critical_lab_delay_flag
        row["lab_specimen_lost_flag"] = lab_specimen_lost_flag
        row["redraw_required_flag"] = redraw_required_flag
        row["redraw_delay_minutes"] = redraw_delay_minutes
        row["diagnostic_bottleneck_score"] = diagnostic_bottleneck_score

        row["pharmacy_verification_delay_minutes"] = pharmacy_verification_delay_minutes
        row["delayed_med_verification_flag"] = delayed_med_verification_flag
        row["medication_shortage_flag"] = medication_shortage_flag
        row["medication_substitution_delay_minutes"] = medication_substitution_delay_minutes
        row["operational_medication_delay"] = operational_medication_delay
        row["operational_medication_delay_minutes"] = operational_medication_delay_minutes
        row["therapeutic_access_delay_score"] = therapeutic_access_delay_score

        row["respiratory_therapy_availability"] = respiratory_therapy_availability
        row["rt_shortage_flag"] = rt_shortage_flag
        row["rapid_response_team_availability"] = rapid_response_team_availability
        row["rrt_saturation_flag"] = rrt_saturation_flag
        row["consult_service_delay_minutes"] = consult_service_delay_minutes
        row["consult_delay_flag"] = consult_delay_flag
        row["procedure_suite_delay_minutes"] = procedure_suite_delay_minutes
        row["procedure_delay_flag"] = procedure_delay_flag
        row["or_emergency_diversion_pressure"] = or_emergency_diversion_pressure

        row["ems_arrival_surge_count"] = ems_arrival_surge_count
        row["ambulance_diversion_pressure"] = ambulance_diversion_pressure
        row["ambulance_diversion_flag"] = ambulance_diversion_flag
        row["concurrent_emergency_load"] = concurrent_emergency_load
        row["multi_emergency_strain_flag"] = multi_emergency_strain_flag

        row["device_shortage_flag"] = device_shortage_flag
        row["telemetry_monitor_overload_score"] = telemetry_monitor_overload_score
        row["telemetry_overload_flag"] = telemetry_overload_flag
        row["isolation_room_shortage_flag"] = isolation_room_shortage_flag
        row["monitoring_reliability_score"] = monitoring_reliability_score
        row["monitoring_blind_spot_score"] = monitoring_blind_spot_score

        row["documentation_backlog_score"] = documentation_backlog_score
        row["documentation_delay_flag"] = documentation_delay_flag

        row["escalation_queue_saturation_score"] = escalation_queue_saturation_score
        row["escalation_queue_state"] = escalation_queue_state
        row["escalation_latency_minutes"] = escalation_latency_minutes
        row["delayed_escalation_flag"] = delayed_escalation_flag
        row["reassessment_delay_minutes"] = reassessment_delay_minutes
        row["delayed_reassessment_flag"] = delayed_reassessment_flag
        row["reassessment_after_intervention_gap_flag"] = reassessment_after_intervention_gap_flag
        row["alert_queue_congestion_score"] = alert_queue_congestion_score
        row["surveillance_fatigue_score"] = surveillance_fatigue_score

        row["resource_constraint_state"] = resource_constraint_state
        row["operational_false_reassurance_flag"] = operational_false_reassurance_flag
        row["operational_cascade_risk"] = operational_cascade_risk
        row["operational_cascade_flag"] = operational_cascade_flag
        row["resource_dependent_deterioration_risk"] = resource_dependent_deterioration_risk
        row["hospital_system_failure_pressure"] = hospital_system_failure_pressure
        row["operational_instability_index"] = operational_instability_index
        row["system_strain_state"] = system_strain_state
        row["bire_operations_attention_required_flag"] = bire_operations_attention_required_flag

        updated_rows.append(row)

    return pd.DataFrame(updated_rows)