"""
BIRE OS Synthetic Longitudinal Outcomes Generator

Chapter 53 doctrine:
No more babying BIRE OS.

Purpose:
Generate synthetic ecosystem outcomes, consequence labels,
contradiction signals, self-doubt signals, false reassurance signals,
and BIRE OS learning targets.

This module does NOT generate clinical truth.

It generates synthetic operational consequence labels used to stress-test:
- lifecycle intelligence
- hidden deterioration awareness
- intervention awareness
- recovery authenticity
- handoff reliability
- operational pressure awareness
- uncertainty handling
- self-contradiction detection
- post-event maturity
- resilient intelligence development

Outcomes are not the end of the simulation.
Outcomes are the feedback signal that tells BIRE OS what it failed to understand.

We Detect What Others Miss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bire.synthetic.config import SYNTHETIC_ECOSYSTEM_CONFIG


def _clip(value, low=0.0, high=1.0):
    return float(np.clip(value, low, high))


def _safe_float(row, col, default=0.0):
    value = row.get(col, default)
    if pd.isna(value):
        return default
    return float(value)


def _safe_bool(row, col, default=False):
    value = row.get(col, default)
    if pd.isna(value):
        return default
    return bool(value)


def _join(items):
    if not items:
        return "none"
    return " | ".join(sorted(set(items)))


def generate_longitudinal_outcomes(df, random_seed=None):
    """
    Generate synthetic longitudinal outcomes and BIRE OS self-interrogation signals.

    Parameters
    ----------
    df:
        DataFrame from operations layer.
    random_seed:
        Optional deterministic seed.

    Returns
    -------
    pd.DataFrame
        Input rows with outcome, contradiction, regret, and learning columns added.
    """

    if random_seed is None:
        random_seed = SYNTHETIC_ECOSYSTEM_CONFIG["random_seed"]

    rng = np.random.default_rng(random_seed)
    rows = []

    for _, original_row in df.iterrows():
        row = original_row.copy()

        care_mode = row.get("care_mode", "OUTPATIENT")
        post_event_state = row.get("post_event_state", "NO_EVENT")
        system_strain_state = row.get("system_strain_state", "stable")
        information_trust_state = row.get(
            "information_trust_state",
            "stable_operational_continuity",
        )
        intervention_trust_state = row.get(
            "intervention_trust_state",
            "response_uncertain",
        )
        recovery_authenticity_state = row.get(
            "recovery_authenticity_state",
            "PARTIAL_RECOVERY_UNCERTAIN",
        )

        fragility = _safe_float(row, "fragility_score", 0.0)
        deterioration = _safe_float(row, "deterioration_tendency", 0.0)
        recovery = _safe_float(row, "recovery_resilience", 0.5)

        rebound = _safe_float(row, "rebound_deterioration_risk", 0.0)
        hidden_instability = _safe_float(row, "hidden_instability_score", 0.0)
        handoff_uncertainty = _safe_float(row, "handoff_uncertainty_score", 0.0)
        handoff_deception = _safe_float(row, "handoff_deception_pressure_score", 0.0)
        continuity_risk = _safe_float(row, "longitudinal_continuity_risk", 0.0)
        memory_decay = _safe_float(row, "operational_memory_decay_score", 0.0)

        operational_instability = _safe_float(row, "operational_instability_index", 0.0)
        cascade_risk = _safe_float(row, "operational_cascade_risk", 0.0)
        hospital_failure_pressure = _safe_float(row, "hospital_system_failure_pressure", 0.0)
        resource_dependent_risk = _safe_float(row, "resource_dependent_deterioration_risk", 0.0)
        monitoring_blind_spot = _safe_float(row, "monitoring_blind_spot_score", 0.0)
        escalation_queue = _safe_float(row, "escalation_queue_saturation_score", 0.0)

        intervention_uncertainty = _safe_float(row, "intervention_uncertainty_score", 0.0)
        therapeutic_pressure = _safe_float(row, "therapeutic_system_pressure_score", 0.0)
        treatment_masking = _safe_float(row, "treatment_masking_risk", 0.0)
        stabilization_durability = _safe_float(row, "stabilization_durability_score", 0.5)

        lab_burden = _safe_float(row, "lab_abnormality_burden", 0.0)
        critical_lab_burden = _safe_float(row, "critical_lab_burden", 0.0)
        imaging_instability = _safe_float(row, "imaging_instability_score", 0.0)
        diagnosis_pressure = _safe_float(row, "diagnosis_risk_pressure_score", 0.0)

        failed_stabilization = _safe_bool(row, "failed_stabilization_flag", False)
        masked_deterioration = _safe_bool(row, "masked_deterioration_flag", False)
        treatment_dependency_loss = _safe_bool(row, "treatment_dependency_visibility_loss", False)
        delayed_escalation = _safe_bool(row, "delayed_escalation_flag", False)
        delayed_reassessment = _safe_bool(row, "delayed_reassessment_flag", False)
        operational_cascade = _safe_bool(row, "operational_cascade_flag", False)
        operational_false_reassurance = _safe_bool(row, "operational_false_reassurance_flag", False)
        icu_level_outside_icu = _safe_bool(row, "icu_level_care_outside_icu_flag", False)
        handoff_false_reassurance = _safe_bool(row, "false_reassurance_from_handoff_flag", False)
        hidden_trend_loss = _safe_bool(row, "hidden_trend_loss_flag", False)
        continuity_rescue = _safe_bool(row, "continuity_rescue_flag", False)

        # =====================================================
        # Deep consequence pressure domains
        # =====================================================

        physiologic_consequence_pressure = _clip(
            fragility * 0.16
            + deterioration * 0.18
            + hidden_instability * 0.18
            + rebound * 0.14
            + min(lab_burden, 10) * 0.020
            + min(critical_lab_burden, 5) * 0.045
            + min(imaging_instability, 10) * 0.020
            + diagnosis_pressure * 0.08
        )

        therapeutic_consequence_pressure = _clip(
            intervention_uncertainty * 0.22
            + therapeutic_pressure * 0.18
            + treatment_masking * 0.18
            + rebound * 0.14
            + (1 - stabilization_durability) * 0.14
            + float(treatment_dependency_loss) * 0.14
        )

        operational_consequence_pressure = _clip(
            operational_instability * 0.22
            + cascade_risk * 0.14
            + hospital_failure_pressure * 0.18
            + resource_dependent_risk * 0.16
            + monitoring_blind_spot * 0.12
            + escalation_queue * 0.10
            + float(icu_level_outside_icu) * 0.08
        )

        continuity_consequence_pressure = _clip(
            handoff_uncertainty * 0.22
            + handoff_deception * 0.18
            + continuity_risk * 0.18
            + memory_decay * 0.16
            + float(handoff_false_reassurance) * 0.14
            + float(hidden_trend_loss) * 0.12
        )

        # =====================================================
        # Ecosystem severity
        # =====================================================

        ecosystem_severity_score = (
            physiologic_consequence_pressure * 0.30
            + therapeutic_consequence_pressure * 0.20
            + operational_consequence_pressure * 0.22
            + continuity_consequence_pressure * 0.18
            + intervention_uncertainty * 0.06
            - recovery * 0.14
        )

        if failed_stabilization:
            ecosystem_severity_score += 0.12

        if masked_deterioration:
            ecosystem_severity_score += 0.10

        if delayed_escalation:
            ecosystem_severity_score += 0.05

        if delayed_reassessment:
            ecosystem_severity_score += 0.05

        ecosystem_severity_score += rng.normal(0, 0.035)
        ecosystem_severity_score = _clip(ecosystem_severity_score)

        # =====================================================
        # Risk signals
        # =====================================================

        readmission_risk = _clip(
            rebound * 0.28
            + operational_instability * 0.16
            + handoff_uncertainty * 0.16
            + fragility * 0.14
            + discharge_instability_score if False else 0.0
        )

        recurrent_deterioration_risk = _clip(
            deterioration * 0.24
            + rebound * 0.26
            + hidden_instability * 0.18
            + float(masked_deterioration) * 0.12
            + float(failed_stabilization) * 0.16
            + intervention_uncertainty * 0.04
        )

        recovery_quality_score = _clip(
            recovery * 0.42
            + stabilization_durability * 0.18
            - operational_instability * 0.14
            - cascade_risk * 0.10
            - intervention_uncertainty * 0.10
            - handoff_uncertainty * 0.09
            - hidden_instability * 0.08
        )

        discharge_instability_score = _clip(
            rebound * 0.22
            + hidden_instability * 0.18
            + float(masked_deterioration) * 0.14
            + handoff_uncertainty * 0.14
            + operational_instability * 0.12
            + float(treatment_dependency_loss) * 0.12
            + therapeutic_pressure * 0.08
        )

        readmission_risk = _clip(
            rebound * 0.28
            + operational_instability * 0.16
            + handoff_uncertainty * 0.16
            + fragility * 0.14
            + discharge_instability_score * 0.16
            + hospital_failure_pressure * 0.10
        )

        # =====================================================
        # Deception / contradiction outcomes
        # =====================================================

        deceived_deterioration_outcome = bool(
            (
                recovery_quality_score >= 0.35
                and hidden_instability >= 0.45
                and recurrent_deterioration_risk >= 0.45
            )
            or (
                operational_false_reassurance
                and hidden_instability >= 0.40
            )
            or handoff_false_reassurance
        )

        early_discharge_instability = bool(
            discharge_instability_score >= 0.48
            and (
                recovery_quality_score >= 0.30
                or recovery_authenticity_state in [
                    "PARTIAL_RECOVERY_UNCERTAIN",
                    "SUPPORT_DEPENDENT_STABILITY",
                    "ARTIFICIAL_RECOVERY_PATTERN",
                ]
            )
        )

        masked_physiology_failure = bool(
            treatment_masking >= 0.55
            or masked_deterioration
            or (
                therapeutic_pressure >= 0.50
                and hidden_instability >= 0.45
            )
        )

        active_deterioration_during_recovery_label = bool(
            recovery_quality_score >= 0.35
            and (
                hidden_instability >= 0.45
                or rebound >= 0.45
                or recurrent_deterioration_risk >= 0.50
            )
        )

        false_reassurance_outcome_flag = bool(
            deceived_deterioration_outcome
            or operational_false_reassurance
            or handoff_false_reassurance
            or early_discharge_instability
        )

        missed_hidden_deterioration_signal = bool(
            hidden_instability >= 0.50
            and (
                final_outcome if False else True
            )
            and (
                delayed_escalation
                or delayed_reassessment
                or monitoring_blind_spot >= 0.45
                or handoff_uncertainty >= 0.45
            )
        )

        premature_resolution_flag = bool(
            recovery_quality_score >= 0.40
            and (
                rebound >= 0.45
                or hidden_instability >= 0.45
                or discharge_instability_score >= 0.50
                or treatment_dependency_loss
            )
        )

        recovery_claim_contradicted_flag = bool(
            recovery_quality_score >= 0.35
            and (
                hidden_instability >= 0.45
                or rebound >= 0.45
                or intervention_uncertainty >= 0.45
                or handoff_deception >= 0.45
                or operational_instability >= 0.55
            )
        )

        contradictory_recovery_signature = bool(
            recovery_quality_score >= 0.35
            and (
                physiologic_consequence_pressure >= 0.50
                or therapeutic_consequence_pressure >= 0.50
                or continuity_consequence_pressure >= 0.50
            )
        )

        hidden_instability_persistence = bool(
            hidden_instability >= 0.50
            and (
                post_event_state != "NO_EVENT"
                or recovery_quality_score >= 0.30
                or rebound >= 0.40
            )
        )

        trajectory_reversal_after_recovery = bool(
            recovery_quality_score >= 0.35
            and rebound >= 0.50
            and recurrent_deterioration_risk >= 0.45
        )

        false_normalization_pattern = bool(
            recovery_quality_score >= 0.40
            and (
                hidden_instability >= 0.45
                or treatment_masking >= 0.50
                or handoff_deception >= 0.45
            )
        )

        dependency_misinterpreted_as_recovery = bool(
            recovery_authenticity_state in [
                "SUPPORT_DEPENDENT_STABILITY",
                "ARTIFICIAL_RECOVERY_PATTERN",
            ]
            or treatment_dependency_loss
            or (
                therapeutic_pressure >= 0.55
                and recovery_quality_score >= 0.35
            )
        )

        reassessment_deficit_outcome = bool(
            delayed_reassessment
            and (
                intervention_uncertainty >= 0.40
                or hidden_instability >= 0.45
                or rebound >= 0.45
            )
        )

        confidence_without_truth = bool(
            recovery_quality_score >= 0.45
            and (
                false_reassurance_outcome_flag
                or outcome_uncertainty_score if False else False
            )
        )

        # temporary uncertainty before final confidence
        preliminary_uncertainty = _clip(
            handoff_uncertainty * 0.18
            + intervention_uncertainty * 0.18
            + operational_instability * 0.14
            + cascade_risk * 0.10
            + hidden_instability * 0.14
            + float(masked_deterioration) * 0.10
            + float(false_reassurance_outcome_flag) * 0.16
        )

        confidence_without_truth = bool(
            recovery_quality_score >= 0.45
            and preliminary_uncertainty >= 0.45
        )

        silent_failure_accumulation = bool(
            (
                handoff_uncertainty >= 0.35
                + operational_instability >= 0.35
                + intervention_uncertainty >= 0.35
            )
            or (
                continuity_consequence_pressure >= 0.45
                and operational_consequence_pressure >= 0.45
            )
        )

        physiologic_operational_divergence = bool(
            (
                recovery_quality_score >= 0.40
                and operational_consequence_pressure >= 0.55
            )
            or (
                operational_instability <= 0.30
                and physiologic_consequence_pressure >= 0.55
            )
        )

        adaptive_failure_pattern = bool(
            therapeutic_pressure >= 0.50
            and hidden_instability >= 0.45
            and recovery_quality_score >= 0.30
        )

        systemically_missed_decline = bool(
            (
                hidden_instability >= 0.45
                and handoff_uncertainty >= 0.35
                and operational_instability >= 0.40
            )
            or (
                diagnosis_pressure >= 0.45
                and imaging_instability >= 4
                and delayed_escalation
            )
        )

        recovery_without_resilience = bool(
            recovery_quality_score >= 0.40
            and (
                rebound >= 0.45
                or operational_consequence_pressure >= 0.50
                or continuity_consequence_pressure >= 0.50
            )
        )

        operationally_amplified_harm = bool(
            operational_consequence_pressure >= 0.55
            or hospital_failure_pressure >= 0.55
            or resource_dependent_risk >= 0.50
        )

        late_truth_recognition = bool(
            (
                delayed_escalation
                or delayed_reassessment
                or handoff_uncertainty >= 0.50
            )
            and ecosystem_severity_score >= 0.55
        )

        stabilized_but_not_safe = bool(
            recovery_quality_score >= 0.40
            and (
                discharge_instability_score >= 0.45
                or hidden_instability >= 0.45
                or rebound >= 0.45
            )
        )

        escalation_without_resolution = bool(
            post_event_state != "NO_EVENT"
            and (
                hidden_instability >= 0.45
                or rebound >= 0.45
                or failed_stabilization
                or intervention_uncertainty >= 0.45
            )
        )

        care_pathway_fragility = bool(
            continuity_consequence_pressure >= 0.45
            and operational_consequence_pressure >= 0.45
        )

        # =====================================================
        # Outcome category / final outcome
        # =====================================================

        if ecosystem_severity_score < 0.18:
            outcome_category = "low_complexity_recovery"
            final_outcome = "discharged_home"

        elif ecosystem_severity_score < 0.35:
            outcome_category = "moderate_complexity_recovery"
            final_outcome = "stepdown_recovery"

        elif ecosystem_severity_score < 0.50:
            outcome_category = "complicated_recovery"
            final_outcome = "operationally_complicated_recovery"

        elif ecosystem_severity_score < 0.68:
            outcome_category = "high_risk_unstable_course"
            final_outcome = "long_stay"

        elif ecosystem_severity_score < 0.82:
            outcome_category = "critical_course"
            final_outcome = "icu_transfer"

        else:
            outcome_category = "severe_synthetic_outcome"
            final_outcome = (
                "simulated_mortality"
                if rng.random() < ecosystem_severity_score * 0.42
                else "failed_stabilization"
            )

        if failed_stabilization:
            final_outcome = "failed_stabilization"

        if deceived_deterioration_outcome and final_outcome in [
            "discharged_home",
            "stepdown_recovery",
            "operationally_complicated_recovery",
        ]:
            final_outcome = "deceived_deterioration"

        if early_discharge_instability and final_outcome in [
            "discharged_home",
            "stepdown_recovery",
        ]:
            final_outcome = "early_discharge_instability"

        if masked_physiology_failure and final_outcome not in [
            "simulated_mortality",
            "failed_stabilization",
        ]:
            final_outcome = "masked_physiology_failure"

        if recurrent_deterioration_risk >= 0.60 and final_outcome not in [
            "simulated_mortality",
            "failed_stabilization",
            "masked_physiology_failure",
        ]:
            final_outcome = "recurrent_deterioration"

        readmitted_30d_flag = bool(rng.random() < readmission_risk)

        if readmitted_30d_flag and final_outcome in [
            "discharged_home",
            "stepdown_recovery",
            "early_discharge_instability",
        ]:
            final_outcome = "readmitted_30d"

        # =====================================================
        # Disposition / transfers
        # =====================================================

        icu_transfer_flag = bool(
            final_outcome == "icu_transfer"
            or (
                care_mode != "ICU"
                and ecosystem_severity_score >= 0.75
            )
        )

        stepdown_transfer_flag = bool(final_outcome == "stepdown_recovery")

        if final_outcome == "discharged_home":
            discharge_disposition = "home"

        elif final_outcome == "readmitted_30d":
            discharge_disposition = "home_then_readmission"

        elif final_outcome == "stepdown_recovery":
            discharge_disposition = "stepdown_or_observation"

        elif final_outcome in [
            "long_stay",
            "operationally_complicated_recovery",
            "deceived_deterioration",
            "early_discharge_instability",
            "masked_physiology_failure",
            "recurrent_deterioration",
        ]:
            discharge_disposition = "extended_care_or_support"

        elif final_outcome in ["icu_transfer", "failed_stabilization"]:
            discharge_disposition = "higher_level_of_care"

        elif final_outcome == "simulated_mortality":
            discharge_disposition = "simulated_death"

        else:
            discharge_disposition = "uncertain_disposition"

        level_of_care_change = "none"

        if icu_transfer_flag:
            level_of_care_change = "escalated_to_icu"

        elif stepdown_transfer_flag:
            level_of_care_change = "stepped_down"

        elif final_outcome == "readmitted_30d":
            level_of_care_change = "returned_after_discharge"

        # =====================================================
        # Failure mode attribution
        # =====================================================

        failure_modes = []

        if masked_physiology_failure:
            failure_modes.append("masked_physiology_failure")

        if deceived_deterioration_outcome:
            failure_modes.append("deceived_deterioration")

        if early_discharge_instability:
            failure_modes.append("early_discharge_instability")

        if failed_stabilization:
            failure_modes.append("failed_stabilization")

        if treatment_dependency_loss or dependency_misinterpreted_as_recovery:
            failure_modes.append("treatment_dependency_misinterpreted")

        if handoff_uncertainty >= 0.45 or continuity_consequence_pressure >= 0.50:
            failure_modes.append("handoff_fragmentation")

        if operational_instability >= 0.55 or operationally_amplified_harm:
            failure_modes.append("operational_pressure_failure")

        if intervention_uncertainty >= 0.45:
            failure_modes.append("intervention_uncertainty")

        if recurrent_deterioration_risk >= 0.60:
            failure_modes.append("recurrent_deterioration_pattern")

        if systemically_missed_decline:
            failure_modes.append("systemically_missed_decline")

        if not failure_modes:
            failure_modes.append("none")

        failure_mode = _join(failure_modes)

        # =====================================================
        # Recovery trust
        # =====================================================

        recovery_trust_state = "trustworthy_recovery"

        if rebound >= 0.40:
            recovery_trust_state = "rebound_risk_present"

        if hidden_instability_persistence:
            recovery_trust_state = "hidden_instability_persistent"

        if masked_physiology_failure:
            recovery_trust_state = "masked_physiology_recovery_unreliable"

        if failed_stabilization:
            recovery_trust_state = "unstable_recovery_path"

        if treatment_dependency_loss or dependency_misinterpreted_as_recovery:
            recovery_trust_state = "treatment_dependency_uncertain"

        if information_trust_state in [
            "high_operational_fragmentation",
            "critical_operational_fragmentation",
        ]:
            recovery_trust_state = "recovery_context_fragmented"

        if recovery_claim_contradicted_flag:
            recovery_trust_state = "recovery_claim_contradicted"

        # =====================================================
        # Self-contradiction / self-doubt
        # =====================================================

        outcome_self_contradiction_score = _clip(
            float(recovery_quality_score >= 0.35) * 0.15
            + hidden_instability * 0.16
            + rebound * 0.13
            + intervention_uncertainty * 0.12
            + handoff_deception * 0.11
            + operational_false_reassurance * 0.12
            + masked_physiology_failure * 0.12
            + recovery_claim_contradicted_flag * 0.09
        )

        outcome_uncertainty_score = _clip(
            handoff_uncertainty * 0.18
            + intervention_uncertainty * 0.18
            + operational_instability * 0.14
            + cascade_risk * 0.10
            + hidden_instability * 0.14
            + float(masked_deterioration) * 0.08
            + float(false_reassurance_outcome_flag) * 0.12
            + outcome_self_contradiction_score * 0.06
        )

        outcome_confidence_score = round(_clip(1.0 - outcome_uncertainty_score), 4)

        bire_self_doubt_required_flag = bool(
            outcome_uncertainty_score >= 0.45
            or outcome_self_contradiction_score >= 0.45
            or recovery_claim_contradicted_flag
            or confidence_without_truth
            or false_reassurance_outcome_flag
        )

        confidence_without_truth = bool(
            outcome_confidence_score >= 0.55
            and (
                recovery_claim_contradicted_flag
                or hidden_instability_persistence
                or false_reassurance_outcome_flag
            )
        )

        synthetic_regret_score = round(
            _clip(
                missed_hidden_deterioration_signal * 0.16
                + delayed_escalation * 0.12
                + delayed_reassessment * 0.10
                + early_discharge_instability * 0.14
                + false_reassurance_outcome_flag * 0.14
                + operationally_amplified_harm * 0.12
                + systemically_missed_decline * 0.12
                + recovery_claim_contradicted_flag * 0.10,
                0,
                1,
            ),
            4,
        )

        synthetic_regret_signal = bool(synthetic_regret_score >= 0.35)

        preventability_pressure_score = round(
            _clip(
                delayed_escalation * 0.14
                + delayed_reassessment * 0.12
                + handoff_uncertainty * 0.14
                + operational_instability * 0.14
                + treatment_dependency_loss * 0.12
                + monitoring_blind_spot * 0.12
                + early_discharge_instability * 0.12
                + false_reassurance_outcome_flag * 0.10,
                0,
                1,
            ),
            4,
        )

        avoidable_risk_signal = bool(preventability_pressure_score >= 0.35)

        # =====================================================
        # BIRE learning signal
        # =====================================================

        upgrade_areas = []

        if missed_hidden_deterioration_signal or hidden_instability_persistence:
            upgrade_areas.append("HVI_hidden_vitals_intelligence")

        if masked_physiology_failure or dependency_misinterpreted_as_recovery:
            upgrade_areas.append("therapeutic_systems_intelligence")

        if treatment_dependency_loss or intervention_trust_state == "treatment_supported_stability":
            upgrade_areas.append("intervention_awareness")

        if handoff_uncertainty >= 0.35 or continuity_consequence_pressure >= 0.45:
            upgrade_areas.append("handoff_continuity_logic")

        if operational_instability >= 0.45 or operational_cascade or operationally_amplified_harm:
            upgrade_areas.append("hospital_operations_pressure_awareness")

        if rebound >= 0.40 or failed_stabilization or recovery_claim_contradicted_flag:
            upgrade_areas.append("post_event_recovery_maturity")

        if outcome_uncertainty_score >= 0.45 or bire_self_doubt_required_flag:
            upgrade_areas.append("uncertainty_confidence_layer")

        if recurrent_deterioration_risk >= 0.55:
            upgrade_areas.append("longitudinal_recurrence_detection")

        if early_discharge_instability:
            upgrade_areas.append("discharge_readiness_intelligence")

        if systemically_missed_decline:
            upgrade_areas.append("cross_subsystem_signal_integration")

        if confidence_without_truth:
            upgrade_areas.append("confidence_without_truth_detection")

        if not upgrade_areas:
            upgrade_areas.append("baseline_monitoring_validation")

        bire_learning_signal = _join(upgrade_areas)

        if bire_self_doubt_required_flag:
            synthetic_ground_truth_label = "self_doubt_required_outcome"
        elif false_reassurance_outcome_flag:
            synthetic_ground_truth_label = "false_reassurance_outcome"
        elif outcome_uncertainty_score >= 0.55:
            synthetic_ground_truth_label = "high_uncertainty_outcome"
        elif ecosystem_severity_score >= 0.65:
            synthetic_ground_truth_label = "high_risk_outcome"
        elif readmitted_30d_flag:
            synthetic_ground_truth_label = "readmission_outcome"
        elif final_outcome in ["discharged_home", "stepdown_recovery"]:
            synthetic_ground_truth_label = "recovery_outcome"
        else:
            synthetic_ground_truth_label = "complex_operational_outcome"

        longitudinal_complexity_score = round(
            _clip(
                ecosystem_severity_score * 0.25
                + operational_instability * 0.14
                + handoff_uncertainty * 0.12
                + intervention_uncertainty * 0.12
                + cascade_risk * 0.10
                + recurrent_deterioration_risk * 0.10
                + outcome_self_contradiction_score * 0.10
                + preventability_pressure_score * 0.07,
                0,
                1,
            ),
            4,
        )

        outcome_teaching_statement = (
            "baseline_monitoring_validated"
        )

        if synthetic_regret_signal:
            outcome_teaching_statement = (
                "More skepticism may have changed this synthetic trajectory."
            )
        elif bire_self_doubt_required_flag:
            outcome_teaching_statement = (
                "Evidence conflict required lower confidence and deeper reassessment."
            )
        elif false_reassurance_outcome_flag:
            outcome_teaching_statement = (
                "Surface reassurance contradicted deeper instability signals."
            )
        elif recovery_claim_contradicted_flag:
            outcome_teaching_statement = (
                "Recovery label conflicted with unresolved instability pressure."
            )

        # =====================================================
        # Append outputs
        # =====================================================

        row["physiologic_consequence_pressure"] = round(physiologic_consequence_pressure, 4)
        row["therapeutic_consequence_pressure"] = round(therapeutic_consequence_pressure, 4)
        row["operational_consequence_pressure"] = round(operational_consequence_pressure, 4)
        row["continuity_consequence_pressure"] = round(continuity_consequence_pressure, 4)

        row["ecosystem_severity_score"] = round(ecosystem_severity_score, 4)
        row["outcome_category"] = outcome_category
        row["final_outcome"] = final_outcome
        row["discharge_disposition"] = discharge_disposition
        row["level_of_care_change"] = level_of_care_change

        row["readmission_risk"] = round(readmission_risk, 4)
        row["readmitted_30d_flag"] = bool(readmitted_30d_flag)
        row["recurrent_deterioration_risk"] = round(recurrent_deterioration_risk, 4)
        row["recovery_quality_score"] = round(recovery_quality_score, 4)
        row["discharge_instability_score"] = round(discharge_instability_score, 4)

        row["icu_transfer_flag"] = bool(icu_transfer_flag)
        row["stepdown_transfer_flag"] = bool(stepdown_transfer_flag)
        row["discharge_instability_flag"] = bool(discharge_instability_score >= 0.45)
        row["unplanned_return_flag"] = bool(readmitted_30d_flag)

        row["deceived_deterioration_outcome"] = deceived_deterioration_outcome
        row["early_discharge_instability"] = early_discharge_instability
        row["masked_physiology_failure"] = masked_physiology_failure
        row["active_deterioration_during_recovery_label"] = active_deterioration_during_recovery_label
        row["false_reassurance_outcome_flag"] = false_reassurance_outcome_flag
        row["missed_hidden_deterioration_signal"] = missed_hidden_deterioration_signal
        row["premature_resolution_flag"] = premature_resolution_flag
        row["recovery_claim_contradicted_flag"] = recovery_claim_contradicted_flag
        row["contradictory_recovery_signature"] = contradictory_recovery_signature
        row["hidden_instability_persistence"] = hidden_instability_persistence
        row["trajectory_reversal_after_recovery"] = trajectory_reversal_after_recovery
        row["false_normalization_pattern"] = false_normalization_pattern
        row["adaptive_failure_pattern"] = adaptive_failure_pattern
        row["dependency_misinterpreted_as_recovery"] = dependency_misinterpreted_as_recovery
        row["systemically_missed_decline"] = systemically_missed_decline
        row["recovery_without_resilience"] = recovery_without_resilience
        row["reassessment_deficit_outcome"] = reassessment_deficit_outcome
        row["confidence_without_truth"] = confidence_without_truth
        row["silent_failure_accumulation"] = silent_failure_accumulation
        row["physiologic_operational_divergence"] = physiologic_operational_divergence
        row["operationally_amplified_harm"] = operationally_amplified_harm
        row["late_truth_recognition"] = late_truth_recognition
        row["stabilized_but_not_safe"] = stabilized_but_not_safe
        row["escalation_without_resolution"] = escalation_without_resolution
        row["care_pathway_fragility"] = care_pathway_fragility

        row["failure_mode"] = failure_mode
        row["avoidable_risk_signal"] = bool(avoidable_risk_signal)
        row["preventability_pressure_score"] = preventability_pressure_score
        row["recovery_trust_state"] = recovery_trust_state

        row["outcome_self_contradiction_score"] = round(outcome_self_contradiction_score, 4)
        row["outcome_confidence_score"] = outcome_confidence_score
        row["outcome_uncertainty_score"] = round(outcome_uncertainty_score, 4)
        row["bire_self_doubt_required_flag"] = bire_self_doubt_required_flag
        row["synthetic_regret_score"] = synthetic_regret_score
        row["synthetic_regret_signal"] = synthetic_regret_signal

        row["synthetic_ground_truth_label"] = synthetic_ground_truth_label
        row["bire_learning_signal"] = bire_learning_signal
        row["recommended_system_upgrade_area"] = upgrade_areas
        row["outcome_teaching_statement"] = outcome_teaching_statement

        row["longitudinal_complexity_score"] = longitudinal_complexity_score
        row["simulated_mortality_flag"] = bool(final_outcome == "simulated_mortality")

        rows.append(row)

    return pd.DataFrame(rows)