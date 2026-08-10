# ============================================================
# BIRE OS — Trajectory-Aware Recovery Intelligence (TARI)
# File: src/bire/tari/policy.py
# Chapter: 66
#
# Created: 2026-07-03
# Updated: 2026-07-03
#
# CHANGED:
# - Initialized TARI policy module
# - Established recovery governance policies
# - Added authority and responsibility definitions
# - Prepared governance framework for recovery trust
#
# Purpose:
# Backend helper for TARI governance policies,
# authority boundaries, subsystem responsibilities,
# and recovery trust governance definitions.
# ============================================================

# ============================================================
# TARI — Recovery Governance Policy
# ============================================================

import pandas as pd

TARI_POLICY = {

    # Core Purpose
    "govern_recovery": True,

    # Core Responsibilities
    "validate_recovery": True,
    "determine_recovery_trust": True,
    "accumulate_recovery_evidence": True,

    # Explicit Non-Responsibilities
    "measure_physiology": False,
    "interpret_trajectory": False,
    "forecast_patient_outcome": False,
    "retain_longitudinal_memory": False,
    "calculate_system_confidence": False,

    # Identity
    "highest_authority": "RECOVERY_GOVERNANCE",

    "core_question":
        "Has this patient's recovery earned trust?",

    "core_doctrine":
        "RECOVERY_IS_EARNED",
}


def build_tari_policy_table():
    """
    Build TARI policy table.
    """


    rows = []

    for key, value in TARI_POLICY.items():

        rows.append({
            "policy": key,
            "value": value,
        })

    return pd.DataFrame(rows)

TARI_GOVERNANCE = {

    "govern_recovery": True,
    "validate_recovery": True,
    "evaluate_recovery_evidence": True,
    "determine_recovery_trust": True,

    "measure_physiology": False,
    "interpret_trajectory": False,
    "forecast_patient_outcome": False,
    "retain_longitudinal_memory": False,
    "calculate_system_confidence": False,

    "highest_authority":
        "RECOVERY_GOVERNANCE",

    "core_question":
        "Has this patient's recovery earned trust?"
}


def build_tari_governance_table():
    """
    Build TARI governance table.
    """

    rows = []

    for key, value in TARI_GOVERNANCE.items():

        rows.append({
            "governance": key,
            "value": value,
        })

    return pd.DataFrame(rows)

TARI_RECOVERY_GOVERNANCE_POLICY = {

    # Recovery Governance
    "recovery_is_earned": True,
    "require_accumulated_evidence": True,
    "require_longitudinal_support": True,
    "require_multi_layer_agreement": True,

    # Recovery Protection
    "allow_single_observation_recovery": False,
    "allow_single_vital_recovery": False,
    "allow_assumed_recovery": False,
    "allow_premature_recovery": False,

    # Governance
    "govern_recovery": True,
    "govern_recovery_trust": True,

    # Philosophy
    "policy_question":
        "Has sufficient evidence accumulated to trust recovery?"
}


def build_tari_recovery_governance_policy_table():
    """
    Build TARI recovery governance policy table.
    """

    rows = []

    for key, value in (
        TARI_RECOVERY_GOVERNANCE_POLICY.items()
    ):

        rows.append({
            "policy": key,
            "value": value,
        })

    return pd.DataFrame(rows)

TARI_AUTHORITY_BOUNDARY = {

    # Recovery Governance
    "govern_recovery": True,
    "govern_recovery_trust": True,
    "approve_recovery_confidence": True,

    # Explicit Non-Authorities
    "measure_physiology": False,
    "interpret_trajectory": False,
    "forecast_patient_outcome": False,
    "retain_longitudinal_memory": False,
    "calculate_system_confidence": False,
    "recommend_treatment": False,
    "perform_diagnosis": False,

    # Authority
    "highest_authority":
        "RECOVERY_GOVERNANCE",

    "core_authority_question":
        "Has recovery earned sufficient evidence to be trusted?"
}


def build_tari_authority_boundary_table():
    """
    Build TARI authority boundary table.
    """

    rows = []

    for key, value in (
        TARI_AUTHORITY_BOUNDARY.items()
    ):

        rows.append({
            "authority_boundary": key,
            "value": value,
        })

    return pd.DataFrame(rows)

TARI_INPUT_POLICY = {

    # Primary Intelligence Sources
    "consume_ati_outputs": True,
    "consume_hvi_outputs": True,
    "consume_lpmr_outputs": True,
    "consume_ibpip_outputs": True,
    "consume_confidence_engine_outputs": True,

    # Explicit Non-Inputs
    "consume_raw_vitals": False,
    "consume_raw_laboratory_data": False,
    "consume_raw_imaging": False,
    "consume_raw_notes": False,
    "consume_raw_measurements": False,

    # Identity
    "input_question":
        "What completed intelligence contributes to recovery trust?"
}


def build_tari_input_policy_table():
    """
    Build TARI input policy table.
    """

    rows = []

    for key, value in (
        TARI_INPUT_POLICY.items()
    ):

        rows.append({
            "input_policy": key,
            "value": value,
        })

    return pd.DataFrame(rows)

TARI_OUTPUT_POLICY = {

    # Primary Outputs
    "produce_recovery_trust": True,
    "produce_recovery_governance": True,
    "produce_recovery_validation": True,
    "produce_recovery_trust_state": True,

    # Explicit Non-Outputs
    "produce_measurements": False,
    "produce_trajectory_interpretations": False,
    "produce_patient_forecasts": False,
    "produce_longitudinal_memory": False,
    "produce_system_confidence": False,
    "produce_treatment_recommendations": False,
    "produce_diagnosis": False,

    # Identity
    "output_question":
        "What recovery governance decisions should TARI communicate?"
}


TARI_CONSUMER_POLICY = {

    # Primary Consumers
    "ibpip_can_consume_tari": True,
    "psrv2_can_display_tari_when_authorized": True,
    "future_discharge_intelligence_can_consume_tari": True,
    "future_specialty_recovery_can_consume_tari": True,

    # Explicit Non-Authorities
    "tari_controls_consumers": False,
    "tari_overrides_consumers": False,
    "tari_performs_consumer_tasks": False,

    # Identity
    "consumer_question":
        "Which subsystems may utilize recovery governance decisions?"
}


def build_tari_consumer_policy_table():
    """
    Build TARI consumer policy table.
    """

    rows = []

    for key, value in (
        TARI_CONSUMER_POLICY.items()
    ):

        rows.append({
            "consumer_policy": key,
            "value": value,
        })

    return pd.DataFrame(rows)

TARI_SUBSYSTEM_RELATIONSHIPS = {

    "HVI":
        "Provides hidden physiologic recovery evidence, recovery authenticity, recovery resilience, and hidden instability.",

    "ATI":
        "Provides recovery trajectory behavior, trajectory direction, recovery momentum, and trajectory interpretation.",

    "LPMR":
        "Provides longitudinal patient experience, recovery history, previous false recoveries, and patient-specific recovery patterns.",

    "IBPIP":
        "Provides physiologic interpretation, compensation reserve, patient context, and overall recovery context.",

    "CONFIDENCE_ENGINE":
        "Provides subsystem agreement, recovery confidence, and overall system confidence.",

    "PSRV2":
        "Displays TARI recovery governance outputs when authorized."
}


def build_tari_subsystem_relationships_table():
    """
    Build TARI subsystem relationship table.
    """

    rows = []

    for subsystem, relationship in (
        TARI_SUBSYSTEM_RELATIONSHIPS.items()
    ):

        rows.append({
            "subsystem": subsystem,
            "relationship": relationship,
        })

    return pd.DataFrame(rows)

TARI_GOVERNANCE_DOMAINS = {

    "RECOVERY_AUTHENTICITY":
        "Authenticity of recovery evidence.",

    "RECOVERY_DURABILITY":
        "Durability of sustained recovery.",

    "RECOVERY_RESILIENCE":
        "Ability of recovery to survive physiologic stress.",

    "RECOVERY_STABILITY":
        "Stability of recovery over longitudinal observation.",

    "RECOVERY_CONSISTENCY":
        "Agreement across intelligence layers regarding recovery.",

    "RECOVERY_TRUST":
        "Overall recovery trust accumulated from supporting evidence.",
}


def build_tari_governance_domains_table():
    """
    Build TARI governance domains table.
    """

    rows = []

    for domain, description in (
        TARI_GOVERNANCE_DOMAINS.items()
    ):

        rows.append({
            "domain": domain,
            "description": description,
            "status": "TARI_DOMAIN",
        })

    return pd.DataFrame(rows)


TARI_RECOVERY_TRUST_FORMATION_POLICY = {

    # Recovery Trust Formation
    "recovery_is_earned": True,
    "require_accumulated_evidence": True,
    "require_longitudinal_support": True,
    "require_multi_layer_agreement": True,
    "require_recovery_authenticity": True,
    "require_recovery_resilience": True,
    "require_recovery_stability": True,

    # Explicit Non-Formation Rules
    "single_observation_forms_trust": False,
    "single_vital_forms_trust": False,
    "single_subsystem_forms_trust": False,
    "isolated_improvement_forms_trust": False,

    # Philosophy
    "trust_question":
        "Has recovery accumulated sufficient evidence to earn trust?"
}


def build_tari_recovery_trust_formation_policy_table():
    """
    Build TARI recovery trust formation policy table.
    """

    rows = []

    for key, value in (
        TARI_RECOVERY_TRUST_FORMATION_POLICY.items()
    ):

        rows.append({
            "trust_policy": key,
            "value": value,
        })

    return pd.DataFrame(rows)

TARI_RECOVERY_TRUST_STATES = {

    "RECOVERY_TRUST_GRANTED":
        "Recovery has accumulated sufficient evidence to earn trust.",

    "RECOVERY_TRUST_PROBABLE":
        "Recovery evidence is strengthening but additional evidence is required.",

    "RECOVERY_TRUST_UNVERIFIED":
        "Recovery evidence remains insufficient for trustworthy recovery.",

    "RECOVERY_TRUST_FRAGILE":
        "Recovery currently appears vulnerable to deterioration or contradiction.",

    "RECOVERY_TRUST_DECEPTIVE":
        "Recovery appears inconsistent with accumulated evidence.",

    "RECOVERY_TRUST_REVERSING":
        "Recovery evidence indicates deterioration may be returning.",

    "RECOVERY_TRUST_FAILED":
        "Recovery no longer satisfies the minimum evidence required for trust."
}


def build_tari_recovery_trust_states_table():
    """
    Build TARI recovery trust states table.
    """

    rows = []

    for state, description in (
        TARI_RECOVERY_TRUST_STATES.items()
    ):

        rows.append({
            "trust_state": state,
            "description": description,
        })

    return pd.DataFrame(rows)