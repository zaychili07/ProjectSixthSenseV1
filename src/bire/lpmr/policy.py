# ============================================================
# BIRE OS — Longitudinal Patient Memory Retention (LPMR)
# File: src/bire/lpmr/policy.py
# Chapter: 65
#
# Created: 2026-06-20
# Updated: 2026-06-20
#
# CHANGED:
# - Initialized LPMR governance and policy module
# - Added ownership boundaries
# - Added consumer definitions
# - Added non-responsibility definitions
# - Established LPMR patient memory doctrine
#
# Purpose:
# Governance, ownership, policy, doctrine,
# consumer definitions, and subsystem boundaries
# for Longitudinal Patient Memory Retention (LPMR).
# ============================================================

import pandas as pd


LPMR_POLICY = {

    "retain_longitudinal_patient_experience": True,
    "retain_recovery_history": True,
    "retain_deterioration_history": True,
    "retain_compensation_patterns": True,
    "retain_rebound_patterns": True,
    "retain_stabilization_history": True,
    "retain_patient_signatures": True,

    "store_raw_fragments": False,
    "archive_fragments": False,
    "clean_data": False,
    "perform_interpretation": False,
    "perform_forecasting": False,
    "perform_measurement": False,
    "retrieve_memory": False,

    "core_question": "Have we seen this before?"
}

LPMR_RESPONSIBILITIES = [
    "retain_longitudinal_patient_experience",
    "retain_recovery_history",
    "retain_deterioration_history",
    "retain_compensation_patterns",
    "retain_rebound_patterns",
    "retain_stabilization_history",
    "retain_patient_signatures",
]

LPMR_NON_RESPONSIBILITIES = [
    "store_raw_fragments",
    "archive_fragments",
    "clean_data",
    "perform_interpretation",
    "perform_forecasting",
    "perform_measurement",
    "retrieve_memory",
]

LPMR_CONSUMERS = [
    "IBPIP",
    "ATI",
    "BIRE_FI",
    "HVI",
    "BMS",
]

LPMR_MEMORY_CATEGORIES = {
    "RECOVERY_MEMORY": [
        "repeated_recovery_failure",
        "recovery_durability",
        "temporary_improvement",
        "fragile_stabilization",
        "partial_stabilization",
    ],
    "DETERIORATION_MEMORY": [
        "deterioration_recurrence",
        "deterioration_identity",
        "progressive_baseline_drift",
        "deterioration_acceleration",
    ],
    "COMPENSATION_MEMORY": [
        "compensation_success",
        "compensation_failure",
        "temporary_compensation",
        "compensation_exhaustion",
    ],
    "REBOUND_MEMORY": [
        "rebound_deterioration",
        "rebound_frequency",
        "rebound_severity",
        "rebound_persistence",
    ],
    "INSTABILITY_MEMORY": [
        "chronic_instability_burden",
        "repeated_instability_cycles",
        "longitudinal_instability_signatures",
    ],
    "PATIENT_SIGNATURE_MEMORY": [
        "patient_specific_trajectory_signatures",
        "recurring_physiologic_behavior",
        "recurring_recovery_patterns",
        "recurring_deterioration_patterns",
    ],
}
LPMR_AUTHORITY_BOUNDARIES = {

    "retain_patient_experience": True,
    "retain_patient_patterns": True,
    "retain_patient_signatures": True,

    "perform_measurement": False,
    "perform_interpretation": False,
    "perform_forecasting": False,
    "perform_memory_storage": False,
    "perform_memory_retrieval": False,
    "perform_recommendations": False,

    "highest_authority":
        "PATIENT_EXPERIENCE_RETENTION",

    "core_authority_question":
        "What longitudinal patient experience is worth remembering?"
}

LPMR_INPUT_POLICY = {
    "consume_ibpip_outputs": True,
    "consume_ati_outputs": True,
    "consume_hvi_outputs": True,
    "consume_bire_fi_outputs": True,
    "consume_bms_outputs": True,

    "consume_raw_fragments": False,
    "consume_raw_measurements": False,
    "consume_raw_notes": False,

    "input_question":
        "What patient experiences are worth retaining?",
}
LPMR_OUTPUT_POLICY = {

    "produce_recovery_memory": True,
    "produce_deterioration_memory": True,
    "produce_compensation_memory": True,
    "produce_rebound_memory": True,
    "produce_instability_memory": True,
    "produce_patient_signature_memory": True,

    "produce_measurements": False,
    "produce_interpretations": False,
    "produce_forecasts": False,
    "produce_recommendations": False,

    "output_question":
        "What patient experiences should be retained for future use?"
}

LPMR_CONSUMER_POLICY = {
    "ibpip_can_consume_lpmr": True,
    "ati_can_consume_lpmr": True,
    "bire_fi_can_consume_lpmr": True,
    "hvi_can_consume_lpmr": True,
    "bms_can_consume_lpmr": True,

    "psrv2_can_display_lpmr_when_authorized": True,

    "lpmr_controls_consumers": False,
    "lpmr_overrides_consumers": False,
    "lpmr_performs_consumer_tasks": False,

    "consumer_question":
        "Which subsystems may use retained patient experience?"
}

LPMR_SUBSYSTEM_RELATIONSHIPS = {

    "CMR":
        "Stores historical fragments and archived context",

    "ATI":
        "Consumes retained patient experience for trajectory interpretation",

    "BIRE_FI":
        "Consumes retained patient experience for trajectory forecasting",

    "HVI":
        "Produces hidden physiologic patterns that may later become retained experience",

    "IBPIP":
        "Parent subsystem of LPMR and primary consumer of retained patient experience",

    "BMS":
        "May consume retained patient experience for mode-aware context",

    "PSRV2":
        "May display retained patient experience when authorized",
}

LPMR_DOMAINS = {

    "RECOVERY_DOMAIN":
        "Longitudinal recovery experience",

    "DETERIORATION_DOMAIN":
        "Longitudinal deterioration experience",

    "COMPENSATION_DOMAIN":
        "Longitudinal compensation behavior",

    "REBOUND_DOMAIN":
        "Longitudinal rebound behavior",

    "INSTABILITY_DOMAIN":
        "Longitudinal instability experience",

    "PATIENT_SIGNATURE_DOMAIN":
        "Patient-specific trajectory signatures",
}


def build_lpmr_authority_boundary_table():

    import pandas as pd

    rows = []

    for key, value in LPMR_AUTHORITY_BOUNDARIES.items():

        rows.append({
            "authority_boundary": key,
            "value": value,
        })

    return pd.DataFrame(rows)

def build_lpmr_policy_table():
    """
    Build LPMR governance table.
    """

    rows = []

    for key, value in LPMR_POLICY.items():

        rows.append({
            "policy": key,
            "value": value,
        })

    return pd.DataFrame(rows)


def build_lpmr_governance_table():
    """
    Build LPMR governance responsibility table.
    """

    rows = []

    for item in LPMR_RESPONSIBILITIES:
        rows.append({
            "category": "LPMR_RESPONSIBILITY",
            "item": item,
            "status": "OWNED_BY_LPMR",
        })

    for item in LPMR_NON_RESPONSIBILITIES:
        rows.append({
            "category": "LPMR_NON_RESPONSIBILITY",
            "item": item,
            "status": "NOT_OWNED_BY_LPMR",
        })

    for item in LPMR_CONSUMERS:
        rows.append({
            "category": "LPMR_CONSUMER",
            "item": item,
            "status": "CAN_CONSUME_LPMR_OUTPUTS",
        })

    return pd.DataFrame(rows)


def build_lpmr_memory_category_table():
    """
    Build LPMR memory category governance table.
    """

    rows = []

    for category, items in LPMR_MEMORY_CATEGORIES.items():
        for item in items:
            rows.append({
                "memory_category": category,
                "retained_experience": item,
                "status": "RETAINED_BY_LPMR",
            })

    return pd.DataFrame(rows)


def build_lpmr_input_policy_table():
    """
    Build LPMR input policy governance table.
    """
    rows = []

    for key, value in LPMR_INPUT_POLICY.items():
        rows.append({
            "input_policy": key,
            "value": value,
        })

    return pd.DataFrame(rows)


def build_lpmr_output_policy_table():
    """
    Build LPMR output policy governance table.
    """

    rows = []

    for key, value in LPMR_OUTPUT_POLICY.items():

        rows.append({
            "output_policy": key,
            "value": value,
        })

    return pd.DataFrame(rows)


def build_lpmr_consumer_policy_table():
    """
    Build LPMR consumer policy governance table.
    """

    rows = []

    for key, value in LPMR_CONSUMER_POLICY.items():
        rows.append({
            "consumer_policy": key,
            "value": value,
        })

    return pd.DataFrame(rows)


def build_lpmr_subsystem_relationship_table():
    """
    Build LPMR subsystem relationship table.
    """

    rows = []

    for subsystem, relationship in (
        LPMR_SUBSYSTEM_RELATIONSHIPS.items()
    ):

        rows.append({
            "subsystem": subsystem,
            "relationship": relationship,
        })

    return pd.DataFrame(rows)


def build_lpmr_domain_table():
    """
    Build LPMR domain review table.
    """

    rows = []

    for domain, description in LPMR_DOMAINS.items():

        rows.append({
            "domain": domain,
            "description": description,
            "status": "LPMR_DOMAIN",
        })

    return pd.DataFrame(rows)

LPMR_MEMORY_FORMATION_POLICY = {

    "memory_is_earned": True,

    "requires_longitudinal_significance": True,
    "requires_patient_specific_behavior": True,
    "requires_pattern_persistence": True,

    "single_observation_forms_memory": False,
    "single_encounter_forms_memory": False,

    "retain_repeated_patterns": True,
    "retain_progressive_patterns": True,
    "retain_declining_patterns": True,
    "retain_recurring_patterns": True,

    "memory_question":
        "When does patient experience become longitudinal memory?"
}


def build_lpmr_memory_formation_policy_table():
    """
    Build LPMR memory formation policy table.
    """

    import pandas as pd

    rows = []

    for key, value in (
        LPMR_MEMORY_FORMATION_POLICY.items()
    ):

        rows.append({
            "memory_policy": key,
            "value": value,
        })

    return pd.DataFrame(rows)

LPMR_LONGITUDINAL_PRESERVATION_POLICY = {

    "preserve_recovery_progression": True,
    "preserve_recovery_durability": True,
    "preserve_temporary_improvement": True,
    "preserve_fragile_stabilization": True,
    "preserve_partial_stabilization": True,
    "preserve_rebound_deterioration": True,
    "preserve_compensation_behavior": True,
    "preserve_progressive_baseline_drift": True,
    "preserve_instability_cycles": True,

    "flatten_patient_history": False,
    "flatten_stabilization": False,
    "flatten_recovery": False,

    "policy_question":
        "What longitudinal patient experience must never be simplified?"
}


def build_lpmr_longitudinal_preservation_policy_table():
    """
    Build LPMR longitudinal preservation policy table.
    """

    import pandas as pd

    rows = []

    for key, value in (
        LPMR_LONGITUDINAL_PRESERVATION_POLICY.items()
    ):

        rows.append({
            "preservation_policy": key,
            "value": value,
        })

    return pd.DataFrame(rows)

LPMR_MEMORY_FORMATION_RULES = {
    "memory_requires_longitudinal_evidence": True,
    "single_observation_can_form_memory": False,
    "single_encounter_can_form_memory": False,
    "repeated_pattern_can_form_memory": True,
    "progressive_pattern_can_form_memory": True,
    "recurring_pattern_can_form_memory": True,
    "declining_pattern_can_form_memory": True,
    "formation_doctrine": "MEMORY_IS_EARNED",
    "formation_question": "Has this patient experience earned longitudinal memory status?",
}


def build_lpmr_memory_formation_rules_table():
    import pandas as pd

    return pd.DataFrame([
        {"memory_formation_rule": key, "value": value}
        for key, value in LPMR_MEMORY_FORMATION_RULES.items()
    ])