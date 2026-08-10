# ============================================================
# BIRE OS — Therapeutic Trajectory Intelligence (TTI)
# File: src/bire/tti/policy.py
# Chapter: 67
#
# Created: 2026-07-15
# Updated: 2026-07-15
#
# CHANGED:
# - Initialized TTI policy module
# - Established therapeutic influence policies
# - Added authority and responsibility definitions
# - Prepared governance framework for therapeutic influence
#
# Purpose:
# Backend helper for TTI governance policies,
# authority boundaries, subsystem responsibilities,
# and therapeutic influence interpretation.
# ============================================================

TTI_MISSION = {

    "evaluate_therapeutic_influence": True,
    "evaluate_trajectory_change": True,
    "evaluate_longitudinal_response": True,

    "recommend_treatment": False,
    "recommend_medications": False,
    "recommend_procedures": False,
    "replace_physician_judgment": False,
    "determine_medical_appropriateness": False,
    "assume_causation": False,

    "highest_authority":
        "THERAPEUTIC_INFLUENCE",

    "core_question":
        "What changed after intervention?",

    "core_doctrine":
        "THERAPEUTIC_INFLUENCE_IS_OBSERVED_NOT_ASSUMED",
}


def build_tti_mission_table():

    import pandas as pd

    rows = []

    for key, value in TTI_MISSION.items():

        rows.append({

            "mission": key,
            "value": value,

        })

    return pd.DataFrame(rows)

TTI_THERAPEUTIC_INFLUENCE_POLICY = {

    # Therapeutic Evaluation
    "therapeutic_influence_is_observed": True,
    "require_longitudinal_evidence": True,
    "require_before_after_comparison": True,
    "require_multi_layer_evidence": True,

    # Scientific Integrity
    "assume_causation": False,
    "assume_effectiveness": False,
    "assume_positive_response": False,
    "assume_negative_response": False,

    # Governance
    "evaluate_trajectory_change": True,
    "evaluate_hidden_response": True,
    "evaluate_response_consistency": True,

    # Philosophy
    "policy_question":
        "Is observed trajectory modification consistent with therapeutic influence?"
}


def build_tti_therapeutic_influence_policy_table():

    import pandas as pd

    rows = []

    for key, value in (
        TTI_THERAPEUTIC_INFLUENCE_POLICY.items()
    ):

        rows.append({

            "policy": key,
            "value": value,

        })

    return pd.DataFrame(rows)

TTI_AUTHORITY_BOUNDARIES = {

    # Therapeutic Influence Authority
    "evaluate_therapeutic_influence": True,
    "evaluate_trajectory_modification": True,
    "evaluate_response_patterns": True,

    # Explicit Non-Authorities
    "recommend_treatment": False,
    "recommend_medications": False,
    "recommend_procedures": False,
    "perform_diagnosis": False,
    "replace_physician_judgment": False,
    "determine_medical_appropriateness": False,
    "assume_causation": False,

    # Authority
    "highest_authority":
        "THERAPEUTIC_INFLUENCE",

    "core_authority_question":
        "Is observed trajectory modification consistent with therapeutic influence?"
}


def build_tti_authority_boundaries_table():

    import pandas as pd

    rows = []

    for key, value in (
        TTI_AUTHORITY_BOUNDARIES.items()
    ):

        rows.append({
            "authority_boundary": key,
            "value": value,
        })

    return pd.DataFrame(rows)

TTI_INPUT_POLICY = {

    # Primary Intelligence Sources
    "consume_ati_outputs": True,
    "consume_hvi_outputs": True,
    "consume_ibpip_outputs": True,
    "consume_bire_fi_outputs": True,
    "consume_lpmr_outputs": True,
    "consume_confidence_engine_outputs": True,

    # Explicit Non-Inputs
    "consume_raw_vitals": False,
    "consume_raw_laboratory_data": False,
    "consume_raw_imaging": False,
    "consume_raw_clinical_notes": False,
    "consume_raw_measurements": False,

    # Philosophy
    "input_question":
        "What completed intelligence contributes toward therapeutic influence interpretation?"
}


def build_tti_input_policy_table():

    import pandas as pd

    rows = []

    for key, value in (
        TTI_INPUT_POLICY.items()
    ):

        rows.append({
            "input_policy": key,
            "value": value,
        })

    return pd.DataFrame(rows)

TTI_OUTPUT_POLICY = {

    # Primary Outputs
    "produce_therapeutic_response": True,
    "produce_trajectory_modification": True,
    "produce_response_confidence": True,
    "produce_therapeutic_response_state": True,

    # Explicit Non-Outputs
    "produce_treatment_recommendations": False,
    "produce_medication_recommendations": False,
    "produce_procedure_recommendations": False,
    "produce_diagnosis": False,
    "produce_causation": False,
    "produce_clinical_orders": False,

    # Identity
    "output_question":
        "What therapeutic influence should TTI communicate?"
}


def build_tti_output_policy_table():
    """
    Build TTI output policy table.
    """

    import pandas as pd

    rows = []

    for key, value in TTI_OUTPUT_POLICY.items():

        rows.append({
            "output_policy": key,
            "value": value,
        })

    return pd.DataFrame(rows)

TTI_CONSUMER_POLICY = {

    # Primary Consumers
    "ati_can_consume_tti": True,
    "bire_fi_can_consume_tti": True,
    "lpmr_can_consume_tti": True,
    "tari_can_consume_tti": True,
    "ibpip_can_consume_tti": True,
    "psrv2_can_display_tti_when_authorized": True,

    # Explicit Non-Authorities
    "tti_controls_consumers": False,
    "tti_overrides_consumers": False,
    "tti_performs_consumer_tasks": False,

    # Identity
    "consumer_question":
        "Which subsystems may utilize therapeutic influence intelligence?"
}


def build_tti_consumer_policy_table():
    """
    Build TTI consumer policy table.
    """

    import pandas as pd

    rows = []

    for key, value in (
        TTI_CONSUMER_POLICY.items()
    ):

        rows.append({
            "consumer_policy": key,
            "value": value,
        })

    return pd.DataFrame(rows)

#============================================================
# Chapter 67.7 — TTI Subsystem Relationships
#============================================================

TTI_SUBSYSTEM_RELATIONSHIPS = {

    "hvi_relationship":
        "Measures hidden physiologic behavior that may contribute to therapeutic influence assessment.",

    "ati_relationship":
        "Provides trajectory interpretation and clinical context supporting therapeutic influence evaluation.",

    "ibpip_relationship":
        "Provides patient-specific physiologic interpretation for individualized therapeutic influence evaluation.",

    "lpmr_relationship":
        "Provides longitudinal therapeutic experience and historical response patterns.",

    "bire_fi_relationship":
        "Utilizes therapeutic influence observations during future trajectory evaluation.",

    "confidence_engine_relationship":
        "Communicates confidence and uncertainty associated with therapeutic influence observations.",

    "tari_relationship":
        "May utilize therapeutic influence observations when governing recovery trust.",

    "tti_replaces_other_subsystems": False,

    "relationship_question":
        "How does TTI cooperate with other intelligence layers while preserving subsystem independence?"
}


def build_tti_relationship_table():
    """
    Build TTI subsystem relationship table.
    """

    import pandas as pd

    rows = []

    for key, value in TTI_SUBSYSTEM_RELATIONSHIPS.items():

        rows.append({
            "relationship": key,
            "value": value
        })

    return pd.DataFrame(rows)

#============================================================
# Chapter 67.8 — TTI Domains
#============================================================

TTI_DOMAINS = {

    "primary_domain":
        "Therapeutic Influence Intelligence",

    "primary_focus":
        "Evaluation of observed therapeutic influence on patient trajectory.",

    "secondary_focus":
        "Assessment of therapeutic response consistency while preserving uncertainty.",

    "excluded_domains": [

        "Physiologic Measurement",

        "Trajectory Interpretation",

        "Recovery Governance",

        "Patient Memory",

        "Future Forecasting",

        "Treatment Recommendation",

        "Diagnosis",

        "Clinical Decision Making",

        "Causation Determination"
    ],

    "domain_question":
        "What intelligence domain belongs exclusively to TTI?"
}


def build_tti_domain_table():
    """
    Build TTI domain table.
    """

    import pandas as pd

    rows = []

    for key, value in TTI_DOMAINS.items():

        rows.append({
            "domain": key,
            "value": value
        })

    return pd.DataFrame(rows)

#============================================================
# Chapter 67.9 — Therapeutic Influence Formation
#============================================================

TTI_FORMATION_POLICY = {

    "formation_principle":
        "Therapeutic influence is formed through the integration of multiple independent intelligence observations.",

    "requires_multiple_sources": True,

    "requires_longitudinal_context": True,

    "requires_patient_specific_context": True,

    "preserves_uncertainty": True,

    "permits_single_source_causation": False,

    "formation_question":
        "How is therapeutic influence formed?"
}


def build_tti_formation_table():
    """
    Build TTI formation policy table.
    """

    import pandas as pd

    rows = []

    for key, value in TTI_FORMATION_POLICY.items():

        rows.append({
            "formation_policy": key,
            "value": value
        })

    return pd.DataFrame(rows)

#============================================================
# Chapter 67.10 — Therapeutic Response States
#============================================================

TTI_RESPONSE_STATES = {

    "positive_response":
        "Observed trajectory is consistent with beneficial therapeutic influence.",

    "negative_response":
        "Observed trajectory is consistent with worsening despite intervention.",

    "partial_response":
        "Observed improvement is incomplete or limited.",

    "delayed_response":
        "Therapeutic influence may be emerging but remains incomplete.",

    "transient_response":
        "Observed improvement was temporary and not sustained.",

    "sustained_response":
        "Observed improvement has remained stable over time.",

    "no_observable_response":
        "No measurable therapeutic influence has been observed.",

    "conflicting_response":
        "Available observations provide conflicting evidence regarding therapeutic influence.",

    "uncertain_response":
        "Available evidence is insufficient to determine therapeutic influence.",

    "response_question":
        "How should TTI classify observed therapeutic influence?"
}


def build_tti_response_state_table():
    """
    Build TTI response state table.
    """

    import pandas as pd

    rows = []

    for key, value in TTI_RESPONSE_STATES.items():

        rows.append({
            "response_state": key,
            "description": value
        })

    return pd.DataFrame(rows)

#============================================================
# Chapter 67.11 — TTI Architectural Review
#============================================================

TTI_ARCHITECTURAL_REVIEW = {

    "identity_defined": True,

    "governance_defined": True,

    "authority_boundaries_defined": True,

    "input_policy_defined": True,

    "output_policy_defined": True,

    "consumer_policy_defined": True,

    "subsystem_relationships_defined": True,

    "domain_defined": True,

    "formation_policy_defined": True,

    "response_states_defined": True,

    "implementation_ready": True,

    "review_question":
        "Has TTI been sufficiently defined for implementation?"
}


def build_tti_architectural_review_table():
    """
    Build TTI architectural review table.
    """

    import pandas as pd

    rows = []

    for key, value in TTI_ARCHITECTURAL_REVIEW.items():

        rows.append({
            "review": key,
            "value": value
        })

    return pd.DataFrame(rows)