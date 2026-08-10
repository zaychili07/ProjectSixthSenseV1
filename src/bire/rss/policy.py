#============================================================
# Project Sixth Sense — BIRE OS
# Replay Stability Support (RSS)
#
# File:
#     policy.py
#
# Chapter:
#     68 — Replay Stability Support
#
# Purpose:
#     Define the identity, governance, authority, evidence,
#     output, consumer, relationship, domain, formation,
#     state, and architectural readiness policies governing
#     Replay Stability Support.
#
# Core Responsibility:
#     Transform repeated stability and instability history
#     into replay-derived longitudinal learning.
#
# Governance:
#     RSS operates under NID governance.
#
# Architectural Boundary:
#     LPMR preserves and retrieves earned patient memory.
#     RSS examines repeated experience for recurring lessons.
#
# Does Not:
#     - Predict future deterioration
#     - Diagnose disease
#     - Recommend treatment
#     - Replace LPMR
#     - Override other intelligence layers
#     - Manufacture patterns from insufficient history
#     - Claim confidence ownership
#
# Core Doctrine:
#     Replay may reveal a lesson.
#     Recurrence must earn it.
#============================================================

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


#============================================================
# Shared Policy Table Builder
#============================================================

def _build_rss_policy_table(
    policy: Mapping[str, object],
    key_column: str,
    value_column: str = "value",
) -> pd.DataFrame:
    """
    Build a standard two-column RSS policy table.

    This helper communicates architectural policy only.
    It does not perform replay analysis or generate RSS states.
    """

    return pd.DataFrame(
        [
            {
                key_column: key,
                value_column: value,
            }
            for key, value in policy.items()
        ]
    )


#============================================================
# Chapter 68.1 — RSS Purpose
#============================================================

RSS_PURPOSE_POLICY = {

    # Primary Purpose
    "transform_replay_into_longitudinal_learning": True,

    "analyze_repeated_stability_cycles": True,

    "analyze_repeated_instability_cycles": True,

    "identify_recurring_behavioral_patterns": True,

    "communicate_replay_derived_lessons": True,

    # Scientific Honesty
    "preserve_uncertainty": True,

    "permit_no_meaningful_pattern": True,

    "require_recurrence_before_learning": True,

    "require_accumulated_history": True,

    # Governance
    "operate_under_nid_governance": True,

    # Explicit Non-Purposes
    "predict_future_deterioration": False,

    "diagnose_disease": False,

    "recommend_treatment": False,

    "replace_lpmr": False,

    "override_other_intelligence_layers": False,

    "manufacture_patterns_from_insufficient_history": False,

    # Central Question
    "purpose_question":
        "What can previous instability teach us?"
}


def build_rss_purpose_table() -> pd.DataFrame:
    """
    Build the RSS purpose policy table.
    """

    return _build_rss_policy_table(
        policy=RSS_PURPOSE_POLICY,
        key_column="purpose_policy",
    )


#============================================================
# Chapter 68.1A — RSS Identity
#============================================================

RSS_IDENTITY_POLICY = {

    "subsystem_name":
        "Replay Stability Support",

    "subsystem_abbreviation":
        "RSS",

    "intelligence_identity":
        "Replay Learning Intelligence",

    "system_role":
        "Replay-learning subsystem of BIRE OS.",

    "represents_accumulated_experience":
        True,

    "owns_replay_derived_learning":
        True,

    "owns_patient_memory_storage":
        False,

    "primary_subject":
        (
            "Recurring stability and instability behavior "
            "across accumulated longitudinal experience."
        ),

    "identity_doctrine":
        (
            "Replay may reveal a lesson. "
            "Recurrence must earn it."
        ),

    "operates_under_nid_governance":
        True,

    "identity_question":
        "What has history consistently taught us?"
}


def build_rss_identity_table() -> pd.DataFrame:
    """
    Build the RSS identity table.
    """

    return _build_rss_policy_table(
        policy=RSS_IDENTITY_POLICY,
        key_column="identity",
    )


#============================================================
# Chapter 68.1B — RSS Governance
#============================================================

RSS_GOVERNANCE_POLICY = {

    # NID Governance
    "nid_governs_replay_intelligence":
        True,

    "nid_governs_replay_admission":
        True,

    "nid_governs_replay_communication":
        True,

    "nid_governs_replay_use":
        True,

    "rss_overrides_nid":
        False,

    "rss_self_authorizes_learning":
        False,

    # Evidence Governance
    "require_sufficient_history":
        True,

    "require_recurrence":
        True,

    "require_replay_similarity":
        True,

    "preserve_uncertainty":
        True,

    "preserve_contradictory_history":
        True,

    "preserve_new_pattern_possibility":
        True,

    "permit_no_meaningful_pattern":
        True,

    "permit_insufficient_replay_history":
        True,

    "communicate_evidence_limitations":
        True,

    # Explicit Prohibitions
    "force_history_into_pattern":
        False,

    "treat_single_event_as_recurrence":
        False,

    "convert_replay_into_prediction":
        False,

    "claim_historical_causation":
        False,

    "override_source_intelligence":
        False,

    "governance_question":
        (
            "How must replay-derived learning be governed "
            "before BIRE OS may use it?"
        )
}


def build_rss_governance_table() -> pd.DataFrame:
    """
    Build the RSS governance policy table.
    """

    return _build_rss_policy_table(
        policy=RSS_GOVERNANCE_POLICY,
        key_column="governance_policy",
    )


#============================================================
# Chapter 68.2 — RSS Replay Learning Policy
#============================================================

RSS_REPLAY_LEARNING_POLICY = {

    "learn_from_repeated_behavior":
        True,

    "compare_stability_cycles":
        True,

    "compare_instability_cycles":
        True,

    "evaluate_pattern_recurrence":
        True,

    "evaluate_pattern_similarity":
        True,

    "evaluate_replay_consistency":
        True,

    "evaluate_repeated_outcomes":
        True,

    "evaluate_stability_scarcity":
        True,

    "evaluate_instability_persistence":
        True,

    "evaluate_failed_recovery_replay":
        True,

    "evaluate_escalation_replay":
        True,

    "evaluate_intervention_replay_when_supported":
        True,

    "communicate_replay_derived_lessons":
        True,

    "permit_no_replay_lesson":
        True,

    "preserve_unknown_replay":
        True,

    "learn_from_single_event":
        False,

    "assume_recurrence_from_similarity_alone":
        False,

    "assume_same_outcome_will_repeat":
        False,

    "learning_question":
        "What lesson, if any, has recurrence earned?"
}


def build_rss_replay_learning_policy_table() -> pd.DataFrame:
    """
    Build the RSS replay learning policy table.
    """

    return _build_rss_policy_table(
        policy=RSS_REPLAY_LEARNING_POLICY,
        key_column="replay_learning_policy",
    )


#============================================================
# Chapter 68.3 — RSS Authority Boundaries
#============================================================

RSS_AUTHORITY_POLICY = {

    # Authorized Responsibilities
    "analyze_repeated_longitudinal_cycles":
        True,

    "measure_replay_burden":
        True,

    "measure_replay_recurrence":
        True,

    "measure_replay_consistency":
        True,

    "measure_stability_scarcity":
        True,

    "measure_historical_instability_persistence":
        True,

    "identify_recurring_behavioral_patterns":
        True,

    "assign_replay_pattern_states":
        True,

    "assign_replay_lesson_states":
        True,

    "communicate_replay_evidence_strength":
        True,

    "communicate_replay_derived_lessons":
        True,

    # Unauthorized Responsibilities
    "create_patient_memory":
        False,

    "modify_lpmr_memory":
        False,

    "predict_future_deterioration":
        False,

    "interpret_current_trajectory":
        False,

    "govern_recovery_trust":
        False,

    "determine_therapeutic_effect":
        False,

    "determine_causation":
        False,

    "diagnose_disease":
        False,

    "recommend_treatment":
        False,

    "issue_clinical_orders":
        False,

    "override_other_intelligence_layers":
        False,

    "authority_question":
        (
            "What replay-derived intelligence may RSS "
            "produce without assuming another layer's authority?"
        )
}


def build_rss_authority_table() -> pd.DataFrame:
    """
    Build the RSS authority boundary table.
    """

    return _build_rss_policy_table(
        policy=RSS_AUTHORITY_POLICY,
        key_column="authority_policy",
    )


#============================================================
# Chapter 68.4 — RSS Input Policy
#============================================================

RSS_INPUT_POLICY = {

    # Required Longitudinal Foundation
    "consume_lpmr_earned_memory":
        True,

    "require_patient_identity":
        True,

    "require_encounter_identity":
        True,

    "require_ordered_longitudinal_history":
        True,

    "require_repeated_observations":
        True,

    # Intelligence Inputs
    "consume_ati_trajectory_intelligence":
        True,

    "consume_hvi_hidden_physiology":
        True,

    "consume_lpmr_longitudinal_memory":
        True,

    "consume_tari_recovery_trust":
        True,

    "consume_tti_when_available":
        True,

    "consume_bire_fi_forecast_history":
        True,

    "consume_ibpip_patient_specific_context":
        True,

    "consume_confidence_engine_outputs":
        True,

    # Input Safety
    "accept_only_earned_memory":
        True,

    "preserve_source_provenance":
        True,

    "preserve_missing_history":
        True,

    "preserve_conflicting_history":
        True,

    "accept_single_event_as_replay":
        False,

    "accept_unordered_history_as_complete_replay":
        False,

    "manufacture_missing_replay_events":
        False,

    "input_question":
        (
            "What accumulated longitudinal evidence may "
            "RSS use to evaluate replay?"
        )
}


def build_rss_input_policy_table() -> pd.DataFrame:
    """
    Build the RSS input policy table.
    """

    return _build_rss_policy_table(
        policy=RSS_INPUT_POLICY,
        key_column="input_policy",
    )


#============================================================
# Chapter 68.5 — RSS Output Policy
#============================================================

RSS_OUTPUT_POLICY = {

    # Authorized Outputs
    "produce_replay_pattern_state":
        True,

    "produce_replay_lesson_state":
        True,

    "produce_replay_burden":
        True,

    "produce_replay_recurrence":
        True,

    "produce_replay_consistency":
        True,

    "produce_stability_scarcity":
        True,

    "produce_historical_instability_persistence":
        True,

    "produce_recovery_replay_observation":
        True,

    "produce_escalation_replay_observation":
        True,

    "produce_intervention_replay_when_supported":
        True,

    "produce_replay_evidence_strength":
        True,

    "produce_replay_uncertainty_state":
        True,

    "produce_insufficient_history_state":
        True,

    "produce_no_meaningful_pattern_state":
        True,

    # Confidence Ownership
    "produce_independent_replay_confidence":
        False,

    "provide_evidence_to_confidence_engine":
        True,

    # Explicit Non-Outputs
    "produce_future_prediction":
        False,

    "produce_diagnosis":
        False,

    "produce_treatment_recommendation":
        False,

    "produce_clinical_orders":
        False,

    "produce_causation_claim":
        False,

    "produce_recovery_governance_decision":
        False,

    "output_question":
        "What replay-derived lesson should RSS communicate?"
}


def build_rss_output_policy_table() -> pd.DataFrame:
    """
    Build the RSS output policy table.
    """

    return _build_rss_policy_table(
        policy=RSS_OUTPUT_POLICY,
        key_column="output_policy",
    )


#============================================================
# Chapter 68.6 — RSS Consumer Policy
#============================================================

RSS_CONSUMER_POLICY = {

    "nid_can_consume_rss":
        True,

    "ati_can_consume_rss_context":
        True,

    "tari_can_consume_rss":
        True,

    "tti_can_consume_rss_when_active":
        True,

    "bire_fi_can_consume_rss":
        True,

    "ibpip_can_consume_rss":
        True,

    "confidence_engine_can_consume_rss_evidence":
        True,

    "psrv2_can_display_rss_when_authorized":
        True,

    # Meta-Memory Boundary
    "lpmr_stores_rss_meta_memory_in_v1":
        False,

    # Consumer Independence
    "rss_controls_consumers":
        False,

    "rss_overrides_consumers":
        False,

    "rss_performs_consumer_tasks":
        False,

    "rss_forces_replay_lessons_into_decisions":
        False,

    "consumer_question":
        (
            "Which intelligence layers may use "
            "replay-derived longitudinal learning?"
        )
}


def build_rss_consumer_policy_table() -> pd.DataFrame:
    """
    Build the RSS consumer policy table.
    """

    return _build_rss_policy_table(
        policy=RSS_CONSUMER_POLICY,
        key_column="consumer_policy",
    )


#============================================================
# Chapter 68.7 — RSS Subsystem Relationships
#============================================================

RSS_SUBSYSTEM_RELATIONSHIPS = {

    "nid_relationship":
        (
            "NID governs the admission, communication, and use "
            "of replay-derived intelligence throughout BIRE OS."
        ),

    "lpmr_relationship":
        (
            "LPMR preserves and retrieves earned patient memory. "
            "RSS examines repeated experience across that memory "
            "for recurring lessons."
        ),

    "ati_relationship":
        (
            "ATI contributes interpreted trajectory behavior. "
            "RSS evaluates whether similar trajectory behavior "
            "has repeatedly occurred across historical cycles."
        ),

    "hvi_relationship":
        (
            "HVI contributes hidden physiologic observations. "
            "RSS evaluates whether hidden physiologic behavior "
            "recurs across stability and instability cycles."
        ),

    "tari_relationship":
        (
            "TARI contributes recovery-trust observations. "
            "RSS evaluates repeated recovery durability, failed "
            "recovery, and recurring return to instability."
        ),

    "tti_relationship":
        (
            "TTI may contribute verified therapeutic influence "
            "observations when active. RSS may then evaluate "
            "recurring intervention-response behavior."
        ),

    "bire_fi_relationship":
        (
            "BIRE-FI contributes historical forecast behavior. "
            "RSS may evaluate recurring forecast modification "
            "without producing prediction itself."
        ),

    "ibpip_relationship":
        (
            "IBPIP contributes patient-specific physiologic "
            "context so replay learning remains individualized."
        ),

    "confidence_engine_relationship":
        (
            "RSS produces replay evidence strength, recurrence, "
            "consistency, and uncertainty. The Confidence Engine "
            "determines how much confidence that evidence deserves."
        ),

    "psrv2_relationship":
        (
            "PSRv2 may display authorized replay-derived lessons "
            "without independently interpreting or governing them."
        ),

    "rss_replaces_lpmr":
        False,

    "rss_replaces_other_subsystems":
        False,

    "relationship_question":
        (
            "How does RSS learn from repeated intelligence "
            "while preserving subsystem independence?"
        )
}


def build_rss_relationship_table() -> pd.DataFrame:
    """
    Build the RSS subsystem relationship table.
    """

    return _build_rss_policy_table(
        policy=RSS_SUBSYSTEM_RELATIONSHIPS,
        key_column="relationship",
    )


#============================================================
# Chapter 68.8 — RSS Replay Intelligence Domains
#============================================================

RSS_REPLAY_DOMAINS = {

    "primary_domain":
        "Replay-Derived Longitudinal Learning",

    "primary_focus":
        (
            "Evaluation of recurring stability and instability "
            "behavior across accumulated patient experience."
        ),

    "replay_burden_domain":
        True,

    "replay_recurrence_domain":
        True,

    "replay_consistency_domain":
        True,

    "stability_scarcity_domain":
        True,

    "historical_instability_persistence_domain":
        True,

    "recurring_deterioration_domain":
        True,

    "recovery_replay_domain":
        True,

    "escalation_replay_domain":
        True,

    "intervention_replay_domain_when_supported":
        True,

    "replay_evidence_strength_domain":
        True,

    "excluded_domains": [
        "Patient Memory Storage",
        "Direct Physiologic Measurement",
        "Current Trajectory Interpretation",
        "Future Prediction",
        "Recovery Governance",
        "Therapeutic Effect Determination",
        "Diagnosis",
        "Treatment Recommendation",
        "Clinical Decision Making",
        "Causation Determination",
    ],

    "domain_question":
        "What intelligence domain belongs exclusively to RSS?"
}


def build_rss_domain_table() -> pd.DataFrame:
    """
    Build the RSS replay intelligence domain table.
    """

    return _build_rss_policy_table(
        policy=RSS_REPLAY_DOMAINS,
        key_column="domain",
    )


#============================================================
# Chapter 68.9 — RSS Replay Formation Policy
#============================================================

RSS_REPLAY_FORMATION_POLICY = {

    "formation_principle":
        (
            "Replay-derived lessons are formed through recurrence, "
            "similarity, consistency, and repeated outcome evidence "
            "across earned longitudinal history."
        ),

    "require_accumulated_history":
        True,

    "require_ordered_history":
        True,

    "require_patient_specific_context":
        True,

    "require_comparable_replay_cycles":
        True,

    "require_recurrence":
        True,

    "require_pattern_similarity":
        True,

    "evaluate_repeated_outcomes":
        True,

    "evaluate_cross_cycle_consistency":
        True,

    "evaluate_stability_duration":
        True,

    "evaluate_instability_persistence":
        True,

    "evaluate_recovery_durability":
        True,

    "preserve_missing_history":
        True,

    "preserve_contradictory_history":
        True,

    "preserve_new_pattern_possibility":
        True,

    "permit_no_meaningful_pattern":
        True,

    "permit_insufficient_replay_history":
        True,

    "minimum_recurrence_threshold_is_configurable":
        True,

    "form_replay_lesson_from_single_event":
        False,

    "form_replay_lesson_from_timestamp_proximity_alone":
        False,

    "form_replay_lesson_from_one_favorable_outcome":
        False,

    "form_replay_lesson_from_memory_count_alone":
        False,

    "assume_repeated_history_predicts_future":
        False,

    "require_nid_governance_before_communication":
        True,

    "formation_question":
        "How is a replay-derived historical lesson formed?"
}


def build_rss_replay_formation_table() -> pd.DataFrame:
    """
    Build the RSS replay formation policy table.
    """

    return _build_rss_policy_table(
        policy=RSS_REPLAY_FORMATION_POLICY,
        key_column="formation_policy",
    )


#============================================================
# Chapter 68.10 — RSS Replay Stability States
#============================================================

RSS_REPLAY_PATTERN_STATES = (
    {
        "state_family":
            "PATTERN_MATURITY",

        "state":
            "INSUFFICIENT_REPLAY_HISTORY",

        "description":
            (
                "Accumulated history is insufficient to perform "
                "meaningful replay analysis."
            ),
    },
    {
        "state_family":
            "PATTERN_MATURITY",

        "state":
            "NEW_PATTERN",

        "description":
            (
                "Sufficient comparison history exists, but current "
                "behavior does not resemble an established replay."
            ),
    },
    {
        "state_family":
            "PATTERN_MATURITY",

        "state":
            "EMERGING_PATTERN",

        "description":
            (
                "Early recurrence is present, but consistency or "
                "historical support remains incomplete."
            ),
    },
    {
        "state_family":
            "PATTERN_MATURITY",

        "state":
            "RECURRING_PATTERN",

        "description":
            (
                "A similar stability or instability pattern has "
                "occurred repeatedly across longitudinal history."
            ),
    },
    {
        "state_family":
            "PATTERN_MATURITY",

        "state":
            "CONSISTENT_PATTERN",

        "description":
            (
                "Repeated cycles demonstrate a stable and "
                "consistently supported historical pattern."
            ),
    },
    {
        "state_family":
            "PATTERN_MATURITY",

        "state":
            "NO_MEANINGFUL_PATTERN",

        "description":
            (
                "Sufficient replay history was reviewed, but no "
                "meaningful recurring lesson was supported."
            ),
    },
    {
        "state_family":
            "PATTERN_MATURITY",

        "state":
            "UNKNOWN_REPLAY",

        "description":
            (
                "Replay evidence exists but remains incomplete, "
                "contradictory, unreliable, or difficult to align."
            ),
    },
)


RSS_REPLAY_LESSON_STATES = (
    {
        "state_family":
            "REPLAY_LESSON",

        "state":
            "REPEATED_INSTABILITY",

        "description":
            (
                "Instability has repeatedly returned across "
                "historical cycles."
            ),
    },
    {
        "state_family":
            "REPLAY_LESSON",

        "state":
            "FAILED_RECOVERY_PATTERN",

        "description":
            (
                "Apparent recovery has repeatedly failed to remain "
                "stable across historical cycles."
            ),
    },
    {
        "state_family":
            "REPLAY_LESSON",

        "state":
            "STABILITY_SCARCITY_HIGH",

        "description":
            (
                "Durable stable periods have historically been "
                "rare relative to instability."
            ),
    },
    {
        "state_family":
            "REPLAY_LESSON",

        "state":
            "ESCALATING_REPLAY",

        "description":
            (
                "Repeated instability cycles demonstrate increasing "
                "burden, frequency, persistence, or difficulty "
                "returning to stability."
            ),
    },
    {
        "state_family":
            "REPLAY_LESSON",

        "state":
            "DURABLE_STABILITY_PATTERN",

        "description":
            (
                "Historical cycles repeatedly demonstrate sustained "
                "and durable stability."
            ),
    },
    {
        "state_family":
            "REPLAY_LESSON",

        "state":
            "RECURRING_RECOVERY_PATTERN",

        "description":
            (
                "Recovery behavior has repeatedly appeared across "
                "historical cycles."
            ),
    },
    {
        "state_family":
            "REPLAY_LESSON",

        "state":
            "RECURRING_INTERVENTION_RESPONSE",

        "description":
            (
                "A similar intervention-response pattern has "
                "recurred across verified therapeutic episodes. "
                "This state requires active and valid TTI evidence."
            ),
    },
)


def build_rss_replay_state_table() -> pd.DataFrame:
    """
    Build the standardized RSS replay state table.

    Pattern maturity states communicate how strongly recurrence
    has been established.

    Replay lesson states communicate what repeated history has
    demonstrated.

    The two state families are intentionally separated because
    one pattern maturity state and one or more replay lessons may
    coexist.
    """

    states = [
        *RSS_REPLAY_PATTERN_STATES,
        *RSS_REPLAY_LESSON_STATES,
    ]

    return pd.DataFrame(states)


#============================================================
# Chapter 68.11 — RSS Architectural Review
#============================================================

RSS_ARCHITECTURAL_REVIEW = {

    "purpose_defined":
        True,

    "identity_defined":
        True,

    "governance_defined":
        True,

    "replay_learning_policy_defined":
        True,

    "authority_boundaries_defined":
        True,

    "input_policy_defined":
        True,

    "output_policy_defined":
        True,

    "consumer_policy_defined":
        True,

    "subsystem_relationships_defined":
        True,

    "replay_domains_defined":
        True,

    "replay_formation_policy_defined":
        True,

    "replay_states_defined":
        True,

    "lpmr_memory_boundary_preserved":
        True,

    "confidence_engine_ownership_preserved":
        True,

    "nid_governance_preserved":
        True,

    "insufficient_history_state_supported":
        True,

    "no_meaningful_pattern_state_supported":
        True,

    "architecture_ready_for_implementation":
        True,

    "evidence_review_required_before_activation":
        True,

    "review_question":
        (
            "Has RSS been sufficiently defined for "
            "evidence review and implementation?"
        )
}


def build_rss_architectural_review_table() -> pd.DataFrame:
    """
    Build the RSS architectural readiness review table.
    """

    return _build_rss_policy_table(
        policy=RSS_ARCHITECTURAL_REVIEW,
        key_column="review",
    )


#============================================================
# Public Module Exports
#============================================================

__all__ = [
    "RSS_PURPOSE_POLICY",
    "RSS_IDENTITY_POLICY",
    "RSS_GOVERNANCE_POLICY",
    "RSS_REPLAY_LEARNING_POLICY",
    "RSS_AUTHORITY_POLICY",
    "RSS_INPUT_POLICY",
    "RSS_OUTPUT_POLICY",
    "RSS_CONSUMER_POLICY",
    "RSS_SUBSYSTEM_RELATIONSHIPS",
    "RSS_REPLAY_DOMAINS",
    "RSS_REPLAY_FORMATION_POLICY",
    "RSS_REPLAY_PATTERN_STATES",
    "RSS_REPLAY_LESSON_STATES",
    "RSS_ARCHITECTURAL_REVIEW",
    "build_rss_purpose_table",
    "build_rss_identity_table",
    "build_rss_governance_table",
    "build_rss_replay_learning_policy_table",
    "build_rss_authority_table",
    "build_rss_input_policy_table",
    "build_rss_output_policy_table",
    "build_rss_consumer_policy_table",
    "build_rss_relationship_table",
    "build_rss_domain_table",
    "build_rss_replay_formation_table",
    "build_rss_replay_state_table",
    "build_rss_architectural_review_table",
]