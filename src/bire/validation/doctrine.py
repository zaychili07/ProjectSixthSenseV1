#============================================================
# Project Sixth Sense — BIRE OS
# Chapter 69.1 — System Validation Doctrine and Scope
#============================================================

from __future__ import annotations

import pandas as pd


_BIRE_SYSTEM_VALIDATION_DOCTRINE = [
    {
        "doctrine_id":
            "VALIDATE_SYSTEM_NOT_ISOLATED_LAYER",

        "doctrine_statement":
            (
                "Validation evaluates the complete governed "
                "system rather than isolated component success."
            ),

        "required_behavior":
            (
                "Observe subsystem interactions, limitations, "
                "uncertainty propagation, and authority."
            ),

        "prohibited_interpretation":
            (
                "A locally successful component does not prove "
                "whole-system readiness."
            ),

        "deployment_relevance":
            "SYSTEM_WIDE",
    },
    {
        "doctrine_id":
            "FAILURE_MUST_REMAIN_VISIBLE",

        "doctrine_statement":
            (
                "Failures, exceptions, and unresolved conditions "
                "must remain observable."
            ),

        "required_behavior":
            (
                "Expose the failure state, affected scope, "
                "evidence limitations, and containment status."
            ),

        "prohibited_interpretation":
            (
                "Silent exception handling must not be treated "
                "as successful intelligence production."
            ),

        "deployment_relevance":
            "BLOCKING_IF_VIOLATED",
    },
    {
        "doctrine_id":
            "DEGRADE_BEFORE_FICTION",

        "doctrine_statement":
            (
                "BIRE OS must reduce capability or authority "
                "before manufacturing unsupported intelligence."
            ),

        "required_behavior":
            (
                "Use explicit degraded-operation states and "
                "withhold unsupported outputs."
            ),

        "prohibited_interpretation":
            (
                "Fallback behavior must not invent missing "
                "evidence, certainty, or conclusions."
            ),

        "deployment_relevance":
            "BLOCKING_IF_VIOLATED",
    },
    {
        "doctrine_id":
            "DISAGREEMENT_MUST_BE_PRESERVED",

        "doctrine_statement":
            (
                "Subsystem disagreement is evidence and must "
                "remain visible."
            ),

        "required_behavior":
            (
                "Preserve conflicting observations, responsible "
                "layers, uncertainty, and unresolved state."
            ),

        "prohibited_interpretation":
            (
                "Disagreement must not be averaged or overwritten "
                "into false consensus."
            ),

        "deployment_relevance":
            "BLOCKING_IF_CONCEALED",
    },
    {
        "doctrine_id":
            "AUTHORITY_MUST_REMAIN_LAYER_BOUND",

        "doctrine_statement":
            (
                "Each BIRE OS layer must remain within its "
                "assigned responsibility."
            ),

        "required_behavior":
            (
                "Measurements, interpretations, governance, "
                "memory, display, and replay remain distinct."
            ),

        "prohibited_interpretation":
            (
                "A support layer must not silently become a "
                "prediction or decision authority."
            ),

        "deployment_relevance":
            "BLOCKING_IF_VIOLATED",
    },
    {
        "doctrine_id":
            "UNCERTAINTY_MUST_PROPAGATE",

        "doctrine_statement":
            (
                "Uncertainty and evidence limitations must "
                "survive cross-layer transmission."
            ),

        "required_behavior":
            (
                "Downstream outputs retain relevant missingness, "
                "conflict, censoring, and trust limitations."
            ),

        "prohibited_interpretation":
            (
                "Data movement must not convert uncertainty "
                "into apparent certainty."
            ),

        "deployment_relevance":
            "BLOCKING_IF_LOST",
    },
    {
        "doctrine_id":
            "RERUNS_MUST_BE_DETERMINISTIC",

        "doctrine_statement":
            (
                "Equivalent governed inputs and policies must "
                "produce reproducible structural outputs."
            ),

        "required_behavior":
            (
                "Ordering, identifiers, quarantine decisions, "
                "and validation results remain reproducible."
            ),

        "prohibited_interpretation":
            (
                "Unexplained nondeterministic differences must "
                "not be accepted as normal behavior."
            ),

        "deployment_relevance":
            "REVIEW_OR_BLOCK",
    },
    {
        "doctrine_id":
            "RECOVERY_MUST_NOT_REWRITE_HISTORY",

        "doctrine_statement":
            (
                "Operational recovery may restore execution but "
                "must not silently alter prior observations."
            ),

        "required_behavior":
            (
                "Preserve provenance, prior states, failures, "
                "and the reason for reassessment."
            ),

        "prohibited_interpretation":
            (
                "A restart must not erase evidence that a failure "
                "or disagreement occurred."
            ),

        "deployment_relevance":
            "BLOCKING_IF_VIOLATED",
    },
    {
        "doctrine_id":
            "OBSERVATION_PRECEDES_DEPLOYMENT_DECISION",

        "doctrine_statement":
            (
                "Validation observations must be collected before "
                "deployment authority is considered."
            ),

        "required_behavior":
            (
                "Record aligned behavior, limitations, reviews, "
                "and blocking failures."
            ),

        "prohibited_interpretation":
            (
                "Incomplete validation must not be treated as "
                "evidence of readiness."
            ),

        "deployment_relevance":
            "SYSTEM_WIDE",
    },
    {
        "doctrine_id":
            "FINAL_BINARY_DECISION_ONLY_AT_69_10",

        "doctrine_statement":
            (
                "Binary deployment authorization is reserved "
                "for the final operational boundary."
            ),

        "required_behavior":
            (
                "Intermediate validation uses observational and "
                "governed review states."
            ),

        "prohibited_interpretation":
            (
                "A successful individual test does not authorize "
                "deployment."
            ),

        "deployment_relevance":
            "FINAL_REVIEW_ONLY",
    },
]


_BIRE_SYSTEM_VALIDATION_SCOPE = [
    {
        "validation_domain":
            "ARCHITECTURE_INTEGRITY",

        "validation_question":
            (
                "Do all layers remain within their assigned "
                "responsibilities?"
            ),

        "primary_observation":
            (
                "Authority boundaries, dependency direction, "
                "ownership, and prohibited responsibility transfer."
            ),

        "required_evidence":
            "ARCHITECTURE_INTEGRITY_VALIDATION",

        "unresolved_failure_blocks_deployment":
            True,
    },
    {
        "validation_domain":
            "GOVERNANCE_COMPLIANCE",

        "validation_question":
            (
                "Does system authority remain proportional to "
                "the available evidence?"
            ),

        "primary_observation":
            (
                "Governance gates, deferred capabilities, "
                "uncertainty handling, and authority leakage."
            ),

        "required_evidence":
            "GOVERNANCE_COMPLIANCE_VALIDATION",

        "unresolved_failure_blocks_deployment":
            True,
    },
    {
        "validation_domain":
            "CROSS_LAYER_INTERACTION",

        "validation_question":
            (
                "Do subsystem interactions preserve meaning, "
                "uncertainty, and responsibility?"
            ),

        "primary_observation":
            (
                "Input-output contracts, semantic preservation, "
                "and downstream limitation propagation."
            ),

        "required_evidence":
            "CROSS_LAYER_INTERACTION_VALIDATION",

        "unresolved_failure_blocks_deployment":
            True,
    },
    {
        "validation_domain":
            "INTELLIGENCE_CONSISTENCY",

        "validation_question":
            (
                "Are agreement and disagreement represented "
                "honestly across the system?"
            ),

        "primary_observation":
            (
                "Consensus, contradiction, missing evidence, "
                "subsystem disagreement, and false-consensus risk."
            ),

        "required_evidence":
            "INTELLIGENCE_CONSISTENCY_VALIDATION",

        "unresolved_failure_blocks_deployment":
            True,
    },
    {
        "validation_domain":
            "FAILURE_MODE_CONTAINMENT",

        "validation_question":
            (
                "Can local failures be detected and prevented "
                "from contaminating unrelated intelligence?"
            ),

        "primary_observation":
            (
                "Exception visibility, quarantine, dependency "
                "isolation, and downstream authority withdrawal."
            ),

        "required_evidence":
            "FAILURE_MODE_CONTAINMENT_VALIDATION",

        "unresolved_failure_blocks_deployment":
            True,
    },
    {
        "validation_domain":
            "DEGRADED_OPERATION_AND_RECOVERY",

        "validation_question":
            (
                "Can BIRE OS operate honestly under partial "
                "capability and recover without rewriting history?"
            ),

        "primary_observation":
            (
                "Degraded states, interrupted execution, restart, "
                "replay, reproducibility, and provenance."
            ),

        "required_evidence":
            "DEGRADED_OPERATION_AND_RECOVERY_VALIDATION",

        "unresolved_failure_blocks_deployment":
            True,
    },
    {
        "validation_domain":
            "DEPLOYMENT_BOUNDARY",

        "validation_question":
            (
                "Does final review prevent deployment while "
                "material concerns remain unresolved?"
            ),

        "primary_observation":
            (
                "Validation completeness, blocking findings, "
                "limitations, and final NID authorization."
            ),

        "required_evidence":
            "FINAL_DEPLOYMENT_DECISION_CONTRACT",

        "unresolved_failure_blocks_deployment":
            True,
    },
]


def build_bire_system_validation_doctrine() -> pd.DataFrame:
    """
    Return the Chapter 69 system-validation doctrine.

    Doctrine installation does not execute validation or
    authorize deployment.
    """

    result = pd.DataFrame(
        _BIRE_SYSTEM_VALIDATION_DOCTRINE
    )

    result[
        "validation_doctrine_installed"
    ] = True

    result[
        "operational_deployment_authorized"
    ] = False

    result[
        "nid_authorization_state"
    ] = (
        "NID_CHAPTER_69_VALIDATION_DOCTRINE_INSTALLED"
    )

    return result


def build_bire_system_validation_scope() -> pd.DataFrame:
    """
    Return the governed Chapter 69 validation scope.

    Every listed domain must be addressed before a final
    deployment decision may occur.
    """

    result = pd.DataFrame(
        _BIRE_SYSTEM_VALIDATION_SCOPE
    )

    result[
        "validation_execution_state"
    ] = "NOT_EVALUATED"

    result[
        "validation_evidence_complete"
    ] = False

    result[
        "deployment_decision_authorized"
    ] = False

    result[
        "operational_deployment_authorized"
    ] = False

    result[
        "nid_authorization_state"
    ] = (
        "NID_CHAPTER_69_VALIDATION_SCOPE_INSTALLED"
    )

    return result


def build_bire_system_validation_entry_contract(
    rss_readiness_contract_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Verify the Chapter 68 RSS handoff and establish the
    Chapter 69 validation-entry contract.

    The contract authorizes controlled validation execution only.
    It does not authorize operational deployment.
    """

    if not isinstance(
        rss_readiness_contract_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "rss_readiness_contract_df must be a pandas DataFrame."
        )

    required_columns = {
        "rss_readiness_state",
        "playground_entry_authorized",
        "operational_deployment_authorized",
        "no_prohibited_authority_leak",
        "prohibited_authority_leak_count",
        "next_required_stage",
        "nid_authorization_state",
    }

    missing_columns = sorted(
        required_columns
        - set(rss_readiness_contract_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "Chapter 69 validation entry cannot proceed. "
            f"Missing RSS readiness columns: {missing_columns}"
        )

    if len(rss_readiness_contract_df) != 1:
        raise ValueError(
            "RSS readiness contract must contain exactly one row."
        )

    rss_contract = (
        rss_readiness_contract_df.iloc[0]
    )

    rss_playground_readiness_confirmed = (
        str(
            rss_contract[
                "rss_readiness_state"
            ]
        )
        ==
        "RSS_PLAYGROUND_READY_WITH_GOVERNED_LIMITATIONS"
    )

    rss_playground_entry_authorized = bool(
        rss_contract[
            "playground_entry_authorized"
        ]
    )

    rss_deployment_authority_withheld = (
        not bool(
            rss_contract[
                "operational_deployment_authorized"
            ]
        )
    )

    rss_no_authority_leak_confirmed = bool(
        rss_contract[
            "no_prohibited_authority_leak"
        ]
    )

    rss_authority_leak_count_zero = (
        int(
            rss_contract[
                "prohibited_authority_leak_count"
            ]
        )
        == 0
    )

    chapter_68_to_69_handoff_confirmed = (
        str(
            rss_contract[
                "next_required_stage"
            ]
        )
        ==
        "CHAPTER_69_SYSTEM_VALIDATION_AND_FINAL_REVIEW"
    )

    entry_checks = {
        "rss_playground_readiness_confirmed":
            rss_playground_readiness_confirmed,

        "rss_playground_entry_authorized":
            rss_playground_entry_authorized,

        "rss_deployment_authority_withheld":
            rss_deployment_authority_withheld,

        "rss_no_authority_leak_confirmed":
            rss_no_authority_leak_confirmed,

        "rss_authority_leak_count_zero":
            rss_authority_leak_count_zero,

        "chapter_68_to_69_handoff_confirmed":
            chapter_68_to_69_handoff_confirmed,
    }

    blocking_conditions = [
        check_name
        for check_name, passed in entry_checks.items()
        if not passed
    ]

    chapter_69_entry_authorized = (
        len(blocking_conditions) == 0
    )

    chapter_entry_state = (
        "CHAPTER_69_SYSTEM_VALIDATION_ENTRY_AUTHORIZED"
        if chapter_69_entry_authorized
        else
        "CHAPTER_69_SYSTEM_VALIDATION_ENTRY_BLOCKED"
    )

    return pd.DataFrame(
        [
            {
                "chapter":
                    (
                        "CHAPTER_69_SYSTEM_VALIDATION_"
                        "AND_FINAL_REVIEW"
                    ),

                "chapter_status":
                    (
                        "ACTIVE"
                        if chapter_69_entry_authorized
                        else
                        "BLOCKED"
                    ),

                "chapter_entry_state":
                    chapter_entry_state,

                **entry_checks,

                "blocking_condition_count":
                    len(blocking_conditions),

                "blocking_conditions":
                    (
                        "NONE"
                        if not blocking_conditions
                        else " | ".join(
                            blocking_conditions
                        )
                    ),

                "validation_evidence_collection_authorized":
                    chapter_69_entry_authorized,

                "cross_layer_stress_testing_authorized":
                    chapter_69_entry_authorized,

                "playground_scenario_execution_authorized":
                    chapter_69_entry_authorized,

                "automatic_issue_repair_authorized":
                    False,

                "silent_exception_suppression_authorized":
                    False,

                "deployment_decision_authorized":
                    False,

                "operational_deployment_authorized":
                    False,

                "nid_authorization_state":
                    (
                        "NID_CHAPTER_69_VALIDATION_ENTRY_INSTALLED"
                        if chapter_69_entry_authorized
                        else
                        "NID_CHAPTER_69_ENTRY_REVIEW_REQUIRED"
                    ),

                "next_required_stage":
                    (
                        "CHAPTER_69_2_PSS_PREDEPLOYMENT_"
                        "FRAMEWORK_ADOPTION"
                    ),

                "central_validation_question":
                    "CAN_BIRE_OS_SURVIVE_REALITY",
            }
        ]
    )


__all__ = [
    "build_bire_system_validation_doctrine",
    "build_bire_system_validation_scope",
    "build_bire_system_validation_entry_contract",
]