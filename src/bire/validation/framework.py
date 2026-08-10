#============================================================
# Project Sixth Sense — BIRE OS
# Chapter 69.2 — PSS Predeployment and Lifecycle
#                  Evaluation Framework Adoption
#============================================================

from __future__ import annotations

import pandas as pd


_PSS_CORE_EVALUATION_DOMAINS = [
    {
        "evaluation_domain":
            "TASK_PERFORMANCE",

        "framework_question":
            (
                "Does the system perform its assigned "
                "domain responsibility effectively?"
            ),

        "required_observation":
            (
                "System-specific performance, missed behavior, "
                "false activation, timeliness, and output quality."
            ),

        "non_compensatory_gate":
            False,
    },
    {
        "evaluation_domain":
            "GOVERNANCE_COMPLIANCE",

        "framework_question":
            (
                "Does system authority remain proportional "
                "to available evidence?"
            ),

        "required_observation":
            (
                "Authority leakage, prohibited behavior, "
                "deferred capability activation, and NID gates."
            ),

        "non_compensatory_gate":
            True,
    },
    {
        "evaluation_domain":
            "UNCERTAINTY_HONESTY",

        "framework_question":
            (
                "Does uncertainty remain visible and survive "
                "cross-layer transmission?"
            ),

        "required_observation":
            (
                "Missingness, contradiction, censoring, "
                "confidence limitation, and unsupported certainty."
            ),

        "non_compensatory_gate":
            True,
    },
    {
        "evaluation_domain":
            "ARCHITECTURE_INTEGRITY",

        "framework_question":
            (
                "Do system layers remain within their assigned "
                "responsibilities and dependency directions?"
            ),

        "required_observation":
            (
                "Layer authority, ownership, interfaces, "
                "forbidden responsibility transfer, and coupling."
            ),

        "non_compensatory_gate":
            True,
    },
    {
        "evaluation_domain":
            "CROSS_LAYER_CONSISTENCY",

        "framework_question":
            (
                "Do layer interactions preserve meaning, "
                "disagreement, and evidence limitations?"
            ),

        "required_observation":
            (
                "Agreement, disagreement, semantic preservation, "
                "false consensus, and limitation propagation."
            ),

        "non_compensatory_gate":
            True,
    },
    {
        "evaluation_domain":
            "DEGRADED_OPERATION_RESILIENCE",

        "framework_question":
            (
                "Can the system remain honest when full "
                "capability or complete evidence is unavailable?"
            ),

        "required_observation":
            (
                "Partial evidence, inactive dependencies, "
                "delayed data, corrupted inputs, and degradation."
            ),

        "non_compensatory_gate":
            True,
    },
    {
        "evaluation_domain":
            "FAILURE_CONTAINMENT",

        "framework_question":
            (
                "Are local failures detected, contained, and "
                "prevented from contaminating unrelated outputs?"
            ),

        "required_observation":
            (
                "Failure visibility, quarantine, affected scope, "
                "authority withdrawal, and unsafe propagation."
            ),

        "non_compensatory_gate":
            True,
    },
    {
        "evaluation_domain":
            "DETERMINISTIC_REPRODUCIBILITY",

        "framework_question":
            (
                "Do equivalent governed inputs and policies "
                "produce reproducible structural outputs?"
            ),

        "required_observation":
            (
                "Run manifests, artifact hashes, ordering, "
                "random seeds, outputs, and rerun comparisons."
            ),

        "non_compensatory_gate":
            True,
    },
    {
        "evaluation_domain":
            "GENERALIZATION",

        "framework_question":
            (
                "Does demonstrated improvement extend beyond "
                "familiar or development-influenced scenarios?"
            ),

        "required_observation":
            (
                "Sealed-suite performance, rotating novelty, "
                "adversarial performance, and generalization gap."
            ),

        "non_compensatory_gate":
            True,
    },
    {
        "evaluation_domain":
            "REGRESSION",

        "framework_question":
            (
                "Did an improvement reduce previously "
                "demonstrated capability?"
            ),

        "required_observation":
            (
                "Frozen benchmark comparison, critical "
                "regressions, severity, and affected domains."
            ),

        "non_compensatory_gate":
            True,
    },
    {
        "evaluation_domain":
            "RECOVERY_BEHAVIOR",

        "framework_question":
            (
                "Can execution recover without erasing or "
                "rewriting prior evidence?"
            ),

        "required_observation":
            (
                "Restart, replay, checkpoint restoration, "
                "provenance, prior failure state, and rerun output."
            ),

        "non_compensatory_gate":
            True,
    },
]


_PSS_BENCHMARK_SUITE_REGISTRY = [
    {
        "benchmark_suite":
            "DEVELOPMENT_SUITE",

        "suite_purpose":
            "DEBUGGING_AND_IMPLEMENTATION_SUPPORT",

        "scenario_visibility":
            "VISIBLE_TO_DEVELOPMENT",

        "scenario_reuse_policy":
            "REUSE_PERMITTED",

        "deployment_evidence_eligible":
            False,

        "generalization_evidence_eligible":
            False,
    },
    {
        "benchmark_suite":
            "FROZEN_REGRESSION_SUITE",

        "suite_purpose":
            "VERSION_TO_VERSION_REGRESSION_DETECTION",

        "scenario_visibility":
            "KNOWN_BUT_FROZEN",

        "scenario_reuse_policy":
            "REQUIRED_ACROSS_RELEASE_CANDIDATES",

        "deployment_evidence_eligible":
            True,

        "generalization_evidence_eligible":
            False,
    },
    {
        "benchmark_suite":
            "SEALED_GENERALIZATION_SUITE",

        "suite_purpose":
            "UNSEEN_SCENARIO_GENERALIZATION_TESTING",

        "scenario_visibility":
            "SEALED_FROM_ROUTINE_DEVELOPMENT",

        "scenario_reuse_policy":
            "RESTRICTED_GOVERNED_ACCESS",

        "deployment_evidence_eligible":
            True,

        "generalization_evidence_eligible":
            True,
    },
    {
        "benchmark_suite":
            "ROTATING_NOVELTY_SUITE",

        "suite_purpose":
            "NEW_VARIATIONS_OF_KNOWN_FAILURE_STRUCTURES",

        "scenario_visibility":
            "VERSIONED_AND_ROTATING",

        "scenario_reuse_policy":
            "RETIRED_AFTER_DEVELOPMENT_INFLUENCE",

        "deployment_evidence_eligible":
            True,

        "generalization_evidence_eligible":
            True,
    },
    {
        "benchmark_suite":
            "ADVERSARIAL_PLAYGROUND_SUITE",

        "suite_purpose":
            "HOSTILE_UNCERTAINTY_AND_FAILURE_STRESS_TESTING",

        "scenario_visibility":
            "CONTROLLED_AND_VERSIONED",

        "scenario_reuse_policy":
            "DIFFICULTY_MUST_NOT_DECREASE_SILENTLY",

        "deployment_evidence_eligible":
            True,

        "generalization_evidence_eligible":
            True,
    },
]


_PSS_LIFECYCLE_EVALUATION_PHASES = [
    {
        "phase_order": 1,
        "evaluation_phase": "BASELINE_ESTABLISHMENT",
        "required_output": "IMMUTABLE_REFERENCE_BASELINE",
    },
    {
        "phase_order": 2,
        "evaluation_phase": "RELEASE_CANDIDATE_COMPARISON",
        "required_output": "PREDEPLOYMENT_COMPARATIVE_EVIDENCE",
    },
    {
        "phase_order": 3,
        "evaluation_phase": "CONTROLLED_PLAYGROUND_EXECUTION",
        "required_output": "STRESS_AND_FAILURE_OBSERVATIONS",
    },
    {
        "phase_order": 4,
        "evaluation_phase": "GOVERNED_IMPROVEMENT",
        "required_output": "VERSIONED_CHANGE_AND_RATIONALE",
    },
    {
        "phase_order": 5,
        "evaluation_phase": "REGRESSION_AND_GENERALIZATION_RETEST",
        "required_output": "POST_CHANGE_COMPARATIVE_EVIDENCE",
    },
    {
        "phase_order": 6,
        "evaluation_phase": "HARDER_PLAYGROUND_REENTRY",
        "required_output": "CHALLENGE_ADJUSTED_IMPROVEMENT_EVIDENCE",
    },
    {
        "phase_order": 7,
        "evaluation_phase": "POSTDEPLOYMENT_OBSERVATION",
        "required_output": "SEPARATE_LINKED_OPERATIONAL_EVIDENCE",
    },
]


def build_pss_core_evaluation_domain_registry() -> pd.DataFrame:
    """
    Return the PSS-wide predeployment and lifecycle evaluation
    domains.

    These domains apply to all PSS systems. System-specific
    profiles add domain measurements without replacing the
    global framework.
    """

    result = pd.DataFrame(
        _PSS_CORE_EVALUATION_DOMAINS
    )

    result[
        "pss_framework_requirement"
    ] = True

    result[
        "evaluation_state"
    ] = "NOT_EVALUATED"

    result[
        "deployment_authority_granted"
    ] = False

    result[
        "nid_governance_state"
    ] = (
        "NID_PSS_EVALUATION_DOMAIN_REGISTERED"
    )

    return result


def build_pss_benchmark_suite_registry() -> pd.DataFrame:
    """
    Return the governed PSS benchmark-suite architecture.
    """

    result = pd.DataFrame(
        _PSS_BENCHMARK_SUITE_REGISTRY
    )

    result[
        "suite_registry_state"
    ] = "PSS_BENCHMARK_SUITE_REGISTERED"

    result[
        "benchmark_execution_authorized"
    ] = False

    result[
        "operational_deployment_authorized"
    ] = False

    result[
        "nid_governance_state"
    ] = (
        "NID_PSS_BENCHMARK_SUITE_REGISTERED"
    )

    return result


def build_pss_lifecycle_evaluation_phase_registry() -> pd.DataFrame:
    """
    Return the governed PSS evaluation lifecycle.
    """

    result = pd.DataFrame(
        _PSS_LIFECYCLE_EVALUATION_PHASES
    )

    result[
        "phase_execution_state"
    ] = "NOT_STARTED"

    result[
        "phase_evidence_complete"
    ] = False

    result[
        "operational_deployment_authorized"
    ] = False

    result[
        "nid_governance_state"
    ] = (
        "NID_PSS_LIFECYCLE_PHASE_REGISTERED"
    )

    return result


def build_bire_pss_framework_adoption_contract(
    validation_entry_contract_df: pd.DataFrame,
    rss_readiness_contract_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Formally adopt the PSS predeployment and lifecycle evaluation
    framework for BIRE OS Chapter 69.

    This authorizes framework and manifest construction only.
    Benchmark execution and operational deployment remain
    unauthorized.
    """

    if not isinstance(
        validation_entry_contract_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "validation_entry_contract_df must be a DataFrame."
        )

    if not isinstance(
        rss_readiness_contract_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "rss_readiness_contract_df must be a DataFrame."
        )

    required_entry_columns = {
        "chapter_entry_state",
        "validation_evidence_collection_authorized",
        "playground_scenario_execution_authorized",
        "operational_deployment_authorized",
    }

    missing_entry_columns = sorted(
        required_entry_columns
        - set(validation_entry_contract_df.columns)
    )

    if missing_entry_columns:
        raise KeyError(
            "BIRE PSS framework adoption cannot proceed. "
            f"Missing validation-entry columns: "
            f"{missing_entry_columns}"
        )

    required_rss_columns = {
        "rss_readiness_state",
        "playground_entry_authorized",
        "operational_deployment_authorized",
        "no_prohibited_authority_leak",
    }

    missing_rss_columns = sorted(
        required_rss_columns
        - set(rss_readiness_contract_df.columns)
    )

    if missing_rss_columns:
        raise KeyError(
            "BIRE PSS framework adoption cannot proceed. "
            f"Missing RSS readiness columns: "
            f"{missing_rss_columns}"
        )

    if len(validation_entry_contract_df) != 1:
        raise ValueError(
            "validation_entry_contract_df must contain one row."
        )

    if len(rss_readiness_contract_df) != 1:
        raise ValueError(
            "rss_readiness_contract_df must contain one row."
        )

    validation_entry = (
        validation_entry_contract_df.iloc[0]
    )

    rss_readiness = (
        rss_readiness_contract_df.iloc[0]
    )

    chapter_69_entry_confirmed = (
        validation_entry[
            "chapter_entry_state"
        ]
        ==
        "CHAPTER_69_SYSTEM_VALIDATION_ENTRY_AUTHORIZED"
    )

    validation_collection_confirmed = bool(
        validation_entry[
            "validation_evidence_collection_authorized"
        ]
    )

    rss_playground_ready = (
        rss_readiness[
            "rss_readiness_state"
        ]
        ==
        "RSS_PLAYGROUND_READY_WITH_GOVERNED_LIMITATIONS"
    )

    rss_authority_clean = bool(
        rss_readiness[
            "no_prohibited_authority_leak"
        ]
    )

    deployment_authority_withheld = (
        not bool(
            validation_entry[
                "operational_deployment_authorized"
            ]
        )
        and
        not bool(
            rss_readiness[
                "operational_deployment_authorized"
            ]
        )
    )

    adoption_checks = {
        "chapter_69_entry_confirmed":
            chapter_69_entry_confirmed,

        "validation_evidence_collection_confirmed":
            validation_collection_confirmed,

        "rss_playground_readiness_confirmed":
            rss_playground_ready,

        "rss_authority_clean":
            rss_authority_clean,

        "deployment_authority_withheld":
            deployment_authority_withheld,
    }

    failed_checks = [
        check
        for check, passed in adoption_checks.items()
        if not passed
    ]

    framework_adoption_authorized = (
        len(failed_checks) == 0
    )

    framework_state = (
        "BIRE_PSS_PREDEPLOYMENT_FRAMEWORK_ADOPTED"
        if framework_adoption_authorized
        else
        "BIRE_PSS_FRAMEWORK_ADOPTION_BLOCKED"
    )

    return pd.DataFrame(
        [
            {
                "chapter":
                    "CHAPTER_69_SYSTEM_VALIDATION_AND_FINAL_REVIEW",

                "framework":
                    (
                        "PSS_PREDEPLOYMENT_AND_"
                        "LIFECYCLE_EVALUATION_FRAMEWORK"
                    ),

                "framework_adoption_state":
                    framework_state,

                **adoption_checks,

                "failed_adoption_check_count":
                    len(failed_checks),

                "failed_adoption_checks":
                    (
                        "NONE"
                        if not failed_checks
                        else " | ".join(failed_checks)
                    ),

                "pss_core_domain_count":
                    len(
                        _PSS_CORE_EVALUATION_DOMAINS
                    ),

                "benchmark_suite_count":
                    len(
                        _PSS_BENCHMARK_SUITE_REGISTRY
                    ),

                "lifecycle_phase_count":
                    len(
                        _PSS_LIFECYCLE_EVALUATION_PHASES
                    ),

                "bire_system_profile_required":
                    True,

                "run_manifest_required":
                    True,

                "scenario_isolation_required":
                    True,

                "immutable_evaluation_ledger_required":
                    True,

                "sealed_generalization_required":
                    True,

                "adversarial_playground_required":
                    True,

                "non_compensatory_governance_gates_required":
                    True,

                "challenge_adjusted_comparison_required":
                    True,

                "codex_required":
                    False,

                "codex_integration_authorized":
                    False,

                "codex_integration_state":
                    (
                        "NOT_REQUIRED_FOR_CHAPTER_69_"
                        "OR_PLAYGROUND_ENTRY"
                    ),

                "framework_construction_authorized":
                    framework_adoption_authorized,

                "run_manifest_construction_authorized":
                    framework_adoption_authorized,

                "benchmark_registry_construction_authorized":
                    framework_adoption_authorized,

                "benchmark_execution_authorized":
                    False,

                "playground_baseline_execution_authorized":
                    False,

                "deployment_decision_authorized":
                    False,

                "operational_deployment_authorized":
                    False,

                "nid_authorization_state":
                    (
                        "NID_BIRE_PSS_FRAMEWORK_ADOPTION_INSTALLED"
                        if framework_adoption_authorized
                        else
                        "NID_BIRE_PSS_FRAMEWORK_ADOPTION_REVIEW_REQUIRED"
                    ),

                "next_required_stage":
                    (
                        "CHAPTER_69_3_EVALUATION_RUN_MANIFEST_"
                        "AND_REPRODUCIBILITY_CONTRACT"
                    ),
            }
        ]
    )


__all__ = [
    "build_pss_core_evaluation_domain_registry",
    "build_pss_benchmark_suite_registry",
    "build_pss_lifecycle_evaluation_phase_registry",
    "build_bire_pss_framework_adoption_contract",
]