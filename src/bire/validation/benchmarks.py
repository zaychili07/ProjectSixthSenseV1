#============================================================
# Project Sixth Sense — BIRE OS
# Chapter 69.4 — Benchmark Suite Registry and
#                  Scenario Isolation Policy
#============================================================

from __future__ import annotations

import pandas as pd


_BIRE_BENCHMARK_REGISTRY_VERSION = (
    "BIRE_BENCHMARK_REGISTRY_V1"
)

_PSS_SCENARIO_ISOLATION_POLICY_VERSION = (
    "PSS_SCENARIO_ISOLATION_POLICY_V1"
)

_BIRE_SCENARIO_MANIFEST_SCHEMA_VERSION = (
    "BIRE_SCENARIO_MANIFEST_V1"
)


_BIRE_BENCHMARK_SUITE_SPECIFICATIONS = {
    "DEVELOPMENT_SUITE": {
        "bire_suite_id":
            "BIRE_DEVELOPMENT_SUITE",

        "bire_suite_specification_version":
            "BIRE_DEVELOPMENT_SUITE_V1",

        "initial_scenario_isolation_state":
            "DEVELOPMENT_VISIBLE",

        "fresh_generalization_claim_authorized":
            False,

        "fresh_novelty_claim_authorized":
            False,

        "fresh_adversarial_claim_authorized":
            False,

        "known_scenario_reuse_authorized":
            True,

        "scenario_manifest_required":
            True,

        "intended_bire_use":
            (
                "DEBUGGING_IMPLEMENTATION_AND_"
                "FAILURE_REPRODUCTION"
            ),
    },

    "FROZEN_REGRESSION_SUITE": {
        "bire_suite_id":
            "BIRE_FROZEN_REGRESSION_SUITE",

        "bire_suite_specification_version":
            "BIRE_FROZEN_REGRESSION_SUITE_V1",

        "initial_scenario_isolation_state":
            "KNOWN_FROZEN_REFERENCE",

        "fresh_generalization_claim_authorized":
            False,

        "fresh_novelty_claim_authorized":
            False,

        "fresh_adversarial_claim_authorized":
            False,

        "known_scenario_reuse_authorized":
            True,

        "scenario_manifest_required":
            True,

        "intended_bire_use":
            (
                "VERSION_TO_VERSION_CAPABILITY_"
                "REGRESSION_DETECTION"
            ),
    },

    "SEALED_GENERALIZATION_SUITE": {
        "bire_suite_id":
            "BIRE_SEALED_GENERALIZATION_SUITE",

        "bire_suite_specification_version":
            "BIRE_SEALED_GENERALIZATION_SUITE_V1",

        "initial_scenario_isolation_state":
            "SEALED_UNSEEN",

        "fresh_generalization_claim_authorized":
            True,

        "fresh_novelty_claim_authorized":
            False,

        "fresh_adversarial_claim_authorized":
            False,

        "known_scenario_reuse_authorized":
            False,

        "scenario_manifest_required":
            True,

        "intended_bire_use":
            (
                "UNSEEN_PREDEPLOYMENT_"
                "GENERALIZATION_EVALUATION"
            ),
    },

    "ROTATING_NOVELTY_SUITE": {
        "bire_suite_id":
            "BIRE_ROTATING_NOVELTY_SUITE",

        "bire_suite_specification_version":
            "BIRE_ROTATING_NOVELTY_SUITE_V1",

        "initial_scenario_isolation_state":
            "ROTATING_UNSEEN",

        "fresh_generalization_claim_authorized":
            True,

        "fresh_novelty_claim_authorized":
            True,

        "fresh_adversarial_claim_authorized":
            False,

        "known_scenario_reuse_authorized":
            False,

        "scenario_manifest_required":
            True,

        "intended_bire_use":
            (
                "NEW_VARIATIONS_OF_KNOWN_"
                "UNCERTAINTY_AND_FAILURE_STRUCTURES"
            ),
    },

    "ADVERSARIAL_PLAYGROUND_SUITE": {
        "bire_suite_id":
            "BIRE_ADVERSARIAL_PLAYGROUND_SUITE",

        "bire_suite_specification_version":
            "BIRE_ADVERSARIAL_PLAYGROUND_SUITE_V1",

        "initial_scenario_isolation_state":
            "ADVERSARIAL_SEALED",

        "fresh_generalization_claim_authorized":
            True,

        "fresh_novelty_claim_authorized":
            True,

        "fresh_adversarial_claim_authorized":
            True,

        "known_scenario_reuse_authorized":
            False,

        "scenario_manifest_required":
            True,

        "intended_bire_use":
            (
                "HOSTILE_FULL_SYSTEM_PLAYGROUND_"
                "STRESS_EVALUATION"
            ),
    },
}


_BIRE_SCENARIO_ISOLATION_RULES = [
    {
        "benchmark_suite":
            "DEVELOPMENT_SUITE",

        "pre_evaluation_visibility_state":
            "DEVELOPMENT_VISIBLE",

        "development_access_before_run_authorized":
            True,

        "development_influence_before_run_authorized":
            True,

        "fresh_generalization_evidence_eligible":
            False,

        "fresh_novelty_evidence_eligible":
            False,

        "fresh_adversarial_evidence_eligible":
            False,

        "post_exposure_state":
            "DEVELOPMENT_VISIBLE",

        "reuse_after_exposure_rule":
            "REUSE_PERMITTED_FOR_DEVELOPMENT",

        "contamination_response":
            "NOT_APPLICABLE_ALREADY_DEVELOPMENT_VISIBLE",
    },

    {
        "benchmark_suite":
            "FROZEN_REGRESSION_SUITE",

        "pre_evaluation_visibility_state":
            "KNOWN_FROZEN_REFERENCE",

        "development_access_before_run_authorized":
            True,

        "development_influence_before_run_authorized":
            True,

        "fresh_generalization_evidence_eligible":
            False,

        "fresh_novelty_evidence_eligible":
            False,

        "fresh_adversarial_evidence_eligible":
            False,

        "post_exposure_state":
            "KNOWN_FROZEN_REFERENCE",

        "reuse_after_exposure_rule":
            "REUSE_REQUIRED_FOR_REGRESSION_COMPARABILITY",

        "contamination_response":
            (
                "KNOWN_REFERENCE_STATUS_PRESERVED_"
                "UNSEEN_CLAIM_PROHIBITED"
            ),
    },

    {
        "benchmark_suite":
            "SEALED_GENERALIZATION_SUITE",

        "pre_evaluation_visibility_state":
            "SEALED_UNSEEN",

        "development_access_before_run_authorized":
            False,

        "development_influence_before_run_authorized":
            False,

        "fresh_generalization_evidence_eligible":
            True,

        "fresh_novelty_evidence_eligible":
            False,

        "fresh_adversarial_evidence_eligible":
            False,

        "post_exposure_state":
            "EXPOSED_AFTER_GOVERNED_EVALUATION",

        "reuse_after_exposure_rule":
            (
                "MAY_REUSE_AS_KNOWN_REGRESSION_OR_"
                "DEVELOPMENT_EVIDENCE_ONLY"
            ),

        "contamination_response":
            (
                "PERMANENTLY_REMOVE_FRESH_GENERALIZATION_"
                "ELIGIBILITY_FOR_THIS_SCENARIO_VERSION"
            ),
    },

    {
        "benchmark_suite":
            "ROTATING_NOVELTY_SUITE",

        "pre_evaluation_visibility_state":
            "ROTATING_UNSEEN",

        "development_access_before_run_authorized":
            False,

        "development_influence_before_run_authorized":
            False,

        "fresh_generalization_evidence_eligible":
            True,

        "fresh_novelty_evidence_eligible":
            True,

        "fresh_adversarial_evidence_eligible":
            False,

        "post_exposure_state":
            "EXPOSED_AFTER_GOVERNED_EVALUATION",

        "reuse_after_exposure_rule":
            (
                "RETIRED_FROM_FRESH_NOVELTY_AFTER_"
                "DEVELOPMENT_INFLUENCE"
            ),

        "contamination_response":
            (
                "PERMANENTLY_REMOVE_FRESH_NOVELTY_AND_"
                "GENERALIZATION_ELIGIBILITY_FOR_THIS_VERSION"
            ),
    },

    {
        "benchmark_suite":
            "ADVERSARIAL_PLAYGROUND_SUITE",

        "pre_evaluation_visibility_state":
            "ADVERSARIAL_SEALED",

        "development_access_before_run_authorized":
            False,

        "development_influence_before_run_authorized":
            False,

        "fresh_generalization_evidence_eligible":
            True,

        "fresh_novelty_evidence_eligible":
            True,

        "fresh_adversarial_evidence_eligible":
            True,

        "post_exposure_state":
            "EXPOSED_AFTER_GOVERNED_EVALUATION",

        "reuse_after_exposure_rule":
            (
                "MAY_REPLAY_FOR_REGRESSION_BUT_"
                "NEW_OR_HARDER_VERSION_REQUIRED_FOR_"
                "FRESH_ADVERSARIAL_EVIDENCE"
            ),

        "contamination_response":
            (
                "PERMANENTLY_REMOVE_FRESH_ADVERSARIAL_"
                "GENERALIZATION_ELIGIBILITY_FOR_THIS_VERSION"
            ),
    },
]


_BIRE_SCENARIO_MANIFEST_FIELDS = [
    {
        "field_category":
            "SCENARIO_IDENTITY",

        "field_name":
            "scenario_id",

        "required_for_materialization":
            True,

        "field_purpose":
            "Immutable scenario identity.",
    },
    {
        "field_category":
            "SCENARIO_IDENTITY",

        "field_name":
            "scenario_version",

        "required_for_materialization":
            True,

        "field_purpose":
            "Version of the scenario definition.",
    },
    {
        "field_category":
            "SCENARIO_IDENTITY",

        "field_name":
            "benchmark_suite",

        "required_for_materialization":
            True,

        "field_purpose":
            "Governed benchmark-suite membership.",
    },
    {
        "field_category":
            "SCENARIO_IDENTITY",

        "field_name":
            "scenario_family",

        "required_for_materialization":
            True,

        "field_purpose":
            (
                "Behavioral, uncertainty, corruption, or "
                "failure family represented by the scenario."
            ),
    },
    {
        "field_category":
            "SCENARIO_IDENTITY",

        "field_name":
            "scenario_description",

        "required_for_materialization":
            True,

        "field_purpose":
            "Human-readable scenario description.",
    },
    {
        "field_category":
            "LINEAGE",

        "field_name":
            "parent_scenario_id",

        "required_for_materialization":
            False,

        "field_purpose":
            (
                "Links a derivative or harder scenario "
                "to its originating scenario."
            ),
    },
    {
        "field_category":
            "LINEAGE",

        "field_name":
            "source_origin",

        "required_for_materialization":
            True,

        "field_purpose":
            (
                "Records whether the scenario was synthetic, "
                "derived, historical, or manually authored."
            ),
    },
    {
        "field_category":
            "CONTENT",

        "field_name":
            "scenario_content_sha256",

        "required_for_materialization":
            True,

        "field_purpose":
            "Fingerprints the exact scenario content.",
    },
    {
        "field_category":
            "CONTENT",

        "field_name":
            "generation_seed",

        "required_for_materialization":
            False,

        "field_purpose":
            (
                "Preserves generation randomness when "
                "the scenario is procedurally generated."
            ),
    },
    {
        "field_category":
            "CHALLENGE",

        "field_name":
            "difficulty_profile_version",

        "required_for_materialization":
            True,

        "field_purpose":
            "Identifies the difficulty-definition version.",
    },
    {
        "field_category":
            "CHALLENGE",

        "field_name":
            "difficulty_level",

        "required_for_materialization":
            True,

        "field_purpose":
            "Records the governed scenario difficulty level.",
    },
    {
        "field_category":
            "CHALLENGE",

        "field_name":
            "novelty_profile_version",

        "required_for_materialization":
            True,

        "field_purpose":
            "Identifies the novelty-definition version.",
    },
    {
        "field_category":
            "CHALLENGE",

        "field_name":
            "novelty_level",

        "required_for_materialization":
            True,

        "field_purpose":
            "Records the governed scenario novelty level.",
    },
    {
        "field_category":
            "ISOLATION",

        "field_name":
            "initial_visibility_state",

        "required_for_materialization":
            True,

        "field_purpose":
            "Records the scenario's original isolation state.",
    },
    {
        "field_category":
            "ISOLATION",

        "field_name":
            "current_visibility_state",

        "required_for_materialization":
            True,

        "field_purpose":
            "Records the scenario's current visibility state.",
    },
    {
        "field_category":
            "ISOLATION",

        "field_name":
            "development_access_count",

        "required_for_materialization":
            True,

        "field_purpose":
            (
                "Counts governed development-access events."
            ),
    },
    {
        "field_category":
            "ISOLATION",

        "field_name":
            "development_influence_flag",

        "required_for_materialization":
            True,

        "field_purpose":
            (
                "Indicates whether the scenario materially "
                "influenced system development."
            ),
    },
    {
        "field_category":
            "ISOLATION",

        "field_name":
            "contamination_state",

        "required_for_materialization":
            True,

        "field_purpose":
            (
                "Preserves whether unseen or novelty status "
                "has been compromised."
            ),
    },
    {
        "field_category":
            "EVIDENCE_ELIGIBILITY",

        "field_name":
            "fresh_generalization_evidence_eligible",

        "required_for_materialization":
            True,

        "field_purpose":
            (
                "Controls whether the scenario may support "
                "a fresh generalization claim."
            ),
    },
    {
        "field_category":
            "EVIDENCE_ELIGIBILITY",

        "field_name":
            "fresh_novelty_evidence_eligible",

        "required_for_materialization":
            True,

        "field_purpose":
            (
                "Controls whether the scenario may support "
                "a fresh novelty claim."
            ),
    },
    {
        "field_category":
            "EVIDENCE_ELIGIBILITY",

        "field_name":
            "fresh_adversarial_evidence_eligible",

        "required_for_materialization":
            True,

        "field_purpose":
            (
                "Controls whether the scenario may support "
                "fresh adversarial evidence."
            ),
    },
    {
        "field_category":
            "EVIDENCE_ELIGIBILITY",

        "field_name":
            "regression_reuse_eligible",

        "required_for_materialization":
            True,

        "field_purpose":
            (
                "Controls whether the scenario may later "
                "serve as known regression evidence."
            ),
    },
    {
        "field_category":
            "EVALUATION_HISTORY",

        "field_name":
            "first_governed_evaluation_run_id",

        "required_for_materialization":
            False,

        "field_purpose":
            (
                "Records the first governed run that "
                "evaluated the scenario."
            ),
    },
    {
        "field_category":
            "EVALUATION_HISTORY",

        "field_name":
            "last_governed_evaluation_run_id",

        "required_for_materialization":
            False,

        "field_purpose":
            (
                "Records the most recent governed run."
            ),
    },
    {
        "field_category":
            "RETIREMENT",

        "field_name":
            "retirement_state",

        "required_for_materialization":
            True,

        "field_purpose":
            (
                "Preserves whether the scenario remains "
                "active for its current evidentiary purpose."
            ),
    },
    {
        "field_category":
            "RETIREMENT",

        "field_name":
            "retirement_reason",

        "required_for_materialization":
            False,

        "field_purpose":
            "Records why the scenario was retired or reassigned.",
    },
]


def build_bire_benchmark_suite_registry(
    pss_benchmark_suite_registry_df: pd.DataFrame,
    reproducibility_contract_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create the BIRE OS benchmark-suite registry from the global
    PSS benchmark architecture.

    Suite registration does not authorize execution.
    """

    if not isinstance(
        pss_benchmark_suite_registry_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "pss_benchmark_suite_registry_df must be a DataFrame."
        )

    if not isinstance(
        reproducibility_contract_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "reproducibility_contract_df must be a DataFrame."
        )

    required_pss_columns = {
        "benchmark_suite",
        "suite_purpose",
        "scenario_visibility",
        "scenario_reuse_policy",
        "deployment_evidence_eligible",
        "generalization_evidence_eligible",
        "suite_registry_state",
        "benchmark_execution_authorized",
    }

    missing_pss_columns = sorted(
        required_pss_columns
        - set(
            pss_benchmark_suite_registry_df.columns
        )
    )

    if missing_pss_columns:
        raise KeyError(
            "BIRE benchmark-suite registration cannot proceed. "
            f"Missing PSS registry columns: "
            f"{missing_pss_columns}"
        )

    required_contract_columns = {
        "manifest_schema_installed",
        "run_manifest_construction_authorized",
        "benchmark_execution_authorized",
        "operational_deployment_authorized",
    }

    missing_contract_columns = sorted(
        required_contract_columns
        - set(
            reproducibility_contract_df.columns
        )
    )

    if missing_contract_columns:
        raise KeyError(
            "BIRE benchmark-suite registration cannot proceed. "
            f"Missing reproducibility-contract columns: "
            f"{missing_contract_columns}"
        )

    if len(
        reproducibility_contract_df
    ) != 1:
        raise ValueError(
            "reproducibility_contract_df must contain one row."
        )

    contract = (
        reproducibility_contract_df.iloc[0]
    )

    if not bool(
        contract[
            "manifest_schema_installed"
        ]
    ):
        raise ValueError(
            "Run-manifest schema is not installed."
        )

    if not bool(
        contract[
            "run_manifest_construction_authorized"
        ]
    ):
        raise ValueError(
            "Run-manifest construction is not authorized."
        )

    if bool(
        contract[
            "benchmark_execution_authorized"
        ]
    ):
        raise ValueError(
            "Benchmark execution was unexpectedly authorized "
            "before suite registration."
        )

    expected_suites = set(
        _BIRE_BENCHMARK_SUITE_SPECIFICATIONS
    )

    observed_suites = set(
        pss_benchmark_suite_registry_df[
            "benchmark_suite"
        ].astype(str)
    )

    if observed_suites != expected_suites:
        raise ValueError(
            "PSS benchmark-suite registry does not match "
            "the required BIRE suite set. "
            f"Expected: {sorted(expected_suites)}; "
            f"observed: {sorted(observed_suites)}"
        )

    specifications = (
        pd.DataFrame
        .from_dict(
            _BIRE_BENCHMARK_SUITE_SPECIFICATIONS,
            orient="index",
        )
        .rename_axis(
            "benchmark_suite"
        )
        .reset_index()
    )

    result = (
        pss_benchmark_suite_registry_df
        .merge(
            specifications,
            on="benchmark_suite",
            how="inner",
            validate="one_to_one",
        )
    )

    result[
        "bire_benchmark_registry_version"
    ] = (
        _BIRE_BENCHMARK_REGISTRY_VERSION
    )

    result[
        "scenario_isolation_policy_version"
    ] = (
        _PSS_SCENARIO_ISOLATION_POLICY_VERSION
    )

    result[
        "scenario_manifest_state"
    ] = "NOT_MATERIALIZED"

    result[
        "scenario_count"
    ] = 0

    result[
        "suite_content_fingerprint_sha256"
    ] = pd.NA

    result[
        "scenario_isolation_audit_complete"
    ] = False

    result[
        "suite_execution_readiness_state"
    ] = (
        "BLOCKED_SCENARIO_MANIFEST_REQUIRED"
    )

    result[
        "benchmark_execution_authorized"
    ] = False

    result[
        "playground_execution_authorized"
    ] = False

    result[
        "operational_deployment_authorized"
    ] = False

    result[
        "nid_authorization_state"
    ] = (
        "NID_BIRE_BENCHMARK_SUITE_REGISTERED"
    )

    return (
        result
        .sort_values(
            "benchmark_suite"
        )
        .reset_index(drop=True)
    )


def build_bire_scenario_isolation_policy() -> pd.DataFrame:
    """
    Return the governed BIRE OS scenario-isolation policy.
    """

    result = pd.DataFrame(
        _BIRE_SCENARIO_ISOLATION_RULES
    )

    result[
        "scenario_isolation_policy_version"
    ] = (
        _PSS_SCENARIO_ISOLATION_POLICY_VERSION
    )

    result[
        "scenario_isolation_policy_defined"
    ] = True

    result[
        "unseen_status_restoration_authorized"
    ] = False

    result[
        "silent_contamination_reset_authorized"
    ] = False

    result[
        "benchmark_execution_authorized"
    ] = False

    result[
        "operational_deployment_authorized"
    ] = False

    result[
        "nid_authorization_state"
    ] = (
        "NID_SCENARIO_ISOLATION_POLICY_INSTALLED"
    )

    return result


def build_bire_scenario_manifest_schema() -> pd.DataFrame:
    """
    Return the BIRE OS scenario-manifest schema.

    Scenario materialization remains separate from benchmark
    execution.
    """

    result = pd.DataFrame(
        _BIRE_SCENARIO_MANIFEST_FIELDS
    )

    result[
        "scenario_manifest_schema_version"
    ] = (
        _BIRE_SCENARIO_MANIFEST_SCHEMA_VERSION
    )

    result[
        "scenario_manifest_field_registered"
    ] = True

    result[
        "benchmark_execution_authorized"
    ] = False

    result[
        "operational_deployment_authorized"
    ] = False

    result[
        "nid_authorization_state"
    ] = (
        "NID_BIRE_SCENARIO_MANIFEST_FIELD_REGISTERED"
    )

    return result


def build_bire_benchmark_scenario_isolation_contract(
    framework_adoption_contract_df: pd.DataFrame,
    reproducibility_contract_df: pd.DataFrame,
    benchmark_suite_registry_df: pd.DataFrame,
    scenario_isolation_policy_df: pd.DataFrame,
    scenario_manifest_schema_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Verify installation of the BIRE benchmark registry and
    scenario-isolation controls.

    This authorizes scenario-manifest construction only.
    """

    frames = {
        "framework_adoption_contract_df":
            framework_adoption_contract_df,

        "reproducibility_contract_df":
            reproducibility_contract_df,

        "benchmark_suite_registry_df":
            benchmark_suite_registry_df,

        "scenario_isolation_policy_df":
            scenario_isolation_policy_df,

        "scenario_manifest_schema_df":
            scenario_manifest_schema_df,
    }

    for name, frame in frames.items():
        if not isinstance(
            frame,
            pd.DataFrame,
        ):
            raise TypeError(
                f"{name} must be a pandas DataFrame."
            )

    if len(
        framework_adoption_contract_df
    ) != 1:
        raise ValueError(
            "framework_adoption_contract_df must contain one row."
        )

    if len(
        reproducibility_contract_df
    ) != 1:
        raise ValueError(
            "reproducibility_contract_df must contain one row."
        )

    framework = (
        framework_adoption_contract_df.iloc[0]
    )

    reproducibility = (
        reproducibility_contract_df.iloc[0]
    )

    registered_suite_set = set(
        benchmark_suite_registry_df[
            "benchmark_suite"
        ].astype(str)
    )

    isolation_suite_set = set(
        scenario_isolation_policy_df[
            "benchmark_suite"
        ].astype(str)
    )

    required_suite_set = set(
        _BIRE_BENCHMARK_SUITE_SPECIFICATIONS
    )

    checks = {
        "pss_framework_adopted":
            (
                framework[
                    "framework_adoption_state"
                ]
                ==
                "BIRE_PSS_PREDEPLOYMENT_FRAMEWORK_ADOPTED"
            ),

        "reproducibility_contract_installed":
            bool(
                reproducibility[
                    "manifest_schema_installed"
                ]
            ),

        "all_five_benchmark_suites_registered":
            (
                registered_suite_set
                ==
                required_suite_set
            ),

        "all_five_isolation_policies_registered":
            (
                isolation_suite_set
                ==
                required_suite_set
            ),

        "scenario_manifest_schema_registered":
            bool(
                len(
                    scenario_manifest_schema_df
                ) > 0
                and
                scenario_manifest_schema_df[
                    "scenario_manifest_field_registered"
                ]
                .fillna(False)
                .all()
            ),

        "benchmark_execution_still_withheld":
            bool(
                ~benchmark_suite_registry_df[
                    "benchmark_execution_authorized"
                ]
                .fillna(False)
                .any()
            ),

        "scenario_manifests_not_yet_materialized":
            bool(
                benchmark_suite_registry_df[
                    "scenario_manifest_state"
                ]
                .eq(
                    "NOT_MATERIALIZED"
                )
                .all()
            ),

        "unseen_status_restoration_prohibited":
            bool(
                ~scenario_isolation_policy_df[
                    "unseen_status_restoration_authorized"
                ]
                .fillna(False)
                .any()
            ),

        "silent_contamination_reset_prohibited":
            bool(
                ~scenario_isolation_policy_df[
                    "silent_contamination_reset_authorized"
                ]
                .fillna(False)
                .any()
            ),
    }

    failed_checks = [
        check_name
        for check_name, passed
        in checks.items()
        if not passed
    ]

    contract_installed = (
        len(failed_checks) == 0
    )

    return pd.DataFrame(
        [
            {
                "chapter":
                    (
                        "CHAPTER_69_SYSTEM_VALIDATION_"
                        "AND_FINAL_REVIEW"
                    ),

                "contract":
                    (
                        "BIRE_BENCHMARK_SUITE_AND_"
                        "SCENARIO_ISOLATION_CONTRACT"
                    ),

                "bire_benchmark_registry_version":
                    _BIRE_BENCHMARK_REGISTRY_VERSION,

                "scenario_isolation_policy_version":
                    _PSS_SCENARIO_ISOLATION_POLICY_VERSION,

                "scenario_manifest_schema_version":
                    _BIRE_SCENARIO_MANIFEST_SCHEMA_VERSION,

                **checks,

                "failed_contract_check_count":
                    len(failed_checks),

                "failed_contract_checks":
                    (
                        "NONE"
                        if not failed_checks
                        else " | ".join(
                            failed_checks
                        )
                    ),

                "scenario_access_audit_required":
                    True,

                "scenario_influence_audit_required":
                    True,

                "scenario_contamination_state_required":
                    True,

                "scenario_lineage_required":
                    True,

                "scenario_content_hash_required":
                    True,

                "scenario_difficulty_version_required":
                    True,

                "scenario_novelty_version_required":
                    True,

                "sealed_suite_isolation_required":
                    True,

                "rotating_suite_retirement_required":
                    True,

                "adversarial_suite_rehardening_required":
                    True,

                "scenario_relabeling_restores_unseen_status":
                    False,

                "scenario_copying_restores_unseen_status":
                    False,

                "scenario_reshuffling_restores_unseen_status":
                    False,

                "development_influence_permanently_removes_"
                "fresh_evidence_eligibility":
                    True,

                "scenario_manifest_construction_authorized":
                    contract_installed,

                "scenario_manifest_materialization_authorized":
                    contract_installed,

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
                        "NID_BIRE_SCENARIO_ISOLATION_"
                        "CONTRACT_INSTALLED"
                        if contract_installed
                        else
                        "NID_BIRE_SCENARIO_ISOLATION_"
                        "REVIEW_REQUIRED"
                    ),

                "next_required_stage":
                    (
                        "CHAPTER_69_5_BIRE_OS_"
                        "PREDEPLOYMENT_BENCHMARK_PROFILE"
                    ),
            }
        ]
    )


__all__ = [
    "build_bire_benchmark_suite_registry",
    "build_bire_scenario_isolation_policy",
    "build_bire_scenario_manifest_schema",
    "build_bire_benchmark_scenario_isolation_contract",
]