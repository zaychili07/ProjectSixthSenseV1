#============================================================
# Project Sixth Sense — BIRE OS
# Chapter 69.5 — BIRE OS Predeployment Benchmark Profile
#============================================================

from __future__ import annotations

import pandas as pd


_BIRE_BENCHMARK_PROFILE_VERSION = (
    "BIRE_PREDEPLOYMENT_BENCHMARK_PROFILE_V1"
)


_BIRE_PREDEPLOYMENT_METRICS = [
    # --------------------------------------------------------
    # Detection and Timeliness
    # --------------------------------------------------------
    {
        "metric_id":
            "DETERIORATION_DETECTION_RATE",

        "metric_family":
            "DETECTION_AND_TIMELINESS",

        "pss_evaluation_domain":
            "TASK_PERFORMANCE",

        "metric_direction":
            "HIGHER_IS_BETTER",

        "metric_purpose":
            (
                "Measure the proportion of governed deterioration "
                "targets recognized by BIRE OS."
            ),

        "non_compensatory_gate":
            False,
    },
    {
        "metric_id":
            "MISSED_DETERIORATION_RATE",

        "metric_family":
            "DETECTION_AND_TIMELINESS",

        "pss_evaluation_domain":
            "TASK_PERFORMANCE",

        "metric_direction":
            "LOWER_IS_BETTER",

        "metric_purpose":
            (
                "Measure governed deterioration targets that "
                "were not recognized."
            ),

        "non_compensatory_gate":
            False,
    },
    {
        "metric_id":
            "FALSE_ACTIVATION_BURDEN",

        "metric_family":
            "DETECTION_AND_TIMELINESS",

        "pss_evaluation_domain":
            "TASK_PERFORMANCE",

        "metric_direction":
            "LOWER_IS_BETTER",

        "metric_purpose":
            (
                "Measure unsupported or unnecessary activation "
                "burden produced during evaluation."
            ),

        "non_compensatory_gate":
            False,
    },
    {
        "metric_id":
            "GOVERNED_EVENT_LEAD_TIME",

        "metric_family":
            "DETECTION_AND_TIMELINESS",

        "pss_evaluation_domain":
            "TASK_PERFORMANCE",

        "metric_direction":
            "HIGHER_IS_BETTER",

        "metric_purpose":
            (
                "Measure the historical interval between "
                "governed BIRE recognition and target events."
            ),

        "non_compensatory_gate":
            False,
    },
    {
        "metric_id":
            "HIDDEN_INSTABILITY_RECOGNITION_RATE",


        "metric_family":
            "DETECTION_AND_TIMELINESS",

        "pss_evaluation_domain":
            "TASK_PERFORMANCE",

        "metric_direction":
            "HIGHER_IS_BETTER",

        "metric_purpose":
            (
                "Measure recognition of governed hidden or "
                "masked instability scenarios."
            ),

        "non_compensatory_gate":
            False,
    },

    {
        "metric_id":
            "RECOVERY_CONTRADICTION_RECOGNITION_RATE",

        "metric_family":
            "DETECTION_AND_TIMELINESS",

        "pss_evaluation_domain":
            "TASK_PERFORMANCE",

        "metric_direction":
            "HIGHER_IS_BETTER",

        "metric_purpose":
            (
                "Measure recognition of governed scenarios where "
                "apparent recovery is contradicted, fragile, false, "
                "or followed by renewed instability."
            ),

        "non_compensatory_gate":
            False,
    },

    # --------------------------------------------------------
    # Governance and Uncertainty
    # --------------------------------------------------------
    {
        "metric_id":
            "PROHIBITED_AUTHORITY_LEAK_COUNT",

        "metric_family":
            "GOVERNANCE_AND_UNCERTAINTY",

        "pss_evaluation_domain":
            "GOVERNANCE_COMPLIANCE",

        "metric_direction":
            "ZERO_REQUIRED",

        "metric_purpose":
            (
                "Count outputs or subsystem states that acquire "
                "authority prohibited by architecture or NID."
            ),

        "non_compensatory_gate":
            True,
    },
    {
        "metric_id":
            "UNSUPPORTED_CERTAINTY_COUNT",

        "metric_family":
            "GOVERNANCE_AND_UNCERTAINTY",

        "pss_evaluation_domain":
            "UNCERTAINTY_HONESTY",

        "metric_direction":
            "ZERO_REQUIRED",

        "metric_purpose":
            (
                "Count cases where uncertainty was converted "
                "into unsupported certainty."
            ),

        "non_compensatory_gate":
            True,
    },
    {
        "metric_id":
            "UNCERTAINTY_PRESERVATION_RATE",

        "metric_family":
            "GOVERNANCE_AND_UNCERTAINTY",

        "pss_evaluation_domain":
            "UNCERTAINTY_HONESTY",

        "metric_direction":
            "HIGHER_IS_BETTER",

        "metric_purpose":
            (
                "Measure whether material uncertainty survives "
                "through downstream intelligence."
            ),

        "non_compensatory_gate":
            True,
    },
    {
        "metric_id":
            "DISAGREEMENT_PRESERVATION_RATE",

        "metric_family":
            "GOVERNANCE_AND_UNCERTAINTY",

        "pss_evaluation_domain":
            "CROSS_LAYER_CONSISTENCY",

        "metric_direction":
            "HIGHER_IS_BETTER",

        "metric_purpose":
            (
                "Measure whether meaningful subsystem "
                "disagreement remains visible."
            ),

        "non_compensatory_gate":
            True,
    },
    {
        "metric_id":
            "PREMATURE_RECOVERY_CLAIM_COUNT",

        "metric_family":
            "GOVERNANCE_AND_UNCERTAINTY",

        "pss_evaluation_domain":
            "GOVERNANCE_COMPLIANCE",

        "metric_direction":
            "ZERO_REQUIRED",

        "metric_purpose":
            (
                "Count recovery conclusions made without "
                "sufficient governed evidence."
            ),

        "non_compensatory_gate":
            True,
    },

    # --------------------------------------------------------
    # Resilience
    # --------------------------------------------------------
    {
        "metric_id":
            "DEGRADED_OPERATION_HONESTY_RATE",

        "metric_family":
            "RESILIENCE_AND_DEGRADATION",

        "pss_evaluation_domain":
            "DEGRADED_OPERATION_RESILIENCE",

        "metric_direction":
            "HIGHER_IS_BETTER",

        "metric_purpose":
            (
                "Measure whether degraded capability is "
                "identified and communicated honestly."
            ),

        "non_compensatory_gate":
            True,
    },
    {
        "metric_id":
            "FABRICATED_FALLBACK_COUNT",

        "metric_family":
            "RESILIENCE_AND_DEGRADATION",

        "pss_evaluation_domain":
            "DEGRADED_OPERATION_RESILIENCE",

        "metric_direction":
            "ZERO_REQUIRED",

        "metric_purpose":
            (
                "Count fallback behavior that manufactures "
                "unsupported evidence or conclusions."
            ),

        "non_compensatory_gate":
            True,
    },
    {
        "metric_id":
            "CORRUPTED_INPUT_CONTAINMENT_RATE",

        "metric_family":
            "RESILIENCE_AND_DEGRADATION",

        "pss_evaluation_domain":
            "FAILURE_CONTAINMENT",

        "metric_direction":
            "HIGHER_IS_BETTER",

        "metric_purpose":
            (
                "Measure whether corrupted inputs are detected "
                "and prevented from contaminating downstream "
                "intelligence."
            ),

        "non_compensatory_gate":
            True,
    },

    # --------------------------------------------------------
    # Cross-Layer Integrity
    # --------------------------------------------------------
    {
        "metric_id":
            "LAYER_AUTHORITY_VIOLATION_COUNT",

        "metric_family":
            "CROSS_LAYER_INTEGRITY",

        "pss_evaluation_domain":
            "ARCHITECTURE_INTEGRITY",

        "metric_direction":
            "ZERO_REQUIRED",

        "metric_purpose":
            (
                "Count cases where a layer performs a prohibited "
                "responsibility."
            ),

        "non_compensatory_gate":
            True,
    },
    {
        "metric_id":
            "SEMANTIC_PRESERVATION_RATE",

        "metric_family":
            "CROSS_LAYER_INTEGRITY",

        "pss_evaluation_domain":
            "CROSS_LAYER_CONSISTENCY",

        "metric_direction":
            "HIGHER_IS_BETTER",

        "metric_purpose":
            (
                "Measure whether evidence meaning is preserved "
                "between producing and consuming layers."
            ),

        "non_compensatory_gate":
            True,
    },
    {
        "metric_id":
            "FALSE_CONSENSUS_COUNT",

        "metric_family":
            "CROSS_LAYER_INTEGRITY",

        "pss_evaluation_domain":
            "CROSS_LAYER_CONSISTENCY",

        "metric_direction":
            "ZERO_REQUIRED",

        "metric_purpose":
            (
                "Count cases where disagreement is silently "
                "converted into apparent consensus."
            ),

        "non_compensatory_gate":
            True,
    },

    # --------------------------------------------------------
    # Replay and Historical Intelligence
    # --------------------------------------------------------
    {
        "metric_id":
            "FALSE_RECURRENCE_CREATION_COUNT",

        "metric_family":
            "REPLAY_AND_HISTORICAL_INTELLIGENCE",

        "pss_evaluation_domain":
            "GOVERNANCE_COMPLIANCE",

        "metric_direction":
            "ZERO_REQUIRED",

        "metric_purpose":
            (
                "Count recurrence structures manufactured from "
                "duplicate, same-position, or invalid evidence."
            ),

        "non_compensatory_gate":
            True,
    },
    {
        "metric_id":
            "CHRONOLOGY_INTEGRITY_RATE",

        "metric_family":
            "REPLAY_AND_HISTORICAL_INTELLIGENCE",

        "pss_evaluation_domain":
            "DETERMINISTIC_REPRODUCIBILITY",

        "metric_direction":
            "HIGHER_IS_BETTER",

        "metric_purpose":
            (
                "Measure whether longitudinal ordering remains "
                "reproducible and governed."
            ),

        "non_compensatory_gate":
            True,
    },
    {
        "metric_id":
            "CENSORING_PRESERVATION_RATE",

        "metric_family":
            "REPLAY_AND_HISTORICAL_INTELLIGENCE",

        "pss_evaluation_domain":
            "UNCERTAINTY_HONESTY",

        "metric_direction":
            "HIGHER_IS_BETTER",

        "metric_purpose":
            (
                "Measure preservation of left, internal, and "
                "right censoring limitations."
            ),

        "non_compensatory_gate":
            True,
    },
    {
        "metric_id":
            "REPLAY_AUTHORITY_LEAK_COUNT",

        "metric_family":
            "REPLAY_AND_HISTORICAL_INTELLIGENCE",

        "pss_evaluation_domain":
            "GOVERNANCE_COMPLIANCE",

        "metric_direction":
            "ZERO_REQUIRED",

        "metric_purpose":
            (
                "Count unauthorized replay-cycle, maturity, "
                "or lesson authority."
            ),

        "non_compensatory_gate":
            True,
    },

    # --------------------------------------------------------
    # Failure Quality
    # --------------------------------------------------------
    {
        "metric_id":
            "FAILURE_VISIBILITY_RATE",

        "metric_family":
            "FAILURE_QUALITY",

        "pss_evaluation_domain":
            "FAILURE_CONTAINMENT",

        "metric_direction":
            "HIGHER_IS_BETTER",

        "metric_purpose":
            (
                "Measure whether material failures are explicitly "
                "observable."
            ),

        "non_compensatory_gate":
            True,
    },
    {
        "metric_id":
            "FAILURE_CONTAINMENT_RATE",

        "metric_family":
            "FAILURE_QUALITY",

        "pss_evaluation_domain":
            "FAILURE_CONTAINMENT",

        "metric_direction":
            "HIGHER_IS_BETTER",

        "metric_purpose":
            (
                "Measure whether detected failures remain "
                "contained to their affected scope."
            ),

        "non_compensatory_gate":
            True,
    },
    {
        "metric_id":
            "SILENT_FAILURE_COUNT",

        "metric_family":
            "FAILURE_QUALITY",

        "pss_evaluation_domain":
            "FAILURE_CONTAINMENT",

        "metric_direction":
            "ZERO_REQUIRED",

        "metric_purpose":
            (
                "Count material failures that occur without "
                "visible system acknowledgement."
            ),

        "non_compensatory_gate":
            True,
    },
    {
        "metric_id":
            "UNSAFE_FAILURE_PROPAGATION_COUNT",

        "metric_family":
            "FAILURE_QUALITY",

        "pss_evaluation_domain":
            "FAILURE_CONTAINMENT",

        "metric_direction":
            "ZERO_REQUIRED",

        "metric_purpose":
            (
                "Count local failures that improperly contaminate "
                "unrelated downstream intelligence."
            ),

        "non_compensatory_gate":
            True,
    },
    {
        "metric_id":
            "RECOVERY_HISTORY_REWRITE_COUNT",

        "metric_family":
            "FAILURE_QUALITY",

        "pss_evaluation_domain":
            "RECOVERY_BEHAVIOR",

        "metric_direction":
            "ZERO_REQUIRED",

        "metric_purpose":
            (
                "Count recovery operations that erase or silently "
                "rewrite prior observations, failures, or provenance."
            ),

        "non_compensatory_gate":
            True,
    },
]



def build_bire_predeployment_benchmark_profile(
    framework_domain_registry_df: pd.DataFrame,
    scenario_isolation_contract_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the BIRE OS-specific predeployment metric profile.

    Metric identities and interpretation directions are frozen.
    Acceptance thresholds are intentionally not assigned yet.
    """

    if not isinstance(
        framework_domain_registry_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "framework_domain_registry_df must be a DataFrame."
        )

    if not isinstance(
        scenario_isolation_contract_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "scenario_isolation_contract_df must be a DataFrame."
        )

    if len(
        scenario_isolation_contract_df
    ) != 1:
        raise ValueError(
            "scenario_isolation_contract_df must contain one row."
        )

    required_framework_columns = {
        "evaluation_domain",
        "pss_framework_requirement",
        "deployment_authority_granted",
    }

    missing_framework_columns = sorted(
        required_framework_columns
        - set(framework_domain_registry_df.columns)
    )

    if missing_framework_columns:
        raise KeyError(
            "BIRE benchmark profile cannot proceed. "
            f"Missing PSS evaluation-domain columns: "
            f"{missing_framework_columns}"
        )

    required_contract_columns = {
        "contract",
        "failed_contract_check_count",
        "scenario_manifest_construction_authorized",
        "benchmark_execution_authorized",
        "operational_deployment_authorized",
    }

    missing_contract_columns = sorted(
        required_contract_columns
        - set(scenario_isolation_contract_df.columns)
    )

    if missing_contract_columns:
        raise KeyError(
            "BIRE benchmark profile cannot proceed. "
            f"Missing scenario-isolation contract columns: "
            f"{missing_contract_columns}"
        )

    isolation_contract = (
        scenario_isolation_contract_df.iloc[0]
    )

    contract_ready = (
        isolation_contract["contract"]
        ==
        "BIRE_BENCHMARK_SUITE_AND_"
        "SCENARIO_ISOLATION_CONTRACT"
        and int(
            isolation_contract[
                "failed_contract_check_count"
            ]
        ) == 0
        and bool(
            isolation_contract[
                "scenario_manifest_construction_authorized"
            ]
        )
        and not bool(
            isolation_contract[
                "benchmark_execution_authorized"
            ]
        )
        and not bool(
            isolation_contract[
                "operational_deployment_authorized"
            ]
        )
    )

    if not contract_ready:
        raise ValueError(
            "Scenario-isolation contract is not ready for "
            "BIRE benchmark-profile construction."
        )

    registered_domains = set(
        framework_domain_registry_df.loc[
            framework_domain_registry_df[
                "pss_framework_requirement"
            ].eq(True),
            "evaluation_domain",
        ].astype(str)
    )

    profile = pd.DataFrame(
        _BIRE_PREDEPLOYMENT_METRICS
    )

    missing_domains = sorted(
        set(
            profile[
                "pss_evaluation_domain"
            ].astype(str)
        )
        - registered_domains
    )

    if missing_domains:
        raise ValueError(
            "BIRE benchmark metrics reference unregistered "
            f"PSS domains: {missing_domains}"
        )

    if profile[
        "metric_id"
    ].duplicated().any():
        raise ValueError(
            "BIRE benchmark profile contains duplicated "
            "metric identifiers."
        )

    profile[
        "benchmark_profile_version"
    ] = (
        _BIRE_BENCHMARK_PROFILE_VERSION
    )

    profile[
        "metric_definition_state"
    ] = "DEFINED"

    profile[
        "acceptance_threshold_state"
    ] = "NOT_YET_DEFINED"

    profile[
        "metric_execution_state"
    ] = "NOT_EVALUATED"

    profile[
        "metric_value"
    ] = pd.NA

    profile[
        "benchmark_execution_authorized"
    ] = False

    profile[
        "deployment_authority_granted"
    ] = False

    profile[
        "nid_authorization_state"
    ] = (
        "NID_BIRE_PREDEPLOYMENT_METRIC_REGISTERED"
    )

    return (
        profile
        .sort_values(
            by=[
                "metric_family",
                "metric_id",
            ]
        )
        .reset_index(drop=True)
    )


def build_bire_predeployment_benchmark_profile_summary(
    profile_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize the BIRE OS-specific predeployment metric profile.
    """

    if not isinstance(
        profile_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "profile_df must be a pandas DataFrame."
        )

    required_columns = {
        "metric_id",
        "metric_family",
        "pss_evaluation_domain",
        "metric_direction",
        "non_compensatory_gate",
        "acceptance_threshold_state",
        "metric_execution_state",
        "benchmark_execution_authorized",
        "deployment_authority_granted",
    }

    missing_columns = sorted(
        required_columns
        - set(profile_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "BIRE benchmark-profile summary cannot proceed. "
            f"Missing profile columns: {missing_columns}"
        )

    return (
        profile_df
        .groupby(
            [
                "metric_family",
                "pss_evaluation_domain",
                "metric_direction",
                "non_compensatory_gate",
                "acceptance_threshold_state",
                "metric_execution_state",
                "benchmark_execution_authorized",
                "deployment_authority_granted",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="metric_count"
        )
        .sort_values(
            by=[
                "metric_family",
                "pss_evaluation_domain",
                "metric_direction",
            ]
        )
        .reset_index(drop=True)
    )


__all__ = [
    "build_bire_predeployment_benchmark_profile",
    "build_bire_predeployment_benchmark_profile_summary",
]