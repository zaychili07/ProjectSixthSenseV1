#============================================================
# Project Sixth Sense — BIRE OS
# Chapter 69.6 — Metric Calculation & Acceptance Contract
#============================================================

from __future__ import annotations

import pandas as pd


_BIRE_METRIC_CALCULATION_VERSION = (
    "BIRE_METRIC_CALCULATION_V1"
)

_BIRE_METRIC_ACCEPTANCE_POLICY_VERSION = (
    "BIRE_METRIC_ACCEPTANCE_POLICY_V1"
)


_BIRE_METRIC_CALCULATION_DEFINITIONS = {

    # ========================================================
    # Detection and Timeliness
    # ========================================================

    "DETERIORATION_DETECTION_RATE": {
        "calculation_type":
            "RATE",

        "numerator_definition":
            (
                "Governed deterioration targets recognized "
                "by BIRE OS."
            ),

        "denominator_definition":
            (
                "Governed deterioration targets eligible "
                "for evaluation."
            ),

        "aggregation_rule":
            "NUMERATOR_DIVIDED_BY_DENOMINATOR",

        "measurement_unit":
            "PROPORTION",

        "zero_denominator_state":
            "NOT_TESTABLE_NO_ELIGIBLE_DETERIORATION_TARGETS",
    },

    "MISSED_DETERIORATION_RATE": {
        "calculation_type":
            "RATE",

        "numerator_definition":
            (
                "Governed deterioration targets not recognized "
                "by BIRE OS."
            ),

        "denominator_definition":
            (
                "Governed deterioration targets eligible "
                "for evaluation."
            ),

        "aggregation_rule":
            "NUMERATOR_DIVIDED_BY_DENOMINATOR",

        "measurement_unit":
            "PROPORTION",

        "zero_denominator_state":
            "NOT_TESTABLE_NO_ELIGIBLE_DETERIORATION_TARGETS",
    },

    "FALSE_ACTIVATION_BURDEN": {
        "calculation_type":
            "BURDEN_RATE",

        "numerator_definition":
            (
                "Unsupported or unnecessary BIRE activation "
                "events."
            ),

        "denominator_definition":
            "Eligible evaluation scenarios.",

        "aggregation_rule":
            "ACTIVATION_EVENTS_PER_ELIGIBLE_SCENARIO",

        "measurement_unit":
            "ACTIVATIONS_PER_SCENARIO",

        "zero_denominator_state":
            "NOT_TESTABLE_NO_ELIGIBLE_SCENARIOS",
    },

    "GOVERNED_EVENT_LEAD_TIME": {
        "calculation_type":
            "DISTRIBUTION_SUMMARY",

        "numerator_definition":
            (
                "Target-event time minus first governed "
                "BIRE recognition time for correctly "
                "recognized target events."
            ),

        "denominator_definition":
            (
                "Correctly recognized governed target events "
                "with valid temporal information."
            ),

        "aggregation_rule":
            "MEDIAN_WITH_FULL_DISTRIBUTION_PRESERVED",

        "measurement_unit":
            "REGISTERED_CANONICAL_TIME_UNIT",

        "zero_denominator_state":
            "NOT_TESTABLE_NO_RECOGNIZED_TARGET_EVENTS",
    },

    "HIDDEN_INSTABILITY_RECOGNITION_RATE": {
        "calculation_type":
            "RATE",

        "numerator_definition":
            (
                "Governed hidden or masked instability "
                "scenarios recognized by BIRE OS."
            ),

        "denominator_definition":
            (
                "Governed hidden or masked instability "
                "scenarios eligible for evaluation."
            ),

        "aggregation_rule":
            "NUMERATOR_DIVIDED_BY_DENOMINATOR",

        "measurement_unit":
            "PROPORTION",

        "zero_denominator_state":
            "NOT_TESTABLE_NO_HIDDEN_INSTABILITY_SCENARIOS",
    },

    "RECOVERY_CONTRADICTION_RECOGNITION_RATE": {
        "calculation_type":
            "RATE",

        "numerator_definition":
            (
                "Governed recovery-contradiction scenarios "
                "recognized by BIRE OS."
            ),

        "denominator_definition":
            (
                "Governed scenarios containing false, fragile, "
                "contradicted, or renewed-instability recovery "
                "evidence."
            ),

        "aggregation_rule":
            "NUMERATOR_DIVIDED_BY_DENOMINATOR",

        "measurement_unit":
            "PROPORTION",

        "zero_denominator_state":
            "NOT_TESTABLE_NO_RECOVERY_CONTRADICTION_SCENARIOS",
    },

    # ========================================================
    # Governance and Uncertainty
    # ========================================================

    "PROHIBITED_AUTHORITY_LEAK_COUNT": {
        "calculation_type":
            "COUNT",

        "numerator_definition":
            (
                "Observed prohibited authority activations "
                "across system outputs and subsystem states."
            ),

        "denominator_definition":
            "NOT_APPLICABLE",

        "aggregation_rule":
            "EVENT_COUNT",

        "measurement_unit":
            "COUNT",

        "zero_denominator_state":
            "NOT_APPLICABLE",
    },

    "UNSUPPORTED_CERTAINTY_COUNT": {
        "calculation_type":
            "COUNT",

        "numerator_definition":
            (
                "Cases where uncertainty or incomplete evidence "
                "became unsupported certainty."
            ),

        "denominator_definition":
            "NOT_APPLICABLE",

        "aggregation_rule":
            "EVENT_COUNT",

        "measurement_unit":
            "COUNT",

        "zero_denominator_state":
            "NOT_APPLICABLE",
    },

    "UNCERTAINTY_PRESERVATION_RATE": {
        "calculation_type":
            "RATE",

        "numerator_definition":
            (
                "Eligible uncertainty-bearing observations whose "
                "material uncertainty remained visible downstream."
            ),

        "denominator_definition":
            (
                "Eligible observations containing material "
                "uncertainty or evidence limitation."
            ),

        "aggregation_rule":
            "NUMERATOR_DIVIDED_BY_DENOMINATOR",

        "measurement_unit":
            "PROPORTION",

        "zero_denominator_state":
            "NOT_TESTABLE_NO_UNCERTAINTY_BEARING_SCENARIOS",
    },

    "DISAGREEMENT_PRESERVATION_RATE": {
        "calculation_type":
            "RATE",

        "numerator_definition":
            (
                "Governed subsystem disagreements preserved "
                "without false consensus."
            ),

        "denominator_definition":
            (
                "Governed subsystem disagreements eligible "
                "for evaluation."
            ),

        "aggregation_rule":
            "NUMERATOR_DIVIDED_BY_DENOMINATOR",

        "measurement_unit":
            "PROPORTION",

        "zero_denominator_state":
            "NOT_TESTABLE_NO_SUBSYSTEM_DISAGREEMENT_SCENARIOS",
    },

    "PREMATURE_RECOVERY_CLAIM_COUNT": {
        "calculation_type":
            "COUNT",

        "numerator_definition":
            (
                "Recovery conclusions produced without "
                "sufficient governed recovery evidence."
            ),

        "denominator_definition":
            "NOT_APPLICABLE",

        "aggregation_rule":
            "EVENT_COUNT",

        "measurement_unit":
            "COUNT",

        "zero_denominator_state":
            "NOT_APPLICABLE",
    },

    # ========================================================
    # Resilience and Degradation
    # ========================================================

    "DEGRADED_OPERATION_HONESTY_RATE": {
        "calculation_type":
            "RATE",

        "numerator_definition":
            (
                "Degraded-operation scenarios where reduced "
                "capability or authority was explicitly "
                "communicated."
            ),

        "denominator_definition":
            (
                "Governed scenarios requiring degraded operation."
            ),

        "aggregation_rule":
            "NUMERATOR_DIVIDED_BY_DENOMINATOR",

        "measurement_unit":
            "PROPORTION",

        "zero_denominator_state":
            "NOT_TESTABLE_NO_DEGRADED_OPERATION_SCENARIOS",
    },

    "FABRICATED_FALLBACK_COUNT": {
        "calculation_type":
            "COUNT",

        "numerator_definition":
            (
                "Fallback outputs that manufactured unsupported "
                "evidence or conclusions."
            ),

        "denominator_definition":
            "NOT_APPLICABLE",

        "aggregation_rule":
            "EVENT_COUNT",

        "measurement_unit":
            "COUNT",

        "zero_denominator_state":
            "NOT_APPLICABLE",
    },

    "CORRUPTED_INPUT_CONTAINMENT_RATE": {
        "calculation_type":
            "RATE",

        "numerator_definition":
            (
                "Corrupted-input scenarios detected and "
                "contained before improper downstream "
                "contamination."
            ),

        "denominator_definition":
            (
                "Governed corrupted-input scenarios."
            ),

        "aggregation_rule":
            "NUMERATOR_DIVIDED_BY_DENOMINATOR",

        "measurement_unit":
            "PROPORTION",

        "zero_denominator_state":
            "NOT_TESTABLE_NO_CORRUPTED_INPUT_SCENARIOS",
    },

    # ========================================================
    # Cross-Layer Integrity
    # ========================================================

    "LAYER_AUTHORITY_VIOLATION_COUNT": {
        "calculation_type":
            "COUNT",

        "numerator_definition":
            (
                "Observed cases where a BIRE OS layer performed "
                "a responsibility prohibited by architecture."
            ),

        "denominator_definition":
            "NOT_APPLICABLE",

        "aggregation_rule":
            "EVENT_COUNT",

        "measurement_unit":
            "COUNT",

        "zero_denominator_state":
            "NOT_APPLICABLE",
    },

    "SEMANTIC_PRESERVATION_RATE": {
        "calculation_type":
            "RATE",

        "numerator_definition":
            (
                "Eligible cross-layer transfers preserving the "
                "meaning and limitations of source evidence."
            ),

        "denominator_definition":
            (
                "Governed cross-layer evidence transfers "
                "eligible for evaluation."
            ),

        "aggregation_rule":
            "NUMERATOR_DIVIDED_BY_DENOMINATOR",

        "measurement_unit":
            "PROPORTION",

        "zero_denominator_state":
            "NOT_TESTABLE_NO_CROSS_LAYER_TRANSFER_SCENARIOS",
    },

    "FALSE_CONSENSUS_COUNT": {
        "calculation_type":
            "COUNT",

        "numerator_definition":
            (
                "Cases where disagreement was converted into "
                "unsupported apparent consensus."
            ),

        "denominator_definition":
            "NOT_APPLICABLE",

        "aggregation_rule":
            "EVENT_COUNT",

        "measurement_unit":
            "COUNT",

        "zero_denominator_state":
            "NOT_APPLICABLE",
    },

    # ========================================================
    # Replay and Historical Intelligence
    # ========================================================

    "FALSE_RECURRENCE_CREATION_COUNT": {
        "calculation_type":
            "COUNT",

        "numerator_definition":
            (
                "Recurrence structures created from duplicate, "
                "same-position, invalid, or otherwise "
                "unauthorized evidence."
            ),

        "denominator_definition":
            "NOT_APPLICABLE",

        "aggregation_rule":
            "EVENT_COUNT",

        "measurement_unit":
            "COUNT",

        "zero_denominator_state":
            "NOT_APPLICABLE",
    },

    "CHRONOLOGY_INTEGRITY_RATE": {
        "calculation_type":
            "RATE",

        "numerator_definition":
            (
                "Eligible longitudinal ordering relationships "
                "reconstructed reproducibly under the governed "
                "chronology contract."
            ),

        "denominator_definition":
            (
                "Longitudinal ordering relationships eligible "
                "for chronology evaluation."
            ),

        "aggregation_rule":
            "NUMERATOR_DIVIDED_BY_DENOMINATOR",

        "measurement_unit":
            "PROPORTION",

        "zero_denominator_state":
            "NOT_TESTABLE_NO_LONGITUDINAL_ORDER_RELATIONSHIPS",
    },

    "CENSORING_PRESERVATION_RATE": {
        "calculation_type":
            "RATE",

        "numerator_definition":
            (
                "Eligible censored histories whose censoring "
                "limitations remained preserved."
            ),

        "denominator_definition":
            (
                "Histories containing governed left, internal, "
                "or right censoring."
            ),

        "aggregation_rule":
            "NUMERATOR_DIVIDED_BY_DENOMINATOR",

        "measurement_unit":
            "PROPORTION",

        "zero_denominator_state":
            "NOT_TESTABLE_NO_CENSORED_HISTORY_SCENARIOS",
    },

    "REPLAY_AUTHORITY_LEAK_COUNT": {
        "calculation_type":
            "COUNT",

        "numerator_definition":
            (
                "Unauthorized replay-cycle, maturity, pattern, "
                "or lesson authority activations."
            ),

        "denominator_definition":
            "NOT_APPLICABLE",

        "aggregation_rule":
            "EVENT_COUNT",

        "measurement_unit":
            "COUNT",

        "zero_denominator_state":
            "NOT_APPLICABLE",
    },

    # ========================================================
    # Failure Quality
    # ========================================================

    "FAILURE_VISIBILITY_RATE": {
        "calculation_type":
            "RATE",

        "numerator_definition":
            (
                "Material failure events explicitly surfaced "
                "by BIRE OS."
            ),

        "denominator_definition":
            (
                "Material failure events occurring during "
                "the governed evaluation."
            ),

        "aggregation_rule":
            "NUMERATOR_DIVIDED_BY_DENOMINATOR",

        "measurement_unit":
            "PROPORTION",

        "zero_denominator_state":
            "NOT_TESTABLE_NO_MATERIAL_FAILURE_EVENTS",
    },

    "FAILURE_CONTAINMENT_RATE": {
        "calculation_type":
            "RATE",

        "numerator_definition":
            (
                "Detected material failures successfully "
                "contained to their governed affected scope."
            ),

        "denominator_definition":
            (
                "Detected material failures requiring containment."
            ),

        "aggregation_rule":
            "NUMERATOR_DIVIDED_BY_DENOMINATOR",

        "measurement_unit":
            "PROPORTION",

        "zero_denominator_state":
            "NOT_TESTABLE_NO_DETECTED_FAILURES",
    },

    "SILENT_FAILURE_COUNT": {
        "calculation_type":
            "COUNT",

        "numerator_definition":
            (
                "Material failures occurring without explicit "
                "BIRE OS acknowledgement."
            ),

        "denominator_definition":
            "NOT_APPLICABLE",

        "aggregation_rule":
            "EVENT_COUNT",

        "measurement_unit":
            "COUNT",

        "zero_denominator_state":
            "NOT_APPLICABLE",
    },

    "UNSAFE_FAILURE_PROPAGATION_COUNT": {
        "calculation_type":
            "COUNT",

        "numerator_definition":
            (
                "Local failures improperly contaminating "
                "unrelated downstream intelligence."
            ),

        "denominator_definition":
            "NOT_APPLICABLE",

        "aggregation_rule":
            "EVENT_COUNT",

        "measurement_unit":
            "COUNT",

        "zero_denominator_state":
            "NOT_APPLICABLE",
    },

    "RECOVERY_HISTORY_REWRITE_COUNT": {
        "calculation_type":
            "COUNT",

        "numerator_definition":
            (
                "Recovery operations that erased or silently "
                "rewrote prior observations, failures, or "
                "provenance."
            ),

        "denominator_definition":
            "NOT_APPLICABLE",

        "aggregation_rule":
            "EVENT_COUNT",

        "measurement_unit":
            "COUNT",

        "zero_denominator_state":
            "NOT_APPLICABLE",
    },
}


def _build_acceptance_policy(
    metric_direction: str,
    non_compensatory_gate: bool,
) -> dict[str, object]:
    """
    Derive the governed acceptance-policy type from metric
    direction and non-compensatory status.
    """

    if metric_direction == "ZERO_REQUIRED":
        return {
            "acceptance_policy_type":
                "ARCHITECTURAL_HARD_GATE",

            "acceptance_operator":
                "EQUAL_TO",

            "hard_gate_value":
                0.0,

            "reference_baseline_required":
                False,

            "acceptance_threshold_state":
                "FROZEN_ARCHITECTURAL_HARD_GATE",

            "future_comparison_policy":
                "ZERO_TOLERANCE",
        }

    if non_compensatory_gate:
        return {
            "acceptance_policy_type":
                "NON_COMPENSATORY_BASELINE_REFERENCED_GATE",

            "acceptance_operator":
                (
                    "GREATER_THAN_OR_EQUAL_TO"
                    if metric_direction == "HIGHER_IS_BETTER"
                    else
                    "LESS_THAN_OR_EQUAL_TO"
                ),

            "hard_gate_value":
                pd.NA,

            "reference_baseline_required":
                True,

            "acceptance_threshold_state":
                "PENDING_GOVERNED_REFERENCE_BASELINE",

            "future_comparison_policy":
                (
                    "REFERENCE_DISTRIBUTION_AND_"
                    "NON_REGRESSION_POLICY_REQUIRED"
                ),
        }

    return {
        "acceptance_policy_type":
            "PERFORMANCE_BASELINE_REFERENCED_METRIC",

        "acceptance_operator":
            (
                "GREATER_THAN_OR_EQUAL_TO"
                if metric_direction == "HIGHER_IS_BETTER"
                else
                "LESS_THAN_OR_EQUAL_TO"
            ),

        "hard_gate_value":
            pd.NA,

        "reference_baseline_required":
            True,

        "acceptance_threshold_state":
            "PENDING_GOVERNED_REFERENCE_BASELINE",

        "future_comparison_policy":
            (
                "REFERENCE_DISTRIBUTION_AND_"
                "CHALLENGE_ADJUSTED_COMPARISON_REQUIRED"
            ),
    }


def build_bire_metric_calculation_acceptance_registry(
    profile_df: pd.DataFrame,
    scenario_isolation_contract_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the calculation and acceptance registry for all BIRE OS
    predeployment metrics.

    Calculation rules are frozen before baseline execution.

    ZERO_REQUIRED metrics receive immediate architectural hard
    gates. Other metrics require governed baseline reference
    distributions.

    Benchmark execution remains unauthorized.
    """

    if not isinstance(
        profile_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "profile_df must be a pandas DataFrame."
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

    required_profile_columns = {
        "metric_id",
        "metric_family",
        "pss_evaluation_domain",
        "metric_direction",
        "non_compensatory_gate",
        "metric_definition_state",
        "benchmark_execution_authorized",
        "deployment_authority_granted",
    }

    missing_profile_columns = sorted(
        required_profile_columns
        - set(profile_df.columns)
    )

    if missing_profile_columns:
        raise KeyError(
            "BIRE metric-calculation registry cannot proceed. "
            f"Missing profile columns: {missing_profile_columns}"
        )

    if profile_df[
        "metric_id"
    ].duplicated().any():
        raise ValueError(
            "BIRE metric profile contains duplicated metric IDs."
        )

    scenario_contract = (
        scenario_isolation_contract_df.iloc[0]
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
        - set(
            scenario_isolation_contract_df.columns
        )
    )

    if missing_contract_columns:
        raise KeyError(
            "BIRE metric-calculation registry cannot proceed. "
            "Missing scenario-isolation contract columns: "
            f"{missing_contract_columns}"
        )

    contract_ready = (
        scenario_contract["contract"]
        ==
        "BIRE_BENCHMARK_SUITE_AND_"
        "SCENARIO_ISOLATION_CONTRACT"
        and int(
            scenario_contract[
                "failed_contract_check_count"
            ]
        ) == 0
        and bool(
            scenario_contract[
                "scenario_manifest_construction_authorized"
            ]
        )
        and not bool(
            scenario_contract[
                "benchmark_execution_authorized"
            ]
        )
        and not bool(
            scenario_contract[
                "operational_deployment_authorized"
            ]
        )
    )

    if not contract_ready:
        raise ValueError(
            "Scenario-isolation contract is not ready."
        )

    profile_metric_ids = set(
        profile_df[
            "metric_id"
        ].astype(str)
    )

    calculation_metric_ids = set(
        _BIRE_METRIC_CALCULATION_DEFINITIONS
    )

    missing_calculations = sorted(
        profile_metric_ids
        - calculation_metric_ids
    )

    extra_calculations = sorted(
        calculation_metric_ids
        - profile_metric_ids
    )

    if missing_calculations:
        raise ValueError(
            "Missing metric-calculation definitions: "
            f"{missing_calculations}"
        )

    if extra_calculations:
        raise ValueError(
            "Calculation definitions exist for metrics not "
            f"registered in the profile: {extra_calculations}"
        )

    rows: list[dict[str, object]] = []

    for metric in profile_df.itertuples(
        index=False
    ):
        calculation = (
            _BIRE_METRIC_CALCULATION_DEFINITIONS[
                metric.metric_id
            ]
        )

        acceptance = _build_acceptance_policy(
            metric_direction=(
                metric.metric_direction
            ),
            non_compensatory_gate=bool(
                metric.non_compensatory_gate
            ),
        )

        rows.append(
            {
                "metric_id":
                    metric.metric_id,

                "metric_family":
                    metric.metric_family,

                "pss_evaluation_domain":
                    metric.pss_evaluation_domain,

                "metric_direction":
                    metric.metric_direction,

                "non_compensatory_gate":
                    bool(
                        metric.non_compensatory_gate
                    ),

                **calculation,

                **acceptance,

                "not_testable_handling":
                    "PRESERVE_NOT_TESTABLE_DO_NOT_IMPUTE",

                "baseline_role":
                    (
                        "HARD_GATE_EVALUATION"
                        if metric.metric_direction
                        == "ZERO_REQUIRED"
                        else
                        "ESTABLISH_REFERENCE_DISTRIBUTION"
                    ),

                "metric_calculation_version":
                    _BIRE_METRIC_CALCULATION_VERSION,

                "acceptance_policy_version":
                    _BIRE_METRIC_ACCEPTANCE_POLICY_VERSION,

                "metric_calculation_state":
                    "CALCULATION_RULE_DEFINED",

                "metric_acceptance_policy_state":
                    "ACCEPTANCE_METHOD_DEFINED",

                "metric_execution_state":
                    "NOT_EVALUATED",

                "metric_value":
                    pd.NA,

                "metric_testability_state":
                    "NOT_EVALUATED",

                "benchmark_execution_authorized":
                    False,

                "deployment_authority_granted":
                    False,

                "nid_authorization_state":
                    (
                        "NID_BIRE_METRIC_CALCULATION_"
                        "AND_ACCEPTANCE_POLICY_REGISTERED"
                    ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            by=[
                "metric_family",
                "metric_id",
            ]
        )
        .reset_index(drop=True)
    )


def build_bire_metric_calculation_acceptance_summary(
    registry_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize BIRE OS metric calculation and acceptance policy.
    """

    if not isinstance(
        registry_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "registry_df must be a pandas DataFrame."
        )

    required_columns = {
        "metric_id",
        "metric_family",
        "metric_direction",
        "calculation_type",
        "acceptance_policy_type",
        "acceptance_threshold_state",
        "reference_baseline_required",
        "non_compensatory_gate",
        "metric_execution_state",
        "benchmark_execution_authorized",
    }

    missing_columns = sorted(
        required_columns
        - set(registry_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "Metric calculation summary cannot proceed. "
            f"Missing columns: {missing_columns}"
        )

    return (
        registry_df
        .groupby(
            [
                "metric_family",
                "metric_direction",
                "calculation_type",
                "acceptance_policy_type",
                "acceptance_threshold_state",
                "reference_baseline_required",
                "non_compensatory_gate",
                "metric_execution_state",
                "benchmark_execution_authorized",
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
                "acceptance_policy_type",
                "metric_direction",
            ]
        )
        .reset_index(drop=True)
    )


def build_bire_metric_calculation_acceptance_contract(
    registry_df: pd.DataFrame,
    reproducibility_contract_df: pd.DataFrame,
    scenario_isolation_contract_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Verify that all BIRE OS metric calculations and acceptance
    methods are defined before baseline scenario materialization.

    This does not authorize metric or benchmark execution.
    """

    for name, frame in {
        "registry_df":
            registry_df,
        "reproducibility_contract_df":
            reproducibility_contract_df,
        "scenario_isolation_contract_df":
            scenario_isolation_contract_df,
    }.items():
        if not isinstance(
            frame,
            pd.DataFrame,
        ):
            raise TypeError(
                f"{name} must be a pandas DataFrame."
            )

    if len(
        reproducibility_contract_df
    ) != 1:
        raise ValueError(
            "reproducibility_contract_df must contain one row."
        )

    if len(
        scenario_isolation_contract_df
    ) != 1:
        raise ValueError(
            "scenario_isolation_contract_df must contain one row."
        )

    required_registry_columns = {
        "metric_id",
        "metric_direction",
        "metric_calculation_state",
        "metric_acceptance_policy_state",
        "acceptance_policy_type",
        "acceptance_threshold_state",
        "hard_gate_value",
        "reference_baseline_required",
        "benchmark_execution_authorized",
        "deployment_authority_granted",
    }

    missing_registry_columns = sorted(
        required_registry_columns
        - set(registry_df.columns)
    )

    if missing_registry_columns:
        raise KeyError(
            "Metric acceptance contract cannot proceed. "
            f"Missing registry columns: "
            f"{missing_registry_columns}"
        )

    hard_gate_rows = registry_df.loc[
        registry_df[
            "acceptance_policy_type"
        ].eq(
            "ARCHITECTURAL_HARD_GATE"
        )
    ]

    reference_rows = registry_df.loc[
        registry_df[
            "reference_baseline_required"
        ].eq(True)
    ]

    checks = {
        "all_profile_metrics_registered":
            len(registry_df) == 26,

        "all_metric_ids_unique":
            not registry_df[
                "metric_id"
            ].duplicated().any(),

        "all_calculation_rules_defined":
            registry_df[
                "metric_calculation_state"
            ]
            .eq(
                "CALCULATION_RULE_DEFINED"
            )
            .all(),

        "all_acceptance_methods_defined":
            registry_df[
                "metric_acceptance_policy_state"
            ]
            .eq(
                "ACCEPTANCE_METHOD_DEFINED"
            )
            .all(),

        "all_zero_required_metrics_are_hard_gates":
            bool(
                registry_df.loc[
                    registry_df[
                        "metric_direction"
                    ].eq("ZERO_REQUIRED"),
                    "acceptance_policy_type",
                ]
                .eq(
                    "ARCHITECTURAL_HARD_GATE"
                )
                .all()
            ),

        "all_hard_gate_values_equal_zero":
            bool(
                hard_gate_rows[
                    "hard_gate_value"
                ]
                .fillna(float("nan"))
                .eq(0.0)
                .all()
            ),

        "reference_metrics_require_baseline":
            bool(
                reference_rows[
                    "acceptance_threshold_state"
                ]
                .eq(
                    "PENDING_GOVERNED_REFERENCE_BASELINE"
                )
                .all()
            ),

        "metric_execution_still_withheld":
            bool(
                registry_df[
                    "metric_execution_state"
                ]
                .eq(
                    "NOT_EVALUATED"
                )
                .all()
            ),

        "benchmark_execution_still_withheld":
            bool(
                ~registry_df[
                    "benchmark_execution_authorized"
                ]
                .fillna(False)
                .any()
            ),

        "deployment_authority_still_withheld":
            bool(
                ~registry_df[
                    "deployment_authority_granted"
                ]
                .fillna(False)
                .any()
            ),
    }

    failed_checks = [
        name
        for name, passed in checks.items()
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
                        "BIRE_METRIC_CALCULATION_"
                        "AND_ACCEPTANCE_CONTRACT"
                    ),

                "metric_calculation_version":
                    _BIRE_METRIC_CALCULATION_VERSION,

                "acceptance_policy_version":
                    _BIRE_METRIC_ACCEPTANCE_POLICY_VERSION,

                **checks,

                "registered_metric_count":
                    int(
                        len(registry_df)
                    ),

                "architectural_hard_gate_count":
                    int(
                        len(hard_gate_rows)
                    ),

                "reference_baseline_metric_count":
                    int(
                        len(reference_rows)
                    ),

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

                "hard_gate_thresholds_frozen":
                    contract_installed,

                "reference_metric_calculations_frozen":
                    contract_installed,

                "baseline_reference_method_frozen":
                    contract_installed,

                "not_testable_imputation_authorized":
                    False,

                "metric_definition_in_place_change_authorized":
                    False,

                "scenario_manifest_materialization_authorized":
                    contract_installed,

                "metric_execution_authorized":
                    False,

                "benchmark_execution_authorized":
                    False,

                "playground_execution_authorized":
                    False,

                "deployment_decision_authorized":
                    False,

                "operational_deployment_authorized":
                    False,

                "nid_authorization_state":
                    (
                        "NID_BIRE_METRIC_CALCULATION_"
                        "ACCEPTANCE_CONTRACT_INSTALLED"
                        if contract_installed
                        else
                        "NID_BIRE_METRIC_CALCULATION_"
                        "ACCEPTANCE_REVIEW_REQUIRED"
                    ),

                "next_required_stage":
                    (
                        "CHAPTER_69_7_BASELINE_SCENARIO_"
                        "MANIFEST_MATERIALIZATION"
                    ),
            }
        ]
    )


__all__ = [
    "build_bire_metric_calculation_acceptance_registry",
    "build_bire_metric_calculation_acceptance_summary",
    "build_bire_metric_calculation_acceptance_contract",
]