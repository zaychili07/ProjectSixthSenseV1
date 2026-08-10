#============================================================
# Project Sixth Sense — BIRE OS
# Chapter 69.10 — Final System Verification
#                   & Playground Entry Decision
#============================================================

from __future__ import annotations

import pandas as pd


_BIRE_FINAL_VERIFICATION_VERSION = (
    "BIRE_CHAPTER_69_FINAL_VERIFICATION_V1"
)

_BIRE_PLAYGROUND_ENTRY_POLICY_VERSION = (
    "BIRE_PLAYGROUND_ENTRY_POLICY_V1"
)


_EXPECTED_FINAL_WITHHELD_STATES = {
    "NOT_TESTABLE_IN_FROZEN_BASELINE",
    "SEMANTIC_BINDING_WITHHELD",
    "JOIN_COMPATIBILITY_WITHHELD",
    "DIRECTION_CONFLICT_WITHHELD",
    "NO_VALID_RUNTIME_BINDING",
}


def _require_single_row(
    frame: pd.DataFrame,
    name: str,
) -> pd.Series:
    if not isinstance(
        frame,
        pd.DataFrame,
    ):
        raise TypeError(
            f"{name} must be a pandas DataFrame."
        )

    if len(frame) != 1:
        raise ValueError(
            f"{name} must contain exactly one row."
        )

    return frame.iloc[0]


def build_bire_chapter_69_final_metric_disposition(
    final_results_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize the final disposition of all 26 predeployment
    metrics after frozen-baseline execution.

    No new metric interpretation is introduced here.
    """

    if not isinstance(
        final_results_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "final_results_df must be a DataFrame."
        )

    required_columns = {
        "metric_id",
        "metric_family",
        "metric_value",
        "metric_testability_state",
        "metric_execution_state",
        "metric_result_state",
        "hard_gate_pass",
    }

    missing_columns = sorted(
        required_columns
        - set(final_results_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "Final metric disposition cannot proceed. "
            f"Missing columns: {missing_columns}"
        )

    if final_results_df[
        "metric_id"
    ].duplicated().any():
        raise ValueError(
            "Final results contain duplicate metric IDs."
        )

    result = (
        final_results_df[
            [
                "metric_id",
                "metric_family",
                "metric_value",
                "metric_testability_state",
                "metric_execution_state",
                "metric_result_state",
                "hard_gate_pass",
            ]
        ]
        .copy()
    )

    result[
        "chapter_69_final_metric_state"
    ] = (
        result.apply(
            lambda row: (
                "EXECUTED_HARD_GATE_PASS"
                if (
                    row[
                        "metric_execution_state"
                    ]
                    == "EXECUTED"
                    and row[
                        "metric_result_state"
                    ]
                    == "HARD_GATE_PASS"
                )
                else
                "EXECUTED_HARD_GATE_FAIL"
                if (
                    row[
                        "metric_execution_state"
                    ]
                    == "EXECUTED"
                    and row[
                        "metric_result_state"
                    ]
                    == "HARD_GATE_FAIL"
                )
                else
                str(
                    row[
                        "metric_testability_state"
                    ]
                )
            ),
            axis=1,
        )
    )

    result[
        "playground_test_responsibility"
    ] = (
        result[
            "metric_execution_state"
        ]
        .ne("EXECUTED")
    )

    result[
        "deployment_evidence_complete"
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
        "NID_BIRE_CHAPTER_69_FINAL_METRIC_DISPOSITION_RECORDED"
    )

    return (
        result
        .sort_values(
            by=[
                "playground_test_responsibility",
                "metric_family",
                "metric_id",
            ],
            ascending=[
                True,
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def build_bire_chapter_69_final_verification(
    execution_gate_df: pd.DataFrame,
    result_write_record_df: pd.DataFrame,
    final_results_df: pd.DataFrame,
    metric_acceptance_contract_df: pd.DataFrame,
    baseline_materialization_contract_df: pd.DataFrame,
    reproducibility_contract_df: pd.DataFrame,
    scenario_isolation_contract_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform the final Chapter 69 pre-Playground verification.

    Successful verification authorizes Playground entry and
    Playground preparation only.

    It does not authorize Playground execution or deployment.
    """

    execution_gate = _require_single_row(
        execution_gate_df,
        "execution_gate_df",
    )

    result_record = _require_single_row(
        result_write_record_df,
        "result_write_record_df",
    )

    metric_contract = _require_single_row(
        metric_acceptance_contract_df,
        "metric_acceptance_contract_df",
    )

    materialization = _require_single_row(
        baseline_materialization_contract_df,
        "baseline_materialization_contract_df",
    )

    reproducibility = _require_single_row(
        reproducibility_contract_df,
        "reproducibility_contract_df",
    )

    isolation = _require_single_row(
        scenario_isolation_contract_df,
        "scenario_isolation_contract_df",
    )

    if not isinstance(
        final_results_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "final_results_df must be a DataFrame."
        )

    required_result_columns = {
        "metric_id",
        "metric_value",
        "metric_testability_state",
        "metric_execution_state",
        "metric_result_state",
        "hard_gate_pass",
    }

    missing_result_columns = sorted(
        required_result_columns
        - set(final_results_df.columns)
    )

    if missing_result_columns:
        raise KeyError(
            "Final Chapter 69 verification cannot proceed. "
            f"Missing result columns: {missing_result_columns}"
        )

    executed = (
        final_results_df.loc[
            final_results_df[
                "metric_execution_state"
            ].eq("EXECUTED")
        ]
    )

    unexecuted = (
        final_results_df.loc[
            final_results_df[
                "metric_execution_state"
            ].ne("EXECUTED")
        ]
    )

    executed_hard_gate_failures = (
        executed.loc[
            executed[
                "metric_result_state"
            ].eq("HARD_GATE_FAIL")
        ]
    )

    unexecuted_with_values = (
        unexecuted.loc[
            unexecuted[
                "metric_value"
            ].notna()
        ]
    )

    observed_withheld_states = set(
        unexecuted[
            "metric_testability_state"
        ]
        .dropna()
        .astype(str)
    )

    unknown_withheld_states = sorted(
        observed_withheld_states
        - _EXPECTED_FINAL_WITHHELD_STATES
    )

    run_id = str(
        execution_gate[
            "run_id"
        ]
    )

    result_run_id = str(
        result_record[
            "run_id"
        ]
    )

    checks = {
        "reproducibility_contract_installed":
            bool(
                reproducibility[
                    "run_manifest_construction_authorized"
                ]
            ),

        "scenario_isolation_contract_installed":
            bool(
                isolation[
                    "scenario_manifest_construction_authorized"
                ]
            ),

        "metric_acceptance_contract_installed":
            bool(
                metric_contract[
                    "hard_gate_thresholds_frozen"
                ]
            )
            and bool(
                metric_contract[
                    "reference_metric_calculations_frozen"
                ]
            ),

        "frozen_baseline_materialization_installed":
            bool(
                materialization[
                    "pre_execution_baseline_manifest_"
                    "construction_authorized"
                ]
            ),

        "frozen_baseline_execution_was_authorized":
            bool(
                execution_gate[
                    "frozen_baseline_execution_authorized"
                ]
            ),

        "execution_gate_passed_without_failure":
            int(
                execution_gate[
                    "failed_gate_check_count"
                ]
            )
            == 0,

        "result_record_written_immutably":
            (
                result_record[
                    "result_record_write_state"
                ]
                ==
                "IMMUTABLE_FROZEN_BASELINE_"
                "RESULT_RECORD_WRITTEN"
            ),

        "run_identity_preserved":
            run_id
            == result_run_id,

        "all_26_metrics_preserved":
            len(
                final_results_df
            )
            == 26,

        "all_metric_ids_unique":
            not final_results_df[
                "metric_id"
            ].duplicated().any(),

        "at_least_one_metric_executed":
            len(
                executed
            )
            > 0,

        "no_executed_hard_gate_failure":
            executed_hard_gate_failures.empty,

        "executed_hard_gates_passed":
            bool(
                result_record[
                    "all_executed_hard_gates_passed"
                ]
            ),

        "unexecuted_metrics_have_no_values":
            unexecuted_with_values.empty,

        "unexecuted_states_are_governed":
            len(
                unknown_withheld_states
            )
            == 0,

        "generalization_claims_remained_withheld":
            not bool(
                isolation[
                    "scenario_relabeling_restores_unseen_status"
                ]
            ),

        "playground_was_not_executed_during_baseline":
            not bool(
                result_record[
                    "playground_execution_authorized"
                ]
            ),

        "operational_deployment_remained_withheld":
            not bool(
                result_record[
                    "operational_deployment_authorized"
                ]
            ),
    }

    failed_checks = [
        check_name
        for check_name, passed
        in checks.items()
        if not passed
    ]

    verification_complete = (
        len(
            failed_checks
        )
        == 0
    )

    unresolved_metric_count = int(
        len(
            unexecuted
        )
    )

    return pd.DataFrame(
        [
            {
                "chapter":
                    (
                        "CHAPTER_69_SYSTEM_VALIDATION_"
                        "AND_FINAL_REVIEW"
                    ),

                "review":
                    (
                        "BIRE_CHAPTER_69_FINAL_SYSTEM_"
                        "VERIFICATION"
                    ),

                "final_verification_version":
                    _BIRE_FINAL_VERIFICATION_VERSION,

                "playground_entry_policy_version":
                    _BIRE_PLAYGROUND_ENTRY_POLICY_VERSION,

                "run_id":
                    run_id,

                **checks,

                "registered_metric_count":
                    int(
                        len(
                            final_results_df
                        )
                    ),

                "executed_metric_count":
                    int(
                        len(
                            executed
                        )
                    ),

                "unresolved_metric_count":
                    unresolved_metric_count,

                "executed_hard_gate_failure_count":
                    int(
                        len(
                            executed_hard_gate_failures
                        )
                    ),

                "unknown_withheld_state_count":
                    int(
                        len(
                            unknown_withheld_states
                        )
                    ),

                "unknown_withheld_states":
                    (
                        "NONE"
                        if not unknown_withheld_states
                        else " | ".join(
                            unknown_withheld_states
                        )
                    ),

                "failed_final_check_count":
                    int(
                        len(
                            failed_checks
                        )
                    ),

                "failed_final_checks":
                    (
                        "NONE"
                        if not failed_checks
                        else " | ".join(
                            failed_checks
                        )
                    ),

                "chapter_69_verification_state":
                    (
                        "SYSTEM_VERIFICATION_COMPLETE_"
                        "WITH_GOVERNED_LIMITATIONS"
                        if verification_complete
                        else
                        "SYSTEM_VERIFICATION_INCOMPLETE_"
                        "REVIEW_REQUIRED"
                    ),

                "playground_entry_decision":
                    (
                        "PLAYGROUND_ENTRY_AUTHORIZED_"
                        "WITH_GOVERNED_LIMITATIONS"
                        if verification_complete
                        else
                        "PLAYGROUND_ENTRY_WITHHELD"
                    ),

                "playground_entry_authorized":
                    verification_complete,

                "playground_scenario_materialization_authorized":
                    verification_complete,

                "playground_run_preparation_authorized":
                    verification_complete,

                # The Playground still needs its own run manifest
                # and execution gate.
                "playground_execution_authorized":
                    False,

                "sealed_generalization_execution_authorized":
                    False,

                "rotating_novelty_execution_authorized":
                    False,

                "deployment_decision_authorized":
                    False,

                "operational_deployment_authorized":
                    False,

                "unresolved_metrics_become_"
                "playground_test_responsibility":
                    bool(
                        verification_complete
                        and unresolved_metric_count > 0
                    ),

                "nid_authorization_state":
                    (
                        "NID_BIRE_PLAYGROUND_ENTRY_AUTHORIZED_"
                        "WITH_GOVERNED_LIMITATIONS"
                        if verification_complete
                        else
                        "NID_BIRE_PLAYGROUND_ENTRY_WITHHELD"
                    ),

                "next_required_stage":
                    (
                        "THE_PLAYGROUND"
                        if verification_complete
                        else
                        "CHAPTER_69_FINAL_REMEDIATION"
                    ),
            }
        ]
    )

    __all__ = [

    "build_bire_chapter_69_final_metric_disposition",
    "build_bire_chapter_69_final_verification",
]