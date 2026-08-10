#============================================================
# Project Sixth Sense — BIRE OS
# Chapter 69.8 — Pre-Playground Baseline
#                  Run Manifest & Execution Gate
#============================================================

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import hashlib
import inspect
import uuid

import pandas as pd

from bire.validation.manifest import (
    build_bire_evaluation_run_manifest,
    write_bire_evaluation_run_manifest,
)


_BIRE_BASELINE_PREPARATION_VERSION = (
    "BIRE_PREPLAYGROUND_BASELINE_PREPARATION_V1"
)

_BIRE_BASELINE_EXECUTION_GATE_VERSION = (
    "BIRE_PREPLAYGROUND_BASELINE_EXECUTION_GATE_V1"
)


def _require_single_row(
    frame: pd.DataFrame,
    frame_name: str,
) -> pd.Series:
    if not isinstance(
        frame,
        pd.DataFrame,
    ):
        raise TypeError(
            f"{frame_name} must be a pandas DataFrame."
        )

    if len(frame) != 1:
        raise ValueError(
            f"{frame_name} must contain exactly one row."
        )

    return frame.iloc[0]


def _first_unique_text(
    frame: pd.DataFrame,
    column: str,
) -> str:
    if column not in frame.columns:
        raise KeyError(
            f"Required column missing: {column}"
        )

    values = (
        frame[column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    values = values[
        values.ne("")
    ].unique().tolist()

    if len(values) != 1:
        raise ValueError(
            f"{column} must contain exactly one "
            f"nonempty unique value. Observed: {values}"
        )

    return str(values[0])


def _sha256_file(
    path: str | Path,
) -> str:
    file_path = Path(
        path
    ).expanduser().resolve()

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Artifact does not exist: {file_path}"
        )

    digest = hashlib.sha256()

    with file_path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _generate_baseline_run_id() -> str:
    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    suffix = uuid.uuid4().hex[:10]

    return (
        f"PSSRUN-{timestamp}-{suffix}"
    )


def _invoke_existing_manifest_builder(
    candidate_kwargs: dict[str, object],
) -> pd.DataFrame:
    """
    Invoke the Chapter 69.3 public manifest builder while
    respecting its installed function signature.

    This prevents Chapter 69.8 from duplicating manifest logic.
    """

    signature = inspect.signature(
        build_bire_evaluation_run_manifest
    )

    resolved_kwargs: dict[
        str,
        object,
    ] = {}

    unsupported_required_parameters: list[
        str
    ] = []

    for (
        parameter_name,
        parameter,
    ) in signature.parameters.items():

        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue

        if parameter_name in candidate_kwargs:
            resolved_kwargs[
                parameter_name
            ] = candidate_kwargs[
                parameter_name
            ]

            continue

        if (
            parameter.default
            is inspect.Parameter.empty
        ):
            unsupported_required_parameters.append(
                parameter_name
            )

    if unsupported_required_parameters:
        raise TypeError(
            "The installed Chapter 69.3 manifest builder "
            "requires parameters that Chapter 69.8 does not "
            "currently map: "
            f"{unsupported_required_parameters}. "
            "Inspect build_bire_evaluation_run_manifest() "
            "before proceeding."
        )

    result = (
        build_bire_evaluation_run_manifest(
            **resolved_kwargs
        )
    )

    if not isinstance(
        result,
        pd.DataFrame,
    ):
        raise TypeError(
            "Chapter 69.3 manifest builder did not return "
            "a pandas DataFrame."
        )

    if len(result) != 1:
        raise ValueError(
            "The pre-execution run manifest must contain "
            "exactly one row."
        )

    return result


def build_bire_preplayground_baseline_run_manifest(
    framework_adoption_contract_df: pd.DataFrame,
    reproducibility_contract_df: pd.DataFrame,
    materialization_contract_df: pd.DataFrame,
    scenario_manifest_df: pd.DataFrame,
    scenario_manifest_write_record_df: pd.DataFrame,
    benchmark_suite_registry_df: pd.DataFrame,
    metric_profile_df: pd.DataFrame,
    metric_acceptance_contract_df: pd.DataFrame,
    repository_root: str | Path,
    planned_output_root: str | Path,
    system_version: str = (
        "BIRE_OS_CHAPTER_69_PREPLAYGROUND_RC1"
    ),
    governance_policy_version: str = (
        "BIRE_CHAPTER_69_VALIDATION_GOVERNANCE_V1"
    ),
    nid_policy_version: str = (
        "BIRE_CHAPTER_69_NID_POLICY_V1"
    ),
    random_seed_set: Iterable[int] = (
        6908,
    ),
    configuration_artifacts: Iterable[
        str | Path
    ] = (),
    model_artifacts: Iterable[
        str | Path
    ] = (),
    input_artifacts: Iterable[
        str | Path
    ] = (),
) -> pd.DataFrame:
    """
    Construct the first frozen pre-Playground BIRE OS
    pre-execution evaluation manifest.

    The Chapter 69.3 manifest builder remains the authority
    for manifest construction.
    """

    framework_adoption = _require_single_row(
        framework_adoption_contract_df,
        "framework_adoption_contract_df",
    )

    reproducibility = _require_single_row(
        reproducibility_contract_df,
        "reproducibility_contract_df",
    )

    materialization = _require_single_row(
        materialization_contract_df,
        "materialization_contract_df",
    )

    scenario_write = _require_single_row(
        scenario_manifest_write_record_df,
        "scenario_manifest_write_record_df",
    )

    metric_contract = _require_single_row(
        metric_acceptance_contract_df,
        "metric_acceptance_contract_df",
    )

    if not isinstance(
        scenario_manifest_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "scenario_manifest_df must be a DataFrame."
        )

    if not isinstance(
        benchmark_suite_registry_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "benchmark_suite_registry_df must be a DataFrame."
        )

    if not isinstance(
        metric_profile_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "metric_profile_df must be a DataFrame."
        )

    prerequisite_checks = {
        "reproducibility_contract_ready":
            bool(
                reproducibility[
                    "run_manifest_construction_authorized"
                ]
            ),

        "frozen_baseline_materialized":
            bool(
                materialization[
                    "pre_execution_baseline_manifest_"
                    "construction_authorized"
                ]
            ),

        "scenario_manifest_write_complete":
            (
                scenario_write[
                    "manifest_write_state"
                ]
                ==
                "IMMUTABLE_FROZEN_BASELINE_"
                "SCENARIO_MANIFEST_WRITTEN"
            ),

        "metric_contract_ready":
            (
                int(
                    metric_contract[
                        "failed_contract_check_count"
                    ]
                )
                == 0
            ),

        "metric_execution_still_withheld":
            not bool(
                metric_contract[
                    "metric_execution_authorized"
                ]
            ),

        "benchmark_execution_still_withheld":
            not bool(
                metric_contract[
                    "benchmark_execution_authorized"
                ]
            ),
    }

    failed_prerequisites = [
        check_name
        for check_name, passed
        in prerequisite_checks.items()
        if not passed
    ]

    if failed_prerequisites:
        raise ValueError(
            "Pre-Playground baseline run-manifest "
            "construction is blocked: "
            f"{failed_prerequisites}"
        )

    frozen_suite_rows = (
        benchmark_suite_registry_df.loc[
            benchmark_suite_registry_df[
                "benchmark_suite"
            ].astype(str).eq(
                "FROZEN_REGRESSION_SUITE"
            )
        ]
    )

    if len(
        frozen_suite_rows
    ) != 1:
        raise ValueError(
            "Exactly one FROZEN_REGRESSION_SUITE "
            "registry row is required."
        )

    frozen_suite = (
        frozen_suite_rows.iloc[0]
    )

    suite_version = str(
        frozen_suite[
            "bire_suite_specification_version"
        ]
    )

    scenario_manifest_path = Path(
        str(
            scenario_write[
                "scenario_manifest_path"
            ]
        )
    ).expanduser().resolve()

    scenario_manifest_sha256 = str(
        scenario_write[
            "scenario_manifest_sha256"
        ]
    )

    actual_scenario_hash = (
        _sha256_file(
            scenario_manifest_path
        )
    )

    if (
        actual_scenario_hash
        != scenario_manifest_sha256
    ):
        raise ValueError(
            "Frozen scenario-manifest file hash does not "
            "match its write record."
        )

    difficulty_profile_version = (
        _first_unique_text(
            scenario_manifest_df,
            "difficulty_profile_version",
        )
    )

    novelty_profile_version = (
        _first_unique_text(
            scenario_manifest_df,
            "novelty_profile_version",
        )
    )

    metric_definition_version = (
        _first_unique_text(
            metric_profile_df,
            "benchmark_profile_version",
        )
    )

    scenario_manifest_version = str(
        materialization[
            "baseline_manifest_version"
        ]
    )

    run_id = (
        _generate_baseline_run_id()
    )

    repo_root = str(
        Path(
            repository_root
        )
        .expanduser()
        .resolve()
    )

    output_root = str(
        Path(
            planned_output_root
        )
        .expanduser()
        .resolve()
    )

    combined_input_artifacts = [
        str(
            scenario_manifest_path
        )
    ]

    combined_input_artifacts.extend(
        str(
            Path(path)
            .expanduser()
            .resolve()
        )
        for path in input_artifacts
    )

    combined_input_artifacts = list(
        dict.fromkeys(
            combined_input_artifacts
        )
    )

    # --------------------------------------------------------
    # Candidate argument map for the installed 69.3 builder.
    #
    # Aliases are included intentionally so this wrapper can
    # tolerate minor naming differences without duplicating
    # manifest construction logic.
    # --------------------------------------------------------

    candidate_kwargs: dict[
        str,
        object,
    ] = {

        "framework_adoption_contract_df":
            framework_adoption_contract_df,

        "reproducibility_contract_df":
            reproducibility_contract_df,

        "system_id":
            "BIRE_OS",

        "system_version":
            system_version,

        "run_type":
            "PRE_PLAYGROUND_FROZEN_BASELINE",

        "lifecycle_phase":
            "BASELINE_ESTABLISHMENT",

        "deployment_state":
            "PREDEPLOYMENT",

        "benchmark_suite":
            "FROZEN_REGRESSION_SUITE",

        "benchmark_suite_version":
            suite_version,

        "scenario_manifest_id":
            "BIRE_FROZEN_BASELINE_SCENARIO_MANIFEST",

        "scenario_manifest_version":
            scenario_manifest_version,

        "scenario_manifest_path":
            str(
                scenario_manifest_path
            ),

        "scenario_manifest_sha256":
            scenario_manifest_sha256,

        "difficulty_profile_version":
            difficulty_profile_version,

        "novelty_profile_version":
            novelty_profile_version,

        "metric_definition_version":
            metric_definition_version,

        "governance_policy_version":
            governance_policy_version,

        "nid_policy_version":
            nid_policy_version,

        "random_seed_set":
            list(
                random_seed_set
            ),

        "random_seeds":
            list(
                random_seed_set
            ),

        "seed_set":
            list(
                random_seed_set
            ),

        "configuration_artifacts":
            list(
                configuration_artifacts
            ),

        "configuration_artifact_paths":
            list(
                configuration_artifacts
            ),

        "model_artifacts":
            list(
                model_artifacts
            ),

        "model_artifact_paths":
            list(
                model_artifacts
            ),

        "input_artifacts":
            combined_input_artifacts,

        "input_artifact_paths":
            combined_input_artifacts,

        "planned_output_root":
            output_root,

        "output_root":
            output_root,

        "repository_root":
            repo_root,

        "repo_root":
            repo_root,

        "comparison_baseline_run_id":
            None,

        "parent_run_id":
            None,

        "run_id":
            run_id,
    }

    manifest = (
        _invoke_existing_manifest_builder(
            candidate_kwargs
        )
    )

    manifest_row = (
        manifest.iloc[0]
    )

    if (
        "run_id"
        in manifest.columns
        and str(
            manifest_row[
                "run_id"
            ]
        )
        != run_id
    ):
        raise ValueError(
            "Chapter 69.3 manifest builder changed the "
            "governed baseline run ID."
        )

    if (
        "scenario_manifest_sha256"
        in manifest.columns
        and str(
            manifest_row[
                "scenario_manifest_sha256"
            ]
        )
        != scenario_manifest_sha256
    ):
        raise ValueError(
            "Run manifest scenario hash does not match the "
            "frozen scenario-manifest artifact."
        )

    if (
        "benchmark_execution_authorized"
        in manifest.columns
        and bool(
            manifest_row[
                "benchmark_execution_authorized"
            ]
        )
    ):
        raise ValueError(
            "The pre-execution manifest unexpectedly "
            "authorized benchmark execution."
        )

    return manifest


def write_bire_preplayground_baseline_run_manifest(
    run_manifest_df: pd.DataFrame,
    output_root: str | Path,
) -> pd.DataFrame:
    """
    Write the immutable Chapter 69.3 pre-execution manifest
    and return a normalized BIRE baseline write record.
    """

    manifest = _require_single_row(
        run_manifest_df,
        "run_manifest_df",
    )

    run_id = str(
        manifest[
            "run_id"
        ]
    )

    root = (
        Path(
            output_root
        )
        .expanduser()
        .resolve()
    )

    destination = (
        root
        / run_id
        / "pre_execution_run_manifest.json"
    )

    if destination.exists():
        raise FileExistsError(
            "Pre-execution run manifests are immutable "
            f"and cannot be overwritten: {destination}"
        )

    # Chapter 69.3 remains the authoritative writer.
    write_bire_evaluation_run_manifest(
        run_manifest_df,
        destination,
    )

    if not destination.is_file():
        raise RuntimeError(
            "Chapter 69.3 writer returned without creating "
            "the expected run-manifest artifact."
        )

    manifest_file_sha256 = (
        _sha256_file(
            destination
        )
    )

    return pd.DataFrame(
        [
            {
                "run_id":
                    run_id,

                "run_manifest_path":
                    str(
                        destination
                    ),

                "run_manifest_file_sha256":
                    manifest_file_sha256,

                "manifest_fingerprint_sha256":
                    manifest[
                        "manifest_fingerprint_sha256"
                    ],

                "run_manifest_write_state":
                    (
                        "IMMUTABLE_PRE_EXECUTION_"
                        "BASELINE_RUN_MANIFEST_WRITTEN"
                    ),

                "manifest_overwrite_authorized":
                    False,

                "baseline_execution_authorized":
                    False,

                "playground_execution_authorized":
                    False,

                "operational_deployment_authorized":
                    False,

                "nid_authorization_state":
                    (
                        "NID_BIRE_PRE_EXECUTION_"
                        "BASELINE_MANIFEST_WRITTEN"
                    ),
            }
        ]
    )


def build_bire_preplayground_baseline_execution_gate(
    run_manifest_df: pd.DataFrame,
    run_manifest_write_record_df: pd.DataFrame,
    scenario_manifest_write_record_df: pd.DataFrame,
    materialization_contract_df: pd.DataFrame,
    metric_acceptance_contract_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform the final authorization review before the frozen
    pre-Playground baseline may execute.

    This gate authorizes the frozen baseline only.

    Playground and deployment authority remain withheld.
    """

    manifest = _require_single_row(
        run_manifest_df,
        "run_manifest_df",
    )

    run_write = _require_single_row(
        run_manifest_write_record_df,
        "run_manifest_write_record_df",
    )

    scenario_write = _require_single_row(
        scenario_manifest_write_record_df,
        "scenario_manifest_write_record_df",
    )

    materialization = _require_single_row(
        materialization_contract_df,
        "materialization_contract_df",
    )

    metric_contract = _require_single_row(
        metric_acceptance_contract_df,
        "metric_acceptance_contract_df",
    )

    git_commit_hash = str(
        manifest.get(
            "git_commit_hash",
            "",
        )
    ).strip()

    git_worktree_state = str(
        manifest.get(
            "git_worktree_state",
            "",
        )
    ).strip().upper()

    clean_worktree_states = {
        "CLEAN",
        "CLEAN_WORKTREE",
        "WORKTREE_CLEAN",
    }

    environment_fingerprint = str(
        manifest.get(
            "environment_fingerprint_sha256",
            "",
        )
    ).strip()

    dependency_fingerprint = str(
        manifest.get(
            "dependency_snapshot_sha256",
            "",
        )
    ).strip()

    manifest_fingerprint = str(
        manifest.get(
            "manifest_fingerprint_sha256",
            "",
        )
    ).strip()

    scenario_manifest_hash = str(
        manifest.get(
            "scenario_manifest_sha256",
            "",
        )
    ).strip()

    checks = {
       "metric_acceptance_contract_installed":
    (
        bool(
            metric_contract[
                "hard_gate_thresholds_frozen"
            ]
        )
        and bool(
            metric_contract[
                "reference_metric_calculations_frozen"
            ]
        )
        and bool(
            metric_contract[
                "baseline_reference_method_frozen"
            ]
        )
        and bool(
            metric_contract[
                "scenario_manifest_materialization_authorized"
            ]
        )
    ),

        "hard_gate_thresholds_frozen":
            bool(
            metric_contract[
                "hard_gate_thresholds_frozen"
        ]
    ),

        "reference_metric_calculations_frozen":
            bool(
            metric_contract[
                "reference_metric_calculations_frozen"
        ]
    ),

        "frozen_baseline_materialization_installed":
        (
            bool(
            materialization[
                "all_required_scenarios_materialized"
            ]
        )
        and bool(
            materialization[
                "all_required_scenarios_have_evidence"
            ]
        )
        and bool(
            materialization[
                "all_content_hashes_present"
            ]
        )
    ),

        "baseline_manifest_construction_authorized":
            bool(
            materialization[
                "pre_execution_baseline_manifest_"
                "construction_authorized"
        ]
    ),

        "correct_run_type":
            (
                str(
                    manifest[
                        "run_type"
                    ]
                )
                ==
                "PRE_PLAYGROUND_FROZEN_BASELINE"
            ),

        "correct_lifecycle_phase":
            (
                str(
                    manifest[
                        "lifecycle_phase"
                    ]
                )
                ==
                "BASELINE_ESTABLISHMENT"
            ),

        "correct_benchmark_suite":
            (
                str(
                    manifest[
                        "benchmark_suite"
                    ]
                )
                ==
                "FROZEN_REGRESSION_SUITE"
            ),

        "predeployment_state_preserved":
            (
                str(
                    manifest[
                        "deployment_state"
                    ]
                )
                ==
                "PREDEPLOYMENT"
            ),

        "scenario_manifest_hash_matches":
            (
                scenario_manifest_hash
                ==
                str(
                    scenario_write[
                        "scenario_manifest_sha256"
                    ]
                )
            ),

        "scenario_manifest_is_immutable":
            (
                scenario_write[
                    "manifest_write_state"
                ]
                ==
                "IMMUTABLE_FROZEN_BASELINE_"
                "SCENARIO_MANIFEST_WRITTEN"
            ),

        "run_manifest_written_immutably":
            (
                run_write[
                    "run_manifest_write_state"
                ]
                ==
                "IMMUTABLE_PRE_EXECUTION_"
                "BASELINE_RUN_MANIFEST_WRITTEN"
            ),

        "run_ids_match":
            (
                str(
                    manifest[
                        "run_id"
                    ]
                )
                ==
                str(
                    run_write[
                        "run_id"
                    ]
                )
            ),

        "manifest_fingerprint_present":
            (
                len(
                    manifest_fingerprint
                )
                == 64
            ),

        "git_commit_identified":
            (
                bool(
                    git_commit_hash
                )
                and git_commit_hash.upper()
                not in {
                    "UNKNOWN",
                    "UNAVAILABLE",
                    "NONE",
                }
            ),

        "git_worktree_clean":
            (
                git_worktree_state
                in clean_worktree_states
            ),

        "dependency_fingerprint_present":
            (
                len(
                    dependency_fingerprint
                )
                == 64
            ),

        "environment_fingerprint_present":
            (
                len(
                    environment_fingerprint
                )
                == 64
            ),

        "manifest_benchmark_authority_withheld":
            (
                not bool(
                    manifest.get(
                        "benchmark_execution_authorized",
                        False,
                    )
                )
            ),

        "manifest_deployment_authority_withheld":
            (
                not bool(
                    manifest.get(
                        "operational_deployment_authorized",
                        False,
                    )
                )
            ),
    }

    failed_checks = [
        check_name
        for check_name, passed
        in checks.items()
        if not passed
    ]

    gate_installed = (
        len(
            failed_checks
        )
        == 0
    )

    return pd.DataFrame(
        [
            {
                "chapter":
                    (
                        "CHAPTER_69_SYSTEM_VALIDATION_"
                        "AND_FINAL_REVIEW"
                    ),

                "gate":
                    (
                        "BIRE_PREPLAYGROUND_FROZEN_"
                        "BASELINE_EXECUTION_GATE"
                    ),

                "baseline_preparation_version":
                    _BIRE_BASELINE_PREPARATION_VERSION,

                "baseline_execution_gate_version":
                    _BIRE_BASELINE_EXECUTION_GATE_VERSION,

                "run_id":
                    manifest[
                        "run_id"
                    ],

                **checks,

                "failed_gate_check_count":
                    len(
                        failed_checks
                    ),

                "failed_gate_checks":
                    (
                        "NONE"
                        if not failed_checks
                        else " | ".join(
                            failed_checks
                        )
                    ),

                "frozen_baseline_execution_authorized":
                    gate_installed,

                "metric_execution_within_frozen_"
                "baseline_authorized":
                    gate_installed,

                # Global/general benchmark authority remains
                # withheld. Only this named frozen run is
                # authorized when the gate passes.
                "general_benchmark_execution_authorized":
                    False,

                "sealed_generalization_exposure_authorized":
                    False,

                "rotating_novelty_exposure_authorized":
                    False,

                "adversarial_playground_exposure_authorized":
                    False,

                "playground_execution_authorized":
                    False,

                "deployment_decision_authorized":
                    False,

                "operational_deployment_authorized":
                    False,

                "nid_authorization_state":
                    (
                        "NID_BIRE_FROZEN_BASELINE_"
                        "EXECUTION_AUTHORIZED"
                        if gate_installed
                        else
                        "NID_BIRE_FROZEN_BASELINE_"
                        "EXECUTION_BLOCKED"
                    ),

                "next_required_stage":
                    (
                        "CHAPTER_69_9_PRE_PLAYGROUND_"
                        "BASELINE_EXECUTION_AND_RESULT_RECORD"
                        if gate_installed
                        else
                        "CHAPTER_69_8_BASELINE_EXECUTION_"
                        "GATE_REMEDIATION"
                    ),


            }
        ]
    )


__all__ = [
    "build_bire_preplayground_baseline_run_manifest",
    "write_bire_preplayground_baseline_run_manifest",
    "build_bire_preplayground_baseline_execution_gate",
]