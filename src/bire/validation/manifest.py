#============================================================
# Project Sixth Sense — BIRE OS
# Chapter 69.3 — Evaluation Run Manifest and
#                  Reproducibility Contract
#============================================================

from __future__ import annotations

from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid

import pandas as pd


_PSS_EVALUATION_RUN_MANIFEST_SCHEMA_VERSION = (
    "PSS_EVALUATION_RUN_MANIFEST_V1"
)


_PSS_EVALUATION_RUN_MANIFEST_FIELDS = [
    {
        "field_category": "MANIFEST_IDENTITY",
        "field_name": "manifest_schema_version",
        "required_before_execution": True,
        "field_purpose": (
            "Identifies the schema used to construct the run manifest."
        ),
    },
    {
        "field_category": "MANIFEST_IDENTITY",
        "field_name": "run_id",
        "required_before_execution": True,
        "field_purpose": (
            "Provides the immutable unique identity of the evaluation run."
        ),
    },
    {
        "field_category": "MANIFEST_IDENTITY",
        "field_name": "created_at_utc",
        "required_before_execution": True,
        "field_purpose": (
            "Records when the pre-execution manifest was created."
        ),
    },
    {
        "field_category": "SYSTEM_IDENTITY",
        "field_name": "system_id",
        "required_before_execution": True,
        "field_purpose": (
            "Identifies the PSS system being evaluated."
        ),
    },
    {
        "field_category": "SYSTEM_IDENTITY",
        "field_name": "system_version",
        "required_before_execution": True,
        "field_purpose": (
            "Identifies the exact system release or candidate version."
        ),
    },
    {
        "field_category": "SYSTEM_IDENTITY",
        "field_name": "run_type",
        "required_before_execution": True,
        "field_purpose": (
            "Identifies baseline, comparison, Playground, retest, "
            "re-entry, or postdeployment purpose."
        ),
    },
    {
        "field_category": "SYSTEM_IDENTITY",
        "field_name": "lifecycle_phase",
        "required_before_execution": True,
        "field_purpose": (
            "Links the run to the governed PSS lifecycle phase."
        ),
    },
    {
        "field_category": "SYSTEM_IDENTITY",
        "field_name": "deployment_state",
        "required_before_execution": True,
        "field_purpose": (
            "Distinguishes predeployment, shadow, controlled, "
            "or postdeployment evidence."
        ),
    },
    {
        "field_category": "RUN_LINKAGE",
        "field_name": "comparison_baseline_run_id",
        "required_before_execution": False,
        "field_purpose": (
            "Identifies the baseline used for comparative evaluation."
        ),
    },
    {
        "field_category": "RUN_LINKAGE",
        "field_name": "parent_run_id",
        "required_before_execution": False,
        "field_purpose": (
            "Links a rerun, correction, or follow-up to its prior run."
        ),
    },
    {
        "field_category": "REPOSITORY",
        "field_name": "repository_root",
        "required_before_execution": True,
        "field_purpose": (
            "Records the repository used during evaluation."
        ),
    },
    {
        "field_category": "REPOSITORY",
        "field_name": "git_commit_hash",
        "required_before_execution": True,
        "field_purpose": (
            "Identifies the exact committed repository state."
        ),
    },
    {
        "field_category": "REPOSITORY",
        "field_name": "git_branch",
        "required_before_execution": True,
        "field_purpose": (
            "Records the active repository branch."
        ),
    },
    {
        "field_category": "REPOSITORY",
        "field_name": "git_worktree_state",
        "required_before_execution": True,
        "field_purpose": (
            "Preserves whether uncommitted changes existed."
        ),
    },
    {
        "field_category": "ENVIRONMENT",
        "field_name": "python_version",
        "required_before_execution": True,
        "field_purpose": (
            "Records the Python runtime version."
        ),
    },
    {
        "field_category": "ENVIRONMENT",
        "field_name": "python_executable",
        "required_before_execution": True,
        "field_purpose": (
            "Records the runtime executable used."
        ),
    },
    {
        "field_category": "ENVIRONMENT",
        "field_name": "platform_string",
        "required_before_execution": True,
        "field_purpose": (
            "Records operating-system and architecture information."
        ),
    },
    {
        "field_category": "ENVIRONMENT",
        "field_name": "dependency_snapshot_source",
        "required_before_execution": True,
        "field_purpose": (
            "Identifies whether dependency state came from lock files "
            "or installed distributions."
        ),
    },
    {
        "field_category": "ENVIRONMENT",
        "field_name": "dependency_snapshot_sha256",
        "required_before_execution": True,
        "field_purpose": (
            "Fingerprints the complete dependency state."
        ),
    },
    {
        "field_category": "ENVIRONMENT",
        "field_name": "environment_fingerprint_sha256",
        "required_before_execution": True,
        "field_purpose": (
            "Fingerprints the combined runtime environment."
        ),
    },
    {
        "field_category": "BENCHMARK",
        "field_name": "benchmark_suite",
        "required_before_execution": True,
        "field_purpose": (
            "Identifies the governed benchmark suite."
        ),
    },
    {
        "field_category": "BENCHMARK",
        "field_name": "benchmark_suite_version",
        "required_before_execution": True,
        "field_purpose": (
            "Identifies the exact benchmark-suite version."
        ),
    },
    {
        "field_category": "BENCHMARK",
        "field_name": "scenario_manifest_id",
        "required_before_execution": True,
        "field_purpose": (
            "Identifies the scenario manifest selected for execution."
        ),
    },
    {
        "field_category": "BENCHMARK",
        "field_name": "scenario_manifest_version",
        "required_before_execution": True,
        "field_purpose": (
            "Identifies the exact scenario-manifest version."
        ),
    },
    {
        "field_category": "BENCHMARK",
        "field_name": "scenario_manifest_sha256",
        "required_before_execution": True,
        "field_purpose": (
            "Fingerprints the scenario manifest."
        ),
    },
    {
        "field_category": "BENCHMARK",
        "field_name": "difficulty_profile_version",
        "required_before_execution": True,
        "field_purpose": (
            "Preserves the challenge definition used by the run."
        ),
    },
    {
        "field_category": "BENCHMARK",
        "field_name": "novelty_profile_version",
        "required_before_execution": True,
        "field_purpose": (
            "Preserves the novelty definition used by the run."
        ),
    },
    {
        "field_category": "POLICY",
        "field_name": "metric_definition_version",
        "required_before_execution": True,
        "field_purpose": (
            "Identifies the metric definitions used to judge results."
        ),
    },
    {
        "field_category": "POLICY",
        "field_name": "governance_policy_version",
        "required_before_execution": True,
        "field_purpose": (
            "Identifies the active governance contract."
        ),
    },
    {
        "field_category": "POLICY",
        "field_name": "nid_policy_version",
        "required_before_execution": True,
        "field_purpose": (
            "Identifies the active NID authority policy."
        ),
    },
    {
        "field_category": "RANDOMNESS",
        "field_name": "random_seed_set_json",
        "required_before_execution": True,
        "field_purpose": (
            "Preserves the complete random-seed set."
        ),
    },
    {
        "field_category": "ARTIFACTS",
        "field_name": "configuration_artifacts_json",
        "required_before_execution": True,
        "field_purpose": (
            "Records configuration paths and SHA-256 fingerprints."
        ),
    },
    {
        "field_category": "ARTIFACTS",
        "field_name": "model_artifacts_json",
        "required_before_execution": True,
        "field_purpose": (
            "Records model paths and SHA-256 fingerprints."
        ),
    },
    {
        "field_category": "ARTIFACTS",
        "field_name": "input_artifacts_json",
        "required_before_execution": True,
        "field_purpose": (
            "Records input paths and SHA-256 fingerprints."
        ),
    },
    {
        "field_category": "OUTPUT",
        "field_name": "planned_output_root",
        "required_before_execution": True,
        "field_purpose": (
            "Records where post-execution evidence will be written."
        ),
    },
    {
        "field_category": "IMMUTABILITY",
        "field_name": "manifest_fingerprint_sha256",
        "required_before_execution": True,
        "field_purpose": (
            "Fingerprints the complete locked pre-execution manifest."
        ),
    },
]


def _normalize_json_value(
    value: Any,
) -> Any:
    """
    Convert Python, pandas, pathlib, and scalar values into a
    stable JSON-compatible representation.
    """

    if value is None:
        return None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): _normalize_json_value(item)
            for key, item in sorted(
                value.items(),
                key=lambda pair: str(pair[0]),
            )
        }

    if isinstance(value, (list, tuple, set)):
        return [
            _normalize_json_value(item)
            for item in value
        ]

    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False

    if isinstance(missing, bool) and missing:
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass

    return value


def _canonical_json(
    value: Any,
) -> str:
    return json.dumps(
        _normalize_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _sha256_text(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def _sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        for block in iter(
            lambda: file_handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _run_git_command(
    repository_root: Path,
    arguments: list[str],
) -> str | None:
    try:
        completed = subprocess.run(
            [
                "git",
                *arguments,
            ],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ):
        return None

    return completed.stdout.strip()


def _resolve_repository_root(
    repository_root: str | Path | None,
) -> Path:
    if repository_root is not None:
        return Path(
            repository_root
        ).expanduser().resolve()

    current = Path.cwd().resolve()

    discovered = _run_git_command(
        current,
        [
            "rev-parse",
            "--show-toplevel",
        ],
    )

    if discovered:
        return Path(
            discovered
        ).resolve()

    return current


def _collect_git_metadata(
    repository_root: Path,
) -> dict[str, object]:
    commit_hash = _run_git_command(
        repository_root,
        [
            "rev-parse",
            "HEAD",
        ],
    )

    branch = _run_git_command(
        repository_root,
        [
            "rev-parse",
            "--abbrev-ref",
            "HEAD",
        ],
    )

    status = _run_git_command(
        repository_root,
        [
            "status",
            "--porcelain",
        ],
    )

    if status is None:
        worktree_state = "GIT_STATUS_UNAVAILABLE"
    elif status:
        worktree_state = "DIRTY_UNCOMMITTED_CHANGES_PRESENT"
    else:
        worktree_state = "CLEAN"

    return {
        "repository_root":
            str(repository_root),

        "git_commit_hash":
            commit_hash
            or "GIT_COMMIT_UNAVAILABLE",

        "git_branch":
            branch
            or "GIT_BRANCH_UNAVAILABLE",

        "git_worktree_state":
            worktree_state,
    }


def _collect_dependency_snapshot(
    repository_root: Path,
) -> dict[str, object]:
    lock_file_names = [
        "requirements.txt",
        "requirements-dev.txt",
        "pyproject.toml",
        "poetry.lock",
        "Pipfile.lock",
        "environment.yml",
        "environment.yaml",
        "conda-lock.yml",
        "uv.lock",
    ]

    lock_records: list[dict[str, str]] = []

    for file_name in lock_file_names:
        candidate = (
            repository_root
            / file_name
        )

        if not candidate.is_file():
            continue

        lock_records.append(
            {
                "path":
                    str(
                        candidate.relative_to(
                            repository_root
                        )
                    ),

                "sha256":
                    _sha256_file(
                        candidate
                    ),
            }
        )

    if lock_records:
        snapshot_payload: object = (
            lock_records
        )

        snapshot_source = (
            "REPOSITORY_DEPENDENCY_FILES"
        )

    else:
        distributions: list[str] = []

        for distribution in (
            metadata.distributions()
        ):
            name = (
                distribution.metadata.get(
                    "Name"
                )
                or "UNKNOWN_PACKAGE"
            )

            distributions.append(
                f"{name}=={distribution.version}"
            )

        snapshot_payload = sorted(
            distributions,
            key=str.lower,
        )

        snapshot_source = (
            "INSTALLED_DISTRIBUTION_SNAPSHOT"
        )

    snapshot_json = _canonical_json(
        snapshot_payload
    )

    return {
        "dependency_snapshot_source":
            snapshot_source,

        "dependency_snapshot_sha256":
            _sha256_text(
                snapshot_json
            ),

        "dependency_snapshot_json":
            snapshot_json,
    }


def _build_environment_metadata(
    dependency_snapshot: dict[str, object],
) -> dict[str, object]:
    environment_payload = {
        "python_version":
            sys.version,

        "python_executable":
            sys.executable,

        "platform_string":
            platform.platform(),

        "machine":
            platform.machine(),

        "processor":
            platform.processor(),

        "dependency_snapshot_source":
            dependency_snapshot[
                "dependency_snapshot_source"
            ],

        "dependency_snapshot_sha256":
            dependency_snapshot[
                "dependency_snapshot_sha256"
            ],
    }

    return {
        **environment_payload,

        "environment_fingerprint_sha256":
            _sha256_text(
                _canonical_json(
                    environment_payload
                )
            ),
    }


def _hash_artifact_collection(
    paths: Iterable[str | Path] | None,
    repository_root: Path,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    for raw_path in paths or []:
        path = Path(
            raw_path
        ).expanduser()

        if not path.is_absolute():
            path = (
                repository_root
                / path
            )

        path = path.resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"Evaluation artifact does not exist: {path}"
            )

        if not path.is_file():
            raise ValueError(
                "Evaluation artifact hashing currently supports "
                f"files only: {path}"
            )

        try:
            display_path = str(
                path.relative_to(
                    repository_root
                )
            )
        except ValueError:
            display_path = str(path)

        records.append(
            {
                "path":
                    display_path,

                "size_bytes":
                    int(
                        path.stat().st_size
                    ),

                "sha256":
                    _sha256_file(
                        path
                    ),
            }
        )

    return sorted(
        records,
        key=lambda record: str(
            record["path"]
        ),
    )


def _require_nonempty_text(
    value: object,
    field_name: str,
) -> str:
    normalized = str(
        value
    ).strip()

    if not normalized:
        raise ValueError(
            f"{field_name} must not be empty."
        )

    return normalized


def _manifest_fingerprint(
    record: dict[str, object],
) -> str:
    payload = {
        key: value
        for key, value in record.items()
        if key
        != "manifest_fingerprint_sha256"
    }

    return _sha256_text(
        _canonical_json(
            payload
        )
    )


def build_pss_evaluation_run_manifest_schema() -> pd.DataFrame:
    """
    Return the PSS evaluation-run manifest schema.

    Schema installation does not authorize benchmark execution.
    """

    result = pd.DataFrame(
        _PSS_EVALUATION_RUN_MANIFEST_FIELDS
    )

    result[
        "manifest_schema_version"
    ] = (
        _PSS_EVALUATION_RUN_MANIFEST_SCHEMA_VERSION
    )

    result[
        "schema_field_registered"
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
        "NID_PSS_RUN_MANIFEST_FIELD_REGISTERED"
    )

    return result


def build_bire_evaluation_reproducibility_contract(
    framework_adoption_contract_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Install the BIRE OS evaluation-run reproducibility contract.

    The contract authorizes manifest construction and immutable
    writing only. Benchmark execution remains unauthorized.
    """

    if not isinstance(
        framework_adoption_contract_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "framework_adoption_contract_df must be a DataFrame."
        )

    required_columns = {
        "framework_adoption_state",
        "run_manifest_required",
        "immutable_evaluation_ledger_required",
        "scenario_isolation_required",
        "benchmark_execution_authorized",
        "operational_deployment_authorized",
    }

    missing_columns = sorted(
        required_columns
        - set(
            framework_adoption_contract_df.columns
        )
    )

    if missing_columns:
        raise KeyError(
            "BIRE reproducibility contract cannot proceed. "
            f"Missing framework-adoption columns: "
            f"{missing_columns}"
        )

    if len(
        framework_adoption_contract_df
    ) != 1:
        raise ValueError(
            "framework_adoption_contract_df must contain one row."
        )

    adoption = (
        framework_adoption_contract_df
        .iloc[0]
    )

    checks = {
        "pss_framework_adopted":
            (
                adoption[
                    "framework_adoption_state"
                ]
                ==
                "BIRE_PSS_PREDEPLOYMENT_FRAMEWORK_ADOPTED"
            ),

        "run_manifest_required_confirmed":
            bool(
                adoption[
                    "run_manifest_required"
                ]
            ),

        "immutable_ledger_required_confirmed":
            bool(
                adoption[
                    "immutable_evaluation_ledger_required"
                ]
            ),

        "scenario_isolation_required_confirmed":
            bool(
                adoption[
                    "scenario_isolation_required"
                ]
            ),

        "benchmark_execution_still_withheld":
            not bool(
                adoption[
                    "benchmark_execution_authorized"
                ]
            ),

        "deployment_authority_still_withheld":
            not bool(
                adoption[
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
                        "BIRE_EVALUATION_RUN_MANIFEST_"
                        "AND_REPRODUCIBILITY_CONTRACT"
                    ),

                "manifest_schema_version":
                    (
                        _PSS_EVALUATION_RUN_MANIFEST_SCHEMA_VERSION
                    ),

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

                "manifest_schema_installed":
                    contract_installed,

                "unique_run_id_required":
                    True,

                "pre_execution_manifest_required":
                    True,

                "pre_execution_manifest_lock_required":
                    True,

                "post_execution_result_record_required":
                    True,

                "manifest_in_place_mutation_authorized":
                    False,

                "manifest_overwrite_authorized":
                    False,

                "run_id_reuse_authorized":
                    False,

                "sha256_artifact_fingerprinting_required":
                    True,

                "repository_state_capture_required":
                    True,

                "dirty_worktree_state_capture_required":
                    True,

                "dependency_fingerprint_required":
                    True,

                "environment_fingerprint_required":
                    True,

                "benchmark_version_required":
                    True,

                "scenario_manifest_hash_required":
                    True,

                "difficulty_profile_version_required":
                    True,

                "novelty_profile_version_required":
                    True,

                "metric_definition_version_required":
                    True,

                "governance_policy_version_required":
                    True,

                "nid_policy_version_required":
                    True,

                "random_seed_set_required":
                    True,

                "comparison_run_linkage_required":
                    True,

                "run_manifest_construction_authorized":
                    contract_installed,

                "immutable_manifest_writing_authorized":
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
                        "NID_BIRE_REPRODUCIBILITY_"
                        "CONTRACT_INSTALLED"
                        if contract_installed
                        else
                        "NID_BIRE_REPRODUCIBILITY_"
                        "CONTRACT_REVIEW_REQUIRED"
                    ),

                "next_required_stage":
                    (
                        "CHAPTER_69_4_BENCHMARK_SUITE_"
                        "REGISTRY_AND_SCENARIO_ISOLATION_POLICY"
                    ),
            }
        ]
    )


def build_bire_evaluation_run_manifest(
    framework_adoption_contract_df: pd.DataFrame,
    *,
    system_version: str,
    run_type: str,
    lifecycle_phase: str,
    deployment_state: str,
    benchmark_suite: str,
    benchmark_suite_version: str,
    scenario_manifest_id: str,
    scenario_manifest_version: str,
    scenario_manifest_path: str | Path,
    difficulty_profile_version: str,
    novelty_profile_version: str,
    metric_definition_version: str,
    governance_policy_version: str,
    nid_policy_version: str,
    random_seeds: Iterable[int],
    planned_output_root: str | Path,
    repository_root: str | Path | None = None,
    configuration_paths: Iterable[str | Path] | None = None,
    model_artifact_paths: Iterable[str | Path] | None = None,
    input_artifact_paths: Iterable[str | Path] | None = None,
    comparison_baseline_run_id: str | None = None,
    parent_run_id: str | None = None,
    run_notes: str = "",
    run_id: str | None = None,
) -> pd.DataFrame:
    """
    Build a locked pre-execution BIRE OS evaluation-run manifest.

    This function fingerprints the repository, environment,
    scenario manifest, configurations, models, and inputs.

    It does not execute the benchmark or create a result record.
    """

    if not isinstance(
        framework_adoption_contract_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "framework_adoption_contract_df must be a DataFrame."
        )

    if len(
        framework_adoption_contract_df
    ) != 1:
        raise ValueError(
            "framework_adoption_contract_df must contain one row."
        )

    adoption = (
        framework_adoption_contract_df
        .iloc[0]
    )

    if (
        adoption[
            "framework_adoption_state"
        ]
        !=
        "BIRE_PSS_PREDEPLOYMENT_FRAMEWORK_ADOPTED"
    ):
        raise ValueError(
            "BIRE OS has not completed PSS framework adoption."
        )

    seeds = [
        int(seed)
        for seed in random_seeds
    ]

    if not seeds:
        raise ValueError(
            "random_seeds must contain at least one seed."
        )

    resolved_repository_root = (
        _resolve_repository_root(
            repository_root
        )
    )

    resolved_scenario_manifest = Path(
        scenario_manifest_path
    ).expanduser()

    if not resolved_scenario_manifest.is_absolute():
        resolved_scenario_manifest = (
            resolved_repository_root
            / resolved_scenario_manifest
        )

    resolved_scenario_manifest = (
        resolved_scenario_manifest.resolve()
    )

    if not resolved_scenario_manifest.is_file():
        raise FileNotFoundError(
            "Scenario manifest does not exist: "
            f"{resolved_scenario_manifest}"
        )

    repository_metadata = (
        _collect_git_metadata(
            resolved_repository_root
        )
    )

    dependency_snapshot = (
        _collect_dependency_snapshot(
            resolved_repository_root
        )
    )

    environment_metadata = (
        _build_environment_metadata(
            dependency_snapshot
        )
    )

    configuration_artifacts = (
        _hash_artifact_collection(
            configuration_paths,
            resolved_repository_root,
        )
    )

    model_artifacts = (
        _hash_artifact_collection(
            model_artifact_paths,
            resolved_repository_root,
        )
    )

    input_artifacts = (
        _hash_artifact_collection(
            input_artifact_paths,
            resolved_repository_root,
        )
    )

    created_at_utc = datetime.now(
        timezone.utc
    )

    resolved_run_id = (
        _require_nonempty_text(
            run_id,
            "run_id",
        )
        if run_id is not None
        else
        (
            "PSSRUN-"
            + created_at_utc.strftime(
                "%Y%m%dT%H%M%SZ"
            )
            + "-"
            + uuid.uuid4().hex[:10].upper()
        )
    )

    planned_output_path = Path(
        planned_output_root
    ).expanduser()

    if not planned_output_path.is_absolute():
        planned_output_path = (
            resolved_repository_root
            / planned_output_path
        )

    record: dict[str, object] = {
        "manifest_schema_version":
            _PSS_EVALUATION_RUN_MANIFEST_SCHEMA_VERSION,

        "run_id":
            resolved_run_id,

        "created_at_utc":
            created_at_utc.isoformat(),

        "system_id":
            "BIRE_OS",

        "system_name":
            (
                "BIO_INTELLIGENCE_RISK_ENGINE_"
                "OPERATING_SYSTEM"
            ),

        "system_version":
            _require_nonempty_text(
                system_version,
                "system_version",
            ),

        "run_type":
            _require_nonempty_text(
                run_type,
                "run_type",
            ),

        "lifecycle_phase":
            _require_nonempty_text(
                lifecycle_phase,
                "lifecycle_phase",
            ),

        "deployment_state":
            _require_nonempty_text(
                deployment_state,
                "deployment_state",
            ),

        "comparison_baseline_run_id":
            comparison_baseline_run_id,

        "parent_run_id":
            parent_run_id,

        **repository_metadata,

        **environment_metadata,

        "dependency_snapshot_json":
            dependency_snapshot[
                "dependency_snapshot_json"
            ],

        "benchmark_suite":
            _require_nonempty_text(
                benchmark_suite,
                "benchmark_suite",
            ),

        "benchmark_suite_version":
            _require_nonempty_text(
                benchmark_suite_version,
                "benchmark_suite_version",
            ),

        "scenario_manifest_id":
            _require_nonempty_text(
                scenario_manifest_id,
                "scenario_manifest_id",
            ),

        "scenario_manifest_version":
            _require_nonempty_text(
                scenario_manifest_version,
                "scenario_manifest_version",
            ),

        "scenario_manifest_path":
            str(
                resolved_scenario_manifest
            ),

        "scenario_manifest_sha256":
            _sha256_file(
                resolved_scenario_manifest
            ),

        "difficulty_profile_version":
            _require_nonempty_text(
                difficulty_profile_version,
                "difficulty_profile_version",
            ),

        "novelty_profile_version":
            _require_nonempty_text(
                novelty_profile_version,
                "novelty_profile_version",
            ),

        "metric_definition_version":
            _require_nonempty_text(
                metric_definition_version,
                "metric_definition_version",
            ),

        "governance_policy_version":
            _require_nonempty_text(
                governance_policy_version,
                "governance_policy_version",
            ),

        "nid_policy_version":
            _require_nonempty_text(
                nid_policy_version,
                "nid_policy_version",
            ),

        "random_seed_set_json":
            _canonical_json(
                sorted(
                    set(seeds)
                )
            ),

        "configuration_artifacts_json":
            _canonical_json(
                configuration_artifacts
            ),

        "model_artifacts_json":
            _canonical_json(
                model_artifacts
            ),

        "input_artifacts_json":
            _canonical_json(
                input_artifacts
            ),

        "planned_output_root":
            str(
                planned_output_path.resolve()
            ),

        "run_notes":
            str(run_notes),

        "manifest_stage":
            "PRE_EXECUTION",

        "manifest_lock_state":
            "PRE_EXECUTION_MANIFEST_LOCKED",

        "post_execution_result_record_required":
            True,

        "benchmark_execution_authorized":
            False,

        "operational_deployment_authorized":
            False,

        "nid_authorization_state":
            "NID_PRE_EXECUTION_RUN_MANIFEST_LOCKED",
    }

    record[
        "manifest_fingerprint_sha256"
    ] = _manifest_fingerprint(
        record
    )

    return pd.DataFrame(
        [
            record
        ]
    )


def write_bire_evaluation_run_manifest(
    manifest_df: pd.DataFrame,
    output_path: str | Path,
) -> pd.DataFrame:
    """
    Write a locked evaluation-run manifest as canonical JSON.

    Existing files are never overwritten.
    """

    if not isinstance(
        manifest_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "manifest_df must be a pandas DataFrame."
        )

    if len(manifest_df) != 1:
        raise ValueError(
            "manifest_df must contain exactly one row."
        )

    required_columns = {
        "run_id",
        "manifest_lock_state",
        "manifest_fingerprint_sha256",
    }

    missing_columns = sorted(
        required_columns
        - set(manifest_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "Manifest writing cannot proceed. "
            f"Missing columns: {missing_columns}"
        )

    record = {
        str(key): _normalize_json_value(value)
        for key, value in (
            manifest_df.iloc[0].to_dict()
        ).items()
    }

    if (
        record[
            "manifest_lock_state"
        ]
        !=
        "PRE_EXECUTION_MANIFEST_LOCKED"
    ):
        raise ValueError(
            "Only locked pre-execution manifests may be written."
        )

    recorded_fingerprint = str(
        record[
            "manifest_fingerprint_sha256"
        ]
    )

    recalculated_fingerprint = (
        _manifest_fingerprint(
            record
        )
    )

    if (
        recorded_fingerprint
        != recalculated_fingerprint
    ):
        raise ValueError(
            "Manifest fingerprint verification failed. "
            "The manifest may have changed after locking."
        )

    destination = Path(
        output_path
    ).expanduser().resolve()

    if destination.exists():
        raise FileExistsError(
            "Evaluation manifests are immutable and cannot "
            f"be overwritten: {destination}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = destination.with_name(
        destination.name
        + ".tmp-"
        + uuid.uuid4().hex
    )

    canonical_payload = (
        json.dumps(
            record,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n"
    )

    try:
        temporary_path.write_text(
            canonical_payload,
            encoding="utf-8",
        )

        os.replace(
            temporary_path,
            destination,
        )

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return pd.DataFrame(
        [
            {
                "run_id":
                    record["run_id"],

                "manifest_path":
                    str(destination),

                "manifest_fingerprint_sha256":
                    recorded_fingerprint,

                "manifest_write_state":
                    "IMMUTABLE_MANIFEST_WRITTEN",

                "manifest_overwrite_authorized":
                    False,

                "benchmark_execution_authorized":
                    False,

                "operational_deployment_authorized":
                    False,

                "nid_authorization_state":
                    (
                        "NID_IMMUTABLE_RUN_"
                        "MANIFEST_WRITTEN"
                    ),
            }
        ]
    )


__all__ = [
    "build_pss_evaluation_run_manifest_schema",
    "build_bire_evaluation_reproducibility_contract",
    "build_bire_evaluation_run_manifest",
    "write_bire_evaluation_run_manifest",
]