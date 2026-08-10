#============================================================
# Project Sixth Sense — BIRE OS
# Chapter 69.7 — Frozen Baseline Scenario
#                  Manifest Materialization
#============================================================

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import hashlib
import json
import os
import uuid

import pandas as pd


_BIRE_FROZEN_BASELINE_MANIFEST_VERSION = (
    "BIRE_FROZEN_BASELINE_MANIFEST_V1"
)

_BIRE_BASELINE_DIFFICULTY_PROFILE_VERSION = (
    "BIRE_BASELINE_DIFFICULTY_PROFILE_V1"
)

_BIRE_BASELINE_NOVELTY_PROFILE_VERSION = (
    "BIRE_BASELINE_NOVELTY_PROFILE_V1"
)


_BIRE_FROZEN_BASELINE_SCENARIOS = [
    {
        "scenario_id":
            "BIRE-FROZEN-ESCALATION-001",

        "scenario_family":
            "ESCALATION_EVENT_REFERENCE",

        "scenario_description":
            (
                "Known encounters where governed care "
                "escalation was observed."
            ),

        "source_frame_key":
            "CANONICAL_LPMR",

        "selection_rule_id":
            "CARE_ESCALATION_POSITIVE",

        "required_columns":
            [
                "patient_id",
                "encounter_id",
                "care_escalation_occurred",
            ],

        "target_metrics":
            [
                "DETERIORATION_DETECTION_RATE",
                "MISSED_DETERIORATION_RATE",
                "GOVERNED_EVENT_LEAD_TIME",
            ],

        "required_for_baseline":
            True,
    },

    {
        "scenario_id":
            "BIRE-FROZEN-NONESCALATION-001",

        "scenario_family":
            "NON_ESCALATION_REFERENCE",

        "scenario_description":
            (
                "Known encounters where care escalation "
                "was not observed."
            ),

        "source_frame_key":
            "CANONICAL_LPMR",

        "selection_rule_id":
            "CARE_ESCALATION_NEGATIVE",

        "required_columns":
            [
                "patient_id",
                "encounter_id",
                "care_escalation_occurred",
            ],

        "target_metrics":
            [
                "FALSE_ACTIVATION_BURDEN",
            ],

        "required_for_baseline":
            True,
    },

    {
        "scenario_id":
            "BIRE-FROZEN-HIDDEN-INSTABILITY-001",

        "scenario_family":
            "HIDDEN_INSTABILITY_REFERENCE",

        "scenario_description":
            (
                "Known encounters containing governed masked "
                "or hidden instability evidence."
            ),

        "source_frame_key":
            "CANONICAL_LPMR",

        "selection_rule_id":
            "MASKED_INSTABILITY_POSITIVE",

        "required_columns":
            [
                "patient_id",
                "encounter_id",
                "masked_instability_signal",
            ],

        "target_metrics":
            [
                "HIDDEN_INSTABILITY_RECOGNITION_RATE",
            ],

        "required_for_baseline":
            True,
    },

    {
        "scenario_id":
            "BIRE-FROZEN-REBOUND-001",

        "scenario_family":
            "REBOUND_INSTABILITY_REFERENCE",

        "scenario_description":
            (
                "Known encounters containing observed "
                "stepdown rebound instability."
            ),

        "source_frame_key":
            "CANONICAL_LPMR",

        "selection_rule_id":
            "STEPDOWN_REBOUND_POSITIVE",

        "required_columns":
            [
                "patient_id",
                "encounter_id",
                "stepdown_rebound_flag",
            ],

        "target_metrics":
            [
                "DETERIORATION_DETECTION_RATE",
                "MISSED_DETERIORATION_RATE",
            ],

        "required_for_baseline":
            True,
    },

    {
        "scenario_id":
            "BIRE-FROZEN-FALSE-RECOVERY-001",

        "scenario_family":
            "FALSE_RECOVERY_REFERENCE",

        "scenario_description":
            (
                "Known encounters containing governed "
                "false vital recovery evidence."
            ),

        "source_frame_key":
            "CANONICAL_LPMR",

        "selection_rule_id":
            "FALSE_VITAL_RECOVERY_POSITIVE",

        "required_columns":
            [
                "patient_id",
                "encounter_id",
                "false_vital_recovery_signal",
            ],

        "target_metrics":
            [
                "RECOVERY_CONTRADICTION_RECOGNITION_RATE",
                "PREMATURE_RECOVERY_CLAIM_COUNT",
                "UNSUPPORTED_CERTAINTY_COUNT",
            ],

        "required_for_baseline":
            True,
    },

    {
        "scenario_id":
            "BIRE-FROZEN-RECOVERY-WARNING-001",

        "scenario_family":
            "RECOVERY_AUTHENTICITY_WARNING_REFERENCE",

        "scenario_description":
            (
                "Known encounters where vital recovery "
                "authenticity warning evidence was present."
            ),

        "source_frame_key":
            "CANONICAL_LPMR",

        "selection_rule_id":
            "RECOVERY_AUTHENTICITY_WARNING_POSITIVE",

        "required_columns":
            [
                "patient_id",
                "encounter_id",
                "vital_recovery_authenticity_warning",
            ],

        "target_metrics":
            [
                "RECOVERY_CONTRADICTION_RECOGNITION_RATE",
                "UNCERTAINTY_PRESERVATION_RATE",
            ],

        "required_for_baseline":
            False,
    },

    {
        "scenario_id":
            "BIRE-FROZEN-TEMPORAL-TIE-001",

        "scenario_family":
            "TEMPORAL_TIE_REFERENCE",

        "scenario_description":
            (
                "Known encounter histories containing "
                "primary chronology ties."
            ),

        "source_frame_key":
            "ORDERING_CONTRACT",

        "selection_rule_id":
            "PRIMARY_TEMPORAL_TIE_POSITIVE",

        "required_columns":
            [
                "patient_id",
                "encounter_id",
                "rss_primary_temporal_tie_flag",
            ],

        "target_metrics":
            [
                "CHRONOLOGY_INTEGRITY_RATE",
                "FALSE_RECURRENCE_CREATION_COUNT",
            ],

        "required_for_baseline":
            True,
    },

    {
        "scenario_id":
            "BIRE-FROZEN-CHRONOLOGY-LIMIT-001",

        "scenario_family":
            "CHRONOLOGY_LIMITATION_REFERENCE",

        "scenario_description":
            (
                "Known histories where replay chronology "
                "eligibility is withheld."
            ),

        "source_frame_key":
            "ORDERING_CONTRACT",

        "selection_rule_id":
            "REPLAY_CHRONOLOGY_INELIGIBLE",

        "required_columns":
            [
                "patient_id",
                "encounter_id",
                "rss_replay_chronology_eligible",
            ],

        "target_metrics":
            [
                "CHRONOLOGY_INTEGRITY_RATE",
                "FALSE_RECURRENCE_CREATION_COUNT",
                "REPLAY_AUTHORITY_LEAK_COUNT",
            ],

        "required_for_baseline":
            True,
    },

    {
        "scenario_id":
            "BIRE-FROZEN-CENSORING-001",

        "scenario_family":
            "CENSORED_EPISODE_REFERENCE",

        "scenario_description":
            (
                "Known provisional RSS episodes containing "
                "left, internal, or right censoring."
            ),

        "source_frame_key":
            "RSS_EPISODES",

        "selection_rule_id":
            "ANY_EPISODE_CENSORING",

        "required_columns":
            [
                "rss_episode_id",
                "rss_episode_left_censored_flag",
                "rss_episode_internal_censoring_flag",
                "rss_episode_right_censored_flag",
            ],

        "target_metrics":
            [
                "CENSORING_PRESERVATION_RATE",
                "UNCERTAINTY_PRESERVATION_RATE",
            ],

        "required_for_baseline":
            True,
    },

    {
        "scenario_id":
            "BIRE-FROZEN-RECURRENCE-001",

        "scenario_family":
            "RECURRENT_RELATIONSHIP_REFERENCE",

        "scenario_description":
            (
                "Known governed cross-family relationships "
                "where recurrence was structurally observed."
            ),

        "source_frame_key":
            "RSS_RECURRENCE",

        "selection_rule_id":
            "RELATIONSHIP_RECURRENCE_OBSERVED",

        "required_columns":
            [
                "patient_id",
                "rss_family_pair",
                "rss_governed_alignment_class",
                "rss_relationship_recurrence_observed",
            ],

        "target_metrics":
            [
                "FALSE_RECURRENCE_CREATION_COUNT",
                "REPLAY_AUTHORITY_LEAK_COUNT",
            ],

        "required_for_baseline":
            True,
    },
]


def _canonical_json(
    value: object,
) -> str:
    return json.dumps(
        value,
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


def _normalize_string_series(
    series: pd.Series,
) -> pd.Series:
    return (
        series
        .astype("string")
        .str.strip()
    )


def _coerce_binary_like(
    series: pd.Series,
) -> pd.Series:
    """
    Convert common binary representations into nullable boolean.
    """

    if pd.api.types.is_bool_dtype(
        series.dtype
    ):
        return series.astype("boolean")

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    numeric_values = set(
        numeric.dropna().unique().tolist()
    )

    if numeric_values and numeric_values.issubset(
        {0, 1, 0.0, 1.0}
    ):
        result = pd.Series(
            pd.NA,
            index=series.index,
            dtype="boolean",
        )

        result.loc[
            numeric.eq(1)
        ] = True

        result.loc[
            numeric.eq(0)
        ] = False

        return result

    text = (
        series
        .astype("string")
        .str.strip()
        .str.lower()
    )

    true_values = {
        "true",
        "t",
        "yes",
        "y",
        "1",
    }

    false_values = {
        "false",
        "f",
        "no",
        "n",
        "0",
    }

    result = pd.Series(
        pd.NA,
        index=series.index,
        dtype="boolean",
    )

    result.loc[
        text.isin(true_values)
    ] = True

    result.loc[
        text.isin(false_values)
    ] = False

    return result


def _hash_identity_frame(
    frame: pd.DataFrame,
    identity_columns: list[str],
) -> tuple[int, str]:
    """
    Produce a deterministic hash of unique normalized identities.
    """

    if frame.empty:
        return (
            0,
            _sha256_text(
                "EMPTY_IDENTITY_POPULATION"
            ),
        )

    identity = frame[
        identity_columns
    ].copy()

    for column in identity_columns:
        identity[column] = (
            _normalize_string_series(
                identity[column]
            )
        )

    identity = (
        identity
        .dropna()
        .drop_duplicates()
        .sort_values(
            by=identity_columns,
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    digest = hashlib.sha256()

    for row in identity.itertuples(
        index=False,
        name=None,
    ):
        digest.update(
            (
                "\x1f".join(
                    str(value)
                    for value in row
                )
                + "\n"
            )
            .encode("utf-8")
        )

    return (
        int(len(identity)),
        digest.hexdigest(),
    )


def _scenario_selection_mask(
    frame: pd.DataFrame,
    selection_rule_id: str,
) -> pd.Series:
    """
    Apply one explicit governed baseline selector.
    """

    if selection_rule_id == "CARE_ESCALATION_POSITIVE":
        return (
            _coerce_binary_like(
                frame[
                    "care_escalation_occurred"
                ]
            )
            .eq(True)
            .fillna(False)
        )

    if selection_rule_id == "CARE_ESCALATION_NEGATIVE":
        return (
            _coerce_binary_like(
                frame[
                    "care_escalation_occurred"
                ]
            )
            .eq(False)
            .fillna(False)
        )

    if selection_rule_id == "MASKED_INSTABILITY_POSITIVE":
        return (
            _coerce_binary_like(
                frame[
                    "masked_instability_signal"
                ]
            )
            .eq(True)
            .fillna(False)
        )

    if selection_rule_id == "STEPDOWN_REBOUND_POSITIVE":
        return (
            _coerce_binary_like(
                frame[
                    "stepdown_rebound_flag"
                ]
            )
            .eq(True)
            .fillna(False)
        )

    if selection_rule_id == "FALSE_VITAL_RECOVERY_POSITIVE":
        return (
            _coerce_binary_like(
                frame[
                    "false_vital_recovery_signal"
                ]
            )
            .eq(True)
            .fillna(False)
        )

    if (
        selection_rule_id
        ==
        "RECOVERY_AUTHENTICITY_WARNING_POSITIVE"
    ):
        return (
            _coerce_binary_like(
                frame[
                    "vital_recovery_authenticity_warning"
                ]
            )
            .eq(True)
            .fillna(False)
        )

    if (
        selection_rule_id
        ==
        "PRIMARY_TEMPORAL_TIE_POSITIVE"
    ):
        return (
            _coerce_binary_like(
                frame[
                    "rss_primary_temporal_tie_flag"
                ]
            )
            .eq(True)
            .fillna(False)
        )

    if (
        selection_rule_id
        ==
        "REPLAY_CHRONOLOGY_INELIGIBLE"
    ):
        return (
            _coerce_binary_like(
                frame[
                    "rss_replay_chronology_eligible"
                ]
            )
            .eq(False)
            .fillna(False)
        )

    if selection_rule_id == "ANY_EPISODE_CENSORING":
        return (
            _coerce_binary_like(
                frame[
                    "rss_episode_left_censored_flag"
                ]
            )
            .eq(True)
            .fillna(False)
            |
            _coerce_binary_like(
                frame[
                    "rss_episode_internal_censoring_flag"
                ]
            )
            .eq(True)
            .fillna(False)
            |
            _coerce_binary_like(
                frame[
                    "rss_episode_right_censored_flag"
                ]
            )
            .eq(True)
            .fillna(False)
        )

    if (
        selection_rule_id
        ==
        "RELATIONSHIP_RECURRENCE_OBSERVED"
    ):
        return (
            _coerce_binary_like(
                frame[
                    "rss_relationship_recurrence_observed"
                ]
            )
            .eq(True)
            .fillna(False)
        )

    raise KeyError(
        "Unknown BIRE baseline selection rule: "
        f"{selection_rule_id}"
    )


def _source_identity_columns(
    source_frame_key: str,
) -> list[str]:
    if source_frame_key in {
        "CANONICAL_LPMR",
        "ORDERING_CONTRACT",
    }:
        return [
            "patient_id",
            "encounter_id",
        ]

    if source_frame_key == "RSS_EPISODES":
        return [
            "rss_episode_id",
        ]

    if source_frame_key == "RSS_RECURRENCE":
        return [
            "patient_id",
            "rss_family_pair",
            "rss_governed_alignment_class",
        ]

    raise KeyError(
        f"Unknown source frame key: {source_frame_key}"
    )


def build_bire_frozen_baseline_scenario_catalog(
    metric_registry_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return the frozen baseline scenario definitions before
    materialization.

    This does not expose sealed, rotating, or adversarial suites.
    """

    if not isinstance(
        metric_registry_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "metric_registry_df must be a DataFrame."
        )

    if "metric_id" not in metric_registry_df.columns:
        raise KeyError(
            "metric_registry_df must contain metric_id."
        )

    registered_metrics = set(
        metric_registry_df[
            "metric_id"
        ].astype(str)
    )

    rows: list[dict[str, object]] = []

    for scenario in (
        _BIRE_FROZEN_BASELINE_SCENARIOS
    ):
        undefined_metrics = sorted(
            set(
                scenario[
                    "target_metrics"
                ]
            )
            - registered_metrics
        )

        if undefined_metrics:
            raise ValueError(
                f"{scenario['scenario_id']} references "
                "undefined BIRE metrics: "
                f"{undefined_metrics}"
            )

        rows.append(
            {
                **scenario,

                "benchmark_suite":
                    "FROZEN_REGRESSION_SUITE",

                "scenario_version":
                    "1.0.0",

                "source_origin":
                    "KNOWN_EXISTING_BIRE_EVIDENCE",

                "difficulty_profile_version":
                    _BIRE_BASELINE_DIFFICULTY_PROFILE_VERSION,

                "difficulty_level":
                    "REFERENCE_BASELINE",

                "novelty_profile_version":
                    _BIRE_BASELINE_NOVELTY_PROFILE_VERSION,

                "novelty_level":
                    "KNOWN_REFERENCE",

                "initial_visibility_state":
                    "KNOWN_FROZEN_REFERENCE",

                "fresh_generalization_evidence_eligible":
                    False,

                "fresh_novelty_evidence_eligible":
                    False,

                "fresh_adversarial_evidence_eligible":
                    False,

                "regression_reuse_eligible":
                    True,

                "scenario_materialization_state":
                    "NOT_YET_MATERIALIZED",
            }
        )

    result = pd.DataFrame(rows)

    result[
        "required_columns_json"
    ] = result[
        "required_columns"
    ].apply(
        _canonical_json
    )

    result[
        "target_metrics_json"
    ] = result[
        "target_metrics"
    ].apply(
        _canonical_json
    )

    return (
        result
        .drop(
            columns=[
                "required_columns",
                "target_metrics",
            ]
        )
        .sort_values(
            [
                "required_for_baseline",
                "scenario_id",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def materialize_bire_frozen_baseline_scenario_manifest(
    scenario_catalog_df: pd.DataFrame,
    canonical_df: pd.DataFrame,
    ordering_contract_df: pd.DataFrame,
    episode_df: pd.DataFrame,
    recurrence_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Materialize the known frozen BIRE OS baseline manifest.

    Each scenario records deterministic source and eligible
    population fingerprints.

    No benchmark execution occurs.
    """

    frames = {
        "CANONICAL_LPMR":
            canonical_df,

        "ORDERING_CONTRACT":
            ordering_contract_df,

        "RSS_EPISODES":
            episode_df,

        "RSS_RECURRENCE":
            recurrence_df,
    }

    for name, frame in frames.items():
        if not isinstance(
            frame,
            pd.DataFrame,
        ):
            raise TypeError(
                f"{name} must be a pandas DataFrame."
            )

    if not isinstance(
        scenario_catalog_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "scenario_catalog_df must be a DataFrame."
        )

    required_catalog_columns = {
        "scenario_id",
        "scenario_version",
        "benchmark_suite",
        "scenario_family",
        "scenario_description",
        "source_frame_key",
        "selection_rule_id",
        "required_columns_json",
        "target_metrics_json",
        "required_for_baseline",
        "source_origin",
        "difficulty_profile_version",
        "difficulty_level",
        "novelty_profile_version",
        "novelty_level",
        "initial_visibility_state",
        "regression_reuse_eligible",
    }

    missing_catalog_columns = sorted(
        required_catalog_columns
        - set(
            scenario_catalog_df.columns
        )
    )

    if missing_catalog_columns:
        raise KeyError(
            "Frozen baseline materialization cannot proceed. "
            f"Missing catalog columns: "
            f"{missing_catalog_columns}"
        )

    if scenario_catalog_df[
        "scenario_id"
    ].duplicated().any():
        raise ValueError(
            "Frozen baseline scenario IDs must be unique."
        )

    rows: list[dict[str, object]] = []

    source_hash_cache: dict[
        tuple[str, tuple[str, ...]],
        tuple[int, str],
    ] = {}

    for scenario in (
        scenario_catalog_df.itertuples(
            index=False
        )
    ):
        source_key = (
            scenario.source_frame_key
        )

        if source_key not in frames:
            raise KeyError(
                f"Unknown source frame: {source_key}"
            )

        source = frames[source_key]

        required_columns = json.loads(
            scenario.required_columns_json
        )

        missing_columns = sorted(
            set(required_columns)
            - set(source.columns)
        )

        identity_columns = (
            _source_identity_columns(
                source_key
            )
        )

        missing_identity_columns = sorted(
            set(identity_columns)
            - set(source.columns)
        )

        all_missing = sorted(
            set(
                missing_columns
                + missing_identity_columns
            )
        )

        if all_missing:
            materialization_state = (
                "BLOCKED_REQUIRED_COLUMNS_MISSING"
            )

            eligible_count = 0
            eligible_hash = pd.NA
            source_identity_count = 0
            source_identity_hash = pd.NA

        else:
            cache_key = (
                source_key,
                tuple(identity_columns),
            )

            if cache_key not in source_hash_cache:
                source_hash_cache[
                    cache_key
                ] = _hash_identity_frame(
                    source,
                    identity_columns,
                )

            (
                source_identity_count,
                source_identity_hash,
            ) = source_hash_cache[
                cache_key
            ]

            selection_mask = (
                _scenario_selection_mask(
                    source,
                    scenario.selection_rule_id,
                )
            )

            selected = source.loc[
                selection_mask
            ]

            (
                eligible_count,
                eligible_hash,
            ) = _hash_identity_frame(
                selected,
                identity_columns,
            )

            materialization_state = (
                "MATERIALIZED"
                if eligible_count > 0
                else
                "MATERIALIZED_ZERO_ELIGIBLE_INSTANCES"
            )

        definition_payload = {
            "scenario_id":
                scenario.scenario_id,

            "scenario_version":
                scenario.scenario_version,

            "benchmark_suite":
                scenario.benchmark_suite,

            "scenario_family":
                scenario.scenario_family,

            "source_frame_key":
                source_key,

            "selection_rule_id":
                scenario.selection_rule_id,

            "required_columns_json":
                scenario.required_columns_json,

            "target_metrics_json":
                scenario.target_metrics_json,

            "difficulty_profile_version":
                scenario.difficulty_profile_version,

            "difficulty_level":
                scenario.difficulty_level,

            "novelty_profile_version":
                scenario.novelty_profile_version,

            "novelty_level":
                scenario.novelty_level,

            "eligible_instance_count":
                eligible_count,

            "eligible_instance_identity_sha256":
                (
                    None
                    if pd.isna(
                        eligible_hash
                    )
                    else eligible_hash
                ),
        }

        rows.append(
            {
                "scenario_id":
                    scenario.scenario_id,

                "scenario_version":
                    scenario.scenario_version,

                "benchmark_suite":
                    scenario.benchmark_suite,

                "scenario_family":
                    scenario.scenario_family,

                "scenario_description":
                    scenario.scenario_description,

                "parent_scenario_id":
                    pd.NA,

                "source_origin":
                    scenario.source_origin,

                "source_frame_key":
                    source_key,

                "selection_rule_id":
                    scenario.selection_rule_id,

                "required_columns_json":
                    scenario.required_columns_json,

                "target_metrics_json":
                    scenario.target_metrics_json,

                "required_for_baseline":
                    bool(
                        scenario.required_for_baseline
                    ),

                "missing_required_columns":
                    (
                        "NONE"
                        if not all_missing
                        else " | ".join(
                            all_missing
                        )
                    ),

                "source_identity_population_count":
                    source_identity_count,

                "source_identity_population_sha256":
                    source_identity_hash,

                "eligible_instance_count":
                    eligible_count,

                "eligible_instance_identity_sha256":
                    eligible_hash,

                "scenario_content_sha256":
                    _sha256_text(
                        _canonical_json(
                            definition_payload
                        )
                    ),

                "generation_seed":
                    pd.NA,

                "difficulty_profile_version":
                    scenario.difficulty_profile_version,

                "difficulty_level":
                    scenario.difficulty_level,

                "novelty_profile_version":
                    scenario.novelty_profile_version,

                "novelty_level":
                    scenario.novelty_level,

                "initial_visibility_state":
                    "KNOWN_FROZEN_REFERENCE",

                "current_visibility_state":
                    "KNOWN_FROZEN_REFERENCE",

                "development_access_count":
                    1,

                "development_influence_flag":
                    True,

                "contamination_state":
                    "KNOWN_REFERENCE_NOT_FRESH_EVIDENCE",

                "fresh_generalization_evidence_eligible":
                    False,

                "fresh_novelty_evidence_eligible":
                    False,

                "fresh_adversarial_evidence_eligible":
                    False,

                "regression_reuse_eligible":
                    True,

                "first_governed_evaluation_run_id":
                    pd.NA,

                "last_governed_evaluation_run_id":
                    pd.NA,

                "retirement_state":
                    "ACTIVE_FROZEN_REFERENCE",

                "retirement_reason":
                    pd.NA,

                "scenario_materialization_state":
                    materialization_state,

                "metric_execution_authorized":
                    False,

                "benchmark_execution_authorized":
                    False,

                "playground_execution_authorized":
                    False,

                "operational_deployment_authorized":
                    False,

                "nid_authorization_state":
                    (
                        "NID_BIRE_FROZEN_BASELINE_"
                        "SCENARIO_MATERIALIZED"
                    ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            by=[
                "required_for_baseline",
                "scenario_family",
                "scenario_id",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def build_bire_frozen_baseline_manifest_summary(
    scenario_manifest_df: pd.DataFrame,
    metric_registry_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize frozen baseline scenario materialization and metric
    coverage.

    The summary does not imply metric testability until actual
    benchmark execution occurs.
    """

    if not isinstance(
        scenario_manifest_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "scenario_manifest_df must be a DataFrame."
        )

    if not isinstance(
        metric_registry_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "metric_registry_df must be a DataFrame."
        )

    registered_metrics = set(
        metric_registry_df[
            "metric_id"
        ].astype(str)
    )

    coverage: dict[str, int] = {
        metric_id: 0
        for metric_id in registered_metrics
    }

    for row in (
        scenario_manifest_df.itertuples(
            index=False
        )
    ):
        if (
            row.scenario_materialization_state
            != "MATERIALIZED"
        ):
            continue

        if int(
            row.eligible_instance_count
        ) <= 0:
            continue

        for metric_id in json.loads(
            row.target_metrics_json
        ):
            if metric_id in coverage:
                coverage[metric_id] += 1

    metric_coverage = pd.DataFrame(
        [
            {
                "metric_id":
                    metric_id,

                "baseline_supporting_scenario_count":
                    scenario_count,

                "baseline_scenario_support_state":
                    (
                        "BASELINE_SCENARIO_AVAILABLE"
                        if scenario_count > 0
                        else
                        "NO_FROZEN_BASELINE_SCENARIO"
                    ),
            }
            for metric_id, scenario_count
            in coverage.items()
        ]
    )

    return (
        metric_registry_df[
            [
                "metric_id",
                "metric_family",
                "metric_direction",
                "acceptance_policy_type",
            ]
        ]
        .merge(
            metric_coverage,
            on="metric_id",
            how="left",
            validate="one_to_one",
        )
        .sort_values(
            by=[
                "baseline_scenario_support_state",
                "metric_family",
                "metric_id",
            ]
        )
        .reset_index(drop=True)
    )


def build_bire_frozen_baseline_materialization_contract(
    scenario_manifest_df: pd.DataFrame,
    metric_acceptance_contract_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Verify that required frozen baseline scenarios materialized
    successfully.

    Successful materialization authorizes construction of the
    first pre-execution baseline run manifest.

    Benchmark execution remains withheld.
    """

    if not isinstance(
        scenario_manifest_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "scenario_manifest_df must be a DataFrame."
        )

    if not isinstance(
        metric_acceptance_contract_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "metric_acceptance_contract_df must be a DataFrame."
        )

    if len(
        metric_acceptance_contract_df
    ) != 1:
        raise ValueError(
            "metric_acceptance_contract_df must contain one row."
        )

    metric_contract = (
        metric_acceptance_contract_df.iloc[0]
    )

    required_scenarios = (
        scenario_manifest_df.loc[
            scenario_manifest_df[
                "required_for_baseline"
            ].eq(True)
        ]
    )

    failed_required_scenarios = (
        required_scenarios.loc[
            ~required_scenarios[
                "scenario_materialization_state"
            ].eq("MATERIALIZED")
        ]
    )

    empty_required_scenarios = (
        required_scenarios.loc[
            required_scenarios[
                "eligible_instance_count"
            ]
            .fillna(0)
            .le(0)
        ]
    )

    checks = {
        "metric_acceptance_contract_installed":
            (
                int(
                    metric_contract[
                        "failed_contract_check_count"
                    ]
                )
                == 0
            ),

        "scenario_manifest_not_empty":
            len(
                scenario_manifest_df
            ) > 0,

        "all_required_scenarios_materialized":
            failed_required_scenarios.empty,

        "all_required_scenarios_have_evidence":
            empty_required_scenarios.empty,

        "all_scenario_ids_unique":
            not scenario_manifest_df[
                "scenario_id"
            ].duplicated().any(),

        "all_content_hashes_present":
            scenario_manifest_df[
                "scenario_content_sha256"
            ]
            .notna()
            .all(),

        "fresh_generalization_claims_withheld":
            bool(
                ~scenario_manifest_df[
                    "fresh_generalization_evidence_eligible"
                ]
                .fillna(False)
                .any()
            ),

        "fresh_novelty_claims_withheld":
            bool(
                ~scenario_manifest_df[
                    "fresh_novelty_evidence_eligible"
                ]
                .fillna(False)
                .any()
            ),

        "fresh_adversarial_claims_withheld":
            bool(
                ~scenario_manifest_df[
                    "fresh_adversarial_evidence_eligible"
                ]
                .fillna(False)
                .any()
            ),

        "benchmark_execution_still_withheld":
            bool(
                ~scenario_manifest_df[
                    "benchmark_execution_authorized"
                ]
                .fillna(False)
                .any()
            ),
    }

    failed_checks = [
        name
        for name, passed
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
                        "BIRE_FROZEN_BASELINE_SCENARIO_"
                        "MATERIALIZATION_CONTRACT"
                    ),

                "baseline_manifest_version":
                    (
                        _BIRE_FROZEN_BASELINE_MANIFEST_VERSION
                    ),

                **checks,

                "registered_scenario_count":
                    int(
                        len(
                            scenario_manifest_df
                        )
                    ),

                "required_baseline_scenario_count":
                    int(
                        len(
                            required_scenarios
                        )
                    ),

                "materialized_scenario_count":
                    int(
                        scenario_manifest_df[
                            "scenario_materialization_state"
                        ]
                        .eq("MATERIALIZED")
                        .sum()
                    ),

                "failed_required_scenario_count":
                    int(
                        len(
                            failed_required_scenarios
                        )
                    ),

                "failed_required_scenarios":
                    (
                        "NONE"
                        if failed_required_scenarios.empty
                        else " | ".join(
                            failed_required_scenarios[
                                "scenario_id"
                            ]
                            .astype(str)
                            .tolist()
                        )
                    ),

                "pre_execution_baseline_manifest_"
                "construction_authorized":
                    contract_installed,

                "baseline_run_manifest_writing_authorized":
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

                "sealed_generalization_exposure_authorized":
                    False,

                "rotating_novelty_exposure_authorized":
                    False,

                "adversarial_playground_exposure_authorized":
                    False,

                "nid_authorization_state":
                    (
                        "NID_BIRE_FROZEN_BASELINE_"
                        "MATERIALIZATION_INSTALLED"
                        if contract_installed
                        else
                        "NID_BIRE_FROZEN_BASELINE_"
                        "MATERIALIZATION_REVIEW_REQUIRED"
                    ),

                "next_required_stage":
                    (
                        "CHAPTER_69_8_PRE_PLAYGROUND_"
                        "BASELINE_RUN_MANIFEST_AND_EXECUTION_GATE"
                    ),
            }
        ]
    )


def write_bire_frozen_baseline_scenario_manifest(
    scenario_manifest_df: pd.DataFrame,
    output_path: str | Path,
) -> pd.DataFrame:
    """
    Write the frozen baseline scenario manifest as immutable JSON.

    Existing files are never overwritten.
    """

    if not isinstance(
        scenario_manifest_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "scenario_manifest_df must be a DataFrame."
        )

    destination = Path(
        output_path
    ).expanduser().resolve()

    if destination.exists():
        raise FileExistsError(
            "Frozen baseline scenario manifests are immutable "
            f"and cannot be overwritten: {destination}"
        )

    records = (
        scenario_manifest_df
        .where(
            pd.notna(
                scenario_manifest_df
            ),
            None,
        )
        .to_dict(
            orient="records"
        )
    )

    payload = {
        "manifest_version":
            _BIRE_FROZEN_BASELINE_MANIFEST_VERSION,

        "benchmark_suite":
            "FROZEN_REGRESSION_SUITE",

        "scenario_count":
            len(records),

        "scenarios":
            records,
    }

    canonical_payload = (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n"
    )

    manifest_sha256 = (
        _sha256_text(
            canonical_payload
        )
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
                "baseline_manifest_version":
                    _BIRE_FROZEN_BASELINE_MANIFEST_VERSION,

                "scenario_manifest_path":
                    str(destination),

                "scenario_manifest_sha256":
                    manifest_sha256,

                "scenario_count":
                    len(records),

                "manifest_write_state":
                    (
                        "IMMUTABLE_FROZEN_BASELINE_"
                        "SCENARIO_MANIFEST_WRITTEN"
                    ),

                "manifest_overwrite_authorized":
                    False,

                "benchmark_execution_authorized":
                    False,

                "nid_authorization_state":
                    (
                        "NID_BIRE_FROZEN_BASELINE_"
                        "MANIFEST_WRITTEN"
                    ),
            }
        ]
    )


__all__ = [
    "build_bire_frozen_baseline_scenario_catalog",
    "materialize_bire_frozen_baseline_scenario_manifest",
    "build_bire_frozen_baseline_manifest_summary",
    "build_bire_frozen_baseline_materialization_contract",
    "write_bire_frozen_baseline_scenario_manifest",
]