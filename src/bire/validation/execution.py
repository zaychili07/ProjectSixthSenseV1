#============================================================
# Project Sixth Sense — BIRE OS
# Chapter 69.9 — Pre-Playground Baseline
#                  Execution & Result Record
#============================================================

from __future__ import annotations

import pandas as pd


_BIRE_BASELINE_EXECUTION_VERSION = (
    "BIRE_PREPLAYGROUND_BASELINE_EXECUTION_V1"
)


def build_bire_frozen_baseline_measurement_plan(
    execution_gate_df: pd.DataFrame,
    metric_registry_df: pd.DataFrame,
    metric_coverage_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the metric-level measurement plan for the authorized
    frozen baseline.

    This does not calculate metric values.
    """

    for name, frame in {
        "execution_gate_df":
            execution_gate_df,
        "metric_registry_df":
            metric_registry_df,
        "metric_coverage_df":
            metric_coverage_df,
    }.items():
        if not isinstance(
            frame,
            pd.DataFrame,
        ):
            raise TypeError(
                f"{name} must be a pandas DataFrame."
            )

    if len(
        execution_gate_df
    ) != 1:
        raise ValueError(
            "execution_gate_df must contain exactly one row."
        )

    gate = (
        execution_gate_df.iloc[0]
    )

    if not bool(
        gate[
            "frozen_baseline_execution_authorized"
        ]
    ):
        raise ValueError(
            "Frozen baseline execution is not authorized."
        )

    if not bool(
        gate[
            "metric_execution_within_frozen_"
            "baseline_authorized"
        ]
    ):
        raise ValueError(
            "Frozen baseline metric execution is not authorized."
        )

    required_registry_columns = {
        "metric_id",
        "metric_family",
        "metric_direction",
        "calculation_type",
        "measurement_unit",
        "aggregation_rule",
        "acceptance_policy_type",
        "acceptance_threshold_state",
        "hard_gate_value",
        "reference_baseline_required",
        "non_compensatory_gate",
    }

    missing_registry_columns = sorted(
        required_registry_columns
        - set(metric_registry_df.columns)
    )

    if missing_registry_columns:
        raise KeyError(
            "Measurement planning cannot proceed. "
            f"Missing metric-registry columns: "
            f"{missing_registry_columns}"
        )

    required_coverage_columns = {
        "metric_id",
        "baseline_supporting_scenario_count",
        "baseline_scenario_support_state",
    }

    missing_coverage_columns = sorted(
        required_coverage_columns
        - set(metric_coverage_df.columns)
    )

    if missing_coverage_columns:
        raise KeyError(
            "Measurement planning cannot proceed. "
            f"Missing coverage columns: "
            f"{missing_coverage_columns}"
        )

    if metric_registry_df[
        "metric_id"
    ].duplicated().any():
        raise ValueError(
            "Metric registry contains duplicate metric IDs."
        )

    if metric_coverage_df[
        "metric_id"
    ].duplicated().any():
        raise ValueError(
            "Metric coverage contains duplicate metric IDs."
        )

    plan = (
        metric_registry_df[
            [
                "metric_id",
                "metric_family",
                "metric_direction",
                "calculation_type",
                "measurement_unit",
                "aggregation_rule",
                "acceptance_policy_type",
                "acceptance_threshold_state",
                "hard_gate_value",
                "reference_baseline_required",
                "non_compensatory_gate",
            ]
        ]
        .merge(
            metric_coverage_df[
                [
                    "metric_id",
                    "baseline_supporting_scenario_count",
                    "baseline_scenario_support_state",
                ]
            ],
            on="metric_id",
            how="left",
            validate="one_to_one",
        )
    )

    plan[
        "baseline_supporting_scenario_count"
    ] = (
        plan[
            "baseline_supporting_scenario_count"
        ]
        .fillna(0)
        .astype(int)
    )

    plan[
        "measurement_evidence_available"
    ] = (
        plan[
            "baseline_supporting_scenario_count"
        ]
        .gt(0)
    )

    plan[
        "baseline_measurement_state"
    ] = (
        plan[
            "measurement_evidence_available"
        ]
        .map(
            {
                True:
                    "BASELINE_MEASUREMENT_EVIDENCE_AVAILABLE",

                False:
                    "NOT_TESTABLE_IN_FROZEN_BASELINE",
            }
        )
    )

    # --------------------------------------------------------
    # A scenario population alone does not tell us which
    # runtime output constitutes the numerator / observation.
    # That binding is deliberately explicit.
    # --------------------------------------------------------

    plan[
        "runtime_measurement_binding_state"
    ] = (
        plan[
            "measurement_evidence_available"
        ]
        .map(
            {
                True:
                    "MEASUREMENT_BINDING_REQUIRED",

                False:
                    "NOT_APPLICABLE_NO_BASELINE_EVIDENCE",
            }
        )
    )

    plan[
        "metric_value"
    ] = pd.NA

    plan[
        "metric_testability_state"
    ] = "NOT_YET_EXECUTED"

    plan[
        "metric_result_state"
    ] = "NOT_YET_EXECUTED"

    plan[
        "baseline_execution_version"
    ] = (
        _BIRE_BASELINE_EXECUTION_VERSION
    )

    plan[
        "run_id"
    ] = gate[
        "run_id"
    ]

    plan[
        "frozen_baseline_execution_authorized"
    ] = True

    plan[
        "playground_execution_authorized"
    ] = False

    plan[
        "operational_deployment_authorized"
    ] = False

    plan[
        "nid_authorization_state"
    ] = (
        "NID_BIRE_FROZEN_BASELINE_"
        "MEASUREMENT_PLAN_INSTALLED"
    )

    return (
        plan
        .sort_values(
            by=[
                "measurement_evidence_available",
                "metric_family",
                "metric_id",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def build_bire_frozen_baseline_measurement_plan_summary(
    measurement_plan_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize metric support before execution.
    """

    if not isinstance(
        measurement_plan_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "measurement_plan_df must be a DataFrame."
        )

    required_columns = {
        "metric_id",
        "metric_family",
        "measurement_evidence_available",
        "baseline_measurement_state",
        "runtime_measurement_binding_state",
    }

    missing_columns = sorted(
        required_columns
        - set(measurement_plan_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "Measurement-plan summary cannot proceed. "
            f"Missing columns: {missing_columns}"
        )

    return (
        measurement_plan_df
        .groupby(
            [
                "metric_family",
                "measurement_evidence_available",
                "baseline_measurement_state",
                "runtime_measurement_binding_state",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="metric_count"
        )
        .sort_values(
            by=[
                "measurement_evidence_available",
                "metric_family",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


#============================================================
# Chapter 69.9B — Runtime Measurement Binding Discovery
#============================================================

import re


_BIRE_RUNTIME_BINDING_DISCOVERY_VERSION = (
    "BIRE_RUNTIME_BINDING_DISCOVERY_V1"
)


_BIRE_METRIC_BINDING_SEARCH_TERMS = {
    "DETERIORATION_DETECTION_RATE": [
        "deterioration",
        "care_escalation",
        "escalation",
        "risk",
        "alert",
        "recognition",
        "detected",
    ],

    "MISSED_DETERIORATION_RATE": [
        "deterioration",
        "care_escalation",
        "escalation",
        "missed",
        "risk",
        "alert",
        "recognition",
    ],

    "FALSE_ACTIVATION_BURDEN": [
        "false_activation",
        "activation",
        "false_positive",
        "alert",
        "unsupported",
    ],

    "GOVERNED_EVENT_LEAD_TIME": [
        "lead_time",
        "event_time",
        "alert_time",
        "recognition_time",
        "years_since_sim_start",
        "temporal",
        "time",
    ],

    "HIDDEN_INSTABILITY_RECOGNITION_RATE": [
        "hidden_instability",
        "masked_instability",
        "hidden",
        "deception",
        "instability",
    ],

    "RECOVERY_CONTRADICTION_RECOGNITION_RATE": [
        "false_recovery",
        "recovery_contradiction",
        "recovery_authenticity",
        "recovery_warning",
        "re_escalation",
        "recovery",
    ],

    "PREMATURE_RECOVERY_CLAIM_COUNT": [
        "recovery_claim",
        "recovery_authenticity",
        "recovery_trust",
        "recovery_state",
        "recovery",
    ],

    "UNCERTAINTY_PRESERVATION_RATE": [
        "uncertainty",
        "confidence",
        "evidence_completeness",
        "trust",
        "limitation",
    ],

    "UNSUPPORTED_CERTAINTY_COUNT": [
        "unsupported_certainty",
        "certainty",
        "confidence",
        "uncertainty",
        "trust",
    ],

    "CENSORING_PRESERVATION_RATE": [
        "censor",
        "censored",
        "left_censored",
        "right_censored",
        "internal_censoring",
    ],

    "CHRONOLOGY_INTEGRITY_RATE": [
        "chronology",
        "temporal_order",
        "sequence_order",
        "replay_chronology",
        "order_position",
        "ordering",
    ],

    "FALSE_RECURRENCE_CREATION_COUNT": [
        "recurrence",
        "same_position",
        "distinct_position",
        "duplicate",
        "relationship_recurrence",
    ],

    "REPLAY_AUTHORITY_LEAK_COUNT": [
        "replay_cycle",
        "pattern_maturity",
        "replay_lesson",
        "lesson",
        "authority",
    ],
}


_VALIDATION_FRAME_NAME_TERMS = {
    "validation",
    "benchmark",
    "manifest",
    "contract",
    "profile",
    "coverage",
    "measurement_plan",
    "scenario_catalog",
    "scenario_manifest",
    "readiness",
}


def _normalize_binding_name(
    value: object,
) -> str:
    text = str(
        value
    ).strip().lower()

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    )

    return text.strip("_")


def _binding_match_score(
    column_name: str,
    search_terms: list[str],
) -> tuple[int, list[str]]:
    """
    Score a runtime column against metric-specific search terms.

    Exact or phrase matches receive stronger weight than
    individual token overlap.
    """

    normalized_column = (
        _normalize_binding_name(
            column_name
        )
    )

    column_tokens = set(
        normalized_column.split("_")
    )

    matched_terms: list[str] = []

    score = 0

    for raw_term in search_terms:
        term = _normalize_binding_name(
            raw_term
        )

        if not term:
            continue

        if term == normalized_column:
            score += 10
            matched_terms.append(
                raw_term
            )
            continue

        if term in normalized_column:
            score += 5
            matched_terms.append(
                raw_term
            )
            continue

        term_tokens = set(
            term.split("_")
        )

        overlap = (
            term_tokens
            & column_tokens
        )

        if overlap:
            score += len(
                overlap
            )

            matched_terms.append(
                raw_term
            )

    return (
        score,
        sorted(
            set(
                matched_terms
            )
        ),
    )


def _is_validation_derived_frame(
    frame_name: str,
) -> bool:
    normalized = (
        _normalize_binding_name(
            frame_name
        )
    )

    return any(
        term in normalized
        for term in _VALIDATION_FRAME_NAME_TERMS
    )


def build_bire_runtime_measurement_binding_discovery(
    measurement_plan_df: pd.DataFrame,
    runtime_frames: dict[str, pd.DataFrame],
    top_candidates_per_metric: int = 12,
) -> pd.DataFrame:
    """
    Discover candidate runtime columns for the 13 metrics
    supported by the frozen baseline.

    This is column-level discovery only.

    No candidate receives measurement authority here.
    """

    if not isinstance(
        measurement_plan_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "measurement_plan_df must be a DataFrame."
        )

    if not isinstance(
        runtime_frames,
        dict,
    ):
        raise TypeError(
            "runtime_frames must be a dictionary."
        )

    if top_candidates_per_metric <= 0:
        raise ValueError(
            "top_candidates_per_metric must be positive."
        )

    required_plan_columns = {
        "metric_id",
        "metric_family",
        "measurement_evidence_available",
        "baseline_measurement_state",
        "runtime_measurement_binding_state",
    }

    missing_columns = sorted(
        required_plan_columns
        - set(
            measurement_plan_df.columns
        )
    )

    if missing_columns:
        raise KeyError(
            "Runtime binding discovery cannot proceed. "
            f"Missing measurement-plan columns: "
            f"{missing_columns}"
        )

    supported = (
        measurement_plan_df.loc[
            measurement_plan_df[
                "measurement_evidence_available"
            ].eq(True)
        ]
        .copy()
    )

    expected_supported_metrics = set(
        supported[
            "metric_id"
        ].astype(str)
    )

    configured_metrics = set(
        _BIRE_METRIC_BINDING_SEARCH_TERMS
    )

    missing_search_terms = sorted(
        expected_supported_metrics
        - configured_metrics
    )

    if missing_search_terms:
        raise ValueError(
            "Supported metrics lack binding-search terms: "
            f"{missing_search_terms}"
        )

    rows: list[
        dict[str, object]
    ] = []

    for metric in supported.itertuples(
        index=False
    ):
        search_terms = (
            _BIRE_METRIC_BINDING_SEARCH_TERMS[
                metric.metric_id
            ]
        )

        candidates: list[
            dict[str, object]
        ] = []

        for (
            frame_name,
            frame,
        ) in runtime_frames.items():

            if not isinstance(
                frame,
                pd.DataFrame,
            ):
                continue

            validation_derived = (
                _is_validation_derived_frame(
                    frame_name
                )
            )

            for column in frame.columns:
                (
                    score,
                    matched_terms,
                ) = _binding_match_score(
                    column_name=str(
                        column
                    ),
                    search_terms=search_terms,
                )

                if score <= 0:
                    continue

                candidates.append(
                    {
                        "metric_id":
                            metric.metric_id,

                        "metric_family":
                            metric.metric_family,

                        "runtime_frame_name":
                            str(
                                frame_name
                            ),

                        "runtime_column_name":
                            str(
                                column
                            ),

                        "runtime_column_dtype":
                            str(
                                frame[
                                    column
                                ].dtype
                            ),

                        "runtime_frame_row_count":
                            int(
                                len(
                                    frame
                                )
                            ),

                        "binding_match_score":
                            int(
                                score
                            ),

                        "matched_search_terms":
                            " | ".join(
                                matched_terms
                            ),

                        "validation_derived_frame":
                            bool(
                                validation_derived
                            ),

                        "eligible_for_binding_review":
                            not bool(
                                validation_derived
                            ),

                        "binding_authority_state":
                            "CANDIDATE_REVIEW_REQUIRED",

                        "metric_execution_authorized":
                            False,

                        "benchmark_execution_authorized":
                            False,

                        "playground_execution_authorized":
                            False,

                        "nid_authorization_state":
                            (
                                "NID_BIRE_RUNTIME_BINDING_"
                                "CANDIDATE_DISCOVERED"
                            ),
                    }
                )

        candidate_df = pd.DataFrame(
            candidates
        )

        if candidate_df.empty:
            rows.append(
                {
                    "metric_id":
                        metric.metric_id,

                    "metric_family":
                        metric.metric_family,

                    "runtime_frame_name":
                        pd.NA,

                    "runtime_column_name":
                        pd.NA,

                    "runtime_column_dtype":
                        pd.NA,

                    "runtime_frame_row_count":
                        pd.NA,

                    "binding_match_score":
                        0,

                    "matched_search_terms":
                        "NONE",

                    "validation_derived_frame":
                        False,

                    "eligible_for_binding_review":
                        False,

                    "binding_authority_state":
                        "NO_RUNTIME_CANDIDATE_DISCOVERED",

                    "metric_execution_authorized":
                        False,

                    "benchmark_execution_authorized":
                        False,

                    "playground_execution_authorized":
                        False,

                    "nid_authorization_state":
                        (
                            "NID_BIRE_RUNTIME_BINDING_"
                            "CANDIDATE_REVIEW_REQUIRED"
                        ),
                }
            )

            continue

        candidate_df = (
            candidate_df
            .sort_values(
                by=[
                    "eligible_for_binding_review",
                    "binding_match_score",
                    "runtime_frame_name",
                    "runtime_column_name",
                ],
                ascending=[
                    False,
                    False,
                    True,
                    True,
                ],
            )
            .head(
                top_candidates_per_metric
            )
        )

        rows.extend(
            candidate_df.to_dict(
                orient="records"
            )
        )

    result = pd.DataFrame(
        rows
    )

    result[
        "binding_discovery_version"
    ] = (
        _BIRE_RUNTIME_BINDING_DISCOVERY_VERSION
    )

    return (
        result
        .sort_values(
            by=[
                "metric_id",
                "eligible_for_binding_review",
                "binding_match_score",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )


def build_bire_runtime_measurement_binding_discovery_summary(
    discovery_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize candidate runtime evidence discovered for each
    frozen-baseline-supported metric.
    """

    if not isinstance(
        discovery_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "discovery_df must be a DataFrame."
        )

    required_columns = {
        "metric_id",
        "runtime_frame_name",
        "runtime_column_name",
        "eligible_for_binding_review",
        "binding_match_score",
    }

    missing_columns = sorted(
        required_columns
        - set(
            discovery_df.columns
        )
    )

    if missing_columns:
        raise KeyError(
            "Binding discovery summary cannot proceed. "
            f"Missing columns: {missing_columns}"
        )

    rows: list[
        dict[str, object]
    ] = []

    for (
        metric_id,
        group,
    ) in discovery_df.groupby(
        "metric_id",
        sort=True,
    ):
        eligible = group.loc[
            group[
                "eligible_for_binding_review"
            ].eq(True)
        ]

        rows.append(
            {
                "metric_id":
                    metric_id,

                "candidate_count":
                    int(
                        len(
                            group.loc[
                                group[
                                    "runtime_column_name"
                                ].notna()
                            ]
                        )
                    ),

                "eligible_candidate_count":
                    int(
                        len(
                            eligible
                        )
                    ),

                "highest_binding_match_score":
                    (
                        int(
                            eligible[
                                "binding_match_score"
                            ].max()
                        )
                        if not eligible.empty
                        else 0
                    ),

                "runtime_binding_discovery_state":
                    (
                        "RUNTIME_CANDIDATES_DISCOVERED"
                        if not eligible.empty
                        else
                        "NO_ELIGIBLE_RUNTIME_CANDIDATE"
                    ),

                "binding_authority_granted":
                    False,

                "metric_execution_authorized":
                    False,
            }
        )

    return pd.DataFrame(
        rows
    )

#============================================================
# Chapter 69.9C — Runtime Measurement Binding Review
#============================================================

import json


_BIRE_RUNTIME_BINDING_REVIEW_VERSION = (
    "BIRE_RUNTIME_BINDING_REVIEW_V1"
)


def _is_temporary_runtime_frame_name(
    frame_name: str,
) -> bool:
    """
    Detect notebook-generated anonymous names such as
    _217, _222, etc.
    """

    normalized = str(
        frame_name
    ).strip()

    if re.fullmatch(
        r"_\d+",
        normalized,
    ):
        return True

    return normalized.startswith(
        "_"
    )


def _candidate_series_profile(
    series: pd.Series,
) -> dict[str, object]:
    """
    Return compact diagnostics for semantic binding review.
    """

    non_null = (
        series.dropna()
    )

    non_null_count = int(
        len(non_null)
    )

    non_null_percent = (
        float(
            non_null_count
            / len(series)
            * 100.0
        )
        if len(series) > 0
        else 0.0
    )

    unique_count = int(
        non_null.nunique(
            dropna=True
        )
    )

    sample_values = (
        non_null
        .astype(str)
        .drop_duplicates()
        .head(8)
        .tolist()
    )

    numeric = pd.to_numeric(
        non_null,
        errors="coerce",
    )

    numeric_non_null = (
        numeric.dropna()
    )

    return {
        "non_null_count":
            non_null_count,

        "non_null_percent":
            round(
                non_null_percent,
                6,
            ),

        "unique_value_count":
            unique_count,

        "sample_values_json":
            json.dumps(
                sample_values,
                ensure_ascii=False,
            ),

        "numeric_min":
            (
                float(
                    numeric_non_null.min()
                )
                if not numeric_non_null.empty
                else pd.NA
            ),

        "numeric_max":
            (
                float(
                    numeric_non_null.max()
                )
                if not numeric_non_null.empty
                else pd.NA
            ),
    }


def build_bire_runtime_measurement_binding_review(
    discovery_df: pd.DataFrame,
    runtime_frames: dict[str, pd.DataFrame],
    candidates_per_metric: int = 5,
) -> pd.DataFrame:
    """
    Narrow discovered runtime candidates for semantic review.

    Temporary notebook objects and validation-derived sources
    cannot receive measurement authority.

    This function does not authorize bindings.
    """

    if not isinstance(
        discovery_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "discovery_df must be a DataFrame."
        )

    if not isinstance(
        runtime_frames,
        dict,
    ):
        raise TypeError(
            "runtime_frames must be a dictionary."
        )

    required_columns = {
        "metric_id",
        "metric_family",
        "runtime_frame_name",
        "runtime_column_name",
        "runtime_column_dtype",
        "binding_match_score",
        "matched_search_terms",
        "validation_derived_frame",
        "eligible_for_binding_review",
    }

    missing_columns = sorted(
        required_columns
        - set(discovery_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "Runtime binding review cannot proceed. "
            f"Missing discovery columns: "
            f"{missing_columns}"
        )

    rows: list[
        dict[str, object]
    ] = []

    for candidate in (
        discovery_df.itertuples(
            index=False
        )
    ):
        frame_name = str(
            candidate.runtime_frame_name
        )

        column_name = str(
            candidate.runtime_column_name
        )

        temporary_frame = (
            _is_temporary_runtime_frame_name(
                frame_name
            )
        )

        source_exists = (
            frame_name
            in runtime_frames
            and isinstance(
                runtime_frames[
                    frame_name
                ],
                pd.DataFrame,
            )
        )

        column_exists = (
            source_exists
            and column_name
            in runtime_frames[
                frame_name
            ].columns
        )

        review_eligible = (
            bool(
                candidate.eligible_for_binding_review
            )
            and not bool(
                candidate.validation_derived_frame
            )
            and not temporary_frame
            and column_exists
        )

        profile = {
            "non_null_count":
                pd.NA,

            "non_null_percent":
                pd.NA,

            "unique_value_count":
                pd.NA,

            "sample_values_json":
                "[]",

            "numeric_min":
                pd.NA,

            "numeric_max":
                pd.NA,
        }

        if column_exists:
            profile = (
                _candidate_series_profile(
                    runtime_frames[
                        frame_name
                    ][
                        column_name
                    ]
                )
            )

        rows.append(
            {
                "metric_id":
                    candidate.metric_id,

                "metric_family":
                    candidate.metric_family,

                "runtime_frame_name":
                    frame_name,

                "runtime_column_name":
                    column_name,

                "runtime_column_dtype":
                    candidate.runtime_column_dtype,

                "binding_match_score":
                    int(
                        candidate.binding_match_score
                    ),

                "matched_search_terms":
                    candidate.matched_search_terms,

                "temporary_runtime_frame":
                    temporary_frame,

                "validation_derived_frame":
                    bool(
                        candidate.validation_derived_frame
                    ),

                "runtime_source_exists":
                    source_exists,

                "runtime_column_exists":
                    column_exists,

                "eligible_for_semantic_review":
                    review_eligible,

                **profile,

                "semantic_role":
                    "NOT_YET_ASSIGNED",

                "semantic_equivalence_state":
                    "REVIEW_REQUIRED",

                "binding_authority_granted":
                    False,

                "metric_execution_authorized":
                    False,

                "nid_authorization_state":
                    (
                        "NID_BIRE_RUNTIME_BINDING_"
                        "SEMANTIC_REVIEW_REQUIRED"
                    ),
            }
        )

    review = pd.DataFrame(
        rows
    )

    review[
        "binding_review_version"
    ] = (
        _BIRE_RUNTIME_BINDING_REVIEW_VERSION
    )

    eligible = (
        review.loc[
            review[
                "eligible_for_semantic_review"
            ].eq(True)
        ]
        .sort_values(
            by=[
                "metric_id",
                "binding_match_score",
                "non_null_percent",
                "runtime_frame_name",
                "runtime_column_name",
            ],
            ascending=[
                True,
                False,
                False,
                True,
                True,
            ],
        )
        .groupby(
            "metric_id",
            sort=False,
            group_keys=False,
        )
        .head(
            candidates_per_metric
        )
    )

    return (
        eligible
        .reset_index(drop=True)
    )

#============================================================
# Chapter 69.9D — Runtime Measurement
#                   Semantic Binding Disposition
#============================================================

_BIRE_SEMANTIC_BINDING_DISPOSITION_VERSION = (
    "BIRE_SEMANTIC_BINDING_DISPOSITION_V1"
)


_BIRE_BASELINE_BINDING_DISPOSITIONS = [
    {
        "metric_id":
            "CENSORING_PRESERVATION_RATE",

        "binding_disposition":
            "DERIVED_BINDING_REQUIRED",

        "required_evidence_roles":
            (
                "SOURCE_CENSORING_STATE | "
                "DOWNSTREAM_CENSORING_PRESERVATION"
            ),

        "review_finding":
            (
                "Direct episode censoring flags exist, but "
                "preservation requires comparison with "
                "downstream historical evidence."
            ),
    },
    {
        "metric_id":
            "CHRONOLOGY_INTEGRITY_RATE",

        "binding_disposition":
            "DERIVED_BINDING_REQUIRED",

        "required_evidence_roles":
            (
                "TEMPORAL_ORDER | SEQUENCE_ORDER | "
                "REPLAY_CHRONOLOGY_ELIGIBILITY"
            ),

        "review_finding":
            (
                "Governed chronology positions and replay "
                "eligibility exist and require combined evaluation."
            ),
    },
    {
        "metric_id":
            "DETERIORATION_DETECTION_RATE",

        "binding_disposition":
            "DERIVED_BINDING_REQUIRED",

        "required_evidence_roles":
            "DETERIORATION_TARGET | BIRE_RECOGNITION",

        "review_finding":
            (
                "Risk fields were discovered but do not independently "
                "prove governed deterioration recognition."
            ),
    },
    {
        "metric_id":
            "FALSE_ACTIVATION_BURDEN",

        "binding_disposition":
            "NO_VALID_RUNTIME_BINDING_YET",

        "required_evidence_roles":
            (
                "BIRE_ACTIVATION_EVENT | "
                "ACTIVATION_SUPPORT_VALIDITY"
            ),

        "review_finding":
            (
                "Discovered activation_dependency fields do not "
                "represent false activation events."
            ),
    },
    {
        "metric_id":
            "FALSE_RECURRENCE_CREATION_COUNT",

        "binding_disposition":
            "DERIVED_BINDING_REQUIRED",

        "required_evidence_roles":
            (
                "RECURRENCE_OUTPUT | DISTINCT_POSITION_PROVENANCE | "
                "DUPLICATE_OR_SAME_POSITION_PROTECTION"
            ),

        "review_finding":
            (
                "Recurrence evidence exists, but false recurrence "
                "requires provenance comparison."
            ),
    },
    {
        "metric_id":
            "GOVERNED_EVENT_LEAD_TIME",

        "binding_disposition":
            "DIRECTION_CONFLICT_REVIEW_REQUIRED",

        "required_evidence_roles":
            (
                "FIRST_GOVERNED_RECOGNITION_TIME | "
                "TARGET_EVENT_TIME"
            ),

        "review_finding":
            (
                "time_to_recognition_delay_minutes is a delay "
                "measure and cannot substitute for a higher-is-better "
                "lead-time metric."
            ),
    },
    {
        "metric_id":
            "HIDDEN_INSTABILITY_RECOGNITION_RATE",

        "binding_disposition":
            "DERIVED_BINDING_REQUIRED",

        "required_evidence_roles":
            (
                "HIDDEN_INSTABILITY_TARGET | "
                "BIRE_HIDDEN_INSTABILITY_RECOGNITION"
            ),

        "review_finding":
            (
                "Target and recognition evidence must remain "
                "separate before rate calculation."
            ),
    },
    {
        "metric_id":
            "MISSED_DETERIORATION_RATE",

        "binding_disposition":
            "DERIVED_BINDING_REQUIRED",

        "required_evidence_roles":
            "DETERIORATION_TARGET | BIRE_RECOGNITION",

        "review_finding":
            (
                "The missed rate requires the same governed target "
                "and recognition pairing as detection rate."
            ),
    },
    {
        "metric_id":
            "PREMATURE_RECOVERY_CLAIM_COUNT",

        "binding_disposition":
            "DERIVED_BINDING_REQUIRED",

        "required_evidence_roles":
            (
                "RECOVERY_CLAIM | "
                "RECOVERY_EVIDENCE_SUFFICIENCY"
            ),

        "review_finding":
            (
                "Recovery authenticity and trust states provide "
                "context but do not alone prove a premature claim."
            ),
    },
    {
        "metric_id":
            "RECOVERY_CONTRADICTION_RECOGNITION_RATE",

        "binding_disposition":
            "DERIVED_BINDING_REQUIRED",

        "required_evidence_roles":
            (
                "RECOVERY_CONTRADICTION_TARGET | "
                "BIRE_CONTRADICTION_RECOGNITION"
            ),

        "review_finding":
            (
                "vital_recovery_authenticity_warning is a strong "
                "recognition candidate but must be evaluated against "
                "the governed contradiction target population."
            ),
    },
    {
        "metric_id":
            "REPLAY_AUTHORITY_LEAK_COUNT",

        "binding_disposition":
            "MULTI_FIELD_HARD_GATE_REQUIRED",

        "required_evidence_roles":
            (
                "REPLAY_CYCLE_AUTHORITY | PATTERN_MATURITY_AUTHORITY | "
                "REPLAY_LESSON_AUTHORITY"
            ),

        "review_finding":
            (
                "Replay lesson authority is explicitly withheld, "
                "but every prohibited replay-authority family must "
                "be checked before the hard gate can pass."
            ),
    },
    {
        "metric_id":
            "UNCERTAINTY_PRESERVATION_RATE",

        "binding_disposition":
            "DERIVED_BINDING_REQUIRED",

        "required_evidence_roles":
            (
                "SOURCE_UNCERTAINTY_STATE | "
                "DOWNSTREAM_UNCERTAINTY_STATE"
            ),

        "review_finding":
            (
                "Evidence trust states exist, but preservation "
                "requires source-to-downstream comparison."
            ),
    },
    {
        "metric_id":
            "UNSUPPORTED_CERTAINTY_COUNT",

        "binding_disposition":
            "NO_VALID_RUNTIME_BINDING_YET",

        "required_evidence_roles":
            (
                "MATERIAL_UNCERTAINTY_STATE | "
                "DOWNSTREAM_CERTAINTY_ASSERTION"
            ),

        "review_finding":
            (
                "Uncertainty scores measure uncertainty; they do "
                "not independently identify unsupported certainty."
            ),
    },
]


def build_bire_runtime_semantic_binding_disposition(
    measurement_plan_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Install semantic binding dispositions for all frozen-baseline
    supported metrics.

    No metric execution authority is granted here.
    """

    if not isinstance(
        measurement_plan_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "measurement_plan_df must be a DataFrame."
        )

    supported = (
        measurement_plan_df.loc[
            measurement_plan_df[
                "measurement_evidence_available"
            ].eq(True)
        ]
        .copy()
    )

    dispositions = pd.DataFrame(
        _BIRE_BASELINE_BINDING_DISPOSITIONS
    )

    supported_ids = set(
        supported["metric_id"].astype(str)
    )

    disposition_ids = set(
        dispositions["metric_id"].astype(str)
    )

    if supported_ids != disposition_ids:
        raise ValueError(
            "Semantic binding dispositions do not exactly match "
            "the frozen-baseline-supported metric set. "
            f"Missing: {sorted(supported_ids - disposition_ids)}; "
            f"extra: {sorted(disposition_ids - supported_ids)}"
        )

    result = (
        supported[
            [
                "metric_id",
                "metric_family",
                "acceptance_policy_type",
                "baseline_supporting_scenario_count",
            ]
        ]
        .merge(
            dispositions,
            on="metric_id",
            how="one_to_one"
            if False
            else "inner",
            validate="one_to_one",
        )
    )

    result[
        "binding_disposition_version"
    ] = (
        _BIRE_SEMANTIC_BINDING_DISPOSITION_VERSION
    )

    result[
        "semantic_disposition_installed"
    ] = True

    result[
        "binding_authority_granted"
    ] = False

    result[
        "metric_execution_authorized"
    ] = False

    result[
        "playground_execution_authorized"
    ] = False

    result[
        "nid_authorization_state"
    ] = (
        "NID_BIRE_RUNTIME_SEMANTIC_BINDING_"
        "DISPOSITION_INSTALLED"
    )

    return (
        result
        .sort_values(
            by=[
                "binding_disposition",
                "metric_family",
                "metric_id",
            ]
        )
        .reset_index(drop=True)
    )

#============================================================
# Chapter 69.9E — Evidence-Role Binding Resolution
#============================================================

_BIRE_EVIDENCE_ROLE_RESOLUTION_VERSION = (
    "BIRE_EVIDENCE_ROLE_RESOLUTION_V1"
)


_BIRE_EVIDENCE_ROLE_SEARCH_TERMS = {
    "DETERIORATION_TARGET": [
        "care_escalation_occurred",
        "deterioration_event",
        "escalation_occurred",
        "deterioration",
    ],

    "BIRE_RECOGNITION": [
        "deterioration_recognized",
        "escalation_recognized",
        "alert_status",
        "escalation_flag",
        "risk_flag",
    ],

    "HIDDEN_INSTABILITY_TARGET": [
        "masked_instability_signal",
        "hidden_instability",
    ],

    "BIRE_HIDDEN_INSTABILITY_RECOGNITION": [
        "hidden_instability_recognition",
        "masked_instability",
        "instability_warning",
        "instability_flag",
    ],

    "RECOVERY_CONTRADICTION_TARGET": [
        "false_vital_recovery_signal",
        "false_recovery",
        "recovery_contradiction",
    ],

    "BIRE_CONTRADICTION_RECOGNITION": [
        "vital_recovery_authenticity_warning",
        "recovery_authenticity_warning",
        "false_recovery_warning",
    ],

    "RECOVERY_CLAIM": [
        "recovery_claim",
        "recovery_state",
        "recovery_status",
        "recovery_candidate",
    ],

    "RECOVERY_EVIDENCE_SUFFICIENCY": [
        "recovery_authenticity_state",
        "recovery_trust_state",
        "recovery_evidence",
        "recovery_authenticity_warning",
    ],

    "SOURCE_UNCERTAINTY_STATE": [
        "evidence_trust_state",
        "uncertainty_score",
        "confidence_state",
    ],

    "DOWNSTREAM_UNCERTAINTY_STATE": [
        "evidence_trust_state",
        "uncertainty_score",
        "confidence_state",
    ],

    "SOURCE_CENSORING_STATE": [
        "rss_episode_left_censored_flag",
        "rss_episode_internal_censoring_flag",
        "rss_episode_right_censored_flag",
        "censored",
    ],

    "DOWNSTREAM_CENSORING_PRESERVATION": [
        "censoring_context",
        "left_censored_alignment",
        "right_censored_alignment",
        "recurrence_evidence_limit_state",
    ],

    "TEMPORAL_ORDER": [
        "rss_temporal_order_position",
        "temporal_order_position",
        "years_since_sim_start",
    ],

    "SEQUENCE_ORDER": [
        "rss_sequence_order_position",
        "sequence_order_position",
        "encounter_sequence",
    ],

    "REPLAY_CHRONOLOGY_ELIGIBILITY": [
        "rss_replay_chronology_eligible",
        "rss_patient_replay_chronology_eligible",
        "replay_chronology_eligible",
    ],

    "RECURRENCE_OUTPUT": [
        "rss_relationship_recurrence_observed",
        "relationship_recurrence_observed",
        "recurrence_observed",
    ],

    "DISTINCT_POSITION_PROVENANCE": [
        "distinct_position",
        "historical_position",
        "recurrence_position",
    ],

    "DUPLICATE_OR_SAME_POSITION_PROTECTION": [
        "same_position",
        "same_position_compression",
        "duplicate",
        "distinct_position",
    ],

    "FIRST_GOVERNED_RECOGNITION_TIME": [
        "first_governed_recognition_time",
        "recognition_time",
        "time_to_recognition",
    ],

    "TARGET_EVENT_TIME": [
        "target_event_time",
        "event_time",
        "years_since_sim_start",
    ],

    "REPLAY_CYCLE_AUTHORITY": [
        "rss_replay_cycle_generation_authorized",
        "replay_cycle_authorized",
        "replay_cycle",
    ],

    "PATTERN_MATURITY_AUTHORITY": [
        "rss_pattern_maturity_authorized",
        "pattern_maturity_authorized",
        "pattern_maturity",
    ],

    "REPLAY_LESSON_AUTHORITY": [
        "rss_replay_lesson_generation_authorized",
        "replay_lesson_generation_authorized",
        "replay_lesson",
    ],
}


_BIRE_EVIDENCE_ROLE_FRAME_HINTS = {
    "DETERIORATION_TARGET": [
        "lpmr",
        "canonical",
    ],

    "BIRE_RECOGNITION": [
        "ati",
        "hvi",
        "intelligence",
    ],

    "HIDDEN_INSTABILITY_TARGET": [
        "lpmr",
        "canonical",
    ],

    "BIRE_HIDDEN_INSTABILITY_RECOGNITION": [
        "hvi",
        "ati",
    ],

    "RECOVERY_CONTRADICTION_TARGET": [
        "lpmr",
        "canonical",
    ],

    "BIRE_CONTRADICTION_RECOGNITION": [
        "ati",
        "hvi",
    ],

    "RECOVERY_CLAIM": [
        "ati",
        "hvi",
    ],

    "RECOVERY_EVIDENCE_SUFFICIENCY": [
        "ati",
        "hvi",
    ],

    "SOURCE_UNCERTAINTY_STATE": [
        "context",
        "lpmr",
    ],

    "DOWNSTREAM_UNCERTAINTY_STATE": [
        "ati",
        "hvi",
    ],

    "SOURCE_CENSORING_STATE": [
        "controlled_binary_episode_construction",
    ],

    "DOWNSTREAM_CENSORING_PRESERVATION": [
        "cross_family",
        "recurrence",
    ],

    "TEMPORAL_ORDER": [
        "chronology_candidate_agreement",
    ],

    "SEQUENCE_ORDER": [
        "chronology_candidate_agreement",
    ],

    "REPLAY_CHRONOLOGY_ELIGIBILITY": [
        "chronology",
        "controlled_binary_episode_construction",
    ],

    "RECURRENCE_OUTPUT": [
        "relationship_recurrence",
    ],

    "DISTINCT_POSITION_PROVENANCE": [
        "relationship_recurrence",
    ],

    "DUPLICATE_OR_SAME_POSITION_PROTECTION": [
        "relationship_recurrence",
    ],

    "FIRST_GOVERNED_RECOGNITION_TIME": [
        "ati",
    ],

    "TARGET_EVENT_TIME": [
        "lpmr",
        "canonical",
    ],

    "REPLAY_CYCLE_AUTHORITY": [
        "rss",
        "governance",
    ],

    "PATTERN_MATURITY_AUTHORITY": [
        "rss",
        "governance",
    ],

    "REPLAY_LESSON_AUTHORITY": [
        "rss",
        "governance",
    ],
}


def _evidence_role_frame_hint_score(
    evidence_role: str,
    frame_name: str,
) -> int:
    hints = (
        _BIRE_EVIDENCE_ROLE_FRAME_HINTS.get(
            evidence_role,
            [],
        )
    )

    normalized_frame = (
        _normalize_binding_name(
            frame_name
        )
    )

    return int(
        sum(
            3
            for hint in hints
            if _normalize_binding_name(
                hint
            )
            in normalized_frame
        )
    )


def build_bire_runtime_evidence_role_resolution(
    semantic_disposition_df: pd.DataFrame,
    runtime_frames: dict[str, pd.DataFrame],
    candidates_per_role: int = 3,
) -> pd.DataFrame:
    """
    Search runtime evidence by explicitly required semantic role.

    No binding authority is granted here.
    """

    if not isinstance(
        semantic_disposition_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "semantic_disposition_df must be a DataFrame."
        )

    if not isinstance(
        runtime_frames,
        dict,
    ):
        raise TypeError(
            "runtime_frames must be a dictionary."
        )

    if candidates_per_role <= 0:
        raise ValueError(
            "candidates_per_role must be positive."
        )

    required_columns = {
        "metric_id",
        "metric_family",
        "binding_disposition",
        "required_evidence_roles",
    }

    missing_columns = sorted(
        required_columns
        - set(
            semantic_disposition_df.columns
        )
    )

    if missing_columns:
        raise KeyError(
            "Evidence-role resolution cannot proceed. "
            f"Missing columns: {missing_columns}"
        )

    rows: list[
        dict[str, object]
    ] = []

    for metric in (
        semantic_disposition_df.itertuples(
            index=False
        )
    ):
        if (
            metric.binding_disposition
            ==
            "NO_VALID_RUNTIME_BINDING_YET"
        ):
            rows.append(
                {
                    "metric_id":
                        metric.metric_id,

                    "metric_family":
                        metric.metric_family,

                    "binding_disposition":
                        metric.binding_disposition,

                    "evidence_role":
                        "UNRESOLVED_METRIC_BINDING",

                    "runtime_frame_name":
                        pd.NA,

                    "runtime_column_name":
                        pd.NA,

                    "runtime_column_dtype":
                        pd.NA,

                    "semantic_match_score":
                        0,

                    "frame_hint_score":
                        0,

                    "combined_review_score":
                        0,

                    "non_null_percent":
                        pd.NA,

                    "unique_value_count":
                        pd.NA,

                    "sample_values_json":
                        "[]",

                    "role_candidate_state":
                        "BINDING_WITHHELD_NO_VALID_RUNTIME_SOURCE",

                    "binding_authority_granted":
                        False,

                    "metric_execution_authorized":
                        False,
                }
            )

            continue

        evidence_roles = [
            role.strip()
            for role in str(
                metric.required_evidence_roles
            ).split("|")
            if role.strip()
        ]

        for evidence_role in evidence_roles:

            if (
                evidence_role
                not in
                _BIRE_EVIDENCE_ROLE_SEARCH_TERMS
            ):
                raise KeyError(
                    "No evidence-role search policy exists for "
                    f"{evidence_role}"
                )

            search_terms = (
                _BIRE_EVIDENCE_ROLE_SEARCH_TERMS[
                    evidence_role
                ]
            )

            lightweight_candidates: list[
                dict[str, object]
            ] = []

            for (
                frame_name,
                frame,
            ) in runtime_frames.items():

                if not isinstance(
                    frame,
                    pd.DataFrame,
                ):
                    continue

                if _is_validation_derived_frame(
                    frame_name
                ):
                    continue

                if _is_temporary_runtime_frame_name(
                    frame_name
                ):
                    continue

                frame_hint_score = (
                    _evidence_role_frame_hint_score(
                        evidence_role,
                        frame_name,
                    )
                )

                for column in frame.columns:
                    (
                        semantic_score,
                        matched_terms,
                    ) = _binding_match_score(
                        column_name=str(
                            column
                        ),
                        search_terms=search_terms,
                    )

                    if semantic_score <= 0:
                        continue

                    lightweight_candidates.append(
                        {
                            "runtime_frame_name":
                                str(
                                    frame_name
                                ),

                            "runtime_column_name":
                                str(
                                    column
                                ),

                            "runtime_column_dtype":
                                str(
                                    frame[
                                        column
                                    ].dtype
                                ),

                            "semantic_match_score":
                                int(
                                    semantic_score
                                ),

                            "frame_hint_score":
                                int(
                                    frame_hint_score
                                ),

                            "combined_review_score":
                                int(
                                    semantic_score
                                    + frame_hint_score
                                ),

                            "matched_search_terms":
                                " | ".join(
                                    matched_terms
                                ),
                        }
                    )

            if not lightweight_candidates:
                rows.append(
                    {
                        "metric_id":
                            metric.metric_id,

                        "metric_family":
                            metric.metric_family,

                        "binding_disposition":
                            metric.binding_disposition,

                        "evidence_role":
                            evidence_role,

                        "runtime_frame_name":
                            pd.NA,

                        "runtime_column_name":
                            pd.NA,

                        "runtime_column_dtype":
                            pd.NA,

                        "semantic_match_score":
                            0,

                        "frame_hint_score":
                            0,

                        "combined_review_score":
                            0,

                        "matched_search_terms":
                            "NONE",

                        "non_null_percent":
                            pd.NA,

                        "unique_value_count":
                            pd.NA,

                        "sample_values_json":
                            "[]",

                        "role_candidate_state":
                            "NO_RUNTIME_ROLE_CANDIDATE",

                        "binding_authority_granted":
                            False,

                        "metric_execution_authorized":
                            False,
                    }
                )

                continue

            candidate_df = (
                pd.DataFrame(
                    lightweight_candidates
                )
                .sort_values(
                    by=[
                        "combined_review_score",
                        "semantic_match_score",
                        "runtime_frame_name",
                        "runtime_column_name",
                    ],
                    ascending=[
                        False,
                        False,
                        True,
                        True,
                    ],
                )
                .head(
                    candidates_per_role
                )
            )

            for candidate in (
                candidate_df.itertuples(
                    index=False
                )
            ):
                series = runtime_frames[
                    candidate.runtime_frame_name
                ][
                    candidate.runtime_column_name
                ]

                profile = (
                    _candidate_series_profile(
                        series
                    )
                )

                rows.append(
                    {
                        "metric_id":
                            metric.metric_id,

                        "metric_family":
                            metric.metric_family,

                        "binding_disposition":
                            metric.binding_disposition,

                        "evidence_role":
                            evidence_role,

                        "runtime_frame_name":
                            candidate.runtime_frame_name,

                        "runtime_column_name":
                            candidate.runtime_column_name,

                        "runtime_column_dtype":
                            candidate.runtime_column_dtype,

                        "semantic_match_score":
                            candidate.semantic_match_score,

                        "frame_hint_score":
                            candidate.frame_hint_score,

                        "combined_review_score":
                            candidate.combined_review_score,

                        "matched_search_terms":
                            candidate.matched_search_terms,

                        **profile,

                        "role_candidate_state":
                            "ROLE_CANDIDATE_REVIEW_REQUIRED",

                        "binding_authority_granted":
                            False,

                        "metric_execution_authorized":
                            False,
                    }
                )

    result = pd.DataFrame(
        rows
    )

    result[
        "evidence_role_resolution_version"
    ] = (
        _BIRE_EVIDENCE_ROLE_RESOLUTION_VERSION
    )

    result[
        "playground_execution_authorized"
    ] = False

    result[
        "operational_deployment_authorized"
    ] = False

    result[
        "nid_authorization_state"
    ] = (
        "NID_BIRE_EVIDENCE_ROLE_RESOLUTION_REVIEW"
    )

    return (
        result
        .sort_values(
            by=[
                "metric_id",
                "evidence_role",
                "combined_review_score",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )


def build_bire_runtime_evidence_role_resolution_summary(
    resolution_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize evidence-role candidate availability by metric.
    """

    if not isinstance(
        resolution_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "resolution_df must be a DataFrame."
        )

    rows = []

    for (
        metric_id,
        group,
    ) in resolution_df.groupby(
        "metric_id",
        sort=True,
    ):

        unresolved_metric = bool(
            group[
                "evidence_role"
            ]
            .eq(
                "UNRESOLVED_METRIC_BINDING"
            )
            .any()
        )

        role_rows = group.loc[
            ~group[
                "evidence_role"
            ].eq(
                "UNRESOLVED_METRIC_BINDING"
            )
        ]

        required_role_count = int(
            role_rows[
                "evidence_role"
            ].nunique()
        )

        roles_with_candidates = int(
            role_rows.loc[
                role_rows[
                    "runtime_column_name"
                ].notna(),
                "evidence_role",
            ]
            .nunique()
        )

        rows.append(
            {
                "metric_id":
                    metric_id,

                "required_evidence_role_count":
                    required_role_count,

                "roles_with_runtime_candidates":
                    roles_with_candidates,

                "all_required_roles_have_candidates":
                    (
                        not unresolved_metric
                        and required_role_count > 0
                        and required_role_count
                        == roles_with_candidates
                    ),

                "runtime_role_resolution_state":
                    (
                        "ALL_REQUIRED_ROLES_HAVE_CANDIDATES"
                        if (
                            not unresolved_metric
                            and required_role_count > 0
                            and required_role_count
                            == roles_with_candidates
                        )
                        else
                        "ROLE_RESOLUTION_INCOMPLETE"
                    ),

                "binding_authority_granted":
                    False,

                "metric_execution_authorized":
                    False,
            }
        )

    return pd.DataFrame(
        rows
    )
#============================================================
# Chapter 69.9G — Runtime Binding Authority
#                   & Compatibility Audit
#============================================================

_BIRE_BINDING_AUTHORITY_AUDIT_VERSION = (
    "BIRE_BINDING_AUTHORITY_AUDIT_V1"
)


_BIRE_BINDING_AUTHORITY_POLICY = {
    "FALSE_RECURRENCE_CREATION_COUNT":
        {
            "binding_authority_state":
                "BINDING_AUTHORIZED",

            "authority_reason":
                (
                    "Recurrence output, distinct-position provenance, "
                    "and same-position protection are present within "
                    "the same governed RSS recurrence frame."
                ),
        },

    "REPLAY_AUTHORITY_LEAK_COUNT":
        {
            "binding_authority_state":
                "BINDING_AUTHORIZED",

            "authority_reason":
                (
                    "Replay-cycle, pattern-maturity, and replay-lesson "
                    "authority fields directly represent prohibited "
                    "RSS authority states."
                ),
        },

    "RECOVERY_CONTRADICTION_RECOGNITION_RATE":
        {
            "binding_authority_state":
                "JOIN_COMPATIBILITY_REQUIRED",

            "authority_reason":
                (
                    "The target and recognition fields are semantically "
                    "appropriate but require identity-aligned comparison "
                    "between LPMR and ATI."
                ),
        },

    "UNCERTAINTY_PRESERVATION_RATE":
        {
            "binding_authority_state":
                "JOIN_COMPATIBILITY_REQUIRED",

            "authority_reason":
                (
                    "Source and downstream trust states are semantically "
                    "appropriate but require identity-aligned comparison."
                ),
        },

    "CENSORING_PRESERVATION_RATE":
        {
            "binding_authority_state":
                "JOIN_COMPATIBILITY_REQUIRED",

            "authority_reason":
                (
                    "Source censoring is episode-grain while downstream "
                    "censoring context is relationship-grain; lineage "
                    "compatibility must be demonstrated."
                ),
        },

    "CHRONOLOGY_INTEGRITY_RATE":
        {
            "binding_authority_state":
                "JOIN_COMPATIBILITY_REQUIRED",

            "authority_reason":
                (
                    "Temporal and sequence positions are strong candidates, "
                    "but replay eligibility originates at a different "
                    "runtime grain."
                ),
        },

    "DETERIORATION_DETECTION_RATE":
        {
            "binding_authority_state":
                "SEMANTIC_BINDING_WITHHELD",

            "authority_reason":
                (
                    "re_escalation_risk_flag expresses risk context and "
                    "does not establish governed deterioration recognition."
                ),
        },

    "MISSED_DETERIORATION_RATE":
        {
            "binding_authority_state":
                "SEMANTIC_BINDING_WITHHELD",

            "authority_reason":
                (
                    "Missed deterioration cannot be calculated without a "
                    "valid governed deterioration-recognition binding."
                ),
        },

    "HIDDEN_INSTABILITY_RECOGNITION_RATE":
        {
            "binding_authority_state":
                "SEMANTIC_BINDING_WITHHELD",

            "authority_reason":
                (
                    "discharge_instability_flag does not establish semantic "
                    "equivalence to recognition of masked instability."
                ),
        },

    "PREMATURE_RECOVERY_CLAIM_COUNT":
        {
            "binding_authority_state":
                "SEMANTIC_BINDING_WITHHELD",

            "authority_reason":
                (
                    "Recovery state and authenticity context do not alone "
                    "prove that BIRE issued a premature recovery claim."
                ),
        },

    "GOVERNED_EVENT_LEAD_TIME":
        {
            "binding_authority_state":
                "DIRECTION_CONFLICT_WITHHELD",

            "authority_reason":
                (
                    "time_to_recognition_delay_minutes measures delay and "
                    "cannot substitute for the frozen higher-is-better "
                    "lead-time definition."
                ),
        },

    "FALSE_ACTIVATION_BURDEN":
        {
            "binding_authority_state":
                "NO_VALID_RUNTIME_BINDING",

            "authority_reason":
                (
                    "No governed runtime source currently identifies both "
                    "activation events and unsupported activation status."
                ),
        },

    "UNSUPPORTED_CERTAINTY_COUNT":
        {
            "binding_authority_state":
                "NO_VALID_RUNTIME_BINDING",

            "authority_reason":
                (
                    "Uncertainty scores do not identify downstream "
                    "unsupported certainty assertions."
                ),
        },
}


def build_bire_runtime_binding_authority_audit(
    semantic_disposition_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Install governed authority dispositions for all 13
    frozen-baseline-supported metrics.

    Only bindings with direct semantic sufficiency receive
    immediate execution-binding authority.
    """

    if not isinstance(
        semantic_disposition_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "semantic_disposition_df must be a DataFrame."
        )

    required_columns = {
        "metric_id",
        "metric_family",
        "binding_disposition",
    }

    missing_columns = sorted(
        required_columns
        - set(
            semantic_disposition_df.columns
        )
    )

    if missing_columns:
        raise KeyError(
            "Binding authority audit cannot proceed. "
            f"Missing columns: {missing_columns}"
        )

    observed_metrics = set(
        semantic_disposition_df[
            "metric_id"
        ].astype(str)
    )

    policy_metrics = set(
        _BIRE_BINDING_AUTHORITY_POLICY
    )

    if observed_metrics != policy_metrics:
        raise ValueError(
            "Binding authority policy does not exactly match "
            "the supported baseline metric set. "
            f"Missing: {sorted(observed_metrics - policy_metrics)}; "
            f"extra: {sorted(policy_metrics - observed_metrics)}"
        )

    policy = pd.DataFrame(
        [
            {
                "metric_id":
                    metric_id,

                **definition,
            }
            for metric_id, definition
            in _BIRE_BINDING_AUTHORITY_POLICY.items()
        ]
    )

    result = (
        semantic_disposition_df[
            [
                "metric_id",
                "metric_family",
                "binding_disposition",
                "required_evidence_roles",
            ]
        ]
        .merge(
            policy,
            on="metric_id",
            how="inner",
            validate="one_to_one",
        )
    )

    result[
        "binding_authority_granted"
    ] = result[
        "binding_authority_state"
    ].eq(
        "BINDING_AUTHORIZED"
    )

    result[
        "join_compatibility_review_required"
    ] = result[
        "binding_authority_state"
    ].eq(
        "JOIN_COMPATIBILITY_REQUIRED"
    )

    result[
        "metric_execution_authorized"
    ] = result[
        "binding_authority_granted"
    ]

    result[
        "binding_authority_audit_version"
    ] = (
        _BIRE_BINDING_AUTHORITY_AUDIT_VERSION
    )

    result[
        "playground_execution_authorized"
    ] = False

    result[
        "operational_deployment_authorized"
    ] = False

    result[
        "nid_authorization_state"
    ] = (
        result[
            "binding_authority_granted"
        ]
        .map(
            {
                True:
                    "NID_BIRE_RUNTIME_BINDING_AUTHORIZED",

                False:
                    "NID_BIRE_RUNTIME_BINDING_WITHHELD",
            }
        )
    )

    return (
        result
        .sort_values(
            by=[
                "binding_authority_state",
                "metric_family",
                "metric_id",
            ]
        )
        .reset_index(drop=True)
    )


def build_bire_runtime_binding_authority_summary(
    binding_audit_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize runtime binding authority after semantic review.
    """

    if not isinstance(
        binding_audit_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "binding_audit_df must be a DataFrame."
        )

    return (
        binding_audit_df
        .groupby(
            [
                "binding_authority_state",
                "binding_authority_granted",
                "join_compatibility_review_required",
                "metric_execution_authorized",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="metric_count"
        )
        .sort_values(
            by="binding_authority_state"
        )
        .reset_index(drop=True)
    )

#============================================================
# Chapter 69.9H — Runtime Join Compatibility Audit
#============================================================

_BIRE_JOIN_COMPATIBILITY_AUDIT_VERSION = (
    "BIRE_RUNTIME_JOIN_COMPATIBILITY_AUDIT_V1"
)


_BIRE_JOIN_COMPATIBILITY_POLICY = {
    "RECOVERY_CONTRADICTION_RECOGNITION_RATE": {
        "left_frame":
            "lpmr_df",

        "right_frame":
            "ati_df",

        "left_value_column":
            "false_vital_recovery_signal",

        "right_value_column":
            "vital_recovery_authenticity_warning",

        "allowed_join_key_sets": [
            [
                "patient_id",
                "encounter_id",
            ],
        ],

        "required_grain":
            "PATIENT_ENCOUNTER",
    },

    "UNCERTAINTY_PRESERVATION_RATE": {
        "left_frame":
            "context_df",

        "right_frame":
            "ati_df",

        "left_value_column":
            "evidence_trust_state",

        "right_value_column":
            "evidence_trust_state",

        "allowed_join_key_sets": [
            [
                "patient_id",
                "encounter_id",
            ],
        ],

        "required_grain":
            "PATIENT_ENCOUNTER",
    },

    "CENSORING_PRESERVATION_RATE": {
        "left_frame":
            "rss_controlled_binary_episode_construction",

        "right_frame":
            "rss_cross_family_relationship_recurrence",

        "left_value_column":
            "rss_episode_left_censored_flag",

        "right_value_column":
            (
                "rss_relationship_recurrence_"
                "evidence_limit_state"
            ),

        "allowed_join_key_sets": [
            [
                "rss_episode_id",
            ],
            [
                "patient_id",
                "encounter_id",
            ],
        ],

        "required_grain":
            "EPISODE_TO_RELATIONSHIP_LINEAGE",
    },

    "CHRONOLOGY_INTEGRITY_RATE": {
        "left_frame":
            "rss_chronology_candidate_agreement",

        "right_frame":
            "rss_controlled_binary_episode_construction",

        "left_value_column":
            "rss_temporal_order_position",

        "right_value_column":
            "rss_patient_replay_chronology_eligible",

        "allowed_join_key_sets": [
            [
                "patient_id",
                "encounter_id",
            ],
            [
                "patient_id",
                "encounter_sequence",
            ],
        ],

        "required_grain":
            "PATIENT_ENCOUNTER_CHRONOLOGY",
    },
}


def _normalized_identity_frame(
    frame: pd.DataFrame,
    key_columns: list[str],
) -> pd.DataFrame:
    """
    Normalize governed identity columns for deterministic
    compatibility review.
    """

    result = frame[
        key_columns
    ].copy()

    for column in key_columns:
        result[column] = (
            result[column]
            .astype("string")
            .str.strip()
        )

    return result


def _classify_join_cardinality(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keys: list[str],
) -> str:
    left_unique = (
        not left.duplicated(
            subset=keys
        ).any()
    )

    right_unique = (
        not right.duplicated(
            subset=keys
        ).any()
    )

    if left_unique and right_unique:
        return "ONE_TO_ONE"

    if left_unique and not right_unique:
        return "ONE_TO_MANY"

    if not left_unique and right_unique:
        return "MANY_TO_ONE"

    return "MANY_TO_MANY"


def build_bire_runtime_join_compatibility_audit(
    binding_authority_df: pd.DataFrame,
    runtime_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Audit identity and grain compatibility for metrics whose
    runtime bindings require joins.

    Compatibility is granted only when:

    - a governed key set exists in both sources,
    - both identity populations match completely,
    - and the join does not produce uncontrolled many-to-many
      expansion.

    Metric execution does not occur here.
    """

    if not isinstance(
        binding_authority_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "binding_authority_df must be a DataFrame."
        )

    if not isinstance(
        runtime_frames,
        dict,
    ):
        raise TypeError(
            "runtime_frames must be a dictionary."
        )

    join_metrics = (
        binding_authority_df.loc[
            binding_authority_df[
                "binding_authority_state"
            ].eq(
                "JOIN_COMPATIBILITY_REQUIRED"
            )
        ]
    )

    observed_metrics = set(
        join_metrics[
            "metric_id"
        ].astype(str)
    )

    expected_metrics = set(
        _BIRE_JOIN_COMPATIBILITY_POLICY
    )

    if observed_metrics != expected_metrics:
        raise ValueError(
            "Join compatibility policy does not match the "
            "metrics requiring join review. "
            f"Missing: {sorted(observed_metrics - expected_metrics)}; "
            f"extra: {sorted(expected_metrics - observed_metrics)}"
        )

    rows: list[
        dict[str, object]
    ] = []

    for metric_id in sorted(
        expected_metrics
    ):
        policy = (
            _BIRE_JOIN_COMPATIBILITY_POLICY[
                metric_id
            ]
        )

        left_name = policy[
            "left_frame"
        ]

        right_name = policy[
            "right_frame"
        ]

        left_exists = (
            left_name in runtime_frames
            and isinstance(
                runtime_frames[left_name],
                pd.DataFrame,
            )
        )

        right_exists = (
            right_name in runtime_frames
            and isinstance(
                runtime_frames[right_name],
                pd.DataFrame,
            )
        )

        left_column_exists = False
        right_column_exists = False

        selected_keys = None

        if left_exists:
            left_column_exists = (
                policy[
                    "left_value_column"
                ]
                in runtime_frames[
                    left_name
                ].columns
            )

        if right_exists:
            right_column_exists = (
                policy[
                    "right_value_column"
                ]
                in runtime_frames[
                    right_name
                ].columns
            )

        if left_exists and right_exists:
            left_columns = set(
                runtime_frames[
                    left_name
                ].columns
            )

            right_columns = set(
                runtime_frames[
                    right_name
                ].columns
            )

            for key_set in policy[
                "allowed_join_key_sets"
            ]:
                if (
                    set(key_set).issubset(
                        left_columns
                    )
                    and
                    set(key_set).issubset(
                        right_columns
                    )
                ):
                    selected_keys = list(
                        key_set
                    )
                    break

        if (
            not left_exists
            or not right_exists
            or not left_column_exists
            or not right_column_exists
            or selected_keys is None
        ):
            rows.append(
                {
                    "metric_id":
                        metric_id,

                    "left_runtime_frame":
                        left_name,

                    "right_runtime_frame":
                        right_name,

                    "left_value_column":
                        policy[
                            "left_value_column"
                        ],

                    "right_value_column":
                        policy[
                            "right_value_column"
                        ],

                    "required_grain":
                        policy[
                            "required_grain"
                        ],

                    "selected_join_keys":
                        (
                            "NONE"
                            if selected_keys is None
                            else " | ".join(
                                selected_keys
                            )
                        ),

                    "left_source_present":
                        left_exists,

                    "right_source_present":
                        right_exists,

                    "left_value_column_present":
                        left_column_exists,

                    "right_value_column_present":
                        right_column_exists,

                    "left_identity_count":
                        pd.NA,

                    "right_identity_count":
                        pd.NA,

                    "matched_identity_count":
                        pd.NA,

                    "left_identity_match_percent":
                        pd.NA,

                    "right_identity_match_percent":
                        pd.NA,

                    "join_cardinality":
                        "NOT_EVALUATED",

                    "join_compatibility_state":
                        (
                            "NO_GOVERNED_JOIN_KEY_AVAILABLE"
                            if selected_keys is None
                            else
                            "SOURCE_OR_VALUE_COLUMN_MISSING"
                        ),

                    "join_compatibility_authorized":
                        False,

                    "metric_execution_authorized":
                        False,
                }
            )

            continue

        left = _normalized_identity_frame(
            runtime_frames[
                left_name
            ],
            selected_keys,
        )

        right = _normalized_identity_frame(
            runtime_frames[
                right_name
            ],
            selected_keys,
        )

        left = (
            left
            .dropna(
                subset=selected_keys
            )
        )

        right = (
            right
            .dropna(
                subset=selected_keys
            )
        )

        cardinality = (
            _classify_join_cardinality(
                left,
                right,
                selected_keys,
            )
        )

        left_identity = (
            left[
                selected_keys
            ]
            .drop_duplicates()
        )

        right_identity = (
            right[
                selected_keys
            ]
            .drop_duplicates()
        )

        matched_identity = (
            left_identity
            .merge(
                right_identity,
                on=selected_keys,
                how="inner",
                validate="one_to_one",
            )
        )

        left_count = int(
            len(
                left_identity
            )
        )

        right_count = int(
            len(
                right_identity
            )
        )

        matched_count = int(
            len(
                matched_identity
            )
        )

        left_match_percent = (
            matched_count
            / left_count
            * 100.0
            if left_count > 0
            else 0.0
        )

        right_match_percent = (
            matched_count
            / right_count
            * 100.0
            if right_count > 0
            else 0.0
        )

        complete_identity_alignment = (
            left_count > 0
            and right_count > 0
            and matched_count == left_count
            and matched_count == right_count
        )

        controlled_cardinality = (
            cardinality
            != "MANY_TO_MANY"
        )

        compatibility_authorized = (
            complete_identity_alignment
            and controlled_cardinality
        )

        rows.append(
            {
                "metric_id":
                    metric_id,

                "left_runtime_frame":
                    left_name,

                "right_runtime_frame":
                    right_name,

                "left_value_column":
                    policy[
                        "left_value_column"
                    ],

                "right_value_column":
                    policy[
                        "right_value_column"
                    ],

                "required_grain":
                    policy[
                        "required_grain"
                    ],

                "selected_join_keys":
                    " | ".join(
                        selected_keys
                    ),

                "left_source_present":
                    True,

                "right_source_present":
                    True,

                "left_value_column_present":
                    True,

                "right_value_column_present":
                    True,

                "left_identity_count":
                    left_count,

                "right_identity_count":
                    right_count,

                "matched_identity_count":
                    matched_count,

                "left_identity_match_percent":
                    round(
                        left_match_percent,
                        6,
                    ),

                "right_identity_match_percent":
                    round(
                        right_match_percent,
                        6,
                    ),

                "join_cardinality":
                    cardinality,

                "join_compatibility_state":
                    (
                        "JOIN_COMPATIBLE"
                        if compatibility_authorized
                        else
                        "JOIN_COMPATIBILITY_WITHHELD"
                    ),

                "join_compatibility_authorized":
                    compatibility_authorized,

                "metric_execution_authorized":
                    compatibility_authorized,
            }
        )

    result = pd.DataFrame(
        rows
    )

    result[
        "join_compatibility_audit_version"
    ] = (
        _BIRE_JOIN_COMPATIBILITY_AUDIT_VERSION
    )

    result[
        "playground_execution_authorized"
    ] = False

    result[
        "operational_deployment_authorized"
    ] = False

    result[
        "nid_authorization_state"
    ] = (
        result[
            "join_compatibility_authorized"
        ]
        .map(
            {
                True:
                    "NID_BIRE_RUNTIME_JOIN_COMPATIBLE",

                False:
                    "NID_BIRE_RUNTIME_JOIN_WITHHELD",
            }
        )
    )

    return result


def build_bire_runtime_join_compatibility_summary(
    join_audit_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize the compatibility outcome for join-dependent
    baseline metrics.
    """

    if not isinstance(
        join_audit_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "join_audit_df must be a DataFrame."
        )

    return (
        join_audit_df
        .groupby(
            [
                "join_compatibility_state",
                "join_compatibility_authorized",
                "metric_execution_authorized",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="metric_count"
        )
        .sort_values(
            by=[
                "join_compatibility_authorized",
                "join_compatibility_state",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


    #============================================================
# Chapter 69.9I — Authorized Frozen Baseline Metric Execution
#============================================================

_BIRE_FROZEN_BASELINE_METRIC_EXECUTION_VERSION = (
    "BIRE_FROZEN_BASELINE_METRIC_EXECUTION_V1"
)


def _coerce_bool_for_baseline(
    series: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(
        series.dtype
    ):
        return series.astype(
            "boolean"
        )

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

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

    text = (
        series
        .astype("string")
        .str.strip()
        .str.lower()
    )

    result.loc[
        text.isin(
            {
                "true",
                "t",
                "yes",
                "y",
            }
        )
    ] = True

    result.loc[
        text.isin(
            {
                "false",
                "f",
                "no",
                "n",
            }
        )
    ] = False

    return result


def execute_bire_authorized_frozen_baseline_metrics(
    execution_gate_df: pd.DataFrame,
    binding_authority_df: pd.DataFrame,
    join_compatibility_df: pd.DataFrame,
    measurement_plan_df: pd.DataFrame,
    runtime_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Execute only frozen-baseline metrics that earned binding
    authority.

    All other BIRE metrics remain explicitly unexecuted.
    """

    if not isinstance(
        execution_gate_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "execution_gate_df must be a DataFrame."
        )

    if len(
        execution_gate_df
    ) != 1:
        raise ValueError(
            "execution_gate_df must contain one row."
        )

    gate = (
        execution_gate_df.iloc[0]
    )

    if not bool(
        gate[
            "frozen_baseline_execution_authorized"
        ]
    ):
        raise ValueError(
            "Frozen baseline execution is not authorized."
        )

    if not bool(
        gate[
            "metric_execution_within_frozen_"
            "baseline_authorized"
        ]
    ):
        raise ValueError(
            "Frozen baseline metric execution is not authorized."
        )

    for name, frame in {
        "binding_authority_df":
            binding_authority_df,
        "join_compatibility_df":
            join_compatibility_df,
        "measurement_plan_df":
            measurement_plan_df,
    }.items():
        if not isinstance(
            frame,
            pd.DataFrame,
        ):
            raise TypeError(
                f"{name} must be a DataFrame."
            )

    if not isinstance(
        runtime_frames,
        dict,
    ):
        raise TypeError(
            "runtime_frames must be a dictionary."
        )

    direct_authorized = set(
        binding_authority_df.loc[
            binding_authority_df[
                "binding_authority_granted"
            ].eq(True),
            "metric_id",
        ].astype(str)
    )

    join_authorized = set(
        join_compatibility_df.loc[
            join_compatibility_df[
                "join_compatibility_authorized"
            ].eq(True),
            "metric_id",
        ].astype(str)
    )

    executable_metrics = (
        direct_authorized
        | join_authorized
    )

    expected_executable = {
        "FALSE_RECURRENCE_CREATION_COUNT",
        "REPLAY_AUTHORITY_LEAK_COUNT",
    }

    if (
        executable_metrics
        != expected_executable
    ):
        raise ValueError(
            "Executable frozen-baseline metric set changed "
            "unexpectedly. "
            f"Expected: {sorted(expected_executable)}; "
            f"observed: {sorted(executable_metrics)}"
        )

    # --------------------------------------------------------
    # FALSE_RECURRENCE_CREATION_COUNT
    # --------------------------------------------------------

    recurrence_frame_name = (
        "rss_cross_family_relationship_recurrence"
    )

    if (
        recurrence_frame_name
        not in runtime_frames
    ):
        raise KeyError(
            f"Missing runtime frame: {recurrence_frame_name}"
        )

    recurrence = runtime_frames[
        recurrence_frame_name
    ]

    recurrence_required = {
        "rss_relationship_recurrence_observed",
        "rss_distinct_historical_position_count",
        "rss_same_position_alignment_compression_count",
    }

    recurrence_missing = sorted(
        recurrence_required
        - set(
            recurrence.columns
        )
    )

    if recurrence_missing:
        raise KeyError(
            "False-recurrence execution cannot proceed. "
            f"Missing columns: {recurrence_missing}"
        )

    recurrence_observed = (
        _coerce_bool_for_baseline(
            recurrence[
                "rss_relationship_recurrence_observed"
            ]
        )
        .fillna(False)
    )

    distinct_positions = pd.to_numeric(
        recurrence[
            "rss_distinct_historical_position_count"
        ],
        errors="coerce",
    )

    same_position_compression = pd.to_numeric(
        recurrence[
            "rss_same_position_alignment_compression_count"
        ],
        errors="coerce",
    ).fillna(0)

    false_recurrence_mask = (
        recurrence_observed
        &
        (
            distinct_positions.isna()
            |
            distinct_positions.lt(2)
        )
    )

    false_recurrence_count = int(
        false_recurrence_mask.sum()
    )

    recurrence_observed_count = int(
        recurrence_observed.sum()
    )

    same_position_compression_total = int(
        same_position_compression.sum()
    )

    # --------------------------------------------------------
    # REPLAY_AUTHORITY_LEAK_COUNT
    # --------------------------------------------------------

    alignment_frame_name = (
        "rss_cross_family_alignment_governance"
    )

    if (
        alignment_frame_name
        not in runtime_frames
    ):
        raise KeyError(
            f"Missing runtime frame: {alignment_frame_name}"
        )

    alignment = runtime_frames[
        alignment_frame_name
    ]

    authority_required_alignment = {
        "rss_replay_cycle_formation_authorized",
        "rss_replay_lesson_generation_authorized",
    }

    missing_alignment = sorted(
        authority_required_alignment
        - set(
            alignment.columns
        )
    )

    if missing_alignment:
        raise KeyError(
            "Replay-authority execution cannot proceed. "
            f"Missing alignment-governance columns: "
            f"{missing_alignment}"
        )

    pattern_frame_name = (
        "rss_cross_family_relationship_recurrence"
    )

    pattern_frame = runtime_frames[
        pattern_frame_name
    ]

    pattern_column = (
        "rss_pattern_maturity_assignment_authorized"
    )

    if (
        pattern_column
        not in pattern_frame.columns
    ):
        raise KeyError(
            "Replay-authority execution cannot proceed. "
            f"Missing column: {pattern_column}"
        )

    cycle_authority = (
        _coerce_bool_for_baseline(
            alignment[
                "rss_replay_cycle_formation_authorized"
            ]
        )
        .fillna(False)
    )

    lesson_authority = (
        _coerce_bool_for_baseline(
            alignment[
                "rss_replay_lesson_generation_authorized"
            ]
        )
        .fillna(False)
    )

    maturity_authority = (
        _coerce_bool_for_baseline(
            pattern_frame[
                pattern_column
            ]
        )
        .fillna(False)
    )

    cycle_leak_count = int(
        cycle_authority.sum()
    )

    lesson_leak_count = int(
        lesson_authority.sum()
    )

    maturity_leak_count = int(
        maturity_authority.sum()
    )

    replay_authority_leak_count = int(
        cycle_leak_count
        + lesson_leak_count
        + maturity_leak_count
    )

    executed_values = {
        "FALSE_RECURRENCE_CREATION_COUNT":
            {
                "metric_value":
                    false_recurrence_count,

                "execution_detail":
                    (
                        "Recurrence observed with fewer than two "
                        "valid distinct historical positions."
                    ),

                "supporting_observation_count":
                    recurrence_observed_count,

                "auxiliary_audit_value":
                    same_position_compression_total,
            },

        "REPLAY_AUTHORITY_LEAK_COUNT":
            {
                "metric_value":
                    replay_authority_leak_count,

                "execution_detail":
                    (
                        "Sum of unauthorized replay-cycle, "
                        "pattern-maturity, and replay-lesson "
                        "authority activations."
                    ),

                "supporting_observation_count":
                    int(
                        len(alignment)
                        + len(
                            pattern_frame
                        )
                    ),

                "auxiliary_audit_value":
                    (
                        f"cycle={cycle_leak_count}; "
                        f"maturity={maturity_leak_count}; "
                        f"lesson={lesson_leak_count}"
                    ),
            },
    }

    binding_state_lookup = (
        binding_authority_df
        .set_index(
            "metric_id"
        )[
            "binding_authority_state"
        ]
        .astype(str)
        .to_dict()
    )

    rows = []

    for metric in (
        measurement_plan_df.itertuples(
            index=False
        )
    ):
        metric_id = str(
            metric.metric_id
        )

        if metric_id in executed_values:
            evidence = (
                executed_values[
                    metric_id
                ]
            )

            metric_value = (
                evidence[
                    "metric_value"
                ]
            )

            hard_gate_pass = (
                metric_value == 0
            )

            rows.append(
                {
                    "run_id":
                        gate[
                            "run_id"
                        ],

                    "metric_id":
                        metric_id,

                    "metric_family":
                        metric.metric_family,

                    "metric_value":
                        metric_value,

                    "metric_testability_state":
                        "TESTABLE_EXECUTED",

                    "metric_execution_state":
                        "EXECUTED",

                    "metric_result_state":
                        (
                            "HARD_GATE_PASS"
                            if hard_gate_pass
                            else
                            "HARD_GATE_FAIL"
                        ),

                    "binding_authority_state":
                        binding_state_lookup.get(
                            metric_id,
                            "BINDING_AUTHORIZED",
                        ),

                    "execution_detail":
                        evidence[
                            "execution_detail"
                        ],

                    "supporting_observation_count":
                        evidence[
                            "supporting_observation_count"
                        ],

                    "auxiliary_audit_value":
                        evidence[
                            "auxiliary_audit_value"
                        ],

                    "hard_gate_pass":
                        hard_gate_pass,

                    "baseline_reference_established":
                        False,
                }
            )

            continue

        if not bool(
            metric.measurement_evidence_available
        ):
            state = (
                "NOT_TESTABLE_IN_FROZEN_BASELINE"
            )

        else:
            state = (
                binding_state_lookup.get(
                    metric_id,
                    "BINDING_WITHHELD",
                )
            )

        rows.append(
            {
                "run_id":
                    gate[
                        "run_id"
                    ],

                "metric_id":
                    metric_id,

                "metric_family":
                    metric.metric_family,

                "metric_value":
                    pd.NA,

                "metric_testability_state":
                    state,

                "metric_execution_state":
                    "NOT_EXECUTED",

                "metric_result_state":
                    "NO_RESULT",

                "binding_authority_state":
                    state,

                "execution_detail":
                    (
                        "Metric value withheld because execution "
                        "authority was not earned for this frozen "
                        "baseline."
                    ),

                "supporting_observation_count":
                    pd.NA,

                "auxiliary_audit_value":
                    pd.NA,

                "hard_gate_pass":
                    pd.NA,

                "baseline_reference_established":
                    False,
            }
        )

    result = pd.DataFrame(
        rows
    )

    result[
        "baseline_metric_execution_version"
    ] = (
        _BIRE_FROZEN_BASELINE_METRIC_EXECUTION_VERSION
    )

    result[
        "playground_execution_authorized"
    ] = False

    result[
        "operational_deployment_authorized"
    ] = False

    result[
        "nid_authorization_state"
    ] = (
        "NID_BIRE_FROZEN_BASELINE_METRIC_RESULT_RECORDED"
    )

    return (
        result
        .sort_values(
            by=[
                "metric_execution_state",
                "metric_family",
                "metric_id",
            ]
        )
        .reset_index(drop=True)
    )


__all__ = [
    "build_bire_frozen_baseline_measurement_plan",
    "build_bire_frozen_baseline_measurement_plan_summary",
    "build_bire_runtime_measurement_binding_discovery",
    "build_bire_runtime_measurement_binding_discovery_summary",
    "build_bire_runtime_measurement_binding_review",
    "build_bire_runtime_semantic_binding_disposition",
    "build_bire_runtime_evidence_role_resolution",
    "build_bire_runtime_evidence_role_resolution_summary",
    "build_bire_runtime_binding_authority_audit",
    "build_bire_runtime_binding_authority_summary",
    "build_bire_runtime_join_compatibility_audit",
    "build_bire_runtime_join_compatibility_summary",
    "execute_bire_authorized_frozen_baseline_metrics",

]
