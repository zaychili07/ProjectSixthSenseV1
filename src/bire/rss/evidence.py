#============================================================
# Project Sixth Sense — BIRE OS
# Replay Stability Support (RSS)
#
# File:
#     evidence.py
#
# Chapter:
#     68.12 — RSS Replay Evidence Discovery
#
# Purpose:
#     Discover existing longitudinal evidence that may support
#     patient-specific replay learning.
#
# Responsibilities:
#     - Identify patient and encounter identifiers
#     - Identify longitudinal ordering candidates
#     - Identify stability and instability evidence
#     - Identify recovery and escalation evidence
#     - Identify upstream replay context
#     - Identify confidence, quality, and provenance context
#     - Quarantine derived replay labels
#     - Preserve TTI-dependent intervention replay boundaries
#
# Does Not:
#     - Generate replay pattern states
#     - Generate replay lesson states
#     - Predict future deterioration
#     - Manufacture recurrence
#     - Treat similarity as recurrence
#     - Claim historical causation
#
# Governance:
#     All discovered evidence remains subject to NID review.
#
# Core Doctrine:
#     Replay may reveal a lesson.
#     Recurrence must earn it.
#============================================================

from __future__ import annotations

from collections.abc import Mapping
import re

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_numeric_dtype,
)


#============================================================
# RSS Evidence Vocabulary
#============================================================

_RSS_EXACT_IDENTIFIER_COLUMNS = {
    "patient_id",
    "encounter_id",
    "episode_id",
    "memory_id",
    "event_id",
    "replay_id",
}


_RSS_EXACT_ORDER_COLUMNS = {
    "timestamp",
    "datetime",
    "date",
    "event_time",
    "observation_time",
    "recorded_at",
    "charttime",
    "time_index",
    "time_step",
    "sequence",
    "sequence_id",
    "event_order",
    "step",
    "years_since_sim_start",
}


_RSS_ORDER_TOKENS = {
    "timestamp",
    "datetime",
    "charttime",
    "sequence",
    "ordered",
    "order",
    "index",
    "step",
    "chronology",
    "longitudinal",
}


_RSS_DURATION_TOKENS = {
    "duration",
    "minutes",
    "minute",
    "hours",
    "hour",
    "days",
    "day",
    "weeks",
    "week",
    "months",
    "month",
    "years",
    "year",
    "interval",
    "elapsed",
    "since",
}


_RSS_STABILITY_TOKENS = {
    "stable",
    "stability",
    "stabilization",
    "durability",
    "durable",
    "scarcity",
    "reserve",
}


_RSS_INSTABILITY_TOKENS = {
    "instability",
    "unstable",
    "deterioration",
    "deteriorating",
    "worsening",
    "volatility",
    "volatile",
    "collapse",
    "rebound",
    "decompensation",
    "failure",
}


_RSS_RECOVERY_TOKENS = {
    "recovery",
    "recovering",
    "recovered",
    "resilience",
    "reversal",
    "restoration",
}


_RSS_ESCALATION_TOKENS = {
    "escalation",
    "escalating",
    "escalated",
    "re_escalation",
    "reescalation",
    "critical",
    "emergency",
    "transfer",
    "readmission",
}


_RSS_REPLAY_TOKENS = {
    "replay",
    "recurring",
    "recurrence",
    "repeated",
    "repeat",
    "historical",
    "history",
    "prior",
    "previous",
    "persistence",
    "longitudinal",
    "cycle",
    "cycles",
    "pattern",
}


_RSS_OUTCOME_TOKENS = {
    "outcome",
    "occurred",
    "event",
    "discharge",
    "readmission",
    "mortality",
    "survival",
    "resolved",
    "resolution",
    "return",
}


_RSS_BURDEN_TOKENS = {
    "burden",
    "count",
    "frequency",
    "rate",
    "density",
    "scarcity",
    "persistence",
    "proportion",
    "ratio",
}


_RSS_CONFIDENCE_QUALITY_TOKENS = {
    "confidence",
    "uncertainty",
    "trust",
    "completeness",
    "missingness",
    "conflict",
    "contradiction",
    "reliability",
    "quality",
    "provenance",
    "agreement",
}


_RSS_FORECAST_TOKENS = {
    "forecast",
    "prediction",
    "predicted",
    "horizon",
    "future",
}


_RSS_INTERVENTION_TOKENS = {
    "intervention",
    "therapeutic",
    "therapy",
    "treatment",
    "medication",
    "procedure",
    "response",
}


# Existing fields that already encode an RSS-like conclusion
# must not independently generate the same conclusion again.
_RSS_DERIVED_LABEL_TOKENS = {
    "pattern_state",
    "replay_state",
    "lesson_state",
    "insufficient_replay_history",
    "new_pattern",
    "emerging_pattern",
    "recurring_pattern",
    "consistent_pattern",
    "no_meaningful_pattern",
    "unknown_replay",
    "repeated_instability",
    "failed_recovery_pattern",
    "stability_scarcity_high",
    "escalating_replay",
    "durable_stability_pattern",
    "recurring_recovery_pattern",
    "recurring_intervention_response",
}


def _normalize_rss_column_name(
    column_name: str,
) -> str:
    """
    Normalize mixed column naming into lowercase snake-style text.
    """

    normalized = re.sub(
        r"(?<=[a-z0-9])(?=[A-Z])",
        "_",
        str(column_name),
    )

    return re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        normalized,
    ).strip("_").lower()


def _tokenize_rss_column(
    column_name: str,
) -> set[str]:
    """
    Convert a column name into semantic tokens.
    """

    normalized = _normalize_rss_column_name(
        column_name
    )

    return set(
        re.findall(
            r"[a-z0-9]+",
            normalized,
        )
    )


def _contains_rss_phrase(
    normalized_name: str,
    phrases: set[str],
) -> bool:
    """
    Check exact normalized phrases and compound state names.
    """

    return any(
        phrase in normalized_name
        for phrase in phrases
    )


def _is_rss_binary_series(
    series: pd.Series,
) -> bool:
    """
    Determine whether a series is boolean or numeric binary.
    """

    non_null = series.dropna()

    if non_null.empty:
        return False

    if is_bool_dtype(series):
        return True

    if is_numeric_dtype(series):
        values = set(
            pd.to_numeric(
                non_null,
                errors="coerce",
            )
            .dropna()
            .unique()
            .tolist()
        )

        return values.issubset({
            0,
            1,
            0.0,
            1.0,
        })

    return False


def _get_rss_active_count(
    series: pd.Series,
) -> int | None:
    """
    Return active count for boolean or binary evidence fields.
    """

    if not _is_rss_binary_series(series):
        return None

    non_null = series.dropna()

    if is_bool_dtype(series):
        return int(
            non_null.astype(bool).sum()
        )

    numeric = pd.to_numeric(
        non_null,
        errors="coerce",
    )

    return int(
        numeric.eq(1).sum()
    )


def _classify_rss_evidence_column(
    column_name: str,
) -> dict[str, object] | None:
    """
    Classify one possible RSS evidence column.

    Candidate discovery does not establish replay validity.
    Every candidate remains subject to semantic and NID review.
    """

    normalized = _normalize_rss_column_name(
        column_name
    )

    tokens = _tokenize_rss_column(
        column_name
    )

    if normalized in _RSS_EXACT_IDENTIFIER_COLUMNS:
        return {
            "candidate_role":
                "IDENTIFIER",
            "evidence_domains":
                "IDENTITY",
            "rss_v1_candidate":
                True,
            "activation_dependency":
                "NONE",
            "evidence_leakage_risk":
                False,
            "manual_verification_required":
                False,
            "nid_review_required":
                True,
        }

    if normalized in _RSS_EXACT_ORDER_COLUMNS:
        return {
            "candidate_role":
                "LONGITUDINAL_ORDER_CANDIDATE",
            "evidence_domains":
                "TEMPORAL_ORDER",
            "rss_v1_candidate":
                True,
            "activation_dependency":
                "ORDER_VALIDATION",
            "evidence_leakage_risk":
                False,
            "manual_verification_required":
                True,
            "nid_review_required":
                True,
        }

    if _contains_rss_phrase(
        normalized,
        _RSS_DERIVED_LABEL_TOKENS,
    ):
        return {
            "candidate_role":
                "DERIVED_REPLAY_LABEL_CONTEXT",
            "evidence_domains":
                "PRECOMPUTED_REPLAY_CONCLUSION",
            "rss_v1_candidate":
                False,
            "activation_dependency":
                "LEAKAGE_REVIEW",
            "evidence_leakage_risk":
                True,
            "manual_verification_required":
                True,
            "nid_review_required":
                True,
        }

    matched_domains: list[str] = []

    has_stability = bool(
        tokens.intersection(
            _RSS_STABILITY_TOKENS
        )
    )

    has_instability = bool(
        tokens.intersection(
            _RSS_INSTABILITY_TOKENS
        )
    )

    has_recovery = bool(
        tokens.intersection(
            _RSS_RECOVERY_TOKENS
        )
    )

    has_escalation = bool(
        tokens.intersection(
            _RSS_ESCALATION_TOKENS
        )
    )

    has_replay = bool(
        tokens.intersection(
            _RSS_REPLAY_TOKENS
        )
    )

    has_outcome = bool(
        tokens.intersection(
            _RSS_OUTCOME_TOKENS
        )
    )

    has_burden = bool(
        tokens.intersection(
            _RSS_BURDEN_TOKENS
        )
    )

    has_quality = bool(
        tokens.intersection(
            _RSS_CONFIDENCE_QUALITY_TOKENS
        )
    )

    has_forecast = bool(
        tokens.intersection(
            _RSS_FORECAST_TOKENS
        )
    )

    has_intervention = bool(
        tokens.intersection(
            _RSS_INTERVENTION_TOKENS
        )
    )

    has_order = bool(
        tokens.intersection(
            _RSS_ORDER_TOKENS
        )
    )

    has_duration = bool(
        tokens.intersection(
            _RSS_DURATION_TOKENS
        )
    )

    if has_stability:
        matched_domains.append(
            "STABILITY"
        )

    if has_instability:
        matched_domains.append(
            "INSTABILITY"
        )

    if has_recovery:
        matched_domains.append(
            "RECOVERY"
        )

    if has_escalation:
        matched_domains.append(
            "ESCALATION"
        )

    if has_replay:
        matched_domains.append(
            "REPLAY_HISTORY"
        )

    if has_outcome:
        matched_domains.append(
            "OUTCOME"
        )

    if has_burden:
        matched_domains.append(
            "BURDEN"
        )

    if has_quality:
        matched_domains.append(
            "EVIDENCE_QUALITY"
        )

    if has_forecast:
        matched_domains.append(
            "FORECAST_HISTORY"
        )

    if has_intervention:
        matched_domains.append(
            "INTERVENTION_CONTEXT"
        )

    if has_order:
        matched_domains.append(
            "TEMPORAL_ORDER"
        )

    if has_duration:
        matched_domains.append(
            "DURATION"
        )

    if not matched_domains:
        return None

    # --------------------------------------------------------
    # Primary candidate role
    # --------------------------------------------------------

    if has_intervention:
        candidate_role = (
            "TTI_DEPENDENT_INTERVENTION_REPLAY_CONTEXT"
        )
        rss_v1_candidate = False
        activation_dependency = (
            "TTI_RESPONSE_ENGINE"
        )

    elif has_forecast:
        candidate_role = (
            "FORECAST_HISTORY_CONTEXT"
        )
        rss_v1_candidate = True
        activation_dependency = (
            "HISTORICAL_USE_VALIDATION"
        )

    elif has_order:
        candidate_role = (
            "LONGITUDINAL_ORDER_CANDIDATE"
        )
        rss_v1_candidate = True
        activation_dependency = (
            "ORDER_VALIDATION"
        )

    elif has_stability and not has_instability:
        candidate_role = (
            "STABILITY_EVIDENCE"
        )
        rss_v1_candidate = True
        activation_dependency = "NONE"

    elif has_instability:
        candidate_role = (
            "INSTABILITY_EVIDENCE"
        )
        rss_v1_candidate = True
        activation_dependency = "NONE"

    elif has_recovery:
        candidate_role = (
            "RECOVERY_EVIDENCE"
        )
        rss_v1_candidate = True
        activation_dependency = "NONE"

    elif has_escalation:
        candidate_role = (
            "ESCALATION_EVIDENCE"
        )
        rss_v1_candidate = True
        activation_dependency = "NONE"

    elif has_replay:
        candidate_role = (
            "UPSTREAM_REPLAY_CONTEXT"
        )
        rss_v1_candidate = True
        activation_dependency = (
            "PROVENANCE_VALIDATION"
        )

    elif has_outcome:
        candidate_role = (
            "OUTCOME_CONTEXT"
        )
        rss_v1_candidate = True
        activation_dependency = "NONE"

    elif has_quality:
        candidate_role = (
            "CONFIDENCE_QUALITY_CONTEXT"
        )
        rss_v1_candidate = True
        activation_dependency = (
            "CONFIDENCE_ENGINE"
        )

    elif has_duration:
        candidate_role = (
            "DURATION_OR_INTERVAL_CONTEXT"
        )
        rss_v1_candidate = True
        activation_dependency = (
            "TEMPORAL_SEMANTIC_VALIDATION"
        )

    else:
        candidate_role = "REVIEW_CONTEXT"
        rss_v1_candidate = False
        activation_dependency = (
            "MANUAL_REVIEW"
        )

    return {
        "candidate_role":
            candidate_role,
        "evidence_domains":
            " | ".join(
                matched_domains
            ),
        "rss_v1_candidate":
            rss_v1_candidate,
        "activation_dependency":
            activation_dependency,
        "evidence_leakage_risk":
            False,
        "manual_verification_required":
            True,
        "nid_review_required":
            True,
    }


def build_rss_replay_evidence_inventory(
    source_frames: Mapping[
        str,
        pd.DataFrame,
    ],
) -> pd.DataFrame:
    """
    Discover possible RSS replay evidence across BIRE OS sources.

    This function identifies:
    - identifiers
    - longitudinal ordering candidates
    - stability and instability evidence
    - recovery and escalation evidence
    - replay and outcome context
    - confidence and provenance context
    - TTI-dependent intervention context
    - possible evidence-leakage fields

    This function does not form replay cycles or generate RSS states.
    """

    if not isinstance(
        source_frames,
        Mapping,
    ):
        raise TypeError(
            "source_frames must be a mapping of "
            "source names to pandas DataFrames."
        )

    if not source_frames:
        raise ValueError(
            "source_frames cannot be empty."
        )

    rows: list[dict[str, object]] = []

    for source_name, source_df in (
        source_frames.items()
    ):
        if not isinstance(
            source_df,
            pd.DataFrame,
        ):
            raise TypeError(
                f"{source_name} must be a pandas DataFrame."
            )

        for column in source_df.columns:
            classification = (
                _classify_rss_evidence_column(
                    column
                )
            )

            if classification is None:
                continue

            series = source_df[column]

            rows.append({
                "source":
                    source_name,
                "column":
                    column,
                **classification,
                "dtype":
                    str(series.dtype),
                "row_count":
                    len(source_df),
                "non_null_count":
                    int(
                        series.notna().sum()
                    ),
                "non_null_percent":
                    round(
                        series.notna().mean()
                        * 100,
                        3,
                    ),
                "unique_count":
                    int(
                        series.nunique(
                            dropna=True
                        )
                    ),
                "active_count":
                    _get_rss_active_count(
                        series
                    ),
                "sample_values":
                    " | ".join(
                        series
                        .dropna()
                        .drop_duplicates()
                        .astype(str)
                        .head(5)
                        .tolist()
                    ),
            })

    result_columns = [
        "source",
        "column",
        "candidate_role",
        "evidence_domains",
        "rss_v1_candidate",
        "activation_dependency",
        "evidence_leakage_risk",
        "manual_verification_required",
        "nid_review_required",
        "dtype",
        "row_count",
        "non_null_count",
        "non_null_percent",
        "unique_count",
        "active_count",
        "sample_values",
    ]

    if not rows:
        return pd.DataFrame(
            columns=result_columns
        )

    result = pd.DataFrame(rows)

    role_order = {
        "IDENTIFIER": 0,
        "LONGITUDINAL_ORDER_CANDIDATE": 1,
        "STABILITY_EVIDENCE": 2,
        "INSTABILITY_EVIDENCE": 3,
        "RECOVERY_EVIDENCE": 4,
        "ESCALATION_EVIDENCE": 5,
        "UPSTREAM_REPLAY_CONTEXT": 6,
        "OUTCOME_CONTEXT": 7,
        "DURATION_OR_INTERVAL_CONTEXT": 8,
        "CONFIDENCE_QUALITY_CONTEXT": 9,
        "FORECAST_HISTORY_CONTEXT": 10,
        (
            "TTI_DEPENDENT_"
            "INTERVENTION_REPLAY_CONTEXT"
        ): 11,
        "DERIVED_REPLAY_LABEL_CONTEXT": 12,
        "REVIEW_CONTEXT": 13,
    }

    result["_role_order"] = (
        result[
            "candidate_role"
        ]
        .map(role_order)
        .fillna(99)
    )

    return (
        result
        .sort_values(
            by=[
                "_role_order",
                "rss_v1_candidate",
                "source",
                "non_null_count",
                "column",
            ],
            ascending=[
                True,
                False,
                True,
                False,
                True,
            ],
        )
        .drop(
            columns="_role_order"
        )
        .reset_index(drop=True)
    )

#============================================================
# Chapter 68.13 — RSS Replay Evidence Qualification
#============================================================

_RSS_PRIMARY_REPLAY_EVIDENCE_ROLES = {
    "STABILITY_EVIDENCE",
    "INSTABILITY_EVIDENCE",
    "RECOVERY_EVIDENCE",
    "ESCALATION_EVIDENCE",
}


_RSS_QUALIFICATION_STATE_BY_ROLE = {
    "IDENTIFIER":
        "STRUCTURAL_FOUNDATION",

    "LONGITUDINAL_ORDER_CANDIDATE":
        "TEMPORAL_CANDIDATE_REQUIRES_VALIDATION",

    "STABILITY_EVIDENCE":
        "ELIGIBLE_FOR_REPLAY_FAMILY_REVIEW",

    "INSTABILITY_EVIDENCE":
        "ELIGIBLE_FOR_REPLAY_FAMILY_REVIEW",

    "RECOVERY_EVIDENCE":
        "ELIGIBLE_FOR_REPLAY_FAMILY_REVIEW",

    "ESCALATION_EVIDENCE":
        "ELIGIBLE_FOR_REPLAY_FAMILY_REVIEW",

    "UPSTREAM_REPLAY_CONTEXT":
        "PROVENANCE_REVIEW_REQUIRED",

    "OUTCOME_CONTEXT":
        "OUTCOME_ALIGNMENT_REVIEW_REQUIRED",

    "DURATION_OR_INTERVAL_CONTEXT":
        "TEMPORAL_SEMANTIC_REVIEW_REQUIRED",

    "CONFIDENCE_QUALITY_CONTEXT":
        "SUPPORTING_EVIDENCE_CONTEXT",

    "FORECAST_HISTORY_CONTEXT":
        "HISTORICAL_USE_REVIEW_REQUIRED",

    "TTI_DEPENDENT_INTERVENTION_REPLAY_CONTEXT":
        "DEFERRED_TTI_DEPENDENCY",

    "DERIVED_REPLAY_LABEL_CONTEXT":
        "QUARANTINED_EVIDENCE_LEAKAGE",

    "REVIEW_CONTEXT":
        "MANUAL_SEMANTIC_REVIEW_REQUIRED",
}


_RSS_QUALIFICATION_PRIORITY = {
    "STRUCTURAL_FOUNDATION": 0,
    "TEMPORAL_CANDIDATE_REQUIRES_VALIDATION": 1,
    "ELIGIBLE_FOR_REPLAY_FAMILY_REVIEW": 2,
    "OUTCOME_ALIGNMENT_REVIEW_REQUIRED": 3,
    "TEMPORAL_SEMANTIC_REVIEW_REQUIRED": 4,
    "SUPPORTING_EVIDENCE_CONTEXT": 5,
    "PROVENANCE_REVIEW_REQUIRED": 6,
    "HISTORICAL_USE_REVIEW_REQUIRED": 7,
    "DEFERRED_TTI_DEPENDENCY": 8,
    "QUARANTINED_EVIDENCE_LEAKAGE": 9,
    "MANUAL_SEMANTIC_REVIEW_REQUIRED": 10,
    "DEFERRED_OR_UNQUALIFIED": 11,
}


def build_rss_replay_evidence_qualification(
    evidence_inventory_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Qualify discovered RSS evidence candidates.

    Qualification identifies:
    - structural foundation fields
    - temporal candidates
    - primary RSS v1 evidence families
    - supporting context
    - activation dependencies
    - evidence-leakage fields
    - cross-source overlap requiring canonical-source review

    This function does not verify source independence, form replay
    cycles, or generate RSS pattern and lesson states.
    """

    if not isinstance(
        evidence_inventory_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "evidence_inventory_df must be a pandas DataFrame."
        )

    required_columns = {
        "source",
        "column",
        "candidate_role",
        "evidence_domains",
        "rss_v1_candidate",
        "activation_dependency",
        "evidence_leakage_risk",
        "manual_verification_required",
        "nid_review_required",
        "dtype",
        "row_count",
        "non_null_count",
        "non_null_percent",
        "unique_count",
        "active_count",
        "sample_values",
    }

    missing_columns = sorted(
        required_columns
        - set(evidence_inventory_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "RSS evidence qualification cannot proceed. "
            f"Missing inventory columns: {missing_columns}"
        )

    qualification = (
        evidence_inventory_df
        .copy()
    )

    # --------------------------------------------------------
    # Cross-source column overlap
    # --------------------------------------------------------

    source_count_by_column = (
        qualification
        .groupby(
            "column",
            dropna=False,
        )["source"]
        .nunique()
    )

    sources_by_column = (
        qualification
        .groupby(
            "column",
            dropna=False,
        )["source"]
        .agg(
            lambda values: " | ".join(
                sorted(
                    {
                        str(value)
                        for value in values
                    }
                )
            )
        )
    )

    qualification[
        "source_count_for_column"
    ] = (
        qualification["column"]
        .map(source_count_by_column)
        .astype(int)
    )

    qualification[
        "sources_with_column"
    ] = (
        qualification["column"]
        .map(sources_by_column)
    )

    qualification[
        "cross_source_overlap_state"
    ] = "SINGLE_SOURCE_COLUMN"

    qualification.loc[
        qualification[
            "source_count_for_column"
        ].gt(1),
        "cross_source_overlap_state",
    ] = "MULTI_SOURCE_OVERLAP_REVIEW_REQUIRED"

    # --------------------------------------------------------
    # Qualification state
    # --------------------------------------------------------

    qualification[
        "qualification_state"
    ] = (
        qualification[
            "candidate_role"
        ]
        .map(
            _RSS_QUALIFICATION_STATE_BY_ROLE
        )
        .fillna(
            "DEFERRED_OR_UNQUALIFIED"
        )
    )

    # Leakage always overrides other qualification outcomes.
    leakage_mask = (
        qualification[
            "evidence_leakage_risk"
        ].eq(True)
    )

    qualification.loc[
        leakage_mask,
        "qualification_state",
    ] = "QUARANTINED_EVIDENCE_LEAKAGE"

    # --------------------------------------------------------
    # RSS v1 replay-family eligibility
    # --------------------------------------------------------

    qualification[
        "eligible_for_rss_v1_family_review"
    ] = (
        qualification[
            "candidate_role"
        ].isin(
            _RSS_PRIMARY_REPLAY_EVIDENCE_ROLES
        )
        &
        qualification[
            "rss_v1_candidate"
        ].eq(True)
        &
        ~qualification[
            "evidence_leakage_risk"
        ].eq(True)
    )

    # --------------------------------------------------------
    # Canonical-source and evidence-independence status
    # --------------------------------------------------------

    qualification[
        "canonical_source_state"
    ] = "SINGLE_SOURCE_CANDIDATE"

    shared_identifier_mask = (
        qualification[
            "candidate_role"
        ].eq("IDENTIFIER")
        &
        qualification[
            "source_count_for_column"
        ].gt(1)
    )

    qualification.loc[
        shared_identifier_mask,
        "canonical_source_state",
    ] = "SHARED_STRUCTURAL_KEY"

    overlapping_evidence_mask = (
        ~qualification[
            "candidate_role"
        ].eq("IDENTIFIER")
        &
        qualification[
            "source_count_for_column"
        ].gt(1)
    )

    qualification.loc[
        overlapping_evidence_mask,
        "canonical_source_state",
    ] = "CANONICAL_SOURCE_SELECTION_REQUIRED"

    qualification[
        "independent_evidence_status"
    ] = "SINGLE_SOURCE_INDEPENDENCE_NOT_APPLICABLE"

    qualification.loc[
        qualification[
            "source_count_for_column"
        ].gt(1),
        "independent_evidence_status",
    ] = "MULTI_SOURCE_INDEPENDENCE_NOT_VERIFIED"

    # --------------------------------------------------------
    # Qualification ordering
    # --------------------------------------------------------

    qualification[
        "_qualification_priority"
    ] = (
        qualification[
            "qualification_state"
        ]
        .map(
            _RSS_QUALIFICATION_PRIORITY
        )
        .fillna(99)
    )

    return (
        qualification
        .sort_values(
            by=[
                "_qualification_priority",
                "eligible_for_rss_v1_family_review",
                "source_count_for_column",
                "source",
                "column",
            ],
            ascending=[
                True,
                False,
                False,
                True,
                True,
            ],
        )
        .drop(
            columns="_qualification_priority"
        )
        .reset_index(drop=True)
    )


def build_rss_replay_evidence_qualification_summary(
    qualification_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize RSS evidence qualification outcomes.
    """

    if not isinstance(
        qualification_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "qualification_df must be a pandas DataFrame."
        )

    required_columns = {
        "qualification_state",
        "candidate_role",
        "cross_source_overlap_state",
        "eligible_for_rss_v1_family_review",
    }

    missing_columns = sorted(
        required_columns
        - set(qualification_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "RSS qualification summary cannot be built. "
            f"Missing columns: {missing_columns}"
        )

    return (
        qualification_df
        .groupby(
            [
                "qualification_state",
                "candidate_role",
                "cross_source_overlap_state",
                "eligible_for_rss_v1_family_review",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            by=[
                "eligible_for_rss_v1_family_review",
                "count",
                "qualification_state",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

#============================================================
# Chapter 68.14 — RSS Canonical Replay Substrate Selection
#============================================================

_RSS_V1_CANONICAL_REPLAY_SOURCE = "LPMR_SOURCE"


_RSS_SOURCE_SELECTION_PRIORITY = {
    "CANONICAL_REPLAY_SUBSTRATE_FIELD": 0,
    "NONINDEPENDENT_ACCESS_COPY_EXCLUDED": 1,
    "SUPPLEMENTAL_SINGLE_SOURCE_REVIEW_REQUIRED": 2,
    "CANONICAL_SELECTION_DEFERRED_NO_LPMR_COPY": 3,
}


def build_rss_canonical_replay_substrate_selection(
    qualification_df: pd.DataFrame,
    canonical_source: str = (
        _RSS_V1_CANONICAL_REPLAY_SOURCE
    ),
) -> pd.DataFrame:
    """
    Select the canonical RSS v1 replay substrate.

    Eligible evidence available through LPMR is admitted through
    LPMR as the canonical longitudinal access point.

    Copies of the same eligible field appearing in other enriched
    sources are preserved for provenance but receive no independent
    evidentiary weight.

    Eligible fields unavailable through LPMR remain deferred for
    supplemental source-admission review.

    This function does not:
    - verify longitudinal ordering
    - form replay cycles
    - calculate recurrence
    - generate RSS states
    """

    if not isinstance(
        qualification_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "qualification_df must be a pandas DataFrame."
        )

    required_columns = {
        "source",
        "column",
        "candidate_role",
        "evidence_domains",
        "eligible_for_rss_v1_family_review",
        "source_count_for_column",
        "sources_with_column",
        "cross_source_overlap_state",
        "dtype",
        "non_null_count",
        "non_null_percent",
        "unique_count",
        "active_count",
        "sample_values",
    }

    missing_columns = sorted(
        required_columns
        - set(qualification_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "RSS canonical substrate selection cannot proceed. "
            f"Missing qualification columns: {missing_columns}"
        )

    eligible = (
        qualification_df.loc[
            qualification_df[
                "eligible_for_rss_v1_family_review"
            ].eq(True)
        ]
        .copy()
    )

    if eligible.empty:
        return pd.DataFrame(
            columns=[
                *qualification_df.columns,
                "canonical_replay_source",
                "canonical_source_available_for_column",
                "source_selection_state",
                "admitted_to_rss_v1_substrate",
                "independent_evidence_weight",
                "source_selection_reason",
            ]
        )

    canonical_availability = (
        eligible
        .groupby(
            "column",
            dropna=False,
        )["source"]
        .transform(
            lambda values: bool(
                values.eq(
                    canonical_source
                ).any()
            )
        )
    )

    eligible[
        "canonical_replay_source"
    ] = canonical_source

    eligible[
        "canonical_source_available_for_column"
    ] = canonical_availability

    eligible[
        "source_selection_state"
    ] = (
        "CANONICAL_SELECTION_DEFERRED_NO_LPMR_COPY"
    )

    # --------------------------------------------------------
    # Canonical LPMR substrate fields
    # --------------------------------------------------------

    canonical_field_mask = (
        eligible["source"].eq(
            canonical_source
        )
        &
        eligible[
            "canonical_source_available_for_column"
        ].eq(True)
    )

    eligible.loc[
        canonical_field_mask,
        "source_selection_state",
    ] = "CANONICAL_REPLAY_SUBSTRATE_FIELD"

    # --------------------------------------------------------
    # Other copies of fields already available through LPMR
    # --------------------------------------------------------

    access_copy_mask = (
        ~eligible["source"].eq(
            canonical_source
        )
        &
        eligible[
            "canonical_source_available_for_column"
        ].eq(True)
    )

    eligible.loc[
        access_copy_mask,
        "source_selection_state",
    ] = "NONINDEPENDENT_ACCESS_COPY_EXCLUDED"

    # --------------------------------------------------------
    # Eligible single-source fields unavailable in LPMR
    # --------------------------------------------------------

    supplemental_single_source_mask = (
        ~eligible[
            "canonical_source_available_for_column"
        ].eq(True)
        &
        eligible[
            "source_count_for_column"
        ].eq(1)
    )

    eligible.loc[
        supplemental_single_source_mask,
        "source_selection_state",
    ] = (
        "SUPPLEMENTAL_SINGLE_SOURCE_"
        "REVIEW_REQUIRED"
    )

    # --------------------------------------------------------
    # Admission and evidentiary weight
    # --------------------------------------------------------

    eligible[
        "admitted_to_rss_v1_substrate"
    ] = eligible[
        "source_selection_state"
    ].eq(
        "CANONICAL_REPLAY_SUBSTRATE_FIELD"
    )

    evidence_weight = pd.Series(
        pd.NA,
        index=eligible.index,
        dtype="Float64",
    )

    evidence_weight.loc[
        canonical_field_mask
    ] = 1.0

    evidence_weight.loc[
        access_copy_mask
    ] = 0.0

    eligible[
        "independent_evidence_weight"
    ] = evidence_weight

    # --------------------------------------------------------
    # Human-readable selection rationale
    # --------------------------------------------------------

    eligible[
        "source_selection_reason"
    ] = (
        "Eligible field is unavailable through the canonical "
        "LPMR replay substrate. Supplemental source admission "
        "remains deferred."
    )

    eligible.loc[
        canonical_field_mask,
        "source_selection_reason",
    ] = (
        "Eligible field is admitted through LPMR as the "
        "canonical RSS v1 longitudinal replay substrate."
    )

    eligible.loc[
        access_copy_mask,
        "source_selection_reason",
    ] = (
        "The same eligible field is available through LPMR. "
        "This source copy is preserved for provenance but "
        "excluded from independent evidentiary weight."
    )

    eligible.loc[
        supplemental_single_source_mask,
        "source_selection_reason",
    ] = (
        "Eligible field appears in one non-LPMR source and "
        "requires separate provenance and admission review."
    )

    eligible[
        "_source_selection_priority"
    ] = (
        eligible[
            "source_selection_state"
        ]
        .map(
            _RSS_SOURCE_SELECTION_PRIORITY
        )
        .fillna(99)
    )

    return (
        eligible
        .sort_values(
            by=[
                "_source_selection_priority",
                "candidate_role",
                "column",
                "source",
            ],
            ascending=[
                True,
                True,
                True,
                True,
            ],
        )
        .drop(
            columns="_source_selection_priority"
        )
        .reset_index(drop=True)
    )


def build_rss_canonical_replay_substrate_summary(
    substrate_selection_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize RSS canonical replay substrate selection.
    """

    if not isinstance(
        substrate_selection_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "substrate_selection_df must be a pandas DataFrame."
        )

    required_columns = {
        "source_selection_state",
        "candidate_role",
        "admitted_to_rss_v1_substrate",
        "independent_evidence_weight",
    }

    missing_columns = sorted(
        required_columns
        - set(substrate_selection_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "RSS canonical substrate summary cannot be built. "
            f"Missing columns: {missing_columns}"
        )

    return (
        substrate_selection_df
        .groupby(
            [
                "source_selection_state",
                "candidate_role",
                "admitted_to_rss_v1_substrate",
                "independent_evidence_weight",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            by=[
                "admitted_to_rss_v1_substrate",
                "count",
                "candidate_role",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

#============================================================
# Chapter 68.15 — RSS Longitudinal Ordering Validation
#============================================================

_RSS_NON_ORDER_TEMPORAL_TOKENS = {
    "delay",
    "lag",
    "duration",
    "waiting",
    "wait",
    "turnaround",
    "recognition",
    "response",
    "elapsed",
    "interval",
}


_RSS_EXPLICIT_ORDER_SUFFIXES = (
    "_timestamp",
    "_datetime",
    "_date",
    "_time_index",
    "_time_step",
    "_sequence",
    "_sequence_id",
    "_event_order",
)


_RSS_ORDER_VALIDATION_PRIORITY = {
    "ELIGIBLE_CANONICAL_ORDER_CANDIDATE": 0,
    "ELIGIBLE_WITH_TIE_BREAKER_REQUIRED": 1,
    "ENCOUNTER_VALUE_CONFLICT_REVIEW_REQUIRED": 2,
    "INCOMPLETE_ORDER_CANDIDATE": 3,
    "LOW_VARIATION_ORDER_CANDIDATE": 4,
    "TEMPORAL_CONTEXT_NOT_ORDER_ANCHOR": 5,
    "UNPARSEABLE_ORDER_CANDIDATE": 6,
    "MANUAL_ORDER_SEMANTIC_REVIEW_REQUIRED": 7,
}


def _classify_rss_order_semantics(
    column_name: str,
) -> str:
    """
    Classify whether a temporal-looking field can semantically
    represent longitudinal order.
    """

    normalized = _normalize_rss_column_name(
        column_name
    )

    tokens = _tokenize_rss_column(
        column_name
    )

    if normalized in _RSS_EXACT_ORDER_COLUMNS:
        return "EXPLICIT_LONGITUDINAL_ORDER"

    if tokens.intersection(
        _RSS_NON_ORDER_TEMPORAL_TOKENS
    ):
        return "TEMPORAL_CONTEXT_NOT_ORDER"

    if normalized.endswith(
        _RSS_EXPLICIT_ORDER_SUFFIXES
    ):
        return "POSSIBLE_LONGITUDINAL_ORDER"

    return "MANUAL_ORDER_SEMANTIC_REVIEW"


def _evaluate_rss_order_value_type(
    series: pd.Series,
) -> dict[str, object]:
    """
    Evaluate whether a candidate contains usable numeric,
    datetime, or sequence-like values.
    """

    non_null = series.dropna()

    if non_null.empty:
        return {
            "order_value_type":
                "NO_OBSERVED_VALUES",
            "parse_success_percent":
                0.0,
        }

    if pd.api.types.is_datetime64_any_dtype(
        series
    ):
        return {
            "order_value_type":
                "DATETIME",
            "parse_success_percent":
                100.0,
        }

    if pd.api.types.is_numeric_dtype(
        series
    ):
        return {
            "order_value_type":
                "NUMERIC_OR_SEQUENCE",
            "parse_success_percent":
                100.0,
        }

    datetime_values = pd.to_datetime(
        non_null,
        errors="coerce",
    )

    datetime_success = round(
        datetime_values.notna().mean()
        * 100,
        3,
    )

    numeric_values = pd.to_numeric(
        non_null,
        errors="coerce",
    )

    numeric_success = round(
        numeric_values.notna().mean()
        * 100,
        3,
    )

    if datetime_success >= numeric_success:
        return {
            "order_value_type":
                "DATETIME_LIKE_TEXT",
            "parse_success_percent":
                datetime_success,
        }

    return {
        "order_value_type":
            "NUMERIC_LIKE_TEXT",
        "parse_success_percent":
            numeric_success,
    }


def build_rss_longitudinal_order_validation(
    qualification_df: pd.DataFrame,
    canonical_df: pd.DataFrame,
    canonical_source: str = "LPMR_SOURCE",
    patient_column: str = "patient_id",
    encounter_column: str = "encounter_id",
) -> pd.DataFrame:
    """
    Validate longitudinal-order candidates available through the
    canonical RSS replay substrate.

    This function evaluates semantics, completeness, value type,
    within-encounter consistency, patient-level variation, and
    tied order values.

    It does not automatically select a final canonical order field
    or form replay cycles.
    """

    if not isinstance(
        qualification_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "qualification_df must be a pandas DataFrame."
        )

    if not isinstance(
        canonical_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "canonical_df must be a pandas DataFrame."
        )

    required_qualification_columns = {
        "source",
        "column",
        "candidate_role",
        "dtype",
        "non_null_count",
        "non_null_percent",
        "unique_count",
        "sample_values",
    }

    missing_qualification_columns = sorted(
        required_qualification_columns
        - set(qualification_df.columns)
    )

    if missing_qualification_columns:
        raise KeyError(
            "RSS longitudinal-order validation cannot proceed. "
            "Missing qualification columns: "
            f"{missing_qualification_columns}"
        )

    missing_key_columns = [
        column
        for column in (
            patient_column,
            encounter_column,
        )
        if column not in canonical_df.columns
    ]

    if missing_key_columns:
        raise KeyError(
            "Canonical replay dataframe is missing key columns: "
            f"{missing_key_columns}"
        )

    candidates = (
        qualification_df.loc[
            qualification_df[
                "source"
            ].eq(canonical_source)
            &
            qualification_df[
                "candidate_role"
            ].eq(
                "LONGITUDINAL_ORDER_CANDIDATE"
            )
        ]
        .drop_duplicates(
            subset="column"
        )
        .copy()
    )

    rows: list[dict[str, object]] = []

    for candidate in candidates.itertuples(
        index=False
    ):
        column = candidate.column

        if column not in canonical_df.columns:
            continue

        series = canonical_df[column]

        semantic_state = (
            _classify_rss_order_semantics(
                column
            )
        )

        value_type_review = (
            _evaluate_rss_order_value_type(
                series
            )
        )

        non_null_count = int(
            series.notna().sum()
        )

        non_null_percent = round(
            series.notna().mean()
            * 100,
            3,
        )

        unique_count = int(
            series.nunique(
                dropna=True
            )
        )

        review_frame = canonical_df[
            [
                patient_column,
                encounter_column,
                column,
            ]
        ].dropna(
            subset=[
                patient_column,
                encounter_column,
            ]
        )

        encounter_value_counts = (
            review_frame
            .groupby(
                [
                    patient_column,
                    encounter_column,
                ],
                dropna=False,
            )[column]
            .nunique(
                dropna=True
            )
        )

        encounter_conflict_count = int(
            encounter_value_counts
            .gt(1)
            .sum()
        )

        encounter_level = (
            review_frame
            .dropna(
                subset=[column]
            )
            .drop_duplicates(
                subset=[
                    patient_column,
                    encounter_column,
                    column,
                ]
            )
            .groupby(
                [
                    patient_column,
                    encounter_column,
                ],
                as_index=False,
                sort=False,
            )
            .first()
        )

        patient_value_counts = (
            encounter_level
            .groupby(
                patient_column,
                dropna=False,
            )[column]
            .nunique(
                dropna=True
            )
        )

        patients_with_multiple_values = int(
            patient_value_counts
            .gt(1)
            .sum()
        )

        patient_order_tie_rows = int(
            encounter_level
            .duplicated(
                subset=[
                    patient_column,
                    column,
                ],
                keep=False,
            )
            .sum()
        )

        parse_success_percent = float(
            value_type_review[
                "parse_success_percent"
            ]
        )

        if (
            semantic_state
            == "TEMPORAL_CONTEXT_NOT_ORDER"
        ):
            validation_state = (
                "TEMPORAL_CONTEXT_NOT_ORDER_ANCHOR"
            )

        elif parse_success_percent < 95.0:
            validation_state = (
                "UNPARSEABLE_ORDER_CANDIDATE"
            )

        elif non_null_percent < 95.0:
            validation_state = (
                "INCOMPLETE_ORDER_CANDIDATE"
            )

        elif encounter_conflict_count > 0:
            validation_state = (
                "ENCOUNTER_VALUE_CONFLICT_"
                "REVIEW_REQUIRED"
            )

        elif patients_with_multiple_values == 0:
            validation_state = (
                "LOW_VARIATION_ORDER_CANDIDATE"
            )

        elif (
            semantic_state
            == "MANUAL_ORDER_SEMANTIC_REVIEW"
        ):
            validation_state = (
                "MANUAL_ORDER_SEMANTIC_"
                "REVIEW_REQUIRED"
            )

        elif patient_order_tie_rows > 0:
            validation_state = (
                "ELIGIBLE_WITH_TIE_BREAKER_REQUIRED"
            )

        else:
            validation_state = (
                "ELIGIBLE_CANONICAL_ORDER_CANDIDATE"
            )

        rows.append({
            "source":
                canonical_source,
            "column":
                column,
            "semantic_state":
                semantic_state,
            "order_value_type":
                value_type_review[
                    "order_value_type"
                ],
            "parse_success_percent":
                parse_success_percent,
            "non_null_count":
                non_null_count,
            "non_null_percent":
                non_null_percent,
            "unique_count":
                unique_count,
            "encounter_value_conflict_count":
                encounter_conflict_count,
            "patients_with_multiple_order_values":
                patients_with_multiple_values,
            "patient_order_tie_rows":
                patient_order_tie_rows,
            "validation_state":
                validation_state,
            "canonical_order_selected":
                False,
            "sample_values":
                candidate.sample_values,
        })

    result_columns = [
        "source",
        "column",
        "semantic_state",
        "order_value_type",
        "parse_success_percent",
        "non_null_count",
        "non_null_percent",
        "unique_count",
        "encounter_value_conflict_count",
        "patients_with_multiple_order_values",
        "patient_order_tie_rows",
        "validation_state",
        "canonical_order_selected",
        "sample_values",
    ]

    if not rows:
        return pd.DataFrame(
            columns=result_columns
        )

    result = pd.DataFrame(rows)

    result["_validation_priority"] = (
        result[
            "validation_state"
        ]
        .map(
            _RSS_ORDER_VALIDATION_PRIORITY
        )
        .fillna(99)
    )

    return (
        result
        .sort_values(
            by=[
                "_validation_priority",
                "non_null_percent",
                "unique_count",
                "column",
            ],
            ascending=[
                True,
                False,
                False,
                True,
            ],
        )
        .drop(
            columns="_validation_priority"
        )
        .reset_index(drop=True)
    )


def build_rss_longitudinal_order_validation_summary(
    order_validation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize RSS longitudinal-order validation outcomes.
    """

    if not isinstance(
        order_validation_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "order_validation_df must be a pandas DataFrame."
        )

    required_columns = {
        "semantic_state",
        "order_value_type",
        "validation_state",
        "canonical_order_selected",
    }

    missing_columns = sorted(
        required_columns
        - set(order_validation_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "RSS order validation summary cannot be built. "
            f"Missing columns: {missing_columns}"
        )

    return (
        order_validation_df
        .groupby(
            [
                "validation_state",
                "semantic_state",
                "order_value_type",
                "canonical_order_selected",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            by=[
                "canonical_order_selected",
                "count",
                "validation_state",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

#============================================================
# Chapter 68.16 — RSS Canonical Ordering Candidate Resolution
#============================================================

_RSS_ORDER_RESOLUTION_PRIORITY = {
    "ENCOUNTER_CHRONOLOGY_CANDIDATE": 0,
    "ENCOUNTER_START_TIEBREAKER_REQUIRED": 1,
    "ENCOUNTER_INTERVAL_OVERLAP_REVIEW_REQUIRED": 2,
    (
        "ENCOUNTER_START_TIEBREAKER_AND_"
        "OVERLAP_REVIEW_REQUIRED"
    ): 3,
    "INCOMPLETE_ENCOUNTER_COVERAGE": 4,
    "LOW_PATIENT_LONGITUDINAL_VARIATION": 5,
    "UNRESOLVED_TEMPORAL_SEMANTICS": 6,
    "UNRESOLVED_UNPARSEABLE_VALUES": 7,
}


def _coerce_rss_order_candidate(
    series: pd.Series,
    preferred_value_type: str,
) -> tuple[pd.Series, str]:
    """
    Coerce an RSS ordering candidate into numeric or datetime form.

    The preferred type is taken from the prior ordering validation.
    No values are manufactured when parsing fails.
    """

    if pd.api.types.is_datetime64_any_dtype(
        series
    ):
        return (
            pd.to_datetime(
                series,
                errors="coerce",
            ),
            "DATETIME",
        )

    if pd.api.types.is_numeric_dtype(
        series
    ):
        return (
            pd.to_numeric(
                series,
                errors="coerce",
            ),
            "NUMERIC_OR_SEQUENCE",
        )

    if preferred_value_type in {
        "NUMERIC_OR_SEQUENCE",
        "NUMERIC_LIKE_TEXT",
    }:
        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        return (
            numeric,
            "NUMERIC_LIKE_TEXT",
        )

    datetime_values = pd.to_datetime(
        series,
        errors="coerce",
    )

    return (
        datetime_values,
        "DATETIME_LIKE_TEXT",
    )


def build_rss_order_candidate_resolution(
    order_validation_df: pd.DataFrame,
    canonical_df: pd.DataFrame,
    patient_column: str = "patient_id",
    encounter_column: str = "encounter_id",
) -> pd.DataFrame:
    """
    Determine whether temporal candidates can support an honest
    encounter-level chronology.

    Multiple values within an encounter are summarized as an
    observed start and end range.

    This function does not automatically select the canonical RSS
    ordering field and does not form replay cycles.
    """

    if not isinstance(
        order_validation_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "order_validation_df must be a pandas DataFrame."
        )

    if not isinstance(
        canonical_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "canonical_df must be a pandas DataFrame."
        )

    required_validation_columns = {
        "column",
        "semantic_state",
        "order_value_type",
        "parse_success_percent",
        "non_null_percent",
        "validation_state",
    }

    missing_validation_columns = sorted(
        required_validation_columns
        - set(order_validation_df.columns)
    )

    if missing_validation_columns:
        raise KeyError(
            "RSS ordering candidate resolution cannot proceed. "
            "Missing validation columns: "
            f"{missing_validation_columns}"
        )

    missing_key_columns = [
        column
        for column in (
            patient_column,
            encounter_column,
        )
        if column not in canonical_df.columns
    ]

    if missing_key_columns:
        raise KeyError(
            "Canonical RSS dataframe is missing key columns: "
            f"{missing_key_columns}"
        )

    rows: list[dict[str, object]] = []

    for candidate in (
        order_validation_df
        .drop_duplicates(
            subset="column"
        )
        .itertuples(index=False)
    ):
        column = candidate.column

        if column not in canonical_df.columns:
            continue

        review = canonical_df[
            [
                patient_column,
                encounter_column,
                column,
            ]
        ].copy()

        for key in (
            patient_column,
            encounter_column,
        ):
            review[key] = (
                review[key]
                .astype("string")
                .str.strip()
            )

        coerced_values, resolved_value_type = (
            _coerce_rss_order_candidate(
                series=review[column],
                preferred_value_type=(
                    candidate.order_value_type
                ),
            )
        )

        review["_rss_order_value"] = (
            coerced_values
        )

        total_encounter_count = int(
            review[
                [
                    patient_column,
                    encounter_column,
                ]
            ]
            .drop_duplicates()
            .shape[0]
        )

        observed = review.dropna(
            subset=["_rss_order_value"]
        )

        observed_encounter_count = int(
            observed[
                [
                    patient_column,
                    encounter_column,
                ]
            ]
            .drop_duplicates()
            .shape[0]
        )

        encounter_coverage_percent = (
            round(
                (
                    observed_encounter_count
                    / total_encounter_count
                )
                * 100,
                3,
            )
            if total_encounter_count
            else 0.0
        )

        parse_success_percent = (
            round(
                review[
                    "_rss_order_value"
                ]
                .notna()
                .mean()
                * 100,
                3,
            )
        )

        # ----------------------------------------------------
        # Build one temporal range per patient encounter.
        # ----------------------------------------------------

        encounter_ranges = (
            observed
            .groupby(
                [
                    patient_column,
                    encounter_column,
                ],
                as_index=False,
                sort=False,
                dropna=False,
            )
            .agg(
                rss_encounter_start=(
                    "_rss_order_value",
                    "min",
                ),
                rss_encounter_end=(
                    "_rss_order_value",
                    "max",
                ),
                rss_encounter_unique_order_values=(
                    "_rss_order_value",
                    "nunique",
                ),
                rss_encounter_observation_count=(
                    "_rss_order_value",
                    "size",
                ),
            )
        )

        encounters_with_temporal_variation = int(
            encounter_ranges[
                "rss_encounter_unique_order_values"
            ]
            .gt(1)
            .sum()
        )

        zero_span_encounters = int(
            encounter_ranges[
                "rss_encounter_start"
            ]
            .eq(
                encounter_ranges[
                    "rss_encounter_end"
                ]
            )
            .sum()
        )

        # ----------------------------------------------------
        # Evaluate patient-level longitudinal usefulness.
        # ----------------------------------------------------

        patient_encounter_counts = (
            encounter_ranges
            .groupby(
                patient_column,
                dropna=False,
            )[encounter_column]
            .nunique()
        )

        patients_with_multiple_encounters = int(
            patient_encounter_counts
            .gt(1)
            .sum()
        )

        patient_start_tie_rows = int(
            encounter_ranges
            .duplicated(
                subset=[
                    patient_column,
                    "rss_encounter_start",
                ],
                keep=False,
            )
            .sum()
        )

        # ----------------------------------------------------
        # Detect interval overlap after chronological sorting.
        # ----------------------------------------------------

        chronology = (
            encounter_ranges
            .sort_values(
                by=[
                    patient_column,
                    "rss_encounter_start",
                    "rss_encounter_end",
                    encounter_column,
                ]
            )
            .reset_index(drop=True)
        )

        chronology[
            "_previous_encounter_end"
        ] = (
            chronology
            .groupby(
                patient_column,
                dropna=False,
            )["rss_encounter_end"]
            .shift(1)
        )

        overlap_mask = (
            chronology[
                "_previous_encounter_end"
            ]
            .notna()
            &
            chronology[
                "rss_encounter_start"
            ]
            .lt(
                chronology[
                    "_previous_encounter_end"
                ]
            )
        )

        overlapping_encounter_count = int(
            overlap_mask.sum()
        )

        # ----------------------------------------------------
        # Resolve structural chronology state.
        # ----------------------------------------------------

        semantic_state = (
            candidate.semantic_state
        )

        if parse_success_percent < 95.0:
            resolution_state = (
                "UNRESOLVED_UNPARSEABLE_VALUES"
            )

        elif encounter_coverage_percent < 95.0:
            resolution_state = (
                "INCOMPLETE_ENCOUNTER_COVERAGE"
            )

        elif patients_with_multiple_encounters == 0:
            resolution_state = (
                "LOW_PATIENT_LONGITUDINAL_VARIATION"
            )

        elif semantic_state == (
            "MANUAL_ORDER_SEMANTIC_REVIEW"
        ):
            resolution_state = (
                "UNRESOLVED_TEMPORAL_SEMANTICS"
            )

        elif (
            patient_start_tie_rows > 0
            and overlapping_encounter_count > 0
        ):
            resolution_state = (
                "ENCOUNTER_START_TIEBREAKER_AND_"
                "OVERLAP_REVIEW_REQUIRED"
            )

        elif patient_start_tie_rows > 0:
            resolution_state = (
                "ENCOUNTER_START_TIEBREAKER_REQUIRED"
            )

        elif overlapping_encounter_count > 0:
            resolution_state = (
                "ENCOUNTER_INTERVAL_OVERLAP_"
                "REVIEW_REQUIRED"
            )

        else:
            resolution_state = (
                "ENCOUNTER_CHRONOLOGY_CANDIDATE"
            )

        rows.append({
            "column":
                column,

            "prior_validation_state":
                candidate.validation_state,

            "semantic_state":
                semantic_state,

            "resolved_order_value_type":
                resolved_value_type,

            "parse_success_percent":
                parse_success_percent,

            "total_encounter_count":
                total_encounter_count,

            "observed_encounter_count":
                observed_encounter_count,

            "encounter_coverage_percent":
                encounter_coverage_percent,

            "encounters_with_temporal_variation":
                encounters_with_temporal_variation,

            "zero_span_encounter_count":
                zero_span_encounters,

            "patients_with_multiple_encounters":
                patients_with_multiple_encounters,

            "patient_start_tie_rows":
                patient_start_tie_rows,

            "overlapping_encounter_count":
                overlapping_encounter_count,

            "resolution_state":
                resolution_state,

            "canonical_order_selected":
                False,
        })

    result_columns = [
        "column",
        "prior_validation_state",
        "semantic_state",
        "resolved_order_value_type",
        "parse_success_percent",
        "total_encounter_count",
        "observed_encounter_count",
        "encounter_coverage_percent",
        "encounters_with_temporal_variation",
        "zero_span_encounter_count",
        "patients_with_multiple_encounters",
        "patient_start_tie_rows",
        "overlapping_encounter_count",
        "resolution_state",
        "canonical_order_selected",
    ]

    if not rows:
        return pd.DataFrame(
            columns=result_columns
        )

    result = pd.DataFrame(rows)

    result["_resolution_priority"] = (
        result[
            "resolution_state"
        ]
        .map(
            _RSS_ORDER_RESOLUTION_PRIORITY
        )
        .fillna(99)
    )

    return (
        result
        .sort_values(
            by=[
                "_resolution_priority",
                "encounter_coverage_percent",
                "patient_start_tie_rows",
                "overlapping_encounter_count",
                "column",
            ],
            ascending=[
                True,
                False,
                True,
                True,
                True,
            ],
        )
        .drop(
            columns="_resolution_priority"
        )
        .reset_index(drop=True)
    )


def build_rss_order_candidate_resolution_summary(
    resolution_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize RSS ordering candidate resolution outcomes.
    """

    if not isinstance(
        resolution_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "resolution_df must be a pandas DataFrame."
        )

    required_columns = {
        "resolution_state",
        "semantic_state",
        "resolved_order_value_type",
        "canonical_order_selected",
    }

    missing_columns = sorted(
        required_columns
        - set(resolution_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "RSS order resolution summary cannot be built. "
            f"Missing columns: {missing_columns}"
        )

    return (
        resolution_df
        .groupby(
            [
                "resolution_state",
                "semantic_state",
                "resolved_order_value_type",
                "canonical_order_selected",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            by=[
                "canonical_order_selected",
                "count",
                "resolution_state",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    #============================================================
# Chapter 68.17 — RSS Chronology Candidate Agreement Review
#============================================================

_RSS_CHRONOLOGY_AGREEMENT_PRIORITY = {
    "ORDER_POSITION_MATCH": 0,
    "ORDER_DIFFERENCE_TIE_RELATED": 1,
    "ORDER_DIFFERENCE_STRICT_REVIEW_REQUIRED": 2,
    "WITHIN_ENCOUNTER_ORDER_CONFLICT_REVIEW_REQUIRED": 3,
    "MISSING_ORDER_VALUE": 4,
}


def build_rss_chronology_candidate_agreement(
    canonical_df: pd.DataFrame,
    patient_column: str = "patient_id",
    encounter_column: str = "encounter_id",
    temporal_column: str = "years_since_sim_start",
    sequence_column: str = "encounter_sequence",
) -> pd.DataFrame:
    """
    Compare the strongest RSS longitudinal-order candidates.

    The comparison evaluates:
    - encounter-level completeness
    - within-encounter consistency
    - independently generated order positions
    - tie-related ordering differences
    - strict ordering differences
    - local chronological reversals
    - patient-level agreement

    encounter_id is used only as a neutral deterministic
    comparison tie-breaker.

    This function does not select the canonical RSS order field
    and does not form replay cycles.
    """

    if not isinstance(
        canonical_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "canonical_df must be a pandas DataFrame."
        )

    required_columns = {
        patient_column,
        encounter_column,
        temporal_column,
        sequence_column,
    }

    missing_columns = sorted(
        required_columns
        - set(canonical_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "RSS chronology agreement review cannot proceed. "
            f"Missing columns: {missing_columns}"
        )

    working = canonical_df[
        [
            patient_column,
            encounter_column,
            temporal_column,
            sequence_column,
        ]
    ].copy()

    for key in (
        patient_column,
        encounter_column,
    ):
        working[key] = (
            working[key]
            .astype("string")
            .str.strip()
        )

    missing_key_mask = (
        working[
            [
                patient_column,
                encounter_column,
            ]
        ]
        .isna()
        .any(axis=1)
        |
        working[
            [
                patient_column,
                encounter_column,
            ]
        ]
        .eq("")
        .any(axis=1)
    )

    if missing_key_mask.any():
        raise ValueError(
            "Canonical RSS replay substrate contains "
            f"{int(missing_key_mask.sum())} rows with "
            "missing patient or encounter identifiers."
        )

    temporal_values, _ = (
        _coerce_rss_order_candidate(
            series=working[temporal_column],
            preferred_value_type=(
                "NUMERIC_OR_SEQUENCE"
            ),
        )
    )

    sequence_values, _ = (
        _coerce_rss_order_candidate(
            series=working[sequence_column],
            preferred_value_type=(
                "NUMERIC_OR_SEQUENCE"
            ),
        )
    )

    working[
        "_rss_temporal_value"
    ] = temporal_values

    working[
        "_rss_sequence_value"
    ] = sequence_values

    # --------------------------------------------------------
    # Reduce source rows to one patient-encounter record while
    # preserving within-encounter variation.
    # --------------------------------------------------------

    encounter_agreement = (
        working
        .groupby(
            [
                patient_column,
                encounter_column,
            ],
            as_index=False,
            sort=False,
            dropna=False,
        )
        .agg(
            rss_source_row_count=(
                "_rss_temporal_value",
                "size",
            ),
            rss_temporal_non_null_count=(
                "_rss_temporal_value",
                "count",
            ),
            rss_temporal_unique_count=(
                "_rss_temporal_value",
                lambda values: int(
                    values.nunique(
                        dropna=True
                    )
                ),
            ),
            rss_temporal_start=(
                "_rss_temporal_value",
                "min",
            ),
            rss_temporal_end=(
                "_rss_temporal_value",
                "max",
            ),
            rss_sequence_non_null_count=(
                "_rss_sequence_value",
                "count",
            ),
            rss_sequence_unique_count=(
                "_rss_sequence_value",
                lambda values: int(
                    values.nunique(
                        dropna=True
                    )
                ),
            ),
            rss_sequence_start=(
                "_rss_sequence_value",
                "min",
            ),
            rss_sequence_end=(
                "_rss_sequence_value",
                "max",
            ),
        )
    )

    encounter_agreement[
        "rss_temporal_within_encounter_conflict"
    ] = (
        encounter_agreement[
            "rss_temporal_unique_count"
        ]
        .gt(1)
    )

    encounter_agreement[
        "rss_sequence_within_encounter_conflict"
    ] = (
        encounter_agreement[
            "rss_sequence_unique_count"
        ]
        .gt(1)
    )

    encounter_agreement[
        "rss_order_value_missing"
    ] = (
        encounter_agreement[
            "rss_temporal_start"
        ]
        .isna()
        |
        encounter_agreement[
            "rss_sequence_start"
        ]
        .isna()
    )

    encounter_agreement[
        "rss_within_encounter_order_conflict"
    ] = (
        encounter_agreement[
            "rss_temporal_within_encounter_conflict"
        ]
        |
        encounter_agreement[
            "rss_sequence_within_encounter_conflict"
        ]
    )

    encounter_agreement[
        "rss_chronology_agreement_eligible"
    ] = (
        ~encounter_agreement[
            "rss_order_value_missing"
        ]
        &
        ~encounter_agreement[
            "rss_within_encounter_order_conflict"
        ]
    )

    eligible = encounter_agreement.loc[
        encounter_agreement[
            "rss_chronology_agreement_eligible"
        ],
        [
            patient_column,
            encounter_column,
            "rss_temporal_start",
            "rss_sequence_start",
        ],
    ].copy()

    # --------------------------------------------------------
    # Generate independent deterministic order positions.
    #
    # encounter_id is used only as a neutral diagnostic
    # tie-breaker. It is not selected as replay chronology.
    # --------------------------------------------------------

    temporal_order = (
        eligible
        .sort_values(
            by=[
                patient_column,
                "rss_temporal_start",
                encounter_column,
            ],
            kind="mergesort",
        )
        .copy()
    )

    temporal_order[
        "rss_temporal_order_position"
    ] = (
        temporal_order
        .groupby(
            patient_column,
            dropna=False,
        )
        .cumcount()
        .add(1)
        .astype("Int64")
    )

    sequence_order = (
        eligible
        .sort_values(
            by=[
                patient_column,
                "rss_sequence_start",
                encounter_column,
            ],
            kind="mergesort",
        )
        .copy()
    )

    sequence_order[
        "rss_sequence_order_position"
    ] = (
        sequence_order
        .groupby(
            patient_column,
            dropna=False,
        )
        .cumcount()
        .add(1)
        .astype("Int64")
    )

    order_positions = (
        temporal_order[
            [
                patient_column,
                encounter_column,
                "rss_temporal_order_position",
            ]
        ]
        .merge(
            sequence_order[
                [
                    patient_column,
                    encounter_column,
                    "rss_sequence_order_position",
                ]
            ],
            on=[
                patient_column,
                encounter_column,
            ],
            how="outer",
            validate="one_to_one",
        )
    )

    # --------------------------------------------------------
    # Identify tied values within each patient's chronology.
    # --------------------------------------------------------

    tie_review = eligible[
        [
            patient_column,
            encounter_column,
            "rss_temporal_start",
            "rss_sequence_start",
        ]
    ].copy()

    tie_review[
        "rss_temporal_patient_tie_flag"
    ] = tie_review.duplicated(
        subset=[
            patient_column,
            "rss_temporal_start",
        ],
        keep=False,
    )

    tie_review[
        "rss_sequence_patient_tie_flag"
    ] = tie_review.duplicated(
        subset=[
            patient_column,
            "rss_sequence_start",
        ],
        keep=False,
    )

    # --------------------------------------------------------
    # Detect local strict reversal:
    #
    # time moves forward while encounter sequence moves backward.
    # --------------------------------------------------------

    reversal_review = (
        eligible
        .sort_values(
            by=[
                patient_column,
                "rss_temporal_start",
                encounter_column,
            ],
            kind="mergesort",
        )
        .copy()
    )

    reversal_review[
        "_rss_previous_temporal_value"
    ] = (
        reversal_review
        .groupby(
            patient_column,
            dropna=False,
        )[
            "rss_temporal_start"
        ]
        .shift(1)
    )

    reversal_review[
        "_rss_previous_sequence_value"
    ] = (
        reversal_review
        .groupby(
            patient_column,
            dropna=False,
        )[
            "rss_sequence_start"
        ]
        .shift(1)
    )

    reversal_review[
        "rss_local_strict_order_reversal_flag"
    ] = (
        reversal_review[
            "rss_temporal_start"
        ]
        .gt(
            reversal_review[
                "_rss_previous_temporal_value"
            ]
        )
        &
        reversal_review[
            "rss_sequence_start"
        ]
        .lt(
            reversal_review[
                "_rss_previous_sequence_value"
            ]
        )
    )

    reversal_review = reversal_review[
        [
            patient_column,
            encounter_column,
            "rss_local_strict_order_reversal_flag",
        ]
    ]

    # --------------------------------------------------------
    # Attach comparison evidence to encounter contracts.
    # --------------------------------------------------------

    encounter_agreement = (
        encounter_agreement
        .merge(
            order_positions,
            on=[
                patient_column,
                encounter_column,
            ],
            how="left",
            validate="one_to_one",
        )
        .merge(
            tie_review[
                [
                    patient_column,
                    encounter_column,
                    "rss_temporal_patient_tie_flag",
                    "rss_sequence_patient_tie_flag",
                ]
            ],
            on=[
                patient_column,
                encounter_column,
            ],
            how="left",
            validate="one_to_one",
        )
        .merge(
            reversal_review,
            on=[
                patient_column,
                encounter_column,
            ],
            how="left",
            validate="one_to_one",
        )
    )

    for column in (
        "rss_temporal_patient_tie_flag",
        "rss_sequence_patient_tie_flag",
        "rss_local_strict_order_reversal_flag",
    ):
        encounter_agreement[column] = (
            encounter_agreement[column]
            .fillna(False)
            .astype(bool)
        )

    encounter_agreement[
        "rss_order_position_delta"
    ] = (
        encounter_agreement[
            "rss_sequence_order_position"
        ]
        -
        encounter_agreement[
            "rss_temporal_order_position"
        ]
    ).astype("Int64")

    encounter_agreement[
        "rss_absolute_order_position_delta"
    ] = (
        encounter_agreement[
            "rss_order_position_delta"
        ]
        .abs()
        .astype("Int64")
    )

    encounter_agreement[
        "rss_tie_related_difference_flag"
    ] = (
        encounter_agreement[
            "rss_temporal_patient_tie_flag"
        ]
        |
        encounter_agreement[
            "rss_sequence_patient_tie_flag"
        ]
    )

    # --------------------------------------------------------
    # Assign encounter-level comparison state.
    # --------------------------------------------------------

    encounter_agreement[
        "rss_chronology_agreement_state"
    ] = "ORDER_REVIEW_UNRESOLVED"

    missing_order_mask = (
        encounter_agreement[
            "rss_order_value_missing"
        ]
    )

    within_encounter_conflict_mask = (
        ~missing_order_mask
        &
        encounter_agreement[
            "rss_within_encounter_order_conflict"
        ]
    )

    eligible_mask = (
        encounter_agreement[
            "rss_chronology_agreement_eligible"
        ]
    )

    exact_match_mask = (
        eligible_mask
        &
        encounter_agreement[
            "rss_order_position_delta"
        ]
        .eq(0)
    )

    strict_difference_mask = (
        eligible_mask
        &
        (
            (
                encounter_agreement[
                    "rss_order_position_delta"
                ]
                .ne(0)
                &
                ~encounter_agreement[
                    "rss_tie_related_difference_flag"
                ]
            )
            |
            encounter_agreement[
                "rss_local_strict_order_reversal_flag"
            ]
        )
    )

    tie_related_difference_mask = (
        eligible_mask
        &
        encounter_agreement[
            "rss_order_position_delta"
        ]
        .ne(0)
        &
        encounter_agreement[
            "rss_tie_related_difference_flag"
        ]
        &
        ~strict_difference_mask
    )

    encounter_agreement.loc[
        missing_order_mask,
        "rss_chronology_agreement_state",
    ] = "MISSING_ORDER_VALUE"

    encounter_agreement.loc[
        within_encounter_conflict_mask,
        "rss_chronology_agreement_state",
    ] = (
        "WITHIN_ENCOUNTER_ORDER_CONFLICT_"
        "REVIEW_REQUIRED"
    )

    encounter_agreement.loc[
        exact_match_mask,
        "rss_chronology_agreement_state",
    ] = "ORDER_POSITION_MATCH"

    encounter_agreement.loc[
        tie_related_difference_mask,
        "rss_chronology_agreement_state",
    ] = "ORDER_DIFFERENCE_TIE_RELATED"

    encounter_agreement.loc[
        strict_difference_mask,
        "rss_chronology_agreement_state",
    ] = (
        "ORDER_DIFFERENCE_STRICT_"
        "REVIEW_REQUIRED"
    )

    # --------------------------------------------------------
    # Patient-level agreement state.
    # --------------------------------------------------------

    patient_review = (
        encounter_agreement
        .groupby(
            patient_column,
            as_index=False,
            dropna=False,
        )
        .agg(
            rss_patient_encounter_count=(
                encounter_column,
                "nunique",
            ),
            rss_patient_has_missing_order=(
                "rss_chronology_agreement_state",
                lambda values: bool(
                    values.eq(
                        "MISSING_ORDER_VALUE"
                    ).any()
                ),
            ),
            rss_patient_has_within_encounter_conflict=(
                "rss_chronology_agreement_state",
                lambda values: bool(
                    values.eq(
                        "WITHIN_ENCOUNTER_ORDER_"
                        "CONFLICT_REVIEW_REQUIRED"
                    ).any()
                ),
            ),
            rss_patient_has_strict_difference=(
                "rss_chronology_agreement_state",
                lambda values: bool(
                    values.eq(
                        "ORDER_DIFFERENCE_STRICT_"
                        "REVIEW_REQUIRED"
                    ).any()
                ),
            ),
            rss_patient_all_positions_match=(
                "rss_chronology_agreement_state",
                lambda values: bool(
                    values.eq(
                        "ORDER_POSITION_MATCH"
                    ).all()
                ),
            ),
        )
    )

    patient_review[
        "rss_patient_chronology_agreement_state"
    ] = (
        "PATIENT_ORDER_TIE_RELATED_DIFFERENCE"
    )

    patient_review.loc[
        patient_review[
            "rss_patient_all_positions_match"
        ],
        "rss_patient_chronology_agreement_state",
    ] = "PATIENT_ORDER_FULLY_ALIGNED"

    single_encounter_mask = (
        patient_review[
            "rss_patient_encounter_count"
        ]
        .eq(1)
        &
        ~patient_review[
            "rss_patient_has_missing_order"
        ]
        &
        ~patient_review[
            "rss_patient_has_within_encounter_conflict"
        ]
    )

    patient_review.loc[
        single_encounter_mask,
        "rss_patient_chronology_agreement_state",
    ] = (
        "PATIENT_SINGLE_ENCOUNTER_"
        "ORDER_NOT_TESTABLE"
    )

    patient_review.loc[
        patient_review[
            "rss_patient_has_strict_difference"
        ],
        "rss_patient_chronology_agreement_state",
    ] = (
        "PATIENT_ORDER_STRICT_DISAGREEMENT_"
        "REVIEW_REQUIRED"
    )

    patient_review.loc[
        patient_review[
            "rss_patient_has_within_encounter_conflict"
        ],
        "rss_patient_chronology_agreement_state",
    ] = (
        "PATIENT_ORDER_WITHIN_ENCOUNTER_"
        "CONFLICT_REVIEW_REQUIRED"
    )

    patient_review.loc[
        patient_review[
            "rss_patient_has_missing_order"
        ],
        "rss_patient_chronology_agreement_state",
    ] = "PATIENT_ORDER_INCOMPLETE"

    encounter_agreement = (
        encounter_agreement
        .merge(
            patient_review[
                [
                    patient_column,
                    "rss_patient_encounter_count",
                    (
                        "rss_patient_chronology_"
                        "agreement_state"
                    ),
                ]
            ],
            on=patient_column,
            how="left",
            validate="many_to_one",
        )
    )

    encounter_agreement[
        "rss_canonical_order_selected"
    ] = False

    encounter_agreement[
        "_agreement_priority"
    ] = (
        encounter_agreement[
            "rss_chronology_agreement_state"
        ]
        .map(
            _RSS_CHRONOLOGY_AGREEMENT_PRIORITY
        )
        .fillna(99)
    )

    return (
        encounter_agreement
        .sort_values(
            by=[
                "_agreement_priority",
                patient_column,
                "rss_temporal_order_position",
                encounter_column,
            ],
            ascending=[
                True,
                True,
                True,
                True,
            ],
            na_position="last",
        )
        .drop(
            columns="_agreement_priority"
        )
        .reset_index(drop=True)
    )


def build_rss_chronology_candidate_agreement_summary(
    agreement_df: pd.DataFrame,
    patient_column: str = "patient_id",
    encounter_column: str = "encounter_id",
    temporal_column: str = "years_since_sim_start",
    sequence_column: str = "encounter_sequence",
) -> pd.DataFrame:
    """
    Summarize agreement between the two strongest RSS chronology
    candidates.

    Canonical chronology is not authorized by this function.
    """

    if not isinstance(
        agreement_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "agreement_df must be a pandas DataFrame."
        )

    required_columns = {
        patient_column,
        encounter_column,
        "rss_chronology_agreement_eligible",
        "rss_chronology_agreement_state",
        "rss_local_strict_order_reversal_flag",
        "rss_patient_chronology_agreement_state",
    }

    missing_columns = sorted(
        required_columns
        - set(agreement_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "RSS chronology agreement summary cannot proceed. "
            f"Missing columns: {missing_columns}"
        )

    encounter_states = (
        agreement_df[
            "rss_chronology_agreement_state"
        ]
        .value_counts()
    )

    patient_states = (
        agreement_df[
            [
                patient_column,
                "rss_patient_chronology_agreement_state",
            ]
        ]
        .drop_duplicates(
            subset=patient_column
        )[
            "rss_patient_chronology_agreement_state"
        ]
        .value_counts()
    )

    strict_difference_count = int(
        encounter_states.get(
            "ORDER_DIFFERENCE_STRICT_"
            "REVIEW_REQUIRED",
            0,
        )
    )

    within_encounter_conflict_count = int(
        encounter_states.get(
            "WITHIN_ENCOUNTER_ORDER_"
            "CONFLICT_REVIEW_REQUIRED",
            0,
        )
    )

    missing_order_count = int(
        encounter_states.get(
            "MISSING_ORDER_VALUE",
            0,
        )
    )

    if (
        strict_difference_count == 0
        and
        within_encounter_conflict_count == 0
        and
        missing_order_count == 0
    ):
        comparison_state = (
            "CHRONOLOGY_CANDIDATES_CONCORDANT_"
            "TIEBREAKER_POLICY_REQUIRED"
        )
    else:
        comparison_state = (
            "CHRONOLOGY_CANDIDATE_"
            "DISAGREEMENT_REVIEW_REQUIRED"
        )

    return pd.DataFrame([
        {
            "temporal_candidate":
                temporal_column,

            "sequence_candidate":
                sequence_column,

            "comparison_state":
                comparison_state,

            "canonical_order_selected":
                False,

            "total_encounter_count":
                len(agreement_df),

            "eligible_encounter_count":
                int(
                    agreement_df[
                        "rss_chronology_agreement_eligible"
                    ]
                    .sum()
                ),

            "exact_position_match_count":
                int(
                    encounter_states.get(
                        "ORDER_POSITION_MATCH",
                        0,
                    )
                ),

            "tie_related_difference_count":
                int(
                    encounter_states.get(
                        "ORDER_DIFFERENCE_TIE_RELATED",
                        0,
                    )
                ),

            "strict_difference_count":
                strict_difference_count,

            "within_encounter_conflict_count":
                within_encounter_conflict_count,

            "missing_order_count":
                missing_order_count,

            "local_strict_reversal_count":
                int(
                    agreement_df[
                        "rss_local_strict_order_reversal_flag"
                    ]
                    .sum()
                ),

            "total_patient_count":
                int(
                    agreement_df[
                        patient_column
                    ]
                    .nunique()
                ),

            "fully_aligned_patient_count":
                int(
                    patient_states.get(
                        "PATIENT_ORDER_FULLY_ALIGNED",
                        0,
                    )
                ),

            "single_encounter_not_testable_count":
                int(
                    patient_states.get(
                        "PATIENT_SINGLE_ENCOUNTER_"
                        "ORDER_NOT_TESTABLE",
                        0,
                    )
                ),

            "tie_related_patient_count":
                int(
                    patient_states.get(
                        "PATIENT_ORDER_TIE_RELATED_"
                        "DIFFERENCE",
                        0,
                    )
                ),

            "strict_disagreement_patient_count":
                int(
                    patient_states.get(
                        "PATIENT_ORDER_STRICT_"
                        "DISAGREEMENT_REVIEW_REQUIRED",
                        0,
                    )
                ),

            "within_encounter_conflict_patient_count":
                int(
                    patient_states.get(
                        "PATIENT_ORDER_WITHIN_ENCOUNTER_"
                        "CONFLICT_REVIEW_REQUIRED",
                        0,
                    )
                ),

            "incomplete_order_patient_count":
                int(
                    patient_states.get(
                        "PATIENT_ORDER_INCOMPLETE",
                        0,
                    )
                ),
        }
    ])

    #============================================================
# Chapter 68.18 — RSS Chronology Exception Review
#============================================================

_RSS_CHRONOLOGY_EXCEPTION_STATES = {
    "ORDER_DIFFERENCE_TIE_RELATED",
    "ORDER_DIFFERENCE_STRICT_REVIEW_REQUIRED",
    "WITHIN_ENCOUNTER_ORDER_CONFLICT_REVIEW_REQUIRED",
    "MISSING_ORDER_VALUE",
}


_RSS_EXCEPTION_REVIEW_ROLE_PRIORITY = {
    "WITHIN_ENCOUNTER_CONFLICT_TRIGGER": 0,
    "STRICT_DIFFERENCE_TRIGGER": 1,
    "TIE_RELATED_DIFFERENCE_TRIGGER": 2,
    "MISSING_ORDER_VALUE_TRIGGER": 3,
    "REVERSAL_PAIR_COUNTERPART": 4,
    "AFFECTED_PATIENT_CONTEXT": 5,
}


def build_rss_chronology_exception_review(
    agreement_df: pd.DataFrame,
    patient_column: str = "patient_id",
    encounter_column: str = "encounter_id",
) -> pd.DataFrame:
    """
    Isolate chronology exceptions and preserve each affected
    patient's complete encounter history.

    The review identifies:
    - tie-related differences
    - strict position differences
    - within-encounter ordering conflicts
    - missing order values
    - both members of local reversal pairs
    - surrounding patient-history context

    This function does not repair chronology, remove encounters,
    or select the canonical RSS ordering field.
    """

    if not isinstance(
        agreement_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "agreement_df must be a pandas DataFrame."
        )

    required_columns = {
        patient_column,
        encounter_column,
        "rss_temporal_start",
        "rss_temporal_end",
        "rss_sequence_start",
        "rss_sequence_end",
        "rss_temporal_unique_count",
        "rss_sequence_unique_count",
        "rss_temporal_order_position",
        "rss_sequence_order_position",
        "rss_order_position_delta",
        "rss_absolute_order_position_delta",
        "rss_temporal_patient_tie_flag",
        "rss_sequence_patient_tie_flag",
        "rss_local_strict_order_reversal_flag",
        "rss_chronology_agreement_state",
        "rss_patient_chronology_agreement_state",
    }

    missing_columns = sorted(
        required_columns
        - set(agreement_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "RSS chronology exception review cannot proceed. "
            f"Missing agreement columns: {missing_columns}"
        )

    exception_mask = (
        agreement_df[
            "rss_chronology_agreement_state"
        ]
        .isin(
            _RSS_CHRONOLOGY_EXCEPTION_STATES
        )
    )

    reversal_mask = (
        agreement_df[
            "rss_local_strict_order_reversal_flag"
        ]
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )

    affected_patients = (
        agreement_df.loc[
            exception_mask | reversal_mask,
            patient_column,
        ]
        .dropna()
        .drop_duplicates()
    )

    review = (
        agreement_df.loc[
            agreement_df[
                patient_column
            ].isin(affected_patients)
        ]
        .copy()
    )

    if review.empty:
        review[
            "rss_exception_trigger_flag"
        ] = pd.Series(
            dtype=bool
        )

        review[
            "rss_reversal_pair_context_flag"
        ] = pd.Series(
            dtype=bool
        )

        review[
            "rss_exception_review_role"
        ] = pd.Series(
            dtype="string"
        )

        review[
            "rss_exception_review_state"
        ] = pd.Series(
            dtype="string"
        )

        review[
            "rss_canonical_order_selected"
        ] = pd.Series(
            dtype=bool
        )

        return review

    # --------------------------------------------------------
    # Sort each affected patient by temporal chronology.
    # --------------------------------------------------------

    review = (
        review
        .sort_values(
            by=[
                patient_column,
                "rss_temporal_order_position",
                "rss_sequence_order_position",
                encounter_column,
            ],
            kind="mergesort",
            na_position="last",
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Identify direct exception-trigger rows.
    # --------------------------------------------------------

    review[
        "rss_exception_trigger_flag"
    ] = (
        review[
            "rss_chronology_agreement_state"
        ]
        .isin(
            _RSS_CHRONOLOGY_EXCEPTION_STATES
        )
        |
        review[
            "rss_local_strict_order_reversal_flag"
        ]
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )

    # --------------------------------------------------------
    # Preserve both members of a local reversal pair.
    #
    # When the later temporal row contains the reversal flag,
    # the immediately preceding temporal row is also relevant.
    # --------------------------------------------------------

    next_row_reversal_flag = (
        review
        .groupby(
            patient_column,
            dropna=False,
        )[
            "rss_local_strict_order_reversal_flag"
        ]
        .shift(-1)
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )

    current_reversal_flag = (
        review[
            "rss_local_strict_order_reversal_flag"
        ]
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )

    review[
        "rss_reversal_pair_context_flag"
    ] = (
        current_reversal_flag
        |
        next_row_reversal_flag
    )

    # --------------------------------------------------------
    # Assign review roles.
    # --------------------------------------------------------

    review[
        "rss_exception_review_role"
    ] = "AFFECTED_PATIENT_CONTEXT"

    reversal_counterpart_mask = (
        review[
            "rss_reversal_pair_context_flag"
        ]
        &
        ~review[
            "rss_exception_trigger_flag"
        ]
    )

    review.loc[
        reversal_counterpart_mask,
        "rss_exception_review_role",
    ] = "REVERSAL_PAIR_COUNTERPART"

    review.loc[
        review[
            "rss_chronology_agreement_state"
        ].eq(
            "ORDER_DIFFERENCE_TIE_RELATED"
        ),
        "rss_exception_review_role",
    ] = "TIE_RELATED_DIFFERENCE_TRIGGER"

    review.loc[
        review[
            "rss_chronology_agreement_state"
        ].eq(
            "ORDER_DIFFERENCE_STRICT_"
            "REVIEW_REQUIRED"
        ),
        "rss_exception_review_role",
    ] = "STRICT_DIFFERENCE_TRIGGER"

    review.loc[
        review[
            "rss_chronology_agreement_state"
        ].eq(
            "WITHIN_ENCOUNTER_ORDER_"
            "CONFLICT_REVIEW_REQUIRED"
        ),
        "rss_exception_review_role",
    ] = "WITHIN_ENCOUNTER_CONFLICT_TRIGGER"

    review.loc[
        review[
            "rss_chronology_agreement_state"
        ].eq(
            "MISSING_ORDER_VALUE"
        ),
        "rss_exception_review_role",
    ] = "MISSING_ORDER_VALUE_TRIGGER"

    # --------------------------------------------------------
    # Attach neighboring encounter context.
    # --------------------------------------------------------

    patient_groups = review.groupby(
        patient_column,
        dropna=False,
    )

    review[
        "rss_previous_encounter_id"
    ] = patient_groups[
        encounter_column
    ].shift(1)

    review[
        "rss_next_encounter_id"
    ] = patient_groups[
        encounter_column
    ].shift(-1)

    review[
        "rss_previous_temporal_start"
    ] = patient_groups[
        "rss_temporal_start"
    ].shift(1)

    review[
        "rss_next_temporal_start"
    ] = patient_groups[
        "rss_temporal_start"
    ].shift(-1)

    review[
        "rss_previous_sequence_start"
    ] = patient_groups[
        "rss_sequence_start"
    ].shift(1)

    review[
        "rss_next_sequence_start"
    ] = patient_groups[
        "rss_sequence_start"
    ].shift(-1)

    review[
        "rss_temporal_gap_from_previous"
    ] = (
        review[
            "rss_temporal_start"
        ]
        -
        review[
            "rss_previous_temporal_start"
        ]
    )

    review[
        "rss_sequence_gap_from_previous"
    ] = (
        review[
            "rss_sequence_start"
        ]
        -
        review[
            "rss_previous_sequence_start"
        ]
    )

    # --------------------------------------------------------
    # Review state only—no automatic correction.
    # --------------------------------------------------------

    review[
        "rss_exception_review_state"
    ] = "AFFECTED_HISTORY_CONTEXT_PRESERVED"

    review.loc[
        review[
            "rss_exception_trigger_flag"
        ],
        "rss_exception_review_state",
    ] = "CHRONOLOGY_EXCEPTION_REQUIRES_REVIEW"

    review.loc[
        reversal_counterpart_mask,
        "rss_exception_review_state",
    ] = "REVERSAL_PAIR_CONTEXT_PRESERVED"

    review[
        "rss_canonical_order_selected"
    ] = False

    review[
        "_rss_exception_role_priority"
    ] = (
        review[
            "rss_exception_review_role"
        ]
        .map(
            _RSS_EXCEPTION_REVIEW_ROLE_PRIORITY
        )
        .fillna(99)
    )

    return (
        review
        .sort_values(
            by=[
                patient_column,
                "rss_temporal_order_position",
                "_rss_exception_role_priority",
                encounter_column,
            ],
            ascending=[
                True,
                True,
                True,
                True,
            ],
            na_position="last",
        )
        .drop(
            columns="_rss_exception_role_priority"
        )
        .reset_index(drop=True)
    )


def build_rss_chronology_exception_summary(
    exception_review_df: pd.DataFrame,
    patient_column: str = "patient_id",
) -> pd.DataFrame:
    """
    Summarize the isolated RSS chronology exception population.

    Canonical ordering remains unauthorized.
    """

    if not isinstance(
        exception_review_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "exception_review_df must be a pandas DataFrame."
        )

    required_columns = {
        patient_column,
        "rss_exception_trigger_flag",
        "rss_reversal_pair_context_flag",
        "rss_exception_review_role",
        "rss_chronology_agreement_state",
        "rss_local_strict_order_reversal_flag",
    }

    missing_columns = sorted(
        required_columns
        - set(exception_review_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "RSS chronology exception summary cannot proceed. "
            f"Missing columns: {missing_columns}"
        )

    trigger_rows = exception_review_df.loc[
        exception_review_df[
            "rss_exception_trigger_flag"
        ].eq(True)
    ]

    chronology_states = (
        trigger_rows[
            "rss_chronology_agreement_state"
        ]
        .value_counts()
    )

    review_roles = (
        exception_review_df[
            "rss_exception_review_role"
        ]
        .value_counts()
    )

    exception_trigger_count = len(
        trigger_rows
    )

    if exception_trigger_count:
        review_state = (
            "CHRONOLOGY_EXCEPTION_SET_ISOLATED_"
            "REVIEW_REQUIRED"
        )
    else:
        review_state = (
            "NO_CHRONOLOGY_EXCEPTIONS_DETECTED"
        )

    return pd.DataFrame([
        {
            "review_state":
                review_state,

            "canonical_order_selected":
                False,

            "affected_patient_count":
                int(
                    exception_review_df[
                        patient_column
                    ]
                    .nunique()
                ),

            "affected_history_encounter_count":
                len(
                    exception_review_df
                ),

            "exception_trigger_encounter_count":
                exception_trigger_count,

            "tie_related_difference_count":
                int(
                    chronology_states.get(
                        "ORDER_DIFFERENCE_TIE_RELATED",
                        0,
                    )
                ),

            "strict_difference_count":
                int(
                    chronology_states.get(
                        "ORDER_DIFFERENCE_STRICT_"
                        "REVIEW_REQUIRED",
                        0,
                    )
                ),

            "within_encounter_conflict_count":
                int(
                    chronology_states.get(
                        "WITHIN_ENCOUNTER_ORDER_"
                        "CONFLICT_REVIEW_REQUIRED",
                        0,
                    )
                ),

            "missing_order_count":
                int(
                    chronology_states.get(
                        "MISSING_ORDER_VALUE",
                        0,
                    )
                ),

            "local_strict_reversal_count":
                int(
                    exception_review_df[
                        "rss_local_strict_order_reversal_flag"
                    ]
                    .astype("boolean")
                    .fillna(False)
                    .sum()
                ),

            "reversal_pair_counterpart_count":
                int(
                    review_roles.get(
                        "REVERSAL_PAIR_COUNTERPART",
                        0,
                    )
                ),

            "affected_patient_context_count":
                int(
                    review_roles.get(
                        "AFFECTED_PATIENT_CONTEXT",
                        0,
                    )
                ),
        }
    ])

__all__ = [
    "build_rss_replay_evidence_inventory",
    "build_rss_replay_evidence_qualification",
    "build_rss_replay_evidence_qualification_summary",
    "build_rss_canonical_replay_substrate_selection",
    "build_rss_canonical_replay_substrate_summary",
    "build_rss_longitudinal_order_validation",
    "build_rss_longitudinal_order_validation_summary",
    "build_rss_order_candidate_resolution",
    "build_rss_order_candidate_resolution_summary",
    "build_rss_chronology_candidate_agreement",
    "build_rss_chronology_candidate_agreement_summary",
    "build_rss_chronology_exception_review",
    "build_rss_chronology_exception_summary",
]