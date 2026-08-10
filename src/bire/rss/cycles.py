#============================================================
# Project Sixth Sense — BIRE OS
# Replay Stability Support (RSS)
#
# File:
#     cycles.py
#
# Chapter:
#     68.20 — RSS Replay Cycle Boundary Signal
#               Qualification
#
# Purpose:
#     Qualify canonical stability, instability, recovery,
#     and escalation evidence before replay-cycle boundaries
#     are formed.
#
# Responsibilities:
#     - Inspect canonical RSS v1 evidence fields
#     - Identify direct binary boundary candidates
#     - Preserve within-encounter signal variation
#     - Identify categorical mapping requirements
#     - Identify numeric threshold requirements
#     - Reject constant or unsuitable boundary signals
#     - Preserve NID governance requirements
#
# Does Not:
#     - Assign encounter-level RSS states
#     - Define numeric thresholds
#     - Form replay cycles
#     - Generate pattern maturity states
#     - Generate replay lesson states
#     - Predict future patient behavior
#
# Governance:
#     Candidate qualification does not authorize boundary use.
#     All boundary policies remain subject to NID governance.
#
# Core Doctrine:
#     Replay may reveal a lesson.
#     Recurrence must earn it.
#============================================================
from __future__ import annotations

import numpy as np
import pandas as pd



import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_numeric_dtype,
)


_RSS_BOUNDARY_FAMILY_BY_ROLE = {
    "STABILITY_EVIDENCE":
        "STABILITY",

    "INSTABILITY_EVIDENCE":
        "INSTABILITY",

    "RECOVERY_EVIDENCE":
        "RECOVERY",

    "ESCALATION_EVIDENCE":
        "ESCALATION",
}


_RSS_BOUNDARY_QUALIFICATION_PRIORITY = {
    "DIRECT_BINARY_BOUNDARY_CANDIDATE": 0,
    (
        "BINARY_WITHIN_ENCOUNTER_VARIATION_"
        "REVIEW_REQUIRED"
    ): 1,
    "CATEGORICAL_STATE_MAPPING_REQUIRED": 2,
    "NUMERIC_THRESHOLD_POLICY_REQUIRED": 3,
    "INSUFFICIENT_SIGNAL_VARIATION": 4,
    "HIGH_CARDINALITY_CONTEXT_NOT_BOUNDARY_READY": 5,
    "CANONICAL_COLUMN_MISSING": 6,
}


def _coerce_rss_binary_signal(
    series: pd.Series,
) -> pd.Series | None:
    """
    Convert a genuinely binary field to pandas nullable boolean.

    Accepted representations include:
    - True / False
    - 1 / 0
    - yes / no
    - true / false strings

    Returns None when the field is not binary.
    """

    non_null = series.dropna()

    if non_null.empty:
        return None

    if is_bool_dtype(series):
        return series.astype("boolean")

    if is_numeric_dtype(series):
        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        observed_values = set(
            numeric
            .dropna()
            .unique()
            .tolist()
        )

        if observed_values.issubset({
            0,
            1,
            0.0,
            1.0,
        }):
            return (
                numeric
                .map({
                    1: True,
                    0: False,
                })
                .astype("boolean")
            )

        return None

    normalized = (
        series
        .astype("string")
        .str.strip()
        .str.lower()
    )

    recognized_values = {
        "true",
        "false",
        "1",
        "0",
        "yes",
        "no",
    }

    observed_values = set(
        normalized
        .dropna()
        .unique()
        .tolist()
    )

    if not observed_values.issubset(
        recognized_values
    ):
        return None

    return (
        normalized
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
        })
        .astype("boolean")
    )


def build_rss_cycle_boundary_signal_qualification(
    substrate_selection_df: pd.DataFrame,
    canonical_df: pd.DataFrame,
    patient_column: str = "patient_id",
    encounter_column: str = "encounter_id",
    maximum_categorical_state_count: int = 25,
) -> pd.DataFrame:
    """
    Qualify canonical RSS replay-family fields for possible use
    during future cycle-boundary formation.

    Candidate families:
    - stability
    - instability
    - recovery
    - escalation

    Binary fields receive encounter-level variation review.

    Categorical and numeric fields remain unauthorized until
    explicit mapping or threshold policies are established.

    This function does not form replay cycles.
    """

    if not isinstance(
        substrate_selection_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "substrate_selection_df must be a pandas DataFrame."
        )

    if not isinstance(
        canonical_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "canonical_df must be a pandas DataFrame."
        )

    if maximum_categorical_state_count < 2:
        raise ValueError(
            "maximum_categorical_state_count must be at least 2."
        )

    required_selection_columns = {
        "source",
        "column",
        "candidate_role",
        "source_selection_state",
        "admitted_to_rss_v1_substrate",
        "independent_evidence_weight",
        "dtype",
        "non_null_count",
        "non_null_percent",
        "unique_count",
        "active_count",
        "sample_values",
    }

    missing_selection_columns = sorted(
        required_selection_columns
        - set(substrate_selection_df.columns)
    )

    if missing_selection_columns:
        raise KeyError(
            "RSS boundary qualification cannot proceed. "
            "Missing substrate-selection columns: "
            f"{missing_selection_columns}"
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

    candidates = (
        substrate_selection_df.loc[
            substrate_selection_df[
                "admitted_to_rss_v1_substrate"
            ].eq(True)
            &
            substrate_selection_df[
                "candidate_role"
            ].isin(
                _RSS_BOUNDARY_FAMILY_BY_ROLE
            )
        ]
        .drop_duplicates(
            subset="column"
        )
        .copy()
    )

    # ------------------------------------------------------------
    # Normalize the canonical patient-encounter keys once.
    #
    # This must match the identifier normalization used by the
    # canonical longitudinal ordering contract.
    # ------------------------------------------------------------

    canonical_keys = canonical_df[
        [
            patient_column,
            encounter_column,
        ]
    ].copy()

    for key in (
        patient_column,
        encounter_column,
    ):
        canonical_keys[key] = (
            canonical_keys[key]
            .astype("string")
            .str.strip()
    )

    missing_key_mask = (
        canonical_keys[
            [
                patient_column,
                encounter_column,
            ]
        ]
        .isna()
        .any(axis=1)
        |
        canonical_keys[
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
            "Canonical RSS dataframe contains "
            f"{int(missing_key_mask.sum())} rows with "
            "missing patient or encounter identifiers."
        )

    canonical_keys = (
        canonical_keys
        .reset_index(drop=True)
    )

    total_encounter_count = int(
        canonical_keys
        .drop_duplicates()
        .shape[0]
    )

    rows: list[dict[str, object]] = []

    for candidate in candidates.itertuples(
        index=False
    ):
        column = candidate.column

        boundary_family = (
            _RSS_BOUNDARY_FAMILY_BY_ROLE[
                candidate.candidate_role
            ]
        )

        if column not in canonical_df.columns:
            rows.append({
                "column":
                    column,

                "boundary_family":
                    boundary_family,

                "candidate_role":
                    candidate.candidate_role,

                "source_selection_state":
                    candidate.source_selection_state,

                "independent_evidence_weight":
                    candidate.independent_evidence_weight,

                "dtype":
                    candidate.dtype,

                "row_non_null_percent":
                    candidate.non_null_percent,

                "unique_count":
                    candidate.unique_count,

                "encounter_non_null_count":
                    pd.NA,

                "encounter_non_null_percent":
                    pd.NA,

                "positive_encounter_count":
                    pd.NA,

                "negative_only_encounter_count":
                    pd.NA,

                "within_encounter_variation_count":
                    pd.NA,

                "value_semantics":
                    "COLUMN_UNAVAILABLE",

                "qualification_state":
                    "CANONICAL_COLUMN_MISSING",

                "eligible_for_boundary_policy_review":
                    False,

                "automatic_boundary_use_authorized":
                    False,

                "replay_cycle_formation_authorized":
                    False,

                "nid_review_required":
                    True,

                "sample_values":
                    candidate.sample_values,
            })

            continue

        series = canonical_df[column]

        non_null = series.dropna()

        observed_unique_count = int(
            non_null.nunique(
                dropna=True
            )
        )

        binary_signal = (
            _coerce_rss_binary_signal(
                series
            )
        )

        encounter_non_null_count: int | object = (
            pd.NA
        )

        encounter_non_null_percent: float | object = (
            pd.NA
        )

        positive_encounter_count: int | object = (
            pd.NA
        )

        negative_only_encounter_count: int | object = (
            pd.NA
        )

        within_encounter_variation_count: int | object = (
            pd.NA
        )

        if observed_unique_count <= 1:
            value_semantics = (
                "CONSTANT_OR_SINGLE_OBSERVED_VALUE"
            )

            qualification_state = (
                "INSUFFICIENT_SIGNAL_VARIATION"
            )

            eligible_for_review = False

        elif binary_signal is not None:
            value_semantics = "BINARY"

            binary_review = canonical_keys.copy()

            binary_review[
                "_rss_boundary_binary_value"
                ] = (
                    binary_signal
                ).reset_index(drop=True)

            encounter_binary = (
                binary_review
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
                    observed_value_count=(
                        "_rss_boundary_binary_value",
                        "count",
                    ),
                    true_value_count=(
                        "_rss_boundary_binary_value",
                        lambda values: int(
                            values.eq(True).sum()
                        ),
                    ),
                    false_value_count=(
                        "_rss_boundary_binary_value",
                        lambda values: int(
                            values.eq(False).sum()
                        ),
                    ),
                )
            )

            observed_encounter_mask = (
                encounter_binary[
                    "observed_value_count"
                ]
                .gt(0)
            )

            positive_encounter_mask = (
                encounter_binary[
                    "true_value_count"
                ]
                .gt(0)
            )

            negative_only_mask = (
                encounter_binary[
                    "true_value_count"
                ]
                .eq(0)
                &
                encounter_binary[
                    "false_value_count"
                ]
                .gt(0)
            )

            mixed_encounter_mask = (
                encounter_binary[
                    "true_value_count"
                ]
                .gt(0)
                &
                encounter_binary[
                    "false_value_count"
                ]
                .gt(0)
            )

            encounter_non_null_count = int(
                observed_encounter_mask.sum()
            )

            encounter_non_null_percent = (
                round(
                    (
                        encounter_non_null_count
                        / total_encounter_count
                    )
                    * 100,
                    3,
                )
                if total_encounter_count
                else 0.0
            )

            positive_encounter_count = int(
                positive_encounter_mask.sum()
            )

            negative_only_encounter_count = int(
                negative_only_mask.sum()
            )

            within_encounter_variation_count = int(
                mixed_encounter_mask.sum()
            )

            if (
                within_encounter_variation_count
                > 0
            ):
                qualification_state = (
                    "BINARY_WITHIN_ENCOUNTER_"
                    "VARIATION_REVIEW_REQUIRED"
                )
            else:
                qualification_state = (
                    "DIRECT_BINARY_BOUNDARY_"
                    "CANDIDATE"
                )

            eligible_for_review = True

        elif is_numeric_dtype(series):
            value_semantics = (
                "NUMERIC_CONTINUOUS_OR_ORDINAL"
            )

            qualification_state = (
                "NUMERIC_THRESHOLD_POLICY_REQUIRED"
            )

            eligible_for_review = True

        elif (
            observed_unique_count
            <= maximum_categorical_state_count
        ):
            value_semantics = (
                "LOW_CARDINALITY_CATEGORICAL"
            )

            qualification_state = (
                "CATEGORICAL_STATE_MAPPING_REQUIRED"
            )

            eligible_for_review = True

        else:
            value_semantics = (
                "HIGH_CARDINALITY_TEXT_OR_CONTEXT"
            )

            qualification_state = (
                "HIGH_CARDINALITY_CONTEXT_"
                "NOT_BOUNDARY_READY"
            )

            eligible_for_review = False

        rows.append({
            "column":
                column,

            "boundary_family":
                boundary_family,

            "candidate_role":
                candidate.candidate_role,

            "source_selection_state":
                candidate.source_selection_state,

            "independent_evidence_weight":
                candidate.independent_evidence_weight,

            "dtype":
                str(series.dtype),

            "row_non_null_percent":
                round(
                    series.notna().mean()
                    * 100,
                    3,
                ),

            "unique_count":
                observed_unique_count,

            "encounter_non_null_count":
                encounter_non_null_count,

            "encounter_non_null_percent":
                encounter_non_null_percent,

            "positive_encounter_count":
                positive_encounter_count,

            "negative_only_encounter_count":
                negative_only_encounter_count,

            "within_encounter_variation_count":
                within_encounter_variation_count,

            "value_semantics":
                value_semantics,

            "qualification_state":
                qualification_state,

            "eligible_for_boundary_policy_review":
                eligible_for_review,

            "automatic_boundary_use_authorized":
                False,

            "replay_cycle_formation_authorized":
                False,

            "nid_review_required":
                True,

            "sample_values":
                " | ".join(
                    non_null
                    .drop_duplicates()
                    .astype(str)
                    .head(5)
                    .tolist()
                ),
        })

    result_columns = [
        "column",
        "boundary_family",
        "candidate_role",
        "source_selection_state",
        "independent_evidence_weight",
        "dtype",
        "row_non_null_percent",
        "unique_count",
        "encounter_non_null_count",
        "encounter_non_null_percent",
        "positive_encounter_count",
        "negative_only_encounter_count",
        "within_encounter_variation_count",
        "value_semantics",
        "qualification_state",
        "eligible_for_boundary_policy_review",
        "automatic_boundary_use_authorized",
        "replay_cycle_formation_authorized",
        "nid_review_required",
        "sample_values",
    ]

    if not rows:
        return pd.DataFrame(
            columns=result_columns
        )

    result = pd.DataFrame(rows)

    result[
        "_qualification_priority"
    ] = (
        result[
            "qualification_state"
        ]
        .map(
            _RSS_BOUNDARY_QUALIFICATION_PRIORITY
        )
        .fillna(99)
    )

    return (
        result
        .sort_values(
            by=[
                "_qualification_priority",
                "boundary_family",
                "row_non_null_percent",
                "unique_count",
                "column",
            ],
            ascending=[
                True,
                True,
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


def build_rss_cycle_boundary_signal_summary(
    qualification_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize RSS replay-cycle boundary signal qualification.

    Replay-cycle formation remains unauthorized.
    """

    if not isinstance(
        qualification_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "qualification_df must be a pandas DataFrame."
        )

    required_columns = {
        "boundary_family",
        "qualification_state",
        "value_semantics",
        "eligible_for_boundary_policy_review",
        "automatic_boundary_use_authorized",
        "replay_cycle_formation_authorized",
    }

    missing_columns = sorted(
        required_columns
        - set(qualification_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "RSS boundary signal summary cannot proceed. "
            f"Missing columns: {missing_columns}"
        )

    return (
        qualification_df
        .groupby(
            [
                "boundary_family",
                "qualification_state",
                "value_semantics",
                "eligible_for_boundary_policy_review",
                "automatic_boundary_use_authorized",
                "replay_cycle_formation_authorized",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            by=[
                "eligible_for_boundary_policy_review",
                "count",
                "boundary_family",
                "qualification_state",
            ],
            ascending=[
                False,
                False,
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    #============================================================
# Chapter 68.20B — RSS Binary Encounter Count
#                   Reconciliation
#============================================================

def reconcile_rss_binary_encounter_counts(
    qualification_df: pd.DataFrame,
    canonical_df: pd.DataFrame,
    patient_column: str = "patient_id",
    encounter_column: str = "encounter_id",
) -> pd.DataFrame:
    """
    Recalculate encounter-level counts for every binary RSS
    boundary candidate using normalized patient-encounter keys.

    This reconciliation prevents whitespace differences in source
    identifiers from creating duplicate encounter groups.

    It does not change signal semantics, authorize boundaries,
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

    result = qualification_df.copy()

    # --------------------------------------------------------
    # Normalize patient-encounter identifiers once.
    # --------------------------------------------------------

    normalized_keys = canonical_df[
        [
            patient_column,
            encounter_column,
        ]
    ].copy()

    for key in (
        patient_column,
        encounter_column,
    ):
        normalized_keys[key] = (
            normalized_keys[key]
            .astype("string")
            .str.strip()
        )

    missing_key_mask = (
        normalized_keys[
            [
                patient_column,
                encounter_column,
            ]
        ]
        .isna()
        .any(axis=1)
        |
        normalized_keys[
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
            "Canonical RSS dataframe contains "
            f"{int(missing_key_mask.sum())} rows with "
            "missing normalized patient-encounter keys."
        )

    normalized_keys = (
        normalized_keys
        .reset_index(drop=True)
    )

    encounter_index = pd.MultiIndex.from_frame(
        normalized_keys
    )

    encounter_codes, unique_encounters = (
        pd.factorize(
            encounter_index,
            sort=False,
        )
    )

    total_encounter_count = len(
        unique_encounters
    )

    result[
        "normalized_encounter_population_count"
    ] = total_encounter_count

    result[
        "binary_count_reconciliation_state"
    ] = "NOT_APPLICABLE_NONBINARY"

    # --------------------------------------------------------
    # Recalculate only binary evidence fields.
    # --------------------------------------------------------

    binary_rows = result[
        "value_semantics"
    ].eq("BINARY")

    for row_index in result.index[
        binary_rows
    ]:
        column = result.at[
            row_index,
            "column",
        ]

        if column not in canonical_df.columns:
            raise KeyError(
                f"Binary RSS candidate {column!r} "
                "is missing from canonical_df."
            )

        binary_signal = (
            _coerce_rss_binary_signal(
                canonical_df[column]
            )
        )

        if binary_signal is None:
            raise ValueError(
                f"{column!r} was classified as binary "
                "but could not be converted to a binary signal."
            )

        binary_signal = (
            binary_signal
            .reset_index(drop=True)
        )

        valid_mask = (
            binary_signal.notna().to_numpy()
        )

        true_mask = (
            binary_signal
            .fillna(False)
            .astype(bool)
            .to_numpy()
        )

        false_mask = (
            valid_mask
            &
            ~true_mask
        )

        observed_by_encounter = np.bincount(
            encounter_codes[valid_mask],
            minlength=total_encounter_count,
        )

        true_by_encounter = np.bincount(
            encounter_codes[
                valid_mask & true_mask
            ],
            minlength=total_encounter_count,
        )

        false_by_encounter = np.bincount(
            encounter_codes[false_mask],
            minlength=total_encounter_count,
        )

        observed_encounter_count = int(
            (
                observed_by_encounter > 0
            ).sum()
        )

        positive_encounter_count = int(
            (
                true_by_encounter > 0
            ).sum()
        )

        negative_only_encounter_count = int(
            (
                (true_by_encounter == 0)
                &
                (false_by_encounter > 0)
            )
            .sum()
        )

        within_encounter_variation_count = int(
            (
                (true_by_encounter > 0)
                &
                (false_by_encounter > 0)
            )
            .sum()
        )

        result.at[
            row_index,
            "encounter_non_null_count",
        ] = observed_encounter_count

        result.at[
            row_index,
            "encounter_non_null_percent",
        ] = round(
            (
                observed_encounter_count
                / total_encounter_count
            )
            * 100,
            3,
        )

        result.at[
            row_index,
            "positive_encounter_count",
        ] = positive_encounter_count

        result.at[
            row_index,
            "negative_only_encounter_count",
        ] = negative_only_encounter_count

        result.at[
            row_index,
            "within_encounter_variation_count",
        ] = within_encounter_variation_count

        if within_encounter_variation_count > 0:
            result.at[
                row_index,
                "qualification_state",
            ] = (
                "BINARY_WITHIN_ENCOUNTER_"
                "VARIATION_REVIEW_REQUIRED"
            )
        else:
            result.at[
                row_index,
                "qualification_state",
            ] = (
                "DIRECT_BINARY_BOUNDARY_CANDIDATE"
            )

        result.at[
            row_index,
            "binary_count_reconciliation_state",
        ] = (
            "NORMALIZED_ENCOUNTER_COUNTS_RECONCILED"
        )

    return result

#============================================================
# Chapter 68.21 — RSS Direct Binary Boundary
#                  Candidate Review
#============================================================

_RSS_DIRECT_BINARY_REVIEW_PRIORITY = {
    "RECLASSIFICATION_REQUIRED_VARIATION_DETECTED": 0,
    "BINARY_COUNT_INTEGRITY_REVIEW_REQUIRED": 1,
    "COVERAGE_REVIEW_REQUIRED": 2,
    "SEMANTIC_REVIEW_REQUIRED_EXTREME_PREVALENCE": 3,
    "SEMANTIC_REVIEW_REQUIRED": 4,
}


def _calculate_rss_percent(
    numerator: int | float,
    denominator: int | float,
) -> float:
    """
    Safely calculate a percentage.
    """

    if denominator <= 0:
        return 0.0

    return round(
        (float(numerator) / float(denominator))
        * 100,
        3,
    )


def _classify_rss_binary_prevalence(
    positive_count: int,
    observed_count: int,
) -> str:
    """
    Describe the structural prevalence of a binary candidate.

    This classification does not determine clinical meaning.
    """

    if observed_count <= 0:
        return "NO_OBSERVED_ENCOUNTERS"

    positive_fraction = (
        positive_count
        / observed_count
    )

    if positive_fraction == 0:
        return "NO_POSITIVE_ENCOUNTERS"

    if positive_fraction < 0.01:
        return "POSITIVE_STATE_EXTREMELY_RARE"

    if positive_fraction < 0.05:
        return "POSITIVE_STATE_RARE"

    if positive_fraction <= 0.95:
        return "POSITIVE_AND_NEGATIVE_STATES_DISTRIBUTED"

    if positive_fraction < 0.99:
        return "POSITIVE_STATE_HIGHLY_PREVALENT"

    if positive_fraction < 1.0:
        return "POSITIVE_STATE_EXTREMELY_PREVALENT"

    return "ALL_OBSERVED_ENCOUNTERS_POSITIVE"


def build_rss_direct_binary_boundary_review(
    qualification_df: pd.DataFrame,
    total_encounter_count: int,
) -> pd.DataFrame:
    """
    Review direct binary RSS boundary candidates.

    This review evaluates:
    - encounter coverage
    - positive and negative prevalence
    - binary-count reconciliation
    - signal balance
    - semantic verification requirements
    - boundary-policy readiness
    - NID authorization status

    This function does not authorize boundary use, assign patient
    states, or form replay cycles.
    """

    if not isinstance(
        qualification_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "qualification_df must be a pandas DataFrame."
        )

    if total_encounter_count <= 0:
        raise ValueError(
            "total_encounter_count must be greater than zero."
        )

    required_columns = {
        "column",
        "boundary_family",
        "candidate_role",
        "dtype",
        "encounter_non_null_count",
        "encounter_non_null_percent",
        "positive_encounter_count",
        "negative_only_encounter_count",
        "within_encounter_variation_count",
        "value_semantics",
        "qualification_state",
        "automatic_boundary_use_authorized",
        "replay_cycle_formation_authorized",
        "nid_review_required",
        "sample_values",
    }

    missing_columns = sorted(
        required_columns
        - set(qualification_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "RSS direct binary review cannot proceed. "
            f"Missing qualification columns: {missing_columns}"
        )

    review = (
        qualification_df.loc[
            qualification_df[
                "qualification_state"
            ].eq(
                "DIRECT_BINARY_BOUNDARY_CANDIDATE"
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    result_columns = [
        "column",
        "boundary_family",
        "candidate_role",
        "dtype",
        "total_encounter_count",
        "observed_encounter_count",
        "observed_encounter_percent",
        "positive_encounter_count",
        "positive_percent_of_observed",
        "positive_percent_of_population",
        "negative_only_encounter_count",
        "negative_percent_of_observed",
        "unobserved_encounter_count",
        "unobserved_encounter_percent",
        "within_encounter_variation_count",
        "binary_count_integrity_state",
        "binary_prevalence_state",
        "direct_binary_review_state",
        "positive_state_semantics_verified",
        "negative_state_semantics_verified",
        "transition_semantics_verified",
        "persistence_policy_defined",
        "cross_signal_conflict_policy_defined",
        "automatic_boundary_use_authorized",
        "replay_cycle_formation_authorized",
        "nid_authorization_state",
        "review_question",
        "sample_values",
    ]

    if review.empty:
        return pd.DataFrame(
            columns=result_columns
        )

    numeric_count_columns = [
        "encounter_non_null_count",
        "positive_encounter_count",
        "negative_only_encounter_count",
        "within_encounter_variation_count",
    ]

    for column in numeric_count_columns:
        review[column] = (
            pd.to_numeric(
                review[column],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )

    rows: list[dict[str, object]] = []

    for candidate in review.itertuples(
        index=False
    ):
        observed_count = int(
            candidate.encounter_non_null_count
        )

        if observed_count > total_encounter_count:
            raise ValueError(
                f"{candidate.column} contains "
                f"{observed_count:,} observed encounter keys, "
                f"which exceeds the canonical encounter population "
                f"of {total_encounter_count:,}. "
                "Verify patient and encounter identifier normalization."
        )

        positive_count = int(
            candidate.positive_encounter_count
        )

        negative_count = int(
            candidate.negative_only_encounter_count
        )

        variation_count = int(
            candidate.within_encounter_variation_count
        )

        unobserved_count = max(
            total_encounter_count
            - observed_count,
            0,
        )

        reconciled_binary_count = (
            positive_count
            + negative_count
        )

        if (
            variation_count == 0
            and reconciled_binary_count
            == observed_count
        ):
            integrity_state = (
                "BINARY_COUNTS_RECONCILED"
            )
        else:
            integrity_state = (
                "BINARY_COUNT_INTEGRITY_"
                "REVIEW_REQUIRED"
            )

        prevalence_state = (
            _classify_rss_binary_prevalence(
                positive_count=positive_count,
                observed_count=observed_count,
            )
        )

        extreme_prevalence_states = {
            "NO_POSITIVE_ENCOUNTERS",
            "POSITIVE_STATE_EXTREMELY_RARE",
            "POSITIVE_STATE_EXTREMELY_PREVALENT",
            "ALL_OBSERVED_ENCOUNTERS_POSITIVE",
        }

        if variation_count > 0:
            direct_review_state = (
                "RECLASSIFICATION_REQUIRED_"
                "VARIATION_DETECTED"
            )

        elif integrity_state != (
            "BINARY_COUNTS_RECONCILED"
        ):
            direct_review_state = (
                "BINARY_COUNT_INTEGRITY_"
                "REVIEW_REQUIRED"
            )

        elif (
            observed_count
            / total_encounter_count
        ) < 0.95:
            direct_review_state = (
                "COVERAGE_REVIEW_REQUIRED"
            )

        elif prevalence_state in (
            extreme_prevalence_states
        ):
            direct_review_state = (
                "SEMANTIC_REVIEW_REQUIRED_"
                "EXTREME_PREVALENCE"
            )

        else:
            direct_review_state = (
                "SEMANTIC_REVIEW_REQUIRED"
            )

        rows.append({
            "column":
                candidate.column,

            "boundary_family":
                candidate.boundary_family,

            "candidate_role":
                candidate.candidate_role,

            "dtype":
                candidate.dtype,

            "total_encounter_count":
                total_encounter_count,

            "observed_encounter_count":
                observed_count,

            "observed_encounter_percent":
                _calculate_rss_percent(
                    observed_count,
                    total_encounter_count,
                ),

            "positive_encounter_count":
                positive_count,

            "positive_percent_of_observed":
                _calculate_rss_percent(
                    positive_count,
                    observed_count,
                ),

            "positive_percent_of_population":
                _calculate_rss_percent(
                    positive_count,
                    total_encounter_count,
                ),

            "negative_only_encounter_count":
                negative_count,

            "negative_percent_of_observed":
                _calculate_rss_percent(
                    negative_count,
                    observed_count,
                ),

            "unobserved_encounter_count":
                unobserved_count,

            "unobserved_encounter_percent":
                _calculate_rss_percent(
                    unobserved_count,
                    total_encounter_count,
                ),

            "within_encounter_variation_count":
                variation_count,

            "binary_count_integrity_state":
                integrity_state,

            "binary_prevalence_state":
                prevalence_state,

            "direct_binary_review_state":
                direct_review_state,

            "positive_state_semantics_verified":
                False,

            "negative_state_semantics_verified":
                False,

            "transition_semantics_verified":
                False,

            "persistence_policy_defined":
                False,

            "cross_signal_conflict_policy_defined":
                False,

            "automatic_boundary_use_authorized":
                False,

            "replay_cycle_formation_authorized":
                False,

            "nid_authorization_state":
                "NID_REVIEW_PENDING",

            "review_question":
                (
                    f"Does True in {candidate.column} "
                    f"represent affirmative "
                    f"{str(candidate.boundary_family).lower()} "
                    "evidence suitable for governed "
                    "replay-cycle boundary use?"
                ),

            "sample_values":
                candidate.sample_values,
        })

    result = pd.DataFrame(rows)

    result[
        "_direct_review_priority"
    ] = (
        result[
            "direct_binary_review_state"
        ]
        .map(
            _RSS_DIRECT_BINARY_REVIEW_PRIORITY
        )
        .fillna(99)
    )

    return (
        result
        .sort_values(
            by=[
                "_direct_review_priority",
                "boundary_family",
                "positive_percent_of_population",
                "column",
            ],
            ascending=[
                True,
                True,
                False,
                True,
            ],
        )
        .drop(
            columns="_direct_review_priority"
        )
        .reset_index(drop=True)
    )


def build_rss_direct_binary_boundary_summary(
    review_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize the direct binary RSS boundary candidate review.

    Boundary use and replay-cycle formation remain unauthorized.
    """

    if not isinstance(
        review_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "review_df must be a pandas DataFrame."
        )

    required_columns = {
        "boundary_family",
        "binary_prevalence_state",
        "direct_binary_review_state",
        "automatic_boundary_use_authorized",
        "replay_cycle_formation_authorized",
        "nid_authorization_state",
    }

    missing_columns = sorted(
        required_columns
        - set(review_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "RSS direct binary summary cannot proceed. "
            f"Missing review columns: {missing_columns}"
        )

    return (
        review_df
        .groupby(
            [
                "boundary_family",
                "direct_binary_review_state",
                "binary_prevalence_state",
                "automatic_boundary_use_authorized",
                "replay_cycle_formation_authorized",
                "nid_authorization_state",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            by=[
                "boundary_family",
                "count",
                "direct_binary_review_state",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    #============================================================
# Chapter 68.22 — RSS Direct Binary Semantic Disposition
#============================================================

_RSS_DIRECT_BINARY_SEMANTIC_POLICY = {

    "care_escalation_occurred": {
        "semantic_disposition":
            "DIRECT_ESCALATION_EVENT_CANDIDATE",

        "evidence_orientation":
            "AFFIRMATIVE_EVENT_EVIDENCE",

        "governed_evidence_domain":
            "ESCALATION_EVENT",

        "direct_boundary_candidate":
            True,

        "event_boundary_review_candidate":
            True,

        "positive_value_meaning":
            "Observed care escalation occurred.",

        "negative_value_meaning":
            (
                "Care escalation was not observed through this "
                "specific flag. Absence does not establish stability."
            ),

        "future_replay_relevance":
            "ESCALATING_REPLAY",

        "semantic_alignment_state":
            "DISCOVERY_FAMILY_CONFIRMED",
    },

    "stepdown_rebound_flag": {
        "semantic_disposition":
            "DIRECT_INSTABILITY_EVENT_CANDIDATE",

        "evidence_orientation":
            "AFFIRMATIVE_EVENT_EVIDENCE",

        "governed_evidence_domain":
            "REBOUND_INSTABILITY_EVENT",

        "direct_boundary_candidate":
            True,

        "event_boundary_review_candidate":
            True,

        "positive_value_meaning":
            (
                "Rebound instability was observed following "
                "stepdown."
            ),

        "negative_value_meaning":
            (
                "Stepdown rebound was not observed through this "
                "specific flag. Absence does not establish stability."
            ),

        "future_replay_relevance":
            "REPEATED_INSTABILITY",

        "semantic_alignment_state":
            "DISCOVERY_FAMILY_CONFIRMED",
    },

    "readmission_flag": {
        "semantic_disposition":
            "OUTCOME_EVENT_CANDIDATE",

        "evidence_orientation":
            "AFFIRMATIVE_OUTCOME_EVIDENCE",

        "governed_evidence_domain":
            "READMISSION_EVENT",

        "direct_boundary_candidate":
            False,

        "event_boundary_review_candidate":
            True,

        "positive_value_meaning":
            "The encounter is associated with readmission.",

        "negative_value_meaning":
            (
                "Readmission was not observed through this flag. "
                "Absence does not establish durable stability."
            ),

        "future_replay_relevance":
            (
                "RECURRING_RETURN_TO_CARE_OR_"
                "FAILED_STABILITY_CONTEXT"
            ),

        "semantic_alignment_state":
            "OUTCOME_EVENT_SPECIALIZATION_REQUIRED",
    },

    "re_escalation_risk_flag": {
        "semantic_disposition":
            "ESCALATION_RISK_CONTEXT",

        "evidence_orientation":
            "RISK_CONTEXT",

        "governed_evidence_domain":
            "RE_ESCALATION_RISK",

        "direct_boundary_candidate":
            False,

        "event_boundary_review_candidate":
            False,

        "positive_value_meaning":
            (
                "Evidence indicates elevated risk of "
                "re-escalation."
            ),

        "negative_value_meaning":
            (
                "Elevated re-escalation risk was not observed "
                "through this flag. Absence does not guarantee "
                "that escalation will not occur."
            ),

        "future_replay_relevance":
            "ESCALATION_RISK_CONTEXT",

        "semantic_alignment_state":
            "CONTEXT_NOT_EVENT",
    },

    "quiet_gap_before_escalation_flag": {
        "semantic_disposition":
            "ESCALATION_PRECURSOR_CONTEXT",

        "evidence_orientation":
            "PRECURSOR_CONTEXT",

        "governed_evidence_domain":
            "PRE_ESCALATION_QUIET_GAP",

        "direct_boundary_candidate":
            False,

        "event_boundary_review_candidate":
            False,

        "positive_value_meaning":
            (
                "A quiet interval preceding escalation was "
                "identified."
            ),

        "negative_value_meaning":
            (
                "The specific quiet-gap precursor was not "
                "observed. Absence does not exclude escalation."
            ),

        "future_replay_relevance":
            "RECURRING_ESCALATION_PRECURSOR",

        "semantic_alignment_state":
            "PRECURSOR_NOT_EVENT",
    },

    "de_escalation_candidate_flag": {
        "semantic_disposition":
            "CANDIDATE_STATUS_CONTEXT",

        "evidence_orientation":
            "ELIGIBILITY_OR_CANDIDATE_CONTEXT",

        "governed_evidence_domain":
            "DE_ESCALATION_CANDIDACY",

        "direct_boundary_candidate":
            False,

        "event_boundary_review_candidate":
            False,

        "positive_value_meaning":
            (
                "The encounter was identified as a candidate "
                "for de-escalation."
            ),

        "negative_value_meaning":
            (
                "De-escalation candidacy was not established "
                "through this flag."
            ),

        "future_replay_relevance":
            "DE_ESCALATION_ELIGIBILITY_CONTEXT",

        "semantic_alignment_state":
            "CANDIDATE_STATUS_NOT_EVENT",
    },

    "false_recovery_risk_flag": {
        "semantic_disposition":
            "RECOVERY_FRAGILITY_CONTEXT",

        "evidence_orientation":
            "RECOVERY_RISK_CONTEXT",

        "governed_evidence_domain":
            "FALSE_RECOVERY_RISK",

        "direct_boundary_candidate":
            False,

        "event_boundary_review_candidate":
            False,

        "positive_value_meaning":
            (
                "Evidence indicates risk that apparent recovery "
                "may be false or fragile."
            ),

        "negative_value_meaning":
            (
                "False-recovery risk was not observed through "
                "this flag. Absence does not prove recovery."
            ),

        "future_replay_relevance":
            "FAILED_RECOVERY_PATTERN_CONTEXT",

        "semantic_alignment_state":
            "RECOVERY_SEMANTIC_INVERSION_PRESERVED",
    },

    "vital_recovery_authenticity_warning": {
        "semantic_disposition":
            "RECOVERY_CONTRADICTION_WARNING",

        "evidence_orientation":
            "RECOVERY_WARNING_OR_CONTRADICTION",

        "governed_evidence_domain":
            "RECOVERY_AUTHENTICITY_WARNING",

        "direct_boundary_candidate":
            False,

        "event_boundary_review_candidate":
            False,

        "positive_value_meaning":
            (
                "Vital-sign recovery may not represent authentic "
                "or trustworthy recovery."
            ),

        "negative_value_meaning":
            (
                "This specific authenticity warning was not "
                "observed. Absence does not prove recovery."
            ),

        "future_replay_relevance":
            "FAILED_RECOVERY_PATTERN_CONTEXT",

        "semantic_alignment_state":
            "RECOVERY_SEMANTIC_INVERSION_PRESERVED",
    },

    "false_vital_recovery_signal": {
        "semantic_disposition":
            "FALSE_RECOVERY_EVIDENCE",

        "evidence_orientation":
            "AFFIRMATIVE_RECOVERY_CONTRADICTION",

        "governed_evidence_domain":
            "FALSE_VITAL_RECOVERY",

        "direct_boundary_candidate":
            False,

        "event_boundary_review_candidate":
            True,

        "positive_value_meaning":
            (
                "Apparent vital-sign recovery was identified as "
                "potentially false or misleading."
            ),

        "negative_value_meaning":
            (
                "False vital recovery was not observed through "
                "this specific signal. Absence does not prove "
                "durable recovery."
            ),

        "future_replay_relevance":
            "FAILED_RECOVERY_PATTERN",

        "semantic_alignment_state":
            "RECOVERY_SEMANTIC_INVERSION_PRESERVED",
    },

    "prior_false_recovery": {
        "semantic_disposition":
            "HISTORICAL_FAILED_RECOVERY_CONTEXT",

        "evidence_orientation":
            "HISTORICAL_CONTEXT",

        "governed_evidence_domain":
            "PRIOR_FALSE_RECOVERY_HISTORY",

        "direct_boundary_candidate":
            False,

        "event_boundary_review_candidate":
            False,

        "positive_value_meaning":
            (
                "Previous patient history contains evidence of "
                "false recovery."
            ),

        "negative_value_meaning":
            (
                "Prior false recovery was not observed through "
                "the available historical flag."
            ),

        "future_replay_relevance":
            "FAILED_RECOVERY_PATTERN",

        "semantic_alignment_state":
            "HISTORICAL_CONTEXT_NOT_CURRENT_BOUNDARY",
    },
}


_RSS_SEMANTIC_DISPOSITION_PRIORITY = {
    "DIRECT_ESCALATION_EVENT_CANDIDATE": 0,
    "DIRECT_INSTABILITY_EVENT_CANDIDATE": 1,
    "OUTCOME_EVENT_CANDIDATE": 2,
    "FALSE_RECOVERY_EVIDENCE": 3,
    "ESCALATION_RISK_CONTEXT": 4,
    "ESCALATION_PRECURSOR_CONTEXT": 5,
    "CANDIDATE_STATUS_CONTEXT": 6,
    "RECOVERY_FRAGILITY_CONTEXT": 7,
    "RECOVERY_CONTRADICTION_WARNING": 8,
    "HISTORICAL_FAILED_RECOVERY_CONTEXT": 9,
    "SEMANTIC_DISPOSITION_UNDEFINED": 10,
}


def build_rss_direct_binary_semantic_disposition(
    direct_binary_review_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign governed semantic roles to reviewed direct binary
    candidates.

    Semantic disposition distinguishes affirmative events from
    risk, precursor, warning, contradiction, candidate status,
    outcome events, and historical context.

    This function does not authorize automatic boundary use,
    define transitions, or form replay cycles.
    """

    if not isinstance(
        direct_binary_review_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "direct_binary_review_df must be a pandas DataFrame."
        )

    required_columns = {
        "column",
        "boundary_family",
        "observed_encounter_count",
        "observed_encounter_percent",
        "positive_encounter_count",
        "positive_percent_of_population",
        "negative_only_encounter_count",
        "binary_count_integrity_state",
        "binary_prevalence_state",
        "direct_binary_review_state",
    }

    missing_columns = sorted(
        required_columns
        - set(direct_binary_review_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "RSS semantic disposition cannot proceed. "
            f"Missing direct-binary review columns: {missing_columns}"
        )

    if direct_binary_review_df[
        "column"
    ].duplicated().any():
        duplicated_columns = (
            direct_binary_review_df.loc[
                direct_binary_review_df[
                    "column"
                ].duplicated(keep=False),
                "column",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "RSS direct-binary review contains duplicated "
            f"candidate columns: {duplicated_columns}"
        )

    rows: list[dict[str, object]] = []

    for candidate in direct_binary_review_df.itertuples(
        index=False
    ):
        semantic_policy = (
            _RSS_DIRECT_BINARY_SEMANTIC_POLICY.get(
                candidate.column
            )
        )

        if semantic_policy is None:
            semantic_policy = {
                "semantic_disposition":
                    "SEMANTIC_DISPOSITION_UNDEFINED",

                "evidence_orientation":
                    "UNRESOLVED",

                "governed_evidence_domain":
                    "UNRESOLVED",

                "direct_boundary_candidate":
                    False,

                "event_boundary_review_candidate":
                    False,

                "positive_value_meaning":
                    (
                        "Positive-value meaning has not been "
                        "semantically defined."
                    ),

                "negative_value_meaning":
                    (
                        "Negative-value meaning has not been "
                        "semantically defined."
                    ),

                "future_replay_relevance":
                    "UNRESOLVED",

                "semantic_alignment_state":
                    "MANUAL_SEMANTIC_REVIEW_REQUIRED",
            }

        rows.append({
            "column":
                candidate.column,

            "discovery_boundary_family":
                candidate.boundary_family,

            **semantic_policy,

            "observed_encounter_count":
                candidate.observed_encounter_count,

            "observed_encounter_percent":
                candidate.observed_encounter_percent,

            "positive_encounter_count":
                candidate.positive_encounter_count,

            "positive_percent_of_population":
                candidate.positive_percent_of_population,

            "negative_only_encounter_count":
                candidate.negative_only_encounter_count,

            "binary_count_integrity_state":
                candidate.binary_count_integrity_state,

            "binary_prevalence_state":
                candidate.binary_prevalence_state,

            "transition_policy_required":
                bool(
                    semantic_policy[
                        "event_boundary_review_candidate"
                    ]
                ),

            "persistence_policy_required":
                bool(
                    semantic_policy[
                        "event_boundary_review_candidate"
                    ]
                ),

            "cross_signal_conflict_policy_required":
                True,

            "automatic_boundary_use_authorized":
                False,

            "replay_cycle_formation_authorized":
                False,

            "nid_authorization_state":
                "NID_REVIEW_PENDING",

            "semantic_disposition_state":
                (
                    "SEMANTIC_ROLE_DEFINED_"
                    "BOUNDARY_AUTHORITY_WITHHELD"
                    if semantic_policy[
                        "semantic_disposition"
                    ]
                    != "SEMANTIC_DISPOSITION_UNDEFINED"
                    else
                    "SEMANTIC_ROLE_UNDEFINED_REVIEW_REQUIRED"
                ),
        })

    result = pd.DataFrame(rows)

    result[
        "_semantic_priority"
    ] = (
        result[
            "semantic_disposition"
        ]
        .map(
            _RSS_SEMANTIC_DISPOSITION_PRIORITY
        )
        .fillna(99)
    )

    return (
        result
        .sort_values(
            by=[
                "_semantic_priority",
                "column",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .drop(
            columns="_semantic_priority"
        )
        .reset_index(drop=True)
    )


def build_rss_direct_binary_semantic_summary(
    disposition_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize RSS direct-binary semantic dispositions.

    Boundary and replay-cycle authority remain withheld.
    """

    if not isinstance(
        disposition_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "disposition_df must be a pandas DataFrame."
        )

    required_columns = {
        "semantic_disposition",
        "evidence_orientation",
        "direct_boundary_candidate",
        "event_boundary_review_candidate",
        "automatic_boundary_use_authorized",
        "replay_cycle_formation_authorized",
        "nid_authorization_state",
    }

    missing_columns = sorted(
        required_columns
        - set(disposition_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "RSS semantic summary cannot proceed. "
            f"Missing disposition columns: {missing_columns}"
        )

    return (
        disposition_df
        .groupby(
            [
                "semantic_disposition",
                "evidence_orientation",
                "direct_boundary_candidate",
                "event_boundary_review_candidate",
                "automatic_boundary_use_authorized",
                "replay_cycle_formation_authorized",
                "nid_authorization_state",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            by=[
                "direct_boundary_candidate",
                "event_boundary_review_candidate",
                "count",
                "semantic_disposition",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    #============================================================
# Chapter 68.23 — RSS Binary Event Transition
#                  Readiness Review
#============================================================

_RSS_EVENT_TRANSITION_POLICY_SCOPE = {
    "DIRECT_ESCALATION_EVENT_CANDIDATE":
        "ESCALATION_EVENT_TRANSITION_POLICY",

    "DIRECT_INSTABILITY_EVENT_CANDIDATE":
        "REBOUND_INSTABILITY_TRANSITION_POLICY",

    "OUTCOME_EVENT_CANDIDATE":
        "READMISSION_OUTCOME_ALIGNMENT_POLICY",

    "FALSE_RECOVERY_EVIDENCE":
        "FALSE_RECOVERY_CONTRADICTION_POLICY",
}


_RSS_EVENT_TRANSITION_REVIEW_PRIORITY = {
    (
        "WITHIN_ENCOUNTER_EVENT_VARIATION_"
        "REVIEW_REQUIRED"
    ): 0,

    (
        "EVENT_TRANSITION_POLICY_AND_"
        "TEMPORAL_TIE_REVIEW_REQUIRED"
    ): 1,

    (
        "EVENT_TRANSITION_AND_"
        "RECURRENCE_POLICY_REQUIRED"
    ): 2,

    "EVENT_TRANSITION_POLICY_REQUIRED": 3,
}


def build_rss_binary_event_transition_readiness(
    semantic_disposition_df: pd.DataFrame,
    canonical_df: pd.DataFrame,
    ordering_contract_df: pd.DataFrame,
    patient_column: str = "patient_id",
    encounter_column: str = "encounter_id",
) -> pd.DataFrame:
    """
    Review transition behavior for event-capable RSS binary fields.

    The review evaluates:
    - positive observations
    - positive-after-nonpositive transitions
    - consecutive positive observations
    - nonpositive-after-positive observations
    - separated recurring positive episodes
    - temporal-tie ambiguity
    - within-encounter variation
    - chronology eligibility

    This function does not define event persistence, authorize
    boundaries, or form replay cycles.
    """

    if not isinstance(
        semantic_disposition_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "semantic_disposition_df must be a pandas DataFrame."
        )

    if not isinstance(
        canonical_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "canonical_df must be a pandas DataFrame."
        )

    if not isinstance(
        ordering_contract_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "ordering_contract_df must be a pandas DataFrame."
        )

    required_disposition_columns = {
        "column",
        "semantic_disposition",
        "evidence_orientation",
        "governed_evidence_domain",
        "event_boundary_review_candidate",
        "positive_value_meaning",
        "negative_value_meaning",
        "automatic_boundary_use_authorized",
        "replay_cycle_formation_authorized",
        "nid_authorization_state",
    }

    missing_disposition_columns = sorted(
        required_disposition_columns
        - set(semantic_disposition_df.columns)
    )

    if missing_disposition_columns:
        raise KeyError(
            "RSS event transition review cannot proceed. "
            "Missing semantic disposition columns: "
            f"{missing_disposition_columns}"
        )

    required_order_columns = {
        patient_column,
        encounter_column,
        "rss_temporal_start",
        "rss_canonical_order_position",
        "rss_primary_temporal_tie_flag",
        "rss_canonical_order_row_eligible",
        "rss_replay_chronology_eligible",
    }

    missing_order_columns = sorted(
        required_order_columns
        - set(ordering_contract_df.columns)
    )

    if missing_order_columns:
        raise KeyError(
            "RSS event transition review cannot proceed. "
            "Missing canonical ordering columns: "
            f"{missing_order_columns}"
        )

    missing_canonical_keys = [
        column
        for column in (
            patient_column,
            encounter_column,
        )
        if column not in canonical_df.columns
    ]

    if missing_canonical_keys:
        raise KeyError(
            "Canonical RSS dataframe is missing key columns: "
            f"{missing_canonical_keys}"
        )

    event_candidates = (
        semantic_disposition_df.loc[
            semantic_disposition_df[
                "event_boundary_review_candidate"
            ].eq(True)
        ]
        .copy()
        .reset_index(drop=True)
    )

    result_columns = [
        "column",
        "semantic_disposition",
        "evidence_orientation",
        "governed_evidence_domain",
        "transition_policy_scope",
        "total_encounter_count",
        "observed_encounter_count",
        "positive_encounter_count",
        "positive_patient_count",
        "multiple_positive_encounter_patient_count",
        "transition_testable_patient_count",
        "first_observation_positive_count",
        "positive_after_nonpositive_count",
        "consecutive_positive_observation_count",
        "nonpositive_after_positive_count",
        "strict_forward_transition_pair_count",
        "temporal_tie_pair_count",
        "temporal_tie_positive_pair_count",
        "temporal_tie_state_change_count",
        "within_encounter_variation_count",
        "positive_episode_start_count",
        "repeated_positive_episode_patient_count",
        "maximum_positive_episode_count_per_patient",
        "repeated_event_behavior_observed",
        "transition_readiness_state",
        "transition_policy_defined",
        "persistence_policy_defined",
        "temporal_tie_policy_defined",
        "cross_signal_conflict_policy_defined",
        "automatic_boundary_use_authorized",
        "replay_cycle_formation_authorized",
        "nid_authorization_state",
        "positive_value_meaning",
        "negative_value_meaning",
    ]

    if event_candidates.empty:
        return pd.DataFrame(
            columns=result_columns
        )

    # --------------------------------------------------------
    # Normalize source patient-encounter keys once.
    # --------------------------------------------------------

    normalized_keys = canonical_df[
        [
            patient_column,
            encounter_column,
        ]
    ].copy()

    for key in (
        patient_column,
        encounter_column,
    ):
        normalized_keys[key] = (
            normalized_keys[key]
            .astype("string")
            .str.strip()
        )

    missing_key_mask = (
        normalized_keys[
            [
                patient_column,
                encounter_column,
            ]
        ]
        .isna()
        .any(axis=1)
        |
        normalized_keys[
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
            "Canonical RSS dataframe contains "
            f"{int(missing_key_mask.sum())} rows with "
            "missing normalized patient-encounter keys."
        )

    normalized_keys = (
        normalized_keys
        .reset_index(drop=True)
    )

    encounter_index = pd.MultiIndex.from_frame(
        normalized_keys
    )

    encounter_codes, unique_encounters = (
        pd.factorize(
            encounter_index,
            sort=False,
        )
    )

    encounter_key_frame = (
        unique_encounters
        .to_frame(index=False)
    )

    encounter_key_frame.columns = [
        patient_column,
        encounter_column,
    ]

    total_encounter_count = len(
        encounter_key_frame
    )

    # --------------------------------------------------------
    # Prepare the governed ordering substrate.
    # --------------------------------------------------------

    order_base = ordering_contract_df[
        [
            patient_column,
            encounter_column,
            "rss_temporal_start",
            "rss_canonical_order_position",
            "rss_primary_temporal_tie_flag",
            "rss_canonical_order_row_eligible",
            "rss_replay_chronology_eligible",
        ]
    ].copy()

    for key in (
        patient_column,
        encounter_column,
    ):
        order_base[key] = (
            order_base[key]
            .astype("string")
            .str.strip()
        )

    if order_base.duplicated(
        subset=[
            patient_column,
            encounter_column,
        ]
    ).any():
        raise ValueError(
            "RSS ordering contract contains duplicated "
            "patient-encounter keys."
        )

    key_alignment = (
        encounter_key_frame[
            [
                patient_column,
                encounter_column,
            ]
        ]
        .merge(
            order_base[
                [
                    patient_column,
                    encounter_column,
                ]
            ],
            on=[
                patient_column,
                encounter_column,
            ],
            how="outer",
            indicator=True,
            validate="one_to_one",
        )
    )

    # The expression above can be difficult to read when
    # formatted; recalculate explicitly for safety.
    unmatched_key_count = int(
        (
            ~key_alignment[
                "_merge"
            ].eq("both")
        )
        .sum()
    )

    if unmatched_key_count:
        raise ValueError(
            "Canonical evidence and ordering contract keys "
            "do not align. "
            f"Unmatched encounters: {unmatched_key_count}"
        )

    rows: list[dict[str, object]] = []

    for candidate in event_candidates.itertuples(
        index=False
    ):
        column = candidate.column

        if column not in canonical_df.columns:
            raise KeyError(
                f"Event-capable RSS field {column!r} "
                "is missing from canonical_df."
            )

        binary_signal = (
            _coerce_rss_binary_signal(
                canonical_df[column]
            )
        )

        if binary_signal is None:
            raise ValueError(
                f"{column!r} was classified as an event-capable "
                "binary field but cannot be converted to binary."
            )

        binary_signal = (
            binary_signal
            .reset_index(drop=True)
        )

        valid_mask = (
            binary_signal
            .notna()
            .to_numpy()
        )

        true_mask = (
            binary_signal
            .fillna(False)
            .astype(bool)
            .to_numpy()
        )

        false_mask = (
            valid_mask
            &
            ~true_mask
        )

        observed_by_encounter = np.bincount(
            encounter_codes[valid_mask],
            minlength=total_encounter_count,
        )

        true_by_encounter = np.bincount(
            encounter_codes[
                valid_mask & true_mask
            ],
            minlength=total_encounter_count,
        )

        false_by_encounter = np.bincount(
            encounter_codes[false_mask],
            minlength=total_encounter_count,
        )

        within_encounter_variation = (
            (true_by_encounter > 0)
            &
            (false_by_encounter > 0)
        )

        event_flag = pd.Series(
            pd.NA,
            index=range(total_encounter_count),
            dtype="boolean",
        )

        event_flag.loc[
            true_by_encounter > 0
        ] = True

        event_flag.loc[
            (true_by_encounter == 0)
            &
            (false_by_encounter > 0)
        ] = False

        event_signal = (
            encounter_key_frame.copy()
        )

        event_signal[
            "rss_event_flag"
        ] = event_flag

        history = (
            order_base
            .merge(
                event_signal,
                on=[
                    patient_column,
                    encounter_column,
                ],
                how="left",
                validate="one_to_one",
            )
            .sort_values(
                by=[
                    patient_column,
                    "rss_canonical_order_position",
                    encounter_column,
                ],
                kind="mergesort",
                na_position="last",
            )
            .reset_index(drop=True)
        )

        patient_groups = history.groupby(
            patient_column,
            dropna=False,
        )

        history[
            "_rss_previous_event_flag"
        ] = patient_groups[
            "rss_event_flag"
        ].shift(1)

        history[
            "_rss_previous_temporal_start"
        ] = patient_groups[
            "rss_temporal_start"
        ].shift(1)

        history[
            "_rss_previous_row_eligible"
        ] = patient_groups[
            "rss_canonical_order_row_eligible"
        ].shift(1)

        history[
            "_rss_patient_order_index"
        ] = patient_groups.cumcount()

        current_observed = (
            history[
                "rss_event_flag"
            ]
            .notna()
        )

        previous_observed = (
            history[
                "_rss_previous_event_flag"
            ]
            .notna()
        )

        pair_eligible = (
            history[
                "rss_canonical_order_row_eligible"
            ]
            .eq(True)
            &
            history[
                "_rss_previous_row_eligible"
            ]
            .fillna(False)
            .eq(True)
            &
            current_observed
            &
            previous_observed
            &
            history[
                "rss_temporal_start"
            ]
            .notna()
            &
            history[
                "_rss_previous_temporal_start"
            ]
            .notna()
        )

        strict_forward_pair = (
            pair_eligible
            &
            history[
                "rss_temporal_start"
            ]
            .gt(
                history[
                    "_rss_previous_temporal_start"
                ]
            )
        )

        temporal_tie_pair = (
            pair_eligible
            &
            history[
                "rss_temporal_start"
            ]
            .eq(
                history[
                    "_rss_previous_temporal_start"
                ]
            )
        )

        current_positive = (
            history[
                "rss_event_flag"
            ]
            .eq(True)
        )

        current_nonpositive = (
            history[
                "rss_event_flag"
            ]
            .eq(False)
        )

        previous_positive = (
            history[
                "_rss_previous_event_flag"
            ]
            .eq(True)
        )

        previous_nonpositive = (
            history[
                "_rss_previous_event_flag"
            ]
            .eq(False)
        )

        positive_after_nonpositive = (
            strict_forward_pair
            &
            current_positive
            &
            previous_nonpositive
        )

        consecutive_positive = (
            strict_forward_pair
            &
            current_positive
            &
            previous_positive
        )

        nonpositive_after_positive = (
            strict_forward_pair
            &
            current_nonpositive
            &
            previous_positive
        )

        first_observation_positive = (
            history[
                "_rss_patient_order_index"
            ]
            .eq(0)
            &
            history[
                "rss_canonical_order_row_eligible"
            ]
            .eq(True)
            &
            current_positive
        )

        temporal_tie_state_change = (
            temporal_tie_pair
            &
            history[
                "rss_event_flag"
            ]
            .ne(
                history[
                    "_rss_previous_event_flag"
                ]
            )
        )

        temporal_tie_positive_pair = (
            temporal_tie_pair
            &
            (
                current_positive
                |
                previous_positive
            )
        )

        positive_episode_start = (
            first_observation_positive
            |
            positive_after_nonpositive
        )

        history[
            "_rss_positive_episode_start"
        ] = positive_episode_start

        positive_counts_by_patient = (
            history
            .groupby(
                patient_column,
                dropna=False,
            )[
                "rss_event_flag"
            ]
            .apply(
                lambda values: int(
                    values.eq(True).sum()
                )
            )
        )

        eligible_counts_by_patient = (
            history
            .groupby(
                patient_column,
                dropna=False,
            )[
                "rss_canonical_order_row_eligible"
            ]
            .sum()
        )

        episode_starts_by_patient = (
            history
            .groupby(
                patient_column,
                dropna=False,
            )[
                "_rss_positive_episode_start"
            ]
            .sum()
        )

        positive_patient_count = int(
            positive_counts_by_patient
            .gt(0)
            .sum()
        )

        multiple_positive_patient_count = int(
            positive_counts_by_patient
            .gt(1)
            .sum()
        )

        transition_testable_patient_count = int(
            eligible_counts_by_patient
            .ge(2)
            .sum()
        )

        repeated_positive_episode_patient_count = int(
            episode_starts_by_patient
            .ge(2)
            .sum()
        )

        maximum_positive_episode_count = int(
            episode_starts_by_patient.max()
            if len(
                episode_starts_by_patient
            )
            else 0
        )

        repeated_event_behavior_observed = bool(
            repeated_positive_episode_patient_count
            > 0
        )

        within_encounter_variation_count = int(
            within_encounter_variation.sum()
        )

        temporal_tie_state_change_count = int(
            temporal_tie_state_change.sum()
        )

        if within_encounter_variation_count > 0:
            transition_readiness_state = (
                "WITHIN_ENCOUNTER_EVENT_VARIATION_"
                "REVIEW_REQUIRED"
            )

        elif temporal_tie_state_change_count > 0:
            transition_readiness_state = (
                "EVENT_TRANSITION_POLICY_AND_"
                "TEMPORAL_TIE_REVIEW_REQUIRED"
            )

        elif repeated_event_behavior_observed:
            transition_readiness_state = (
                "EVENT_TRANSITION_AND_"
                "RECURRENCE_POLICY_REQUIRED"
            )

        else:
            transition_readiness_state = (
                "EVENT_TRANSITION_POLICY_REQUIRED"
            )

        rows.append({
            "column":
                column,

            "semantic_disposition":
                candidate.semantic_disposition,

            "evidence_orientation":
                candidate.evidence_orientation,

            "governed_evidence_domain":
                candidate.governed_evidence_domain,

            "transition_policy_scope":
                _RSS_EVENT_TRANSITION_POLICY_SCOPE.get(
                    candidate.semantic_disposition,
                    "MANUAL_TRANSITION_POLICY_REQUIRED",
                ),

            "total_encounter_count":
                total_encounter_count,

            "observed_encounter_count":
                int(
                    (
                        observed_by_encounter > 0
                    )
                    .sum()
                ),

            "positive_encounter_count":
                int(
                    (
                        true_by_encounter > 0
                    )
                    .sum()
                ),

            "positive_patient_count":
                positive_patient_count,

            "multiple_positive_encounter_patient_count":
                multiple_positive_patient_count,

            "transition_testable_patient_count":
                transition_testable_patient_count,

            "first_observation_positive_count":
                int(
                    first_observation_positive.sum()
                ),

            "positive_after_nonpositive_count":
                int(
                    positive_after_nonpositive.sum()
                ),

            "consecutive_positive_observation_count":
                int(
                    consecutive_positive.sum()
                ),

            "nonpositive_after_positive_count":
                int(
                    nonpositive_after_positive.sum()
                ),

            "strict_forward_transition_pair_count":
                int(
                    strict_forward_pair.sum()
                ),

            "temporal_tie_pair_count":
                int(
                    temporal_tie_pair.sum()
                ),

            "temporal_tie_positive_pair_count":
                int(
                    temporal_tie_positive_pair.sum()
                ),

            "temporal_tie_state_change_count":
                temporal_tie_state_change_count,

            "within_encounter_variation_count":
                within_encounter_variation_count,

            "positive_episode_start_count":
                int(
                    positive_episode_start.sum()
                ),

            "repeated_positive_episode_patient_count":
                repeated_positive_episode_patient_count,

            "maximum_positive_episode_count_per_patient":
                maximum_positive_episode_count,

            "repeated_event_behavior_observed":
                repeated_event_behavior_observed,

            "transition_readiness_state":
                transition_readiness_state,

            "transition_policy_defined":
                False,

            "persistence_policy_defined":
                False,

            "temporal_tie_policy_defined":
                False,

            "cross_signal_conflict_policy_defined":
                False,

            "automatic_boundary_use_authorized":
                False,

            "replay_cycle_formation_authorized":
                False,

            "nid_authorization_state":
                "NID_REVIEW_PENDING",

            "positive_value_meaning":
                candidate.positive_value_meaning,

            "negative_value_meaning":
                candidate.negative_value_meaning,
        })

    result = pd.DataFrame(rows)

    result[
        "_transition_review_priority"
    ] = (
        result[
            "transition_readiness_state"
        ]
        .map(
            _RSS_EVENT_TRANSITION_REVIEW_PRIORITY
        )
        .fillna(99)
    )

    return (
        result
        .sort_values(
            by=[
                "_transition_review_priority",
                "semantic_disposition",
                "column",
            ]
        )
        .drop(
            columns="_transition_review_priority"
        )
        .reset_index(drop=True)
    )


def build_rss_binary_event_transition_summary(
    transition_review_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize RSS event-transition readiness.

    Transition authority and replay-cycle formation remain withheld.
    """

    if not isinstance(
        transition_review_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "transition_review_df must be a pandas DataFrame."
        )

    required_columns = {
        "semantic_disposition",
        "transition_policy_scope",
        "transition_readiness_state",
        "repeated_event_behavior_observed",
        "automatic_boundary_use_authorized",
        "replay_cycle_formation_authorized",
        "nid_authorization_state",
    }

    missing_columns = sorted(
        required_columns
        - set(transition_review_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "RSS event-transition summary cannot proceed. "
            f"Missing review columns: {missing_columns}"
        )

    return (
        transition_review_df
        .groupby(
            [
                "semantic_disposition",
                "transition_policy_scope",
                "transition_readiness_state",
                "repeated_event_behavior_observed",
                "automatic_boundary_use_authorized",
                "replay_cycle_formation_authorized",
                "nid_authorization_state",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            by=[
                "repeated_event_behavior_observed",
                "count",
                "semantic_disposition",
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
# Chapter 68.24 — RSS Binary Event Episode Policy
#============================================================

_RSS_BINARY_EVENT_EPISODE_POLICY = {

    "care_escalation_occurred": {
        "episode_family":
            "ESCALATION_EPISODE",

        "episode_model":
            "STATEFUL_EVENT_EPISODE",

        "onset_rule":
            (
                "A strictly later False-to-True transition "
                "creates an observed escalation episode onset."
            ),

        "positive_continuation_rule":
            (
                "A strictly later True-to-True transition remains "
                "within the current escalation episode."
            ),

        "positive_to_nonpositive_rule":
            (
                "True-to-False means escalation is no longer "
                "observed through this field; it does not prove "
                "resolution, recovery, or stability."
            ),

        "recurrence_rule":
            (
                "A later False-to-True transition following at "
                "least one intervening nonpositive observation "
                "creates a candidate recurring escalation episode."
            ),

        "left_censor_rule":
            (
                "A first-observation positive creates a "
                "left-censored escalation episode."
            ),

        "temporal_tie_rule":
            (
                "Equal states across tied time create no "
                "transition. Different states across tied time "
                "remain unresolved and quarantined."
            ),

        "episode_separation_rule":
            (
                "An intervening strictly later nonpositive "
                "observation separates candidate escalation "
                "episodes."
            ),

        "required_alignment":
            (
                "Escalation context and cross-signal contradiction "
                "review are required before boundary authorization."
            ),

        "persistence_policy_defined":
            True,

        "episode_construction_review_eligible":
            True,

        "episode_policy_readiness_state":
            (
                "ESCALATION_EPISODE_POLICY_DEFINED_"
                "CONFLICT_POLICY_PENDING"
            ),
    },

    "stepdown_rebound_flag": {
        "episode_family":
            "REBOUND_INSTABILITY_EPISODE",

        "episode_model":
            "DISCRETE_EVENT_EPISODE",

        "onset_rule":
            (
                "A strictly later False-to-True transition "
                "creates an observed rebound-instability onset."
            ),

        "positive_continuation_rule":
            (
                "A strictly later True-to-True transition is "
                "preserved as continuation or repeated "
                "documentation of the same rebound event."
            ),

        "positive_to_nonpositive_rule":
            (
                "True-to-False means rebound is no longer "
                "observed through this field; it does not "
                "establish stability."
            ),

        "recurrence_rule":
            (
                "A later False-to-True transition following "
                "intervening nonpositive evidence creates a "
                "candidate recurring rebound episode."
            ),

        "left_censor_rule":
            (
                "A first-observation positive creates a "
                "left-censored rebound episode."
            ),

        "temporal_tie_rule":
            (
                "Equal states across tied time create no "
                "transition. Different states across tied time "
                "remain unresolved and quarantined."
            ),

        "episode_separation_rule":
            (
                "An intervening strictly later nonpositive "
                "observation separates candidate rebound events."
            ),

        "required_alignment":
            (
                "A verified stepdown or de-escalation context "
                "must be aligned before rebound boundary "
                "authority is granted."
            ),

        "persistence_policy_defined":
            True,

        "episode_construction_review_eligible":
            True,

        "episode_policy_readiness_state":
            (
                "REBOUND_EVENT_POLICY_DEFINED_"
                "STEPDOWN_ALIGNMENT_REQUIRED"
            ),
    },

    "false_vital_recovery_signal": {
        "episode_family":
            "FALSE_RECOVERY_CONTRADICTION_EPISODE",

        "episode_model":
            "CONTRADICTION_EPISODE",

        "onset_rule":
            (
                "A strictly later False-to-True transition "
                "creates an observed false-recovery "
                "contradiction onset."
            ),

        "positive_continuation_rule":
            (
                "A strictly later True-to-True transition "
                "remains within the current recovery-"
                "contradiction episode."
            ),

        "positive_to_nonpositive_rule":
            (
                "True-to-False means this contradiction signal "
                "is no longer observed; it does not prove "
                "authentic or durable recovery."
            ),

        "recurrence_rule":
            (
                "A later False-to-True transition following "
                "intervening nonpositive evidence creates a "
                "candidate recurring false-recovery episode."
            ),

        "left_censor_rule":
            (
                "A first-observation positive creates a "
                "left-censored recovery-contradiction episode."
            ),

        "temporal_tie_rule":
            (
                "Equal states across tied time create no "
                "transition. Different states across tied time "
                "remain unresolved and quarantined."
            ),

        "episode_separation_rule":
            (
                "An intervening strictly later nonpositive "
                "observation separates candidate contradiction "
                "episodes."
            ),

        "required_alignment":
            (
                "Affirmative recovery evidence or a recovery "
                "claim must be aligned before this episode may "
                "support failed-recovery replay."
            ),

        "persistence_policy_defined":
            True,

        "episode_construction_review_eligible":
            True,

        "episode_policy_readiness_state":
            (
                "FALSE_RECOVERY_EPISODE_POLICY_DEFINED_"
                "RECOVERY_ALIGNMENT_REQUIRED"
            ),
    },

    "readmission_flag": {
        "episode_family":
            "READMISSION_OUTCOME_EPISODE",

        "episode_model":
            "OUTCOME_EVENT_GRAIN_UNRESOLVED",

        "onset_rule":
            (
                "A strictly later False-to-True transition "
                "creates a candidate readmission onset, pending "
                "return-to-care alignment."
            ),

        "positive_continuation_rule":
            (
                "Consecutive True observations remain outcome-"
                "grain ambiguous and must not yet be merged or "
                "counted as separate readmissions."
            ),

        "positive_to_nonpositive_rule":
            (
                "True-to-False means readmission is no longer "
                "observed through this flag; it does not prove "
                "discharge or durable stability."
            ),

        "recurrence_rule":
            (
                "A later False-to-True transition may represent "
                "another readmission only after discharge and "
                "return-to-care alignment is verified."
            ),

        "left_censor_rule":
            (
                "A first-observation positive is left-censored "
                "and cannot be counted as an observed readmission "
                "onset."
            ),

        "temporal_tie_rule":
            (
                "Equal states across tied time create no "
                "transition. Different states across tied time "
                "remain unresolved and quarantined."
            ),

        "episode_separation_rule":
            (
                "Episode separation remains undefined until "
                "discharge, encounter type, or return-to-care "
                "evidence is aligned."
            ),

        "required_alignment":
            (
                "Discharge and return-to-care evidence or "
                "equivalent encounter-grain validation is "
                "required."
            ),

        "persistence_policy_defined":
            False,

        "episode_construction_review_eligible":
            False,

        "episode_policy_readiness_state":
            (
                "READMISSION_EPISODE_POLICY_DEFINED_"
                "OUTCOME_GRAIN_ALIGNMENT_REQUIRED"
            ),
    },
}


_RSS_EPISODE_POLICY_PRIORITY = {
    "STATEFUL_EVENT_EPISODE": 0,
    "DISCRETE_EVENT_EPISODE": 1,
    "CONTRADICTION_EPISODE": 2,
    "OUTCOME_EVENT_GRAIN_UNRESOLVED": 3,
}


def build_rss_binary_event_episode_policy(
    transition_review_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach field-specific episode policies to the event-capable
    RSS binary signals.

    The policy defines:
    - onset interpretation
    - positive continuation
    - positive-to-nonpositive interpretation
    - recurrence
    - left censoring
    - temporal-tie handling
    - episode separation
    - required evidence alignment

    This function does not construct episodes or authorize
    replay-cycle formation.
    """

    if not isinstance(
        transition_review_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "transition_review_df must be a pandas DataFrame."
        )

    required_columns = {
        "column",
        "semantic_disposition",
        "transition_policy_scope",
        "positive_encounter_count",
        "first_observation_positive_count",
        "positive_after_nonpositive_count",
        "consecutive_positive_observation_count",
        "nonpositive_after_positive_count",
        "temporal_tie_state_change_count",
        "within_encounter_variation_count",
        "repeated_positive_episode_patient_count",
        "maximum_positive_episode_count_per_patient",
    }

    missing_columns = sorted(
        required_columns
        - set(transition_review_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "RSS binary event episode policy cannot proceed. "
            f"Missing transition-review columns: {missing_columns}"
        )

    if transition_review_df["column"].duplicated().any():
        raise ValueError(
            "RSS transition review contains duplicated "
            "event-capable columns."
        )

    undefined_policy_columns = sorted(
        set(
            transition_review_df["column"]
            .astype(str)
        )
        - set(_RSS_BINARY_EVENT_EPISODE_POLICY)
    )

    if undefined_policy_columns:
        raise KeyError(
            "RSS episode policies are undefined for: "
            f"{undefined_policy_columns}"
        )

    rows: list[dict[str, object]] = []

    for candidate in transition_review_df.itertuples(
        index=False
    ):
        policy = (
            _RSS_BINARY_EVENT_EPISODE_POLICY[
                candidate.column
            ]
        )

        structural_transition_ready = (
            int(
                candidate.temporal_tie_state_change_count
            ) == 0
            and
            int(
                candidate.within_encounter_variation_count
            ) == 0
        )

        construction_review_eligible = (
            bool(
                policy[
                    "episode_construction_review_eligible"
                ]
            )
            and structural_transition_ready
        )

        if not structural_transition_ready:
            readiness_state = (
                "EPISODE_POLICY_DEFINED_"
                "STRUCTURAL_EXCEPTION_REVIEW_REQUIRED"
            )
        else:
            readiness_state = policy[
                "episode_policy_readiness_state"
            ]

        rows.append({
            "column":
                candidate.column,

            "semantic_disposition":
                candidate.semantic_disposition,

            "transition_policy_scope":
                candidate.transition_policy_scope,

            "episode_family":
                policy["episode_family"],

            "episode_model":
                policy["episode_model"],

            "onset_rule":
                policy["onset_rule"],

            "positive_continuation_rule":
                policy[
                    "positive_continuation_rule"
                ],

            "positive_to_nonpositive_rule":
                policy[
                    "positive_to_nonpositive_rule"
                ],

            "recurrence_rule":
                policy["recurrence_rule"],

            "left_censor_rule":
                policy["left_censor_rule"],

            "temporal_tie_rule":
                policy["temporal_tie_rule"],

            "episode_separation_rule":
                policy["episode_separation_rule"],

            "required_alignment":
                policy["required_alignment"],

            "positive_encounter_count":
                candidate.positive_encounter_count,

            "left_censored_positive_count":
                candidate.first_observation_positive_count,

            "observed_positive_onset_count":
                candidate.positive_after_nonpositive_count,

            "positive_continuation_count":
                candidate.consecutive_positive_observation_count,

            "positive_to_nonpositive_observation_count":
                candidate.nonpositive_after_positive_count,

            "repeated_positive_episode_patient_count":
                (
                    candidate
                    .repeated_positive_episode_patient_count
                ),

            "maximum_positive_episode_count_per_patient":
                (
                    candidate
                    .maximum_positive_episode_count_per_patient
                ),

            "structural_transition_ready":
                structural_transition_ready,

            "episode_policy_defined":
                True,

            "transition_policy_defined":
                True,

            "persistence_policy_defined":
                policy[
                    "persistence_policy_defined"
                ],

            "recurrence_policy_defined":
                True,

            "left_censor_policy_defined":
                True,

            "temporal_tie_policy_defined":
                True,

            "cross_signal_conflict_policy_defined":
                False,

            "episode_construction_review_eligible":
                construction_review_eligible,

            "episode_policy_readiness_state":
                readiness_state,

            "automatic_boundary_use_authorized":
                False,

            "replay_cycle_formation_authorized":
                False,

            "nid_authorization_state":
                "NID_POLICY_REVIEW_PENDING",
        })

    result = pd.DataFrame(rows)

    result["_episode_policy_priority"] = (
        result["episode_model"]
        .map(_RSS_EPISODE_POLICY_PRIORITY)
        .fillna(99)
    )

    return (
        result
        .sort_values(
            by=[
                "_episode_policy_priority",
                "column",
            ]
        )
        .drop(
            columns="_episode_policy_priority"
        )
        .reset_index(drop=True)
    )


def build_rss_binary_event_episode_policy_summary(
    episode_policy_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize field-specific RSS binary event episode policies.

    Operational boundary authority and replay-cycle formation
    remain withheld.
    """

    if not isinstance(
        episode_policy_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "episode_policy_df must be a pandas DataFrame."
        )

    required_columns = {
        "episode_family",
        "episode_model",
        "episode_policy_readiness_state",
        "episode_construction_review_eligible",
        "persistence_policy_defined",
        "cross_signal_conflict_policy_defined",
        "automatic_boundary_use_authorized",
        "replay_cycle_formation_authorized",
        "nid_authorization_state",
    }

    missing_columns = sorted(
        required_columns
        - set(episode_policy_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "RSS episode policy summary cannot proceed. "
            f"Missing episode-policy columns: {missing_columns}"
        )

    return (
        episode_policy_df
        .groupby(
            [
                "episode_family",
                "episode_model",
                "episode_policy_readiness_state",
                "episode_construction_review_eligible",
                "persistence_policy_defined",
                "cross_signal_conflict_policy_defined",
                "automatic_boundary_use_authorized",
                "replay_cycle_formation_authorized",
                "nid_authorization_state",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            by=[
                "episode_construction_review_eligible",
                "count",
                "episode_family",
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
# Chapter 68.25 — RSS Controlled Binary Episode
#                  Construction
#============================================================

_RSS_CONTROLLED_EPISODE_PRIORITY = {
    "ESCALATION_EPISODE": 0,
    "REBOUND_INSTABILITY_EPISODE": 1,
    "FALSE_RECOVERY_CONTRADICTION_EPISODE": 2,
}


def build_rss_controlled_binary_episode_construction(
    episode_policy_df: pd.DataFrame,
    canonical_df: pd.DataFrame,
    ordering_contract_df: pd.DataFrame,
    patient_column: str = "patient_id",
    encounter_column: str = "encounter_id",
) -> pd.DataFrame:
    """
    Construct provisional binary-event episodes for RSS.

    Only fields explicitly eligible for episode-construction
    review are processed.

    Construction preserves:
    - observed onset
    - left censoring
    - internal censoring after missing evidence
    - positive continuation
    - observed positive-evidence termination
    - right censoring
    - co-temporal encounter grouping
    - recurrence candidacy
    - unresolved alignment and governance requirements

    This function does not authorize final replay cycles or
    generate RSS pattern and lesson states.
    """

    if not isinstance(
        episode_policy_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "episode_policy_df must be a pandas DataFrame."
        )

    if not isinstance(
        canonical_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "canonical_df must be a pandas DataFrame."
        )

    if not isinstance(
        ordering_contract_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "ordering_contract_df must be a pandas DataFrame."
        )

    required_policy_columns = {
        "column",
        "semantic_disposition",
        "episode_family",
        "episode_model",
        "required_alignment",
        "episode_construction_review_eligible",
        "episode_policy_readiness_state",
    }

    missing_policy_columns = sorted(
        required_policy_columns
        - set(episode_policy_df.columns)
    )

    if missing_policy_columns:
        raise KeyError(
            "RSS controlled episode construction cannot proceed. "
            f"Missing episode-policy columns: "
            f"{missing_policy_columns}"
        )

    required_order_columns = {
        patient_column,
        encounter_column,
        "rss_temporal_start",
        "rss_temporal_end",
        "rss_canonical_order_position",
        "rss_canonical_order_row_eligible",
        "rss_replay_chronology_eligible",
        "rss_primary_temporal_tie_flag",
    }

    missing_order_columns = sorted(
        required_order_columns
        - set(ordering_contract_df.columns)
    )

    if missing_order_columns:
        raise KeyError(
            "RSS controlled episode construction cannot proceed. "
            f"Missing ordering-contract columns: "
            f"{missing_order_columns}"
        )

    eligible_policies = (
        episode_policy_df.loc[
            episode_policy_df[
                "episode_construction_review_eligible"
            ].eq(True)
        ]
        .copy()
        .reset_index(drop=True)
    )

    result_columns = [
        patient_column,
        "rss_episode_id",
        "rss_source_signal",
        "rss_semantic_disposition",
        "rss_episode_family",
        "rss_episode_model",
        "rss_episode_ordinal",
        "rss_episode_start_type",
        "rss_episode_end_state",
        "rss_episode_start_time",
        "rss_episode_end_time",
        "rss_episode_start_order_position",
        "rss_episode_end_order_position",
        "rss_episode_first_encounter_id",
        "rss_episode_last_encounter_id",
        "rss_episode_next_nonpositive_time",
        "rss_episode_positive_temporal_group_count",
        "rss_episode_positive_encounter_count",
        "rss_episode_temporal_tie_group_count",
        "rss_episode_contains_temporal_tie",
        "rss_episode_left_censored_flag",
        "rss_episode_observed_onset_flag",
        "rss_episode_internal_censoring_flag",
        "rss_episode_termination_observed_flag",
        "rss_episode_right_censored_flag",
        "rss_patient_total_episode_count",
        "rss_patient_observed_onset_episode_count",
        "rss_recurrence_candidate_flag",
        "rss_patient_replay_chronology_eligible",
        "rss_required_alignment",
        "rss_episode_policy_readiness_state",
        "rss_episode_construction_state",
        "rss_episode_construction_review_eligible",
        "rss_cross_signal_conflict_policy_defined",
        "rss_automatic_boundary_use_authorized",
        "rss_replay_cycle_formation_authorized",
        "rss_nid_authorization_state",
    ]

    if eligible_policies.empty:
        return pd.DataFrame(
            columns=result_columns
        )

    # --------------------------------------------------------
    # Normalize canonical patient-encounter identifiers once.
    # --------------------------------------------------------

    normalized_keys = canonical_df[
        [
            patient_column,
            encounter_column,
        ]
    ].copy()

    for key in (
        patient_column,
        encounter_column,
    ):
        normalized_keys[key] = (
            normalized_keys[key]
            .astype("string")
            .str.strip()
        )

    missing_key_mask = (
        normalized_keys[
            [
                patient_column,
                encounter_column,
            ]
        ]
        .isna()
        .any(axis=1)
        |
        normalized_keys[
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
            "Canonical RSS dataframe contains "
            f"{int(missing_key_mask.sum())} rows with "
            "missing normalized patient-encounter keys."
        )

    normalized_keys = (
        normalized_keys
        .reset_index(drop=True)
    )

    encounter_index = pd.MultiIndex.from_frame(
        normalized_keys
    )

    encounter_codes, unique_encounters = (
        pd.factorize(
            encounter_index,
            sort=False,
        )
    )

    encounter_key_frame = (
        unique_encounters
        .to_frame(index=False)
    )

    encounter_key_frame.columns = [
        patient_column,
        encounter_column,
    ]

    total_encounter_count = len(
        encounter_key_frame
    )

    # --------------------------------------------------------
    # Prepare governed chronology.
    # --------------------------------------------------------

    order_base = ordering_contract_df[
        [
            patient_column,
            encounter_column,
            "rss_temporal_start",
            "rss_temporal_end",
            "rss_canonical_order_position",
            "rss_canonical_order_row_eligible",
            "rss_replay_chronology_eligible",
            "rss_primary_temporal_tie_flag",
        ]
    ].copy()

    for key in (
        patient_column,
        encounter_column,
    ):
        order_base[key] = (
            order_base[key]
            .astype("string")
            .str.strip()
        )

    if order_base.duplicated(
        subset=[
            patient_column,
            encounter_column,
        ]
    ).any():
        raise ValueError(
            "RSS ordering contract contains duplicated "
            "patient-encounter keys."
        )

    key_alignment = (
        encounter_key_frame[
            [
                patient_column,
                encounter_column,
            ]
        ]
        .merge(
            order_base[
                [
                    patient_column,
                    encounter_column,
                ]
            ],
            on=[
                patient_column,
                encounter_column,
            ],
            how="outer",
            indicator=True,
            validate="one_to_one",
        )
    )

    unmatched_key_count = int(
        (
            ~key_alignment["_merge"].eq("both")
        )
        .sum()
    )

    if unmatched_key_count:
        raise ValueError(
            "Canonical evidence and RSS ordering keys "
            "do not align. "
            f"Unmatched encounters: {unmatched_key_count}"
        )

    episode_results: list[pd.DataFrame] = []

    # --------------------------------------------------------
    # Construct each governed provisional episode family.
    # --------------------------------------------------------

    for policy in eligible_policies.itertuples(
        index=False
    ):
        column = policy.column

        if column not in canonical_df.columns:
            raise KeyError(
                f"Episode-eligible RSS field {column!r} "
                "is missing from canonical_df."
            )

        binary_signal = (
            _coerce_rss_binary_signal(
                canonical_df[column]
            )
        )

        if binary_signal is None:
            raise ValueError(
                f"{column!r} is episode eligible but cannot "
                "be converted to a binary signal."
            )

        binary_signal = (
            binary_signal
            .reset_index(drop=True)
        )

        valid_mask = (
            binary_signal
            .notna()
            .to_numpy()
        )

        true_mask = (
            binary_signal
            .fillna(False)
            .astype(bool)
            .to_numpy()
        )

        false_mask = (
            valid_mask
            &
            ~true_mask
        )

        observed_by_encounter = np.bincount(
            encounter_codes[valid_mask],
            minlength=total_encounter_count,
        )

        true_by_encounter = np.bincount(
            encounter_codes[
                valid_mask & true_mask
            ],
            minlength=total_encounter_count,
        )

        false_by_encounter = np.bincount(
            encounter_codes[false_mask],
            minlength=total_encounter_count,
        )

        mixed_encounter_mask = (
            (true_by_encounter > 0)
            &
            (false_by_encounter > 0)
        )

        mixed_encounter_count = int(
            mixed_encounter_mask.sum()
        )

        if mixed_encounter_count:
            raise ValueError(
                f"{column!r} contains "
                f"{mixed_encounter_count:,} encounters with "
                "both positive and negative observations. "
                "Controlled episode construction requires "
                "within-encounter variation review first."
            )

        event_flag = pd.Series(
            pd.NA,
            index=range(total_encounter_count),
            dtype="boolean",
        )

        event_flag.loc[
            true_by_encounter > 0
        ] = True

        event_flag.loc[
            (true_by_encounter == 0)
            &
            (false_by_encounter > 0)
        ] = False

        event_signal = (
            encounter_key_frame.copy()
        )

        event_signal[
            "rss_event_flag"
        ] = event_flag

        history = (
            order_base
            .merge(
                event_signal,
                on=[
                    patient_column,
                    encounter_column,
                ],
                how="left",
                validate="one_to_one",
            )
        )

        # Quarantined chronology rows do not participate in
        # controlled episode construction.
        history = (
            history.loc[
                history[
                    "rss_canonical_order_row_eligible"
                ].eq(True)
            ]
            .copy()
        )

        # ----------------------------------------------------
        # Consolidate co-temporal encounters.
        #
        # Deterministic encounter ordering must not manufacture
        # a transition between encounters sharing the same time.
        # ----------------------------------------------------

        temporal = (
            history
            .groupby(
                [
                    patient_column,
                    "rss_temporal_start",
                ],
                as_index=False,
                sort=False,
                dropna=False,
            )
            .agg(
                rss_temporal_end=(
                    "rss_temporal_end",
                    "max",
                ),
                rss_temporal_first_order_position=(
                    "rss_canonical_order_position",
                    "min",
                ),
                rss_temporal_last_order_position=(
                    "rss_canonical_order_position",
                    "max",
                ),
                rss_temporal_first_encounter_id=(
                    encounter_column,
                    "first",
                ),
                rss_temporal_last_encounter_id=(
                    encounter_column,
                    "last",
                ),
                rss_temporal_encounter_count=(
                    encounter_column,
                    "nunique",
                ),
                rss_temporal_event_true_count=(
                    "rss_event_flag",
                    lambda values: int(
                        values.eq(True).sum()
                    ),
                ),
                rss_temporal_event_false_count=(
                    "rss_event_flag",
                    lambda values: int(
                        values.eq(False).sum()
                    ),
                ),
                rss_temporal_event_missing_count=(
                    "rss_event_flag",
                    lambda values: int(
                        values.isna().sum()
                    ),
                ),
                rss_patient_replay_chronology_eligible=(
                    "rss_replay_chronology_eligible",
                    lambda values: bool(
                        values
                        .fillna(False)
                        .all()
                    ),
                ),
            )
        )

        temporal_conflict_mask = (
            temporal[
                "rss_temporal_event_true_count"
            ].gt(0)
            &
            temporal[
                "rss_temporal_event_false_count"
            ].gt(0)
        )

        temporal_conflict_count = int(
            temporal_conflict_mask.sum()
        )

        if temporal_conflict_count:
            raise ValueError(
                f"{column!r} contains "
                f"{temporal_conflict_count:,} co-temporal "
                "state conflicts. Episode construction must "
                "remain deferred until tie-state review."
            )

        temporal[
            "rss_temporal_event_flag"
        ] = pd.Series(
            pd.NA,
            index=temporal.index,
            dtype="boolean",
        )

        temporal.loc[
            temporal[
                "rss_temporal_event_true_count"
            ].gt(0),
            "rss_temporal_event_flag",
        ] = True

        temporal.loc[
            temporal[
                "rss_temporal_event_true_count"
            ].eq(0)
            &
            temporal[
                "rss_temporal_event_false_count"
            ].gt(0),
            "rss_temporal_event_flag",
        ] = False

        temporal[
            "rss_temporal_tie_group_flag"
        ] = (
            temporal[
                "rss_temporal_encounter_count"
            ]
            .gt(1)
        )

        temporal = (
            temporal
            .sort_values(
                by=[
                    patient_column,
                    "rss_temporal_start",
                    "rss_temporal_first_order_position",
                ],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )

        patient_groups = temporal.groupby(
            patient_column,
            dropna=False,
        )

        temporal[
            "rss_patient_temporal_index"
        ] = patient_groups.cumcount()

        temporal[
            "rss_previous_event_flag"
        ] = patient_groups[
            "rss_temporal_event_flag"
        ].shift(1)

        temporal[
            "rss_next_event_flag"
        ] = patient_groups[
            "rss_temporal_event_flag"
        ].shift(-1)

        temporal[
            "rss_next_temporal_start"
        ] = patient_groups[
            "rss_temporal_start"
        ].shift(-1)

        current_positive = (
            temporal[
                "rss_temporal_event_flag"
            ].eq(True)
        )

        previous_nonpositive = (
            temporal[
                "rss_previous_event_flag"
            ].eq(False)
        )

        previous_missing = (
            temporal[
                "rss_previous_event_flag"
            ].isna()
        )

        first_temporal_observation = (
            temporal[
                "rss_patient_temporal_index"
            ].eq(0)
        )

        left_censored_start = (
            current_positive
            &
            first_temporal_observation
        )

        observed_onset = (
            current_positive
            &
            ~first_temporal_observation
            &
            previous_nonpositive
        )

        internal_censored_start = (
            current_positive
            &
            ~first_temporal_observation
            &
            previous_missing
        )

        temporal[
            "rss_episode_start_flag"
        ] = (
            left_censored_start
            |
            observed_onset
            |
            internal_censored_start
        )

        temporal[
            "rss_episode_ordinal"
        ] = (
            temporal
            .groupby(
                patient_column,
                dropna=False,
            )[
                "rss_episode_start_flag"
            ]
            .cumsum()
            .astype("Int64")
        )

        temporal[
            "rss_episode_start_type"
        ] = pd.Series(
            pd.NA,
            index=temporal.index,
            dtype="string",
        )

        temporal.loc[
            left_censored_start,
            "rss_episode_start_type",
        ] = "LEFT_CENSORED"

        temporal.loc[
            observed_onset,
            "rss_episode_start_type",
        ] = "OBSERVED_ONSET"

        temporal.loc[
            internal_censored_start,
            "rss_episode_start_type",
        ] = "INTERNAL_CENSORED_AFTER_MISSING"

        temporal[
            "rss_next_event_state"
        ] = "MISSING_EVIDENCE"

        temporal.loc[
            temporal[
                "rss_next_temporal_start"
            ].isna(),
            "rss_next_event_state",
        ] = "END_OF_HISTORY"

        temporal.loc[
            temporal[
                "rss_next_event_flag"
            ].eq(True),
            "rss_next_event_state",
        ] = "POSITIVE"

        temporal.loc[
            temporal[
                "rss_next_event_flag"
            ].eq(False),
            "rss_next_event_state",
        ] = "NONPOSITIVE"

        positive_temporal = (
            temporal.loc[
                current_positive
            ]
            .copy()
        )

        if positive_temporal.empty:
            continue

        episodes = (
            positive_temporal
            .groupby(
                [
                    patient_column,
                    "rss_episode_ordinal",
                ],
                as_index=False,
                sort=False,
                dropna=False,
            )
            .agg(
                rss_episode_start_type=(
                    "rss_episode_start_type",
                    "first",
                ),
                rss_episode_start_time=(
                    "rss_temporal_start",
                    "min",
                ),
                rss_episode_end_time=(
                    "rss_temporal_end",
                    "max",
                ),
                rss_episode_start_order_position=(
                    "rss_temporal_first_order_position",
                    "min",
                ),
                rss_episode_end_order_position=(
                    "rss_temporal_last_order_position",
                    "max",
                ),
                rss_episode_first_encounter_id=(
                    "rss_temporal_first_encounter_id",
                    "first",
                ),
                rss_episode_last_encounter_id=(
                    "rss_temporal_last_encounter_id",
                    "last",
                ),
                rss_episode_next_event_state=(
                    "rss_next_event_state",
                    "last",
                ),
                rss_episode_next_nonpositive_time=(
                    "rss_next_temporal_start",
                    "last",
                ),
                rss_episode_positive_temporal_group_count=(
                    "rss_temporal_start",
                    "size",
                ),
                rss_episode_positive_encounter_count=(
                    "rss_temporal_encounter_count",
                    "sum",
                ),
                rss_episode_temporal_tie_group_count=(
                    "rss_temporal_tie_group_flag",
                    "sum",
                ),
                rss_patient_replay_chronology_eligible=(
                    "rss_patient_replay_chronology_eligible",
                    "all",
                ),
            )
        )

        episodes[
            "rss_episode_left_censored_flag"
        ] = episodes[
            "rss_episode_start_type"
        ].eq("LEFT_CENSORED")

        episodes[
            "rss_episode_observed_onset_flag"
        ] = episodes[
            "rss_episode_start_type"
        ].eq("OBSERVED_ONSET")

        episodes[
            "rss_episode_internal_censoring_flag"
        ] = episodes[
            "rss_episode_start_type"
        ].eq(
            "INTERNAL_CENSORED_AFTER_MISSING"
        )

        episodes[
            "rss_episode_termination_observed_flag"
        ] = episodes[
            "rss_episode_next_event_state"
        ].eq("NONPOSITIVE")

        episodes[
            "rss_episode_right_censored_flag"
        ] = episodes[
            "rss_episode_next_event_state"
        ].eq("END_OF_HISTORY")

        episodes[
            "rss_episode_contains_temporal_tie"
        ] = episodes[
            "rss_episode_temporal_tie_group_count"
        ].gt(0)

        episodes[
            "rss_episode_end_state"
        ] = "END_UNRESOLVED"

        episodes.loc[
            episodes[
                "rss_episode_termination_observed_flag"
            ],
            "rss_episode_end_state",
        ] = "POSITIVE_EVIDENCE_TERMINATION_OBSERVED"

        episodes.loc[
            episodes[
                "rss_episode_right_censored_flag"
            ],
            "rss_episode_end_state",
        ] = "RIGHT_CENSORED_END_OF_HISTORY"

        start_component = pd.Series(
            "INTERNAL_CENSORED",
            index=episodes.index,
            dtype="string",
        )

        start_component.loc[
            episodes[
                "rss_episode_left_censored_flag"
            ]
        ] = "LEFT_CENSORED"

        start_component.loc[
            episodes[
                "rss_episode_observed_onset_flag"
            ]
        ] = "OBSERVED_ONSET"

        end_component = pd.Series(
            "END_UNRESOLVED",
            index=episodes.index,
            dtype="string",
        )

        end_component.loc[
            episodes[
                "rss_episode_termination_observed_flag"
            ]
        ] = "TERMINATION_OBSERVED"

        end_component.loc[
            episodes[
                "rss_episode_right_censored_flag"
            ]
        ] = "RIGHT_CENSORED"

        episodes[
            "rss_episode_construction_state"
        ] = (
            "PROVISIONAL_"
            + start_component
            + "_EPISODE_"
            + end_component
        )

        episodes[
            "rss_patient_total_episode_count"
        ] = (
            episodes
            .groupby(
                patient_column,
                dropna=False,
            )[
                "rss_episode_ordinal"
            ]
            .transform("count")
            .astype("Int64")
        )

        episodes[
            "rss_patient_observed_onset_episode_count"
        ] = (
            episodes
            .groupby(
                patient_column,
                dropna=False,
            )[
                "rss_episode_observed_onset_flag"
            ]
            .transform("sum")
            .astype("Int64")
        )

        episodes[
            "rss_recurrence_candidate_flag"
        ] = (
            episodes[
                "rss_patient_observed_onset_episode_count"
            ]
            .ge(2)
            &
            episodes[
                "rss_patient_replay_chronology_eligible"
            ]
            .fillna(False)
        )

        episodes[
            "rss_episode_id"
        ] = (
            episodes[
                patient_column
            ]
            .astype("string")
            + "::"
            + str(
                policy.episode_family
            )
            + "::"
            + episodes[
                "rss_episode_ordinal"
            ]
            .astype("string")
            .str.zfill(4)
        )

        episodes[
            "rss_source_signal"
        ] = column

        episodes[
            "rss_semantic_disposition"
        ] = policy.semantic_disposition

        episodes[
            "rss_episode_family"
        ] = policy.episode_family

        episodes[
            "rss_episode_model"
        ] = policy.episode_model

        episodes[
            "rss_required_alignment"
        ] = policy.required_alignment

        episodes[
            "rss_episode_policy_readiness_state"
        ] = policy.episode_policy_readiness_state

        episodes[
            "rss_episode_construction_review_eligible"
        ] = (
            episodes[
                "rss_patient_replay_chronology_eligible"
            ]
            .fillna(False)
            &
            ~episodes[
                "rss_episode_internal_censoring_flag"
            ]
        )

        episodes[
            "rss_cross_signal_conflict_policy_defined"
        ] = False

        episodes[
            "rss_automatic_boundary_use_authorized"
        ] = False

        episodes[
            "rss_replay_cycle_formation_authorized"
        ] = False

        episodes[
            "rss_nid_authorization_state"
        ] = "NID_EPISODE_REVIEW_PENDING"

        episode_results.append(
            episodes
        )

    if not episode_results:
        return pd.DataFrame(
            columns=result_columns
        )

    result = pd.concat(
        episode_results,
        ignore_index=True,
    )

    result[
        "_rss_episode_priority"
    ] = (
        result[
            "rss_episode_family"
        ]
        .map(
            _RSS_CONTROLLED_EPISODE_PRIORITY
        )
        .fillna(99)
    )

    return (
        result
        .sort_values(
            by=[
                "_rss_episode_priority",
                patient_column,
                "rss_episode_start_time",
                "rss_episode_ordinal",
            ],
            kind="mergesort",
        )
        .drop(
            columns=[
                "_rss_episode_priority",
                "rss_episode_next_event_state",
            ]
        )
        .reset_index(drop=True)
    )


def build_rss_controlled_binary_episode_summary(
    episode_df: pd.DataFrame,
    patient_column: str = "patient_id",
) -> pd.DataFrame:
    """
    Summarize provisional RSS binary event episodes.

    Episode construction remains provisional and unauthorized for
    final replay-cycle use.
    """

    if not isinstance(
        episode_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "episode_df must be a pandas DataFrame."
        )

    required_columns = {
        patient_column,
        "rss_episode_id",
        "rss_episode_family",
        "rss_episode_model",
        "rss_episode_start_type",
        "rss_episode_end_state",
        "rss_episode_contains_temporal_tie",
        "rss_recurrence_candidate_flag",
        "rss_episode_construction_review_eligible",
        "rss_automatic_boundary_use_authorized",
        "rss_replay_cycle_formation_authorized",
        "rss_nid_authorization_state",
    }

    missing_columns = sorted(
        required_columns
        - set(episode_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "RSS controlled episode summary cannot proceed. "
            f"Missing episode columns: {missing_columns}"
        )

    return (
        episode_df
        .groupby(
            [
                "rss_episode_family",
                "rss_episode_model",
                "rss_episode_start_type",
                "rss_episode_end_state",
                "rss_episode_construction_review_eligible",
                "rss_automatic_boundary_use_authorized",
                "rss_replay_cycle_formation_authorized",
                "rss_nid_authorization_state",
            ],
            dropna=False,
        )
        .agg(
            episode_count=(
                "rss_episode_id",
                "nunique",
            ),
            patient_count=(
                patient_column,
                "nunique",
            ),
            recurrence_candidate_episode_count=(
                "rss_recurrence_candidate_flag",
                "sum",
            ),
            temporal_tie_episode_count=(
                "rss_episode_contains_temporal_tie",
                "sum",
            ),
        )
        .reset_index()
        .sort_values(
            by=[
                "rss_episode_family",
                "episode_count",
                "rss_episode_start_type",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    #============================================================
# Chapter 68.26 — RSS Cross-Family Episode Alignment
#                  and Conflict Review
#============================================================

from itertools import combinations


_RSS_EPISODE_FAMILY_PRIORITY = {
    "ESCALATION_EPISODE": 0,
    "REBOUND_INSTABILITY_EPISODE": 1,
    "FALSE_RECOVERY_CONTRADICTION_EPISODE": 2,
}


_RSS_CROSS_FAMILY_ALIGNMENT_POLICY = {
    (
        "ESCALATION_EPISODE",
        "REBOUND_INSTABILITY_EPISODE",
    ): {
        "alignment_interpretation":
            (
                "Escalation and rebound instability were "
                "observed within overlapping patient-history "
                "intervals."
            ),

        "cross_family_evidence_state":
            (
                "ESCALATION_REBOUND_INSTABILITY_"
                "CONVERGENCE_CANDIDATE"
            ),

        "future_replay_relevance":
            (
                "REPEATED_INSTABILITY_OR_"
                "ESCALATING_REPLAY_CONTEXT"
            ),
    },

    (
        "ESCALATION_EPISODE",
        "FALSE_RECOVERY_CONTRADICTION_EPISODE",
    ): {
        "alignment_interpretation":
            (
                "Escalation and false-recovery contradiction "
                "were observed within overlapping patient-"
                "history intervals."
            ),

        "cross_family_evidence_state":
            (
                "ESCALATION_FALSE_RECOVERY_"
                "CONVERGENCE_CANDIDATE"
            ),

        "future_replay_relevance":
            (
                "FAILED_RECOVERY_OR_"
                "ESCALATING_REPLAY_CONTEXT"
            ),
    },

    (
        "REBOUND_INSTABILITY_EPISODE",
        "FALSE_RECOVERY_CONTRADICTION_EPISODE",
    ): {
        "alignment_interpretation":
            (
                "Rebound instability and false-recovery "
                "contradiction were observed within overlapping "
                "patient-history intervals."
            ),

        "cross_family_evidence_state":
            (
                "REBOUND_FALSE_RECOVERY_"
                "CONVERGENCE_CANDIDATE"
            ),

        "future_replay_relevance":
            (
                "FAILED_RECOVERY_OR_"
                "REPEATED_INSTABILITY_CONTEXT"
            ),
    },
}


_RSS_CROSS_FAMILY_RELATION_PRIORITY = {
    "COEXTENSIVE_INTERVAL": 0,
    "COTEMPORAL_POINT_ALIGNMENT": 1,
    "FAMILY_B_POINT_WITHIN_FAMILY_A_INTERVAL": 2,
    "FAMILY_A_POINT_WITHIN_FAMILY_B_INTERVAL": 3,
    "SHARED_START_OVERLAP": 4,
    "SHARED_END_OVERLAP": 5,
    "FAMILY_B_NESTED_WITHIN_FAMILY_A": 6,
    "FAMILY_A_NESTED_WITHIN_FAMILY_B": 7,
    "PARTIAL_INTERVAL_OVERLAP": 8,
    "BOUNDARY_CONTACT": 9,
}


_RSS_CROSS_FAMILY_TIME_ATOL = 1e-12


def _classify_rss_cross_family_temporal_relation(
    frame: pd.DataFrame,
) -> pd.Series:
    """
    Classify cross-family temporal relationships without treating
    every zero-duration intersection as boundary contact.

    Zero-duration relationships are separated into:
    - cotemporal point alignment
    - point event inside another episode interval
    - genuine shared-edge boundary contact
    """

    start_a = frame[
        "rss_episode_start_time_a"
    ]

    end_a = frame[
        "rss_episode_end_time_a"
    ]

    start_b = frame[
        "rss_episode_start_time_b"
    ]

    end_b = frame[
        "rss_episode_end_time_b"
    ]

    overlap_duration = frame[
        "rss_alignment_overlap_duration"
    ]

    def isclose(
        left: pd.Series,
        right: pd.Series | float,
    ) -> pd.Series:
        return pd.Series(
            np.isclose(
                left,
                right,
                rtol=0.0,
                atol=_RSS_CROSS_FAMILY_TIME_ATOL,
                equal_nan=False,
            ),
            index=frame.index,
            dtype=bool,
        )

    same_start = isclose(
        start_a,
        start_b,
    )

    same_end = isclose(
        end_a,
        end_b,
    )

    point_episode_a = isclose(
        start_a,
        end_a,
    )

    point_episode_b = isclose(
        start_b,
        end_b,
    )

    zero_duration_alignment = isclose(
        overlap_duration,
        0.0,
    )

    positive_duration_overlap = (
        overlap_duration
        .gt(_RSS_CROSS_FAMILY_TIME_ATOL)
    )

    family_a_ends_at_family_b_start = isclose(
        end_a,
        start_b,
    )

    family_b_ends_at_family_a_start = isclose(
        end_b,
        start_a,
    )

    relation = pd.Series(
        "UNCLASSIFIED_CROSS_FAMILY_ALIGNMENT",
        index=frame.index,
        dtype="string",
    )

    # --------------------------------------------------------
    # Positive-duration interval relationships.
    # --------------------------------------------------------

    relation.loc[
        positive_duration_overlap
        &
        same_start
        &
        same_end
    ] = "COEXTENSIVE_INTERVAL"

    relation.loc[
        positive_duration_overlap
        &
        same_start
        &
        ~same_end
    ] = "SHARED_START_OVERLAP"

    relation.loc[
        positive_duration_overlap
        &
        ~same_start
        &
        same_end
    ] = "SHARED_END_OVERLAP"

    family_b_nested_within_a = (
        positive_duration_overlap
        &
        start_a.lt(start_b)
        &
        end_a.gt(end_b)
    )

    family_a_nested_within_b = (
        positive_duration_overlap
        &
        start_b.lt(start_a)
        &
        end_b.gt(end_a)
    )

    relation.loc[
        family_b_nested_within_a
    ] = "FAMILY_B_NESTED_WITHIN_FAMILY_A"

    relation.loc[
        family_a_nested_within_b
    ] = "FAMILY_A_NESTED_WITHIN_FAMILY_B"

    remaining_positive_overlap = (
        positive_duration_overlap
        &
        relation.eq(
            "UNCLASSIFIED_CROSS_FAMILY_ALIGNMENT"
        )
    )

    relation.loc[
        remaining_positive_overlap
    ] = "PARTIAL_INTERVAL_OVERLAP"

    # --------------------------------------------------------
    # Zero-duration relationships.
    # --------------------------------------------------------

    cotemporal_point_alignment = (
        zero_duration_alignment
        &
        point_episode_a
        &
        point_episode_b
        &
        same_start
    )

    relation.loc[
        cotemporal_point_alignment
    ] = "COTEMPORAL_POINT_ALIGNMENT"

    family_a_point_within_b = (
        zero_duration_alignment
        &
        point_episode_a
        &
        ~point_episode_b
        &
        start_a.gt(start_b)
        &
        start_a.lt(end_b)
    )

    relation.loc[
        family_a_point_within_b
    ] = "FAMILY_A_POINT_WITHIN_FAMILY_B_INTERVAL"

    family_b_point_within_a = (
        zero_duration_alignment
        &
        point_episode_b
        &
        ~point_episode_a
        &
        start_b.gt(start_a)
        &
        start_b.lt(end_a)
    )

    relation.loc[
        family_b_point_within_a
    ] = "FAMILY_B_POINT_WITHIN_FAMILY_A_INTERVAL"

    genuine_boundary_contact = (
        zero_duration_alignment
        &
        ~cotemporal_point_alignment
        &
        ~family_a_point_within_b
        &
        ~family_b_point_within_a
        &
        (
            family_a_ends_at_family_b_start
            |
            family_b_ends_at_family_a_start
        )
    )

    relation.loc[
        genuine_boundary_contact
    ] = "BOUNDARY_CONTACT"

    unclassified_mask = relation.eq(
        "UNCLASSIFIED_CROSS_FAMILY_ALIGNMENT"
    )

    if unclassified_mask.any():
        unclassified_examples = (
            frame.loc[
                unclassified_mask,
                [
                    "rss_episode_start_time_a",
                    "rss_episode_end_time_a",
                    "rss_episode_start_time_b",
                    "rss_episode_end_time_b",
                    "rss_alignment_overlap_duration",
                ],
            ]
            .head(10)
            .to_dict("records")
        )

        raise ValueError(
            "RSS cross-family temporal relationship "
            "classification left "
            f"{int(unclassified_mask.sum()):,} aligned pairs "
            "unclassified. Examples: "
            f"{unclassified_examples}"
        )

    return relation


def build_rss_cross_family_episode_alignment(
    episode_df: pd.DataFrame,
    patient_column: str = "patient_id",
) -> pd.DataFrame:
    """
    Align provisional RSS episodes across different episode
    families within the same patient history.

    The review evaluates:
    - positive-duration overlap
    - co-temporal point alignment
    - boundary contact
    - nesting
    - shared starts and endings
    - censoring
    - recurrence candidacy
    - evidence convergence

    Only episodes eligible for controlled construction review
    participate.

    This function does not merge episode families, claim
    causation, or authorize replay-cycle formation.
    """

    if not isinstance(
        episode_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "episode_df must be a pandas DataFrame."
        )

    required_columns = {
        patient_column,
        "rss_episode_id",
        "rss_source_signal",
        "rss_episode_family",
        "rss_episode_model",
        "rss_episode_start_type",
        "rss_episode_end_state",
        "rss_episode_start_time",
        "rss_episode_end_time",
        "rss_episode_left_censored_flag",
        "rss_episode_internal_censoring_flag",
        "rss_episode_right_censored_flag",
        "rss_recurrence_candidate_flag",
        "rss_episode_construction_review_eligible",
    }

    missing_columns = sorted(
        required_columns
        - set(episode_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "RSS cross-family episode alignment cannot "
            "proceed. Missing episode columns: "
            f"{missing_columns}"
        )

    if episode_df[
        "rss_episode_id"
    ].duplicated().any():
        raise ValueError(
            "RSS episode dataframe contains duplicated "
            "episode identifiers."
        )

    eligible = (
        episode_df.loc[
            episode_df[
                "rss_episode_construction_review_eligible"
            ].eq(True)
        ]
        .copy()
        .reset_index(drop=True)
    )

    eligible = eligible.loc[
        eligible[
            "rss_episode_start_time"
        ].notna()
        &
        eligible[
            "rss_episode_end_time"
        ].notna()
    ].copy()

    invalid_interval_mask = (
        eligible[
            "rss_episode_start_time"
        ]
        .gt(
            eligible[
                "rss_episode_end_time"
            ]
        )
    )

    if invalid_interval_mask.any():
        raise ValueError(
            "RSS episode dataframe contains "
            f"{int(invalid_interval_mask.sum())} episodes "
            "whose start time occurs after the end time."
        )

    result_columns = [
        patient_column,
        "rss_alignment_id",
        "rss_family_pair",
        "rss_episode_family_a",
        "rss_episode_family_b",
        "rss_episode_id_a",
        "rss_episode_id_b",
        "rss_source_signal_a",
        "rss_source_signal_b",
        "rss_episode_model_a",
        "rss_episode_model_b",
        "rss_episode_start_type_a",
        "rss_episode_start_type_b",
        "rss_episode_end_state_a",
        "rss_episode_end_state_b",
        "rss_episode_start_time_a",
        "rss_episode_end_time_a",
        "rss_episode_start_time_b",
        "rss_episode_end_time_b",
        "rss_alignment_overlap_start",
        "rss_alignment_overlap_end",
        "rss_alignment_overlap_duration",
        "rss_positive_duration_overlap_flag",
        "rss_cotemporal_point_alignment_flag",
        "rss_point_within_interval_flag",
        "rss_boundary_contact_flag",
        "rss_temporal_relation",
        "rss_alignment_contains_left_censoring",
        "rss_alignment_contains_internal_censoring",
        "rss_alignment_contains_right_censoring",
        "rss_alignment_contains_recurrence_candidate",
        "rss_alignment_interpretation",
        "rss_cross_family_evidence_state",
        "rss_future_replay_relevance",
        "rss_direct_semantic_conflict_flag",
        "rss_cross_signal_conflict_policy_defined",
        "rss_alignment_review_state",
        "rss_alignment_operationally_authorized",
        "rss_replay_cycle_formation_authorized",
        "rss_nid_authorization_state",
    ]

    available_families = [
        family
        for family in sorted(
            eligible[
                "rss_episode_family"
            ]
            .dropna()
            .unique()
            .tolist(),
            key=lambda family: (
                _RSS_EPISODE_FAMILY_PRIORITY.get(
                    family,
                    99,
                )
            ),
        )
        if family in _RSS_EPISODE_FAMILY_PRIORITY
    ]

    alignment_results: list[pd.DataFrame] = []

    selected_columns = [
        patient_column,
        "rss_episode_id",
        "rss_source_signal",
        "rss_episode_family",
        "rss_episode_model",
        "rss_episode_start_type",
        "rss_episode_end_state",
        "rss_episode_start_time",
        "rss_episode_end_time",
        "rss_episode_left_censored_flag",
        "rss_episode_internal_censoring_flag",
        "rss_episode_right_censored_flag",
        "rss_recurrence_candidate_flag",
    ]

    for family_a, family_b in combinations(
        available_families,
        2,
    ):
        policy_key = (
            family_a,
            family_b,
        )

        alignment_policy = (
            _RSS_CROSS_FAMILY_ALIGNMENT_POLICY.get(
                policy_key
            )
        )

        if alignment_policy is None:
            continue

        episodes_a = (
            eligible.loc[
                eligible[
                    "rss_episode_family"
                ].eq(family_a),
                selected_columns,
            ]
            .copy()
        )

        episodes_b = (
            eligible.loc[
                eligible[
                    "rss_episode_family"
                ].eq(family_b),
                selected_columns,
            ]
            .copy()
        )

        if episodes_a.empty or episodes_b.empty:
            continue

        pair_candidates = episodes_a.merge(
            episodes_b,
            on=patient_column,
            how="inner",
            suffixes=(
                "_a",
                "_b",
            ),
            validate="many_to_many",
        )

        overlap_or_contact_mask = (
            pair_candidates[
                "rss_episode_start_time_a"
            ]
            .le(
                pair_candidates[
                    "rss_episode_end_time_b"
                ]
            )
            &
            pair_candidates[
                "rss_episode_start_time_b"
            ]
            .le(
                pair_candidates[
                    "rss_episode_end_time_a"
                ]
            )
        )

        aligned = (
            pair_candidates.loc[
                overlap_or_contact_mask
            ]
            .copy()
        )

        if aligned.empty:
            continue

        aligned[
            "rss_alignment_overlap_start"
        ] = aligned[
            [
                "rss_episode_start_time_a",
                "rss_episode_start_time_b",
            ]
        ].max(axis=1)

        aligned[
            "rss_alignment_overlap_end"
        ] = aligned[
            [
                "rss_episode_end_time_a",
                "rss_episode_end_time_b",
            ]
        ].min(axis=1)

        aligned[
            "rss_alignment_overlap_duration"
        ] = (
            aligned[
                "rss_alignment_overlap_end"
            ]
            -
            aligned[
                "rss_alignment_overlap_start"
            ]
        )

        aligned[
            "rss_temporal_relation"
        ] = (
            _classify_rss_cross_family_temporal_relation(
                aligned
            )
        )

        aligned[
            "rss_positive_duration_overlap_flag"
        ] = aligned[
            "rss_alignment_overlap_duration"
        ].gt(
            _RSS_CROSS_FAMILY_TIME_ATOL
        )

        aligned[
            "rss_cotemporal_point_alignment_flag"
        ] = aligned[
            "rss_temporal_relation"
        ].eq(
            "COTEMPORAL_POINT_ALIGNMENT"
        )

        aligned[
            "rss_point_within_interval_flag"
        ] = aligned[
            "rss_temporal_relation"
        ].isin(
            [
                "FAMILY_A_POINT_WITHIN_FAMILY_B_INTERVAL",
                "FAMILY_B_POINT_WITHIN_FAMILY_A_INTERVAL",
            ]
        )

        aligned[
            "rss_boundary_contact_flag"
        ] = aligned[
            "rss_temporal_relation"
        ].eq(
            "BOUNDARY_CONTACT"
        )

        aligned[
            "rss_alignment_contains_left_censoring"
        ] = (
            aligned[
                "rss_episode_left_censored_flag_a"
            ].fillna(False)
            |
            aligned[
                "rss_episode_left_censored_flag_b"
            ].fillna(False)
        )

        aligned[
            "rss_alignment_contains_internal_censoring"
        ] = (
            aligned[
                "rss_episode_internal_censoring_flag_a"
            ].fillna(False)
            |
            aligned[
                "rss_episode_internal_censoring_flag_b"
            ].fillna(False)
        )

        aligned[
            "rss_alignment_contains_right_censoring"
        ] = (
            aligned[
                "rss_episode_right_censored_flag_a"
            ].fillna(False)
            |
            aligned[
                "rss_episode_right_censored_flag_b"
            ].fillna(False)
        )

        aligned[
            "rss_alignment_contains_recurrence_candidate"
        ] = (
            aligned[
                "rss_recurrence_candidate_flag_a"
            ].fillna(False)
            |
            aligned[
                "rss_recurrence_candidate_flag_b"
            ].fillna(False)
        )

        aligned[
            "rss_family_pair"
        ] = (
            family_a
            + " || "
            + family_b
        )

        aligned[
            "rss_episode_family_a"
        ] = family_a

        aligned[
            "rss_episode_family_b"
        ] = family_b

        aligned[
            "rss_alignment_interpretation"
        ] = alignment_policy[
            "alignment_interpretation"
        ]

        aligned[
            "rss_cross_family_evidence_state"
        ] = alignment_policy[
            "cross_family_evidence_state"
        ]

        aligned[
            "rss_future_replay_relevance"
        ] = alignment_policy[
            "future_replay_relevance"
        ]

        # The current episode families all communicate adverse
        # or cautionary evidence. Overlap is therefore treated
        # as possible evidence convergence rather than direct
        # semantic contradiction.
        aligned[
            "rss_direct_semantic_conflict_flag"
        ] = False

        aligned[
            "rss_cross_signal_conflict_policy_defined"
        ] = False

        aligned[
            "rss_alignment_review_state"
        ] = (
            "CROSS_FAMILY_POSITIVE_DURATION_ALIGNMENT_"
            "POLICY_REVIEW_REQUIRED"
        )

        aligned.loc[
            aligned[
                "rss_point_within_interval_flag"
            ],
            "rss_alignment_review_state",
        ] = (
            "CROSS_FAMILY_POINT_WITHIN_INTERVAL_"
            "REVIEW_REQUIRED"
        )

        aligned.loc[
            aligned[
                "rss_cotemporal_point_alignment_flag"
            ],
            "rss_alignment_review_state",
        ] = (
            "CROSS_FAMILY_COTEMPORAL_POINT_ALIGNMENT_"
            "REVIEW_REQUIRED"
        )

        aligned.loc[
            aligned[
                "rss_boundary_contact_flag"
            ],

            "rss_alignment_review_state",
        ] = (
            "CROSS_FAMILY_BOUNDARY_CONTACT_"
            "REVIEW_REQUIRED"
        )

        aligned[
            "rss_alignment_operationally_authorized"
        ] = False

        aligned[
            "rss_replay_cycle_formation_authorized"
        ] = False

        aligned[
            "rss_nid_authorization_state"
        ] = "NID_CROSS_FAMILY_REVIEW_PENDING"

        aligned[
            "rss_alignment_id"
        ] = (
            aligned[
                patient_column
            ]
            .astype("string")
            + "::"
            + aligned[
                "rss_episode_id_a"
            ]
            .astype("string")
            + "||"
            + aligned[
                "rss_episode_id_b"
            ]
            .astype("string")
        )

        alignment_results.append(
            aligned
        )

    if not alignment_results:
        return pd.DataFrame(
            columns=result_columns
        )

    result = pd.concat(
        alignment_results,
        ignore_index=True,
    )

    if result[
        "rss_alignment_id"
    ].duplicated().any():
        raise ValueError(
            "RSS cross-family alignment produced duplicated "
            "episode-pair identifiers."
        )

    result[
        "_family_a_priority"
    ] = (
        result[
            "rss_episode_family_a"
        ]
        .map(
            _RSS_EPISODE_FAMILY_PRIORITY
        )
        .fillna(99)
    )

    result[
        "_family_b_priority"
    ] = (
        result[
            "rss_episode_family_b"
        ]
        .map(
            _RSS_EPISODE_FAMILY_PRIORITY
        )
        .fillna(99)
    )

    result[
        "_temporal_relation_priority"
    ] = (
        result[
            "rss_temporal_relation"
        ]
        .map(
            _RSS_CROSS_FAMILY_RELATION_PRIORITY
        )
        .fillna(99)
    )

    return (
        result[
            result_columns
            + [
                "_family_a_priority",
                "_family_b_priority",
                "_temporal_relation_priority",
            ]
        ]
        .sort_values(
            by=[
                "_family_a_priority",
                "_family_b_priority",
                patient_column,
                "rss_alignment_overlap_start",
                "_temporal_relation_priority",
                "rss_alignment_id",
            ],
            kind="mergesort",
        )
        .drop(
            columns=[
                "_family_a_priority",
                "_family_b_priority",
                "_temporal_relation_priority",
            ]
        )
        .reset_index(drop=True)
    )


def build_rss_cross_family_episode_alignment_summary(
    alignment_df: pd.DataFrame,
    episode_df: pd.DataFrame,
    patient_column: str = "patient_id",
) -> pd.DataFrame:
    """
    Summarize cross-family episode alignment.

    The summary reports alignment prevalence, participating
    episodes, censoring, recurrence context, and semantic
    conflict status for each episode-family pair.

    Operational replay-cycle authority remains withheld.
    """

    if not isinstance(
        alignment_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "alignment_df must be a pandas DataFrame."
        )

    if not isinstance(
        episode_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "episode_df must be a pandas DataFrame."
        )

    required_alignment_columns = {
        patient_column,
        "rss_family_pair",
        "rss_episode_family_a",
        "rss_episode_family_b",
        "rss_episode_id_a",
        "rss_episode_id_b",
        "rss_temporal_relation",
        "rss_positive_duration_overlap_flag",
        "rss_boundary_contact_flag",
        "rss_alignment_contains_left_censoring",
        "rss_alignment_contains_right_censoring",
        "rss_alignment_contains_recurrence_candidate",
        "rss_direct_semantic_conflict_flag",
        "rss_cross_signal_conflict_policy_defined",
        "rss_replay_cycle_formation_authorized",
        "rss_nid_authorization_state",
    }

    missing_alignment_columns = sorted(
        required_alignment_columns
        - set(alignment_df.columns)
    )

    if missing_alignment_columns:
        raise KeyError(
            "RSS cross-family alignment summary cannot "
            "proceed. Missing alignment columns: "
            f"{missing_alignment_columns}"
        )

    required_episode_columns = {
        patient_column,
        "rss_episode_id",
        "rss_episode_family",
        "rss_episode_construction_review_eligible",
    }

    missing_episode_columns = sorted(
        required_episode_columns
        - set(episode_df.columns)
    )

    if missing_episode_columns:
        raise KeyError(
            "RSS cross-family alignment summary cannot "
            "proceed. Missing episode columns: "
            f"{missing_episode_columns}"
        )

    eligible_episodes = episode_df.loc[
        episode_df[
            "rss_episode_construction_review_eligible"
        ].eq(True)
    ].copy()

    rows: list[dict[str, object]] = []

    family_pairs = (
        alignment_df[
            [
                "rss_family_pair",
                "rss_episode_family_a",
                "rss_episode_family_b",
            ]
        ]
        .drop_duplicates()
        .itertuples(index=False)
    )

    for pair in family_pairs:
        pair_review = alignment_df.loc[
            alignment_df[
                "rss_family_pair"
            ].eq(pair.rss_family_pair)
        ]

        family_a_episode_count = int(
            eligible_episodes.loc[
                eligible_episodes[
                    "rss_episode_family"
                ].eq(
                    pair.rss_episode_family_a
                ),
                "rss_episode_id",
            ]
            .nunique()
        )

        family_b_episode_count = int(
            eligible_episodes.loc[
                eligible_episodes[
                    "rss_episode_family"
                ].eq(
                    pair.rss_episode_family_b
                ),
                "rss_episode_id",
            ]
            .nunique()
        )

        participating_family_a_count = int(
            pair_review[
                "rss_episode_id_a"
            ]
            .nunique()
        )

        participating_family_b_count = int(
            pair_review[
                "rss_episode_id_b"
            ]
            .nunique()
        )

        positive_duration_count = int(
            pair_review[
                "rss_positive_duration_overlap_flag"
            ].sum()
        )

        cotemporal_point_count = int(
            pair_review[
                "rss_cotemporal_point_alignment_flag"
            ].sum()
        )

        point_within_interval_count = int(
            pair_review[
                "rss_point_within_interval_flag"
            ].sum()
        )

        boundary_contact_count = int(
            pair_review[
                "rss_boundary_contact_flag"
            ].sum()
        )

        classified_pair_count = (
            positive_duration_count
            +
            cotemporal_point_count
            +
            point_within_interval_count
            +
            boundary_contact_count
        )

        if classified_pair_count != len(pair_review):
            raise ValueError(
                "RSS cross-family relationship categories do not "
                "reconcile for "
                f"{pair.rss_family_pair!r}. "
                f"Aligned pairs: {len(pair_review):,}; "
                f"classified pairs: {classified_pair_count:,}."
            )

        rows.append({
            "rss_family_pair":
                pair.rss_family_pair,

            "rss_episode_family_a":
                pair.rss_episode_family_a,

            "rss_episode_family_b":
                pair.rss_episode_family_b,

            "family_a_eligible_episode_count":
                family_a_episode_count,

            "family_b_eligible_episode_count":
                family_b_episode_count,

            "aligned_episode_pair_count":
                int(len(pair_review)),

            "aligned_patient_count":
                int(
                    pair_review[
                        patient_column
                    ]
                    .nunique()
                ),

            "participating_family_a_episode_count":
                participating_family_a_count,

            "participating_family_b_episode_count":
                participating_family_b_count,

            "family_a_episode_alignment_percent":
                _calculate_rss_percent(
                    participating_family_a_count,
                    family_a_episode_count,
                ),

            "family_b_episode_alignment_percent":
                _calculate_rss_percent(
                    participating_family_b_count,
                    family_b_episode_count,
                ),

            "positive_duration_overlap_pair_count":
                positive_duration_count,

            "point_within_interval_pair_count":
                point_within_interval_count,

            "boundary_contact_pair_count":
                boundary_contact_count,

            "cotemporal_point_alignment_count":
                cotemporal_point_count,

            "temporal_relationship_partition_state":
                "TEMPORAL_RELATIONSHIP_COUNTS_RECONCILED",



            "left_censored_alignment_count":
                int(
                    pair_review[
                        "rss_alignment_contains_left_censoring"
                    ]
                    .sum()
                ),

            "right_censored_alignment_count":
                int(
                    pair_review[
                        "rss_alignment_contains_right_censoring"
                    ]
                    .sum()
                ),

            "recurrence_candidate_alignment_count":
                int(
                    pair_review[
                        "rss_alignment_contains_recurrence_candidate"
                    ]
                    .sum()
                ),

            "direct_semantic_conflict_count":
                int(
                    pair_review[
                        "rss_direct_semantic_conflict_flag"
                    ]
                    .sum()
                ),

            "cross_signal_conflict_policy_defined":
                bool(
                    pair_review[
                        "rss_cross_signal_conflict_policy_defined"
                    ]
                    .all()
                ),

            "replay_cycle_formation_authorized":
                bool(
                    pair_review[
                        "rss_replay_cycle_formation_authorized"
                    ]
                    .all()
                ),

            "nid_authorization_state":
                (
                    "NID_CROSS_FAMILY_REVIEW_PENDING"
                ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            by=[
                "aligned_episode_pair_count",
                "rss_family_pair",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    #============================================================
# Chapter 68.27 — RSS Cross-Family Alignment
#                  Governance Policy
#============================================================

_RSS_CROSS_FAMILY_ALIGNMENT_GOVERNANCE_POLICY = {

    "POSITIVE_DURATION_OVERLAP": {
        "rss_alignment_evidence_role":
            "INTERVAL_CONVERGENCE_EVIDENCE",

        "rss_allowed_contextual_contribution":
            "CROSS_FAMILY_CONVERGENCE_CONTEXT",

        "rss_recurrence_testing_scope":
            "POSITIVE_DURATION_RELATIONSHIP_RECURRENCE",

        "rss_alignment_convergence_evidence_candidate":
            True,

        "rss_relationship_recurrence_testing_eligible":
            True,

        "rss_alignment_policy_state":
            (
                "INTERVAL_CONVERGENCE_POLICY_DEFINED_"
                "RECURRENCE_TESTING_ELIGIBLE"
            ),
    },

    "POINT_WITHIN_INTERVAL": {
        "rss_alignment_evidence_role":
            "EMBEDDED_EVENT_ALIGNMENT_CONTEXT",

        "rss_allowed_contextual_contribution":
            "CROSS_FAMILY_EMBEDDED_EVENT_CONTEXT",

        "rss_recurrence_testing_scope":
            "POINT_WITHIN_INTERVAL_RELATIONSHIP_RECURRENCE",

        "rss_alignment_convergence_evidence_candidate":
            True,

        "rss_relationship_recurrence_testing_eligible":
            True,

        "rss_alignment_policy_state":
            (
                "EMBEDDED_EVENT_ALIGNMENT_POLICY_DEFINED_"
                "RECURRENCE_TESTING_ELIGIBLE"
            ),
    },

    "COTEMPORAL_POINT_ALIGNMENT": {
        "rss_alignment_evidence_role":
            "COTEMPORAL_CONVERGENCE_CONTEXT",

        "rss_allowed_contextual_contribution":
            "CROSS_FAMILY_COTEMPORAL_CONTEXT",

        "rss_recurrence_testing_scope":
            "COTEMPORAL_RELATIONSHIP_RECURRENCE",

        "rss_alignment_convergence_evidence_candidate":
            True,

        "rss_relationship_recurrence_testing_eligible":
            True,

        "rss_alignment_policy_state":
            (
                "COTEMPORAL_ALIGNMENT_POLICY_DEFINED_"
                "RECURRENCE_TESTING_ELIGIBLE"
            ),
    },

    "BOUNDARY_CONTACT": {
        "rss_alignment_evidence_role":
            "BOUNDARY_ADJACENCY_CONTEXT",

        "rss_allowed_contextual_contribution":
            "CROSS_FAMILY_BOUNDARY_ADJACENCY_CONTEXT",

        "rss_recurrence_testing_scope":
            "BOUNDARY_ADJACENCY_RELATIONSHIP_RECURRENCE",

        "rss_alignment_convergence_evidence_candidate":
            False,

        "rss_relationship_recurrence_testing_eligible":
            True,

        "rss_alignment_policy_state":
            (
                "BOUNDARY_ADJACENCY_POLICY_DEFINED_"
                "RECURRENCE_TESTING_ELIGIBLE"
            ),
    },
}


_RSS_ALIGNMENT_GOVERNANCE_PRIORITY = {
    "POSITIVE_DURATION_OVERLAP": 0,
    "POINT_WITHIN_INTERVAL": 1,
    "COTEMPORAL_POINT_ALIGNMENT": 2,
    "BOUNDARY_CONTACT": 3,
}


def build_rss_cross_family_alignment_governance(
    alignment_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply minimum RSS governance to cross-family episode
    alignments.

    The policy determines:
    - governed temporal relationship class
    - permitted contextual contribution
    - convergence-candidate status
    - relationship-specific recurrence-testing eligibility
    - censoring limitations
    - semantic-conflict limitations
    - retained operational restrictions

    This function does not form replay cycles, infer causation,
    assign confidence, or generate replay lessons.
    """

    if not isinstance(
        alignment_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "alignment_df must be a pandas DataFrame."
        )

    required_columns = {
        "rss_alignment_id",
        "rss_family_pair",
        "rss_episode_family_a",
        "rss_episode_family_b",
        "rss_positive_duration_overlap_flag",
        "rss_point_within_interval_flag",
        "rss_cotemporal_point_alignment_flag",
        "rss_boundary_contact_flag",
        "rss_temporal_relation",
        "rss_alignment_contains_left_censoring",
        "rss_alignment_contains_internal_censoring",
        "rss_alignment_contains_right_censoring",
        "rss_alignment_contains_recurrence_candidate",
        "rss_direct_semantic_conflict_flag",
        "rss_replay_cycle_formation_authorized",
    }

    missing_columns = sorted(
        required_columns
        - set(alignment_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "RSS alignment governance cannot proceed. "
            f"Missing alignment columns: {missing_columns}"
        )

    if alignment_df[
        "rss_alignment_id"
    ].duplicated().any():
        raise ValueError(
            "RSS alignment governance received duplicated "
            "alignment identifiers."
        )

    result = alignment_df.copy()

    relationship_flags = [
        "rss_positive_duration_overlap_flag",
        "rss_point_within_interval_flag",
        "rss_cotemporal_point_alignment_flag",
        "rss_boundary_contact_flag",
    ]

    normalized_flags = (
        result[
            relationship_flags
        ]
        .fillna(False)
        .astype(bool)
    )

    relationship_flag_count = (
        normalized_flags
        .sum(axis=1)
    )

    invalid_partition_mask = (
        ~relationship_flag_count.eq(1)
    )

    if invalid_partition_mask.any():
        examples = (
            result.loc[
                invalid_partition_mask,
                [
                    "rss_alignment_id",
                    *relationship_flags,
                    "rss_temporal_relation",
                ],
            ]
            .head(10)
            .to_dict("records")
        )

        raise ValueError(
            "RSS alignment governance requires exactly one "
            "temporal relationship class per alignment. "
            f"Invalid alignments: "
            f"{int(invalid_partition_mask.sum()):,}. "
            f"Examples: {examples}"
        )

    result[
        "rss_governed_alignment_class"
    ] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )

    result.loc[
        normalized_flags[
            "rss_positive_duration_overlap_flag"
        ],
        "rss_governed_alignment_class",
    ] = "POSITIVE_DURATION_OVERLAP"

    result.loc[
        normalized_flags[
            "rss_point_within_interval_flag"
        ],
        "rss_governed_alignment_class",
    ] = "POINT_WITHIN_INTERVAL"

    result.loc[
        normalized_flags[
            "rss_cotemporal_point_alignment_flag"
        ],
        "rss_governed_alignment_class",
    ] = "COTEMPORAL_POINT_ALIGNMENT"

    result.loc[
        normalized_flags[
            "rss_boundary_contact_flag"
        ],
        "rss_governed_alignment_class",
    ] = "BOUNDARY_CONTACT"

    policy_frame = (
        pd.DataFrame
        .from_dict(
            _RSS_CROSS_FAMILY_ALIGNMENT_GOVERNANCE_POLICY,
            orient="index",
        )
        .rename_axis(
            "rss_governed_alignment_class"
        )
        .reset_index()
    )

    result = result.merge(
        policy_frame,
        on="rss_governed_alignment_class",
        how="left",
        validate="many_to_one",
    )

    undefined_policy_mask = (
        result[
            "rss_alignment_evidence_role"
        ]
        .isna()
    )

    if undefined_policy_mask.any():
        undefined_classes = sorted(
            result.loc[
                undefined_policy_mask,
                "rss_governed_alignment_class",
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "RSS alignment governance policy is undefined "
            f"for: {undefined_classes}"
        )

    left_censoring = (
        result[
            "rss_alignment_contains_left_censoring"
        ]
        .fillna(False)
        .astype(bool)
    )

    internal_censoring = (
        result[
            "rss_alignment_contains_internal_censoring"
        ]
        .fillna(False)
        .astype(bool)
    )

    right_censoring = (
        result[
            "rss_alignment_contains_right_censoring"
        ]
        .fillna(False)
        .astype(bool)
    )

    direct_semantic_conflict = (
        result[
            "rss_direct_semantic_conflict_flag"
        ]
        .fillna(False)
        .astype(bool)
    )

    result[
        "rss_alignment_evidence_limit_state"
    ] = "NO_CENSORING_LIMIT_IDENTIFIED"

    result.loc[
        left_censoring
        &
        ~right_censoring
        &
        ~internal_censoring,
        "rss_alignment_evidence_limit_state",
    ] = "LEFT_CENSORING_CONTEXT"

    result.loc[
        right_censoring
        &
        ~left_censoring
        &
        ~internal_censoring,
        "rss_alignment_evidence_limit_state",
    ] = "RIGHT_CENSORING_CONTEXT"

    result.loc[
        left_censoring
        &
        right_censoring
        &
        ~internal_censoring,
        "rss_alignment_evidence_limit_state",
    ] = "LEFT_AND_RIGHT_CENSORING_CONTEXT"

    result.loc[
        internal_censoring,
        "rss_alignment_evidence_limit_state",
    ] = "INTERNAL_CENSORING_LIMITS_ALIGNMENT"

    result[
        "rss_relationship_recurrence_testing_eligible"
    ] = (
        result[
            "rss_relationship_recurrence_testing_eligible"
        ]
        .fillna(False)
        .astype(bool)
        &
        ~internal_censoring
        &
        ~direct_semantic_conflict
    )

    result[
        "rss_alignment_governance_state"
    ] = (
        "ALIGNMENT_GOVERNED_"
        "RECURRENCE_TESTING_ELIGIBLE"
    )

    result.loc[
        result[
            "rss_governed_alignment_class"
        ].eq("BOUNDARY_CONTACT")
        &
        result[
            "rss_relationship_recurrence_testing_eligible"
        ],
        "rss_alignment_governance_state",
    ] = (
        "BOUNDARY_ADJACENCY_GOVERNED_"
        "RECURRENCE_TESTING_ELIGIBLE"
    )

    result.loc[
        internal_censoring,
        "rss_alignment_governance_state",
    ] = (
        "ALIGNMENT_GOVERNED_"
        "INTERNAL_CENSORING_REVIEW_REQUIRED"
    )

    result.loc[
        direct_semantic_conflict,
        "rss_alignment_governance_state",
    ] = (
        "ALIGNMENT_GOVERNED_"
        "DIRECT_SEMANTIC_CONFLICT_REVIEW_REQUIRED"
    )

    result[
        "rss_alignment_governance_policy_defined"
    ] = True

    result[
        "rss_cross_signal_conflict_policy_defined"
    ] = True

    result[
        "rss_episode_family_merge_authorized"
    ] = False

    result[
        "rss_boundary_transition_inference_authorized"
    ] = False

    result[
        "rss_causal_interpretation_authorized"
    ] = False

    result[
        "rss_alignment_operationally_authorized"
    ] = False

    result[
        "rss_replay_cycle_formation_authorized"
    ] = False

    result[
        "rss_replay_lesson_generation_authorized"
    ] = False

    result[
        "rss_nid_authorization_state"
    ] = (
        "NID_ALIGNMENT_GOVERNANCE_POLICY_INSTALLED"
    )

    result[
        "_rss_alignment_governance_priority"
    ] = (
        result[
            "rss_governed_alignment_class"
        ]
        .map(
            _RSS_ALIGNMENT_GOVERNANCE_PRIORITY
        )
        .fillna(99)
    )

    return (
        result
        .sort_values(
            by=[
                "_rss_alignment_governance_priority",
                "rss_family_pair",
                "rss_alignment_overlap_start",
                "rss_alignment_id",
            ],
            kind="mergesort",
        )
        .drop(
            columns=[
                "_rss_alignment_governance_priority",
            ]
        )
        .reset_index(drop=True)
    )


def build_rss_cross_family_alignment_governance_summary(
    governed_alignment_df: pd.DataFrame,
    patient_column: str = "patient_id",
) -> pd.DataFrame:
    """
    Summarize the minimum governance policy applied to
    cross-family RSS episode alignments.

    Recurrence testing may become eligible while operational
    replay-cycle and lesson authority remain withheld.
    """

    if not isinstance(
        governed_alignment_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "governed_alignment_df must be a pandas DataFrame."
        )

    required_columns = {
        patient_column,
        "rss_alignment_id",
        "rss_family_pair",
        "rss_governed_alignment_class",
        "rss_alignment_evidence_role",
        "rss_allowed_contextual_contribution",
        "rss_relationship_recurrence_testing_eligible",
        "rss_alignment_convergence_evidence_candidate",
        "rss_alignment_evidence_limit_state",
        "rss_alignment_contains_recurrence_candidate",
        "rss_direct_semantic_conflict_flag",
        "rss_alignment_governance_policy_defined",
        "rss_cross_signal_conflict_policy_defined",
        "rss_replay_cycle_formation_authorized",
        "rss_replay_lesson_generation_authorized",
        "rss_nid_authorization_state",
    }

    missing_columns = sorted(
        required_columns
        - set(governed_alignment_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "RSS alignment-governance summary cannot proceed. "
            f"Missing governed columns: {missing_columns}"
        )

    return (
        governed_alignment_df
        .groupby(
            [
                "rss_family_pair",
                "rss_governed_alignment_class",
                "rss_alignment_evidence_role",
                "rss_allowed_contextual_contribution",
                "rss_relationship_recurrence_testing_eligible",
                "rss_alignment_convergence_evidence_candidate",
                "rss_alignment_governance_policy_defined",
                "rss_cross_signal_conflict_policy_defined",
                "rss_replay_cycle_formation_authorized",
                "rss_replay_lesson_generation_authorized",
                "rss_nid_authorization_state",
            ],
            dropna=False,
        )
        .agg(
            alignment_count=(
                "rss_alignment_id",
                "nunique",
            ),

            patient_count=(
                patient_column,
                "nunique",
            ),

            recurrence_candidate_context_count=(
                "rss_alignment_contains_recurrence_candidate",
                "sum",
            ),

            censoring_limited_alignment_count=(
                "rss_alignment_evidence_limit_state",
                lambda values: int(
                    values.ne(
                        "NO_CENSORING_LIMIT_IDENTIFIED"
                    ).sum()
                ),
            ),

            direct_semantic_conflict_count=(
                "rss_direct_semantic_conflict_flag",
                "sum",
            ),
        )
        .reset_index()
        .sort_values(
            by=[
                "rss_family_pair",
                "alignment_count",
                "rss_governed_alignment_class",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    #============================================================
# Chapter 68.28 — RSS Cross-Family Relationship
#                  Recurrence Test
#============================================================

_RSS_MINIMUM_DISTINCT_RELATIONSHIP_POSITIONS_FOR_RECURRENCE = 2


_RSS_RELATIONSHIP_RECURRENCE_STATE_PRIORITY = {
    "RELATIONSHIP_RECURRENCE_OBSERVED": 0,
    "SINGLE_DISTINCT_RELATIONSHIP_OCCURRENCE": 1,
    "RELATIONSHIP_RECURRENCE_NOT_TESTABLE": 2,
}


def build_rss_cross_family_relationship_recurrence_test(
    governed_alignment_df: pd.DataFrame,
    patient_column: str = "patient_id",
    temporal_precision: int = 12,
) -> pd.DataFrame:
    """
    Test governed cross-family relationship recurrence.

    Recurrence requires the same:
    - patient
    - episode-family pair
    - governed alignment class

    to appear at two or more distinct historical positions.

    Multiple alignments at the same historical position are
    consolidated into one relationship occurrence.

    This function does not assign pattern maturity, confidence,
    replay lessons, causation, or operational replay authority.
    """

    if not isinstance(
        governed_alignment_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "governed_alignment_df must be a pandas DataFrame."
        )

    if not isinstance(
        temporal_precision,
        int,
    ):
        raise TypeError(
            "temporal_precision must be an integer."
        )

    if not 0 <= temporal_precision <= 15:
        raise ValueError(
            "temporal_precision must be between 0 and 15."
        )

    required_columns = {
        patient_column,
        "rss_alignment_id",
        "rss_family_pair",
        "rss_episode_family_a",
        "rss_episode_family_b",
        "rss_governed_alignment_class",
        "rss_temporal_relation",
        "rss_alignment_overlap_start",
        "rss_alignment_overlap_end",
        "rss_alignment_evidence_role",
        "rss_allowed_contextual_contribution",
        "rss_recurrence_testing_scope",
        "rss_alignment_convergence_evidence_candidate",
        "rss_relationship_recurrence_testing_eligible",
        "rss_alignment_evidence_limit_state",
        "rss_alignment_contains_left_censoring",
        "rss_alignment_contains_internal_censoring",
        "rss_alignment_contains_right_censoring",
        "rss_direct_semantic_conflict_flag",
        "rss_alignment_governance_policy_defined",
        "rss_cross_signal_conflict_policy_defined",
    }

    missing_columns = sorted(
        required_columns
        - set(governed_alignment_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "RSS relationship recurrence testing cannot "
            "proceed. Missing governed alignment columns: "
            f"{missing_columns}"
        )

    if governed_alignment_df[
        "rss_alignment_id"
    ].duplicated().any():
        raise ValueError(
            "RSS relationship recurrence testing received "
            "duplicated alignment identifiers."
        )

    work = governed_alignment_df.copy()

    identity_columns = [
        patient_column,
        "rss_family_pair",
        "rss_episode_family_a",
        "rss_episode_family_b",
        "rss_governed_alignment_class",
    ]

    for column in identity_columns:
        work[column] = (
            work[column]
            .astype("string")
            .str.strip()
        )

    missing_identity_mask = (
        work[
            identity_columns
        ]
        .isna()
        .any(axis=1)
        |
        work[
            identity_columns
        ]
        .eq("")
        .any(axis=1)
    )

    if missing_identity_mask.any():
        raise ValueError(
            "RSS relationship recurrence testing received "
            f"{int(missing_identity_mask.sum()):,} alignments "
            "with missing governed relationship identity."
        )

    work[
        "_rss_alignment_overlap_start_numeric"
    ] = pd.to_numeric(
        work[
            "rss_alignment_overlap_start"
        ],
        errors="coerce",
    )

    work[
        "_rss_alignment_overlap_end_numeric"
    ] = pd.to_numeric(
        work[
            "rss_alignment_overlap_end"
        ],
        errors="coerce",
    )

    missing_historical_position = (
        work[
            "_rss_alignment_overlap_start_numeric"
        ].isna()
        |
        work[
            "_rss_alignment_overlap_end_numeric"
        ].isna()
    )

    invalid_historical_interval = (
        ~missing_historical_position
        &
        work[
            "_rss_alignment_overlap_start_numeric"
        ].gt(
            work[
                "_rss_alignment_overlap_end_numeric"
            ]
        )
    )

    governed_testing_eligible = (
        work[
            "rss_relationship_recurrence_testing_eligible"
        ]
        .fillna(False)
        .astype(bool)
    )

    internal_censoring = (
        work[
            "rss_alignment_contains_internal_censoring"
        ]
        .fillna(False)
        .astype(bool)
    )

    direct_semantic_conflict = (
        work[
            "rss_direct_semantic_conflict_flag"
        ]
        .fillna(False)
        .astype(bool)
    )

    left_censoring = (
        work[
            "rss_alignment_contains_left_censoring"
        ]
        .fillna(False)
        .astype(bool)
    )

    right_censoring = (
        work[
            "rss_alignment_contains_right_censoring"
        ]
        .fillna(False)
        .astype(bool)
    )

    censoring_limited = (
        work[
            "rss_alignment_evidence_limit_state"
        ]
        .astype("string")
        .ne(
            "NO_CENSORING_LIMIT_IDENTIFIED"
        )
    )

    work[
        "_rss_missing_historical_position"
    ] = missing_historical_position

    work[
        "_rss_invalid_historical_interval"
    ] = invalid_historical_interval

    work[
        "_rss_internal_censoring"
    ] = internal_censoring

    work[
        "_rss_direct_semantic_conflict"
    ] = direct_semantic_conflict

    work[
        "_rss_left_censoring"
    ] = left_censoring

    work[
        "_rss_right_censoring"
    ] = right_censoring

    work[
        "_rss_censoring_limited"
    ] = censoring_limited

    work[
        "rss_relationship_recurrence_observation_eligible"
    ] = (
        governed_testing_eligible
        &
        ~missing_historical_position
        &
        ~invalid_historical_interval
        &
        ~internal_censoring
        &
        ~direct_semantic_conflict
    )

    work[
        "_rss_recurrence_test_ineligible"
    ] = (
        ~work[
            "rss_relationship_recurrence_observation_eligible"
        ]
    )

    work[
        "rss_relationship_recurrence_exclusion_state"
    ] = "RECURRENCE_OBSERVATION_ELIGIBLE"

    work.loc[
        ~governed_testing_eligible,
        "rss_relationship_recurrence_exclusion_state",
    ] = "GOVERNANCE_RECURRENCE_GATE_CLOSED"

    work.loc[
        missing_historical_position,
        "rss_relationship_recurrence_exclusion_state",
    ] = "MISSING_HISTORICAL_POSITION"

    work.loc[
        invalid_historical_interval,
        "rss_relationship_recurrence_exclusion_state",
    ] = "INVALID_HISTORICAL_INTERVAL"

    work.loc[
        internal_censoring,
        "rss_relationship_recurrence_exclusion_state",
    ] = "INTERNAL_CENSORING_LIMITS_RECURRENCE_TEST"

    work.loc[
        direct_semantic_conflict,
        "rss_relationship_recurrence_exclusion_state",
    ] = "DIRECT_SEMANTIC_CONFLICT_LIMITS_RECURRENCE_TEST"

    work[
        "_rss_relationship_occurrence_position"
    ] = (
        work[
            "_rss_alignment_overlap_start_numeric"
        ]
        .round(
            temporal_precision
        )
    )

    group_columns = [
        patient_column,
        "rss_family_pair",
        "rss_governed_alignment_class",
    ]

    policy_columns = [
        "rss_episode_family_a",
        "rss_episode_family_b",
        "rss_alignment_evidence_role",
        "rss_allowed_contextual_contribution",
        "rss_recurrence_testing_scope",
        "rss_alignment_convergence_evidence_candidate",
    ]

    policy_consistency = (
        work
        .groupby(
            group_columns,
            dropna=False,
        )[
            policy_columns
        ]
        .nunique(
            dropna=False
        )
    )

    inconsistent_policy_mask = (
        policy_consistency
        .gt(1)
        .any(axis=1)
    )

    if inconsistent_policy_mask.any():
        inconsistent_groups = (
            policy_consistency.loc[
                inconsistent_policy_mask
            ]
            .head(10)
            .reset_index()
            .to_dict("records")
        )

        raise ValueError(
            "RSS relationship recurrence testing found "
            "inconsistent governance policy inside a patient "
            "relationship group. Examples: "
            f"{inconsistent_groups}"
        )

    grouped_alignment = (
        work
        .groupby(
            group_columns,
            as_index=False,
            dropna=False,
        )
        .agg(
            rss_episode_family_a=(
                "rss_episode_family_a",
                "first",
            ),

            rss_episode_family_b=(
                "rss_episode_family_b",
                "first",
            ),

            rss_alignment_evidence_role=(
                "rss_alignment_evidence_role",
                "first",
            ),

            rss_allowed_contextual_contribution=(
                "rss_allowed_contextual_contribution",
                "first",
            ),

            rss_recurrence_testing_scope=(
                "rss_recurrence_testing_scope",
                "first",
            ),

            rss_alignment_convergence_evidence_candidate=(
                "rss_alignment_convergence_evidence_candidate",
                "first",
            ),

            rss_total_alignment_count=(
                "rss_alignment_id",
                "nunique",
            ),

            rss_recurrence_test_eligible_alignment_count=(
                "rss_relationship_recurrence_observation_eligible",
                "sum",
            ),

            rss_recurrence_test_ineligible_alignment_count=(
                "_rss_recurrence_test_ineligible",
                "sum",
            ),

            rss_missing_position_alignment_count=(
                "_rss_missing_historical_position",
                "sum",
            ),

            rss_invalid_interval_alignment_count=(
                "_rss_invalid_historical_interval",
                "sum",
            ),

            rss_internal_censoring_alignment_count=(
                "_rss_internal_censoring",
                "sum",
            ),

            rss_direct_semantic_conflict_alignment_count=(
                "_rss_direct_semantic_conflict",
                "sum",
            ),

            rss_temporal_relation_variant_count=(
                "rss_temporal_relation",
                "nunique",
            ),

            rss_alignment_governance_policy_defined=(
                "rss_alignment_governance_policy_defined",
                "all",
            ),

            rss_cross_signal_conflict_policy_defined=(
                "rss_cross_signal_conflict_policy_defined",
                "all",
            ),
        )
    )

    eligible = work.loc[
        work[
            "rss_relationship_recurrence_observation_eligible"
        ]
    ].copy()

    occurrence_summary_columns = [
        *group_columns,
        "rss_distinct_historical_position_count",
        "rss_first_relationship_position",
        "rss_last_relationship_position",
        "rss_censoring_limited_position_count",
        "rss_left_censored_position_count",
        "rss_right_censored_position_count",
        "rss_max_alignment_count_at_single_position",
        "rss_same_position_alignment_compression_count",
    ]

    if eligible.empty:
        occurrence_summary = pd.DataFrame(
            columns=occurrence_summary_columns
        )

    else:
        relationship_occurrences = (
            eligible
            .groupby(
                [
                    *group_columns,
                    "_rss_relationship_occurrence_position",
                ],
                as_index=False,
                dropna=False,
            )
            .agg(
                rss_relationship_occurrence_end=(
                    "_rss_alignment_overlap_end_numeric",
                    "max",
                ),

                rss_relationship_occurrence_alignment_count=(
                    "rss_alignment_id",
                    "nunique",
                ),

                rss_relationship_occurrence_contains_censoring=(
                    "_rss_censoring_limited",
                    "any",
                ),

                rss_relationship_occurrence_contains_left_censoring=(
                    "_rss_left_censoring",
                    "any",
                ),

                rss_relationship_occurrence_contains_right_censoring=(
                    "_rss_right_censoring",
                    "any",
                ),
            )
            .sort_values(
                by=[
                    *group_columns,
                    "_rss_relationship_occurrence_position",
                    "rss_relationship_occurrence_end",
                ],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )

        relationship_occurrences[
            "rss_relationship_occurrence_ordinal"
        ] = (
            relationship_occurrences
            .groupby(
                group_columns,
                dropna=False,
            )
            .cumcount()
            .add(1)
            .astype("Int64")
        )

        occurrence_summary = (
            relationship_occurrences
            .groupby(
                group_columns,
                as_index=False,
                dropna=False,
            )
            .agg(
                rss_distinct_historical_position_count=(
                    "_rss_relationship_occurrence_position",
                    "nunique",
                ),

                rss_first_relationship_position=(
                    "_rss_relationship_occurrence_position",
                    "min",
                ),

                rss_last_relationship_position=(
                    "_rss_relationship_occurrence_position",
                    "max",
                ),

                rss_censoring_limited_position_count=(
                    "rss_relationship_occurrence_contains_censoring",
                    "sum",
                ),

                rss_left_censored_position_count=(
                    "rss_relationship_occurrence_contains_left_censoring",
                    "sum",
                ),

                rss_right_censored_position_count=(
                    "rss_relationship_occurrence_contains_right_censoring",
                    "sum",
                ),

                rss_max_alignment_count_at_single_position=(
                    "rss_relationship_occurrence_alignment_count",
                    "max",
                ),

                rss_same_position_alignment_compression_count=(
                    "rss_relationship_occurrence_alignment_count",
                    lambda values: int(
                        (
                            values
                            .sub(1)
                            .clip(lower=0)
                        )
                        .sum()
                    ),
                ),
            )
        )

    result = grouped_alignment.merge(
        occurrence_summary,
        on=group_columns,
        how="left",
        validate="one_to_one",
    )

    integer_columns = [
        "rss_total_alignment_count",
        "rss_recurrence_test_eligible_alignment_count",
        "rss_recurrence_test_ineligible_alignment_count",
        "rss_missing_position_alignment_count",
        "rss_invalid_interval_alignment_count",
        "rss_internal_censoring_alignment_count",
        "rss_direct_semantic_conflict_alignment_count",
        "rss_temporal_relation_variant_count",
        "rss_distinct_historical_position_count",
        "rss_censoring_limited_position_count",
        "rss_left_censored_position_count",
        "rss_right_censored_position_count",
        "rss_max_alignment_count_at_single_position",
        "rss_same_position_alignment_compression_count",
    ]

    for column in integer_columns:
        result[column] = (
            result[column]
            .fillna(0)
            .astype("Int64")
        )

    distinct_position_count = (
        result[
            "rss_distinct_historical_position_count"
        ]
    )

    result[
        "rss_relationship_recurrence_testable"
    ] = distinct_position_count.ge(1)

    result[
        "rss_relationship_recurrence_observed"
    ] = distinct_position_count.ge(
        _RSS_MINIMUM_DISTINCT_RELATIONSHIP_POSITIONS_FOR_RECURRENCE
    )

    result[
        "rss_recurrent_historical_position_count"
    ] = (
        distinct_position_count
        .sub(1)
        .clip(lower=0)
        .astype("Int64")
    )

    result[
        "rss_relationship_recurrence_span"
    ] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Float64",
    )

    recurrence_observed_mask = (
        result[
            "rss_relationship_recurrence_observed"
        ]
    )

    result.loc[
        recurrence_observed_mask,
        "rss_relationship_recurrence_span",
    ] = (
        result.loc[
            recurrence_observed_mask,
            "rss_last_relationship_position",
        ]
        -
        result.loc[
            recurrence_observed_mask,
            "rss_first_relationship_position",
        ]
    )

    result[
        "rss_relationship_recurrence_state"
    ] = "RELATIONSHIP_RECURRENCE_NOT_TESTABLE"

    result.loc[
        distinct_position_count.eq(1),
        "rss_relationship_recurrence_state",
    ] = "SINGLE_DISTINCT_RELATIONSHIP_OCCURRENCE"

    result.loc[
        recurrence_observed_mask,
        "rss_relationship_recurrence_state",
    ] = "RELATIONSHIP_RECURRENCE_OBSERVED"

    result[
        "rss_relationship_recurrence_evidence_limit_state"
    ] = "NO_ELIGIBLE_RELATIONSHIP_POSITION"

    single_position_mask = (
        distinct_position_count.eq(1)
    )

    censoring_position_present = (
        result[
            "rss_censoring_limited_position_count"
        ]
        .gt(0)
    )

    result.loc[
        single_position_mask
        &
        ~censoring_position_present,
        "rss_relationship_recurrence_evidence_limit_state",
    ] = "SINGLE_OCCURRENCE_WITHOUT_CENSORING_LIMIT"

    result.loc[
        single_position_mask
        &
        censoring_position_present,
        "rss_relationship_recurrence_evidence_limit_state",
    ] = "SINGLE_OCCURRENCE_WITH_CENSORING_CONTEXT"

    result.loc[
        recurrence_observed_mask
        &
        ~censoring_position_present,
        "rss_relationship_recurrence_evidence_limit_state",
    ] = "RECURRENCE_OBSERVED_WITHOUT_CENSORING_LIMIT"

    result.loc[
        recurrence_observed_mask
        &
        censoring_position_present,
        "rss_relationship_recurrence_evidence_limit_state",
    ] = "RECURRENCE_OBSERVED_WITH_CENSORING_CONTEXT"

    result[
        "rss_relationship_recurrence_review_state"
    ] = (
        "RELATIONSHIP_RECURRENCE_NOT_TESTABLE_"
        "EVIDENCE_INCOMPLETE"
    )

    result.loc[
        single_position_mask,
        "rss_relationship_recurrence_review_state",
    ] = (
        "RELATIONSHIP_OBSERVED_ONCE_"
        "RECURRENCE_NOT_ESTABLISHED"
    )

    result.loc[
        recurrence_observed_mask,
        "rss_relationship_recurrence_review_state",
    ] = (
        "RELATIONSHIP_RECURRENCE_EVIDENCE_OBSERVED_"
        "LESSON_AUTHORITY_WITHHELD"
    )

    result[
        "rss_minimum_distinct_position_threshold"
    ] = (
        _RSS_MINIMUM_DISTINCT_RELATIONSHIP_POSITIONS_FOR_RECURRENCE
    )

    result[
        "rss_relationship_recurrence_test_policy_defined"
    ] = True

    result[
        "rss_pattern_maturity_assignment_authorized"
    ] = False

    result[
        "rss_replay_cycle_formation_authorized"
    ] = False

    result[
        "rss_replay_lesson_generation_authorized"
    ] = False

    result[
        "rss_causal_interpretation_authorized"
    ] = False

    result[
        "rss_nid_authorization_state"
    ] = (
        "NID_RELATIONSHIP_RECURRENCE_TEST_INSTALLED"
    )

    result[
        "_rss_recurrence_state_priority"
    ] = (
        result[
            "rss_relationship_recurrence_state"
        ]
        .map(
            _RSS_RELATIONSHIP_RECURRENCE_STATE_PRIORITY
        )
        .fillna(99)
    )

    return (
        result
        .sort_values(
            by=[
                "_rss_recurrence_state_priority",
                "rss_family_pair",
                "rss_governed_alignment_class",
                patient_column,
            ],
            kind="mergesort",
        )
        .drop(
            columns=[
                "_rss_recurrence_state_priority",
            ]
        )
        .reset_index(drop=True)
    )


def build_rss_cross_family_relationship_recurrence_summary(
    recurrence_df: pd.DataFrame,
    patient_column: str = "patient_id",
) -> pd.DataFrame:
    """
    Summarize governed cross-family relationship recurrence.

    Observed recurrence remains structural evidence only.
    Replay-cycle, pattern-maturity, and lesson authority remain
    withheld.
    """

    if not isinstance(
        recurrence_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "recurrence_df must be a pandas DataFrame."
        )

    required_columns = {
        patient_column,
        "rss_family_pair",
        "rss_governed_alignment_class",
        "rss_alignment_evidence_role",
        "rss_allowed_contextual_contribution",
        "rss_recurrence_testing_scope",
        "rss_alignment_convergence_evidence_candidate",
        "rss_total_alignment_count",
        "rss_recurrence_test_eligible_alignment_count",
        "rss_distinct_historical_position_count",
        "rss_recurrent_historical_position_count",
        "rss_same_position_alignment_compression_count",
        "rss_censoring_limited_position_count",
        "rss_relationship_recurrence_testable",
        "rss_relationship_recurrence_observed",
        "rss_relationship_recurrence_state",
        "rss_relationship_recurrence_test_policy_defined",
        "rss_pattern_maturity_assignment_authorized",
        "rss_replay_cycle_formation_authorized",
        "rss_replay_lesson_generation_authorized",
        "rss_nid_authorization_state",
    }

    missing_columns = sorted(
        required_columns
        - set(recurrence_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "RSS relationship recurrence summary cannot "
            "proceed. Missing recurrence columns: "
            f"{missing_columns}"
        )

    summary_input = recurrence_df.copy()

    summary_input[
        "_rss_single_occurrence_flag"
    ] = summary_input[
        "rss_relationship_recurrence_state"
    ].eq(
        "SINGLE_DISTINCT_RELATIONSHIP_OCCURRENCE"
    )

    summary_input[
        "_rss_not_testable_flag"
    ] = summary_input[
        "rss_relationship_recurrence_state"
    ].eq(
        "RELATIONSHIP_RECURRENCE_NOT_TESTABLE"
    )

    summary_input[
        "_rss_censoring_limited_recurrence_flag"
    ] = (
        summary_input[
            "rss_relationship_recurrence_observed"
        ]
        &
        summary_input[
            "rss_censoring_limited_position_count"
        ]
        .gt(0)
    )

    summary = (
        summary_input
        .groupby(
            [
                "rss_family_pair",
                "rss_governed_alignment_class",
                "rss_alignment_evidence_role",
                "rss_allowed_contextual_contribution",
                "rss_recurrence_testing_scope",
                "rss_alignment_convergence_evidence_candidate",
                "rss_relationship_recurrence_test_policy_defined",
                "rss_pattern_maturity_assignment_authorized",
                "rss_replay_cycle_formation_authorized",
                "rss_replay_lesson_generation_authorized",
                "rss_nid_authorization_state",
            ],
            dropna=False,
        )
        .agg(
            patient_relationship_count=(
                patient_column,
                "size",
            ),

            patient_count=(
                patient_column,
                "nunique",
            ),

            recurrence_testable_patient_count=(
                "rss_relationship_recurrence_testable",
                "sum",
            ),

            recurrence_observed_patient_count=(
                "rss_relationship_recurrence_observed",
                "sum",
            ),

            single_occurrence_patient_count=(
                "_rss_single_occurrence_flag",
                "sum",
            ),

            recurrence_not_testable_patient_count=(
                "_rss_not_testable_flag",
                "sum",
            ),

            total_alignment_count=(
                "rss_total_alignment_count",
                "sum",
            ),

            recurrence_test_eligible_alignment_count=(
                "rss_recurrence_test_eligible_alignment_count",
                "sum",
            ),

            total_distinct_historical_position_count=(
                "rss_distinct_historical_position_count",
                "sum",
            ),

            recurrent_historical_position_count=(
                "rss_recurrent_historical_position_count",
                "sum",
            ),

            maximum_distinct_position_count_per_patient=(
                "rss_distinct_historical_position_count",
                "max",
            ),

            same_position_alignment_compression_count=(
                "rss_same_position_alignment_compression_count",
                "sum",
            ),

            censoring_limited_recurrence_patient_count=(
                "_rss_censoring_limited_recurrence_flag",
                "sum",
            ),
        )
        .reset_index()
    )

    summary[
        "recurrence_observed_percent_of_testable"
    ] = [
        _calculate_rss_percent(
            observed,
            testable,
        )
        for observed, testable in zip(
            summary[
                "recurrence_observed_patient_count"
            ],
            summary[
                "recurrence_testable_patient_count"
            ],
        )
    ]

    return (
        summary
        .sort_values(
            by=[
                "rss_family_pair",
                "recurrence_observed_percent_of_testable",
                "rss_governed_alignment_class",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

__all__ = [
    "build_rss_cycle_boundary_signal_qualification",
    "build_rss_cycle_boundary_signal_summary",
    "reconcile_rss_binary_encounter_counts",
    "build_rss_direct_binary_boundary_review",
    "build_rss_direct_binary_boundary_summary",
    "build_rss_direct_binary_semantic_disposition",
    "build_rss_direct_binary_semantic_summary",
    "build_rss_binary_event_transition_readiness",
    "build_rss_binary_event_transition_summary",
    "build_rss_binary_event_episode_policy",
    "build_rss_binary_event_episode_policy_summary",
    "build_rss_controlled_binary_episode_construction",
    "build_rss_controlled_binary_episode_summary",
    "build_rss_cross_family_episode_alignment",
    "build_rss_cross_family_episode_alignment_summary",
    "build_rss_cross_family_alignment_governance",
    "build_rss_cross_family_alignment_governance_summary",
    "build_rss_cross_family_relationship_recurrence_test",
    "build_rss_cross_family_relationship_recurrence_summary",
]