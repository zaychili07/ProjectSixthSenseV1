# ============================================================
# BIRE OS — Therapeutic Trajectory Intelligence (TTI)
# File: src/bire/tti/features.py
# Chapter: 67
#
# Created: 2026-07-15
# Updated: 2026-07-15
#
# CHANGED:
# - Initialized TTI feature module
# - Prepared backend space for therapeutic influence features
# - Added framework for intervention response evaluation
# - Separated TTI feature creation from scoring utilities
#
# Purpose:
# Backend helper for therapeutic influence evaluation,
# trajectory modification analysis,
# intervention response interpretation,
# and therapeutic influence formation.
# ============================================================

#============================================================
# Chapter 67.15 — TTI Intervention Episode Contract Construction
# Multi-row encounter-safe implementation
#============================================================

from __future__ import annotations

import re

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_numeric_dtype,
)


_TTI_EPISODE_KEYS = [
    "patient_id",
    "encounter_id",
]

_TTI_HEMODYNAMIC_ANCHOR = (
    "intervention_hemodynamic_support"
)


def _coerce_tti_nullable_boolean(
    series: pd.Series,
    source_name: str,
) -> pd.Series:
    """
    Convert an exposure field to pandas nullable boolean.

    Accepts:
    - True / False
    - 1 / 0
    - common string boolean values

    Raises an error when non-boolean exposure values are found.
    """

    if is_bool_dtype(series):
        return series.astype("boolean")

    if is_numeric_dtype(series):
        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        invalid = (
            numeric.notna()
            & ~numeric.isin([0, 1])
        )

        if invalid.any():
            invalid_values = (
                series.loc[invalid]
                .drop_duplicates()
                .astype(str)
                .head(10)
                .tolist()
            )

            raise ValueError(
                f"{source_name} contains non-binary values "
                f"in {_TTI_HEMODYNAMIC_ANCHOR}: "
                f"{invalid_values}"
            )

        return (
            numeric
            .map({
                1: True,
                0: False,
            })
            .astype("boolean")
        )

    normalized = (
        series
        .astype("string")
        .str.strip()
        .str.lower()
    )

    converted = (
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

    invalid = (
        series.notna()
        & converted.isna()
    )

    if invalid.any():
        invalid_values = (
            series.loc[invalid]
            .drop_duplicates()
            .astype(str)
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"{source_name} contains unrecognized boolean "
            f"values in {_TTI_HEMODYNAMIC_ANCHOR}: "
            f"{invalid_values}"
        )

    return converted


def _normalize_tti_source_prefix(
    source_name: str,
) -> str:
    """
    Create a safe lowercase prefix for source-level columns.
    """

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        source_name.lower(),
    ).strip("_")


def _prepare_tti_episode_source(
    df: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    """
    Normalize and aggregate a TTI source to encounter grain.

    Multiple source rows per encounter are preserved through:
    - total source-row count
    - True exposure count
    - False exposure count
    - missing exposure count
    - within-encounter exposure variation state

    This function does not infer intervention timing.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"{source_name} must be a pandas DataFrame."
        )

    required_cols = [
        *_TTI_EPISODE_KEYS,
        _TTI_HEMODYNAMIC_ANCHOR,
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"{source_name} is missing required columns: "
            f"{missing_cols}"
        )

    source = df[required_cols].copy()

    for key in _TTI_EPISODE_KEYS:
        source[key] = (
            source[key]
            .astype("string")
            .str.strip()
        )

    missing_key_mask = (
        source[_TTI_EPISODE_KEYS]
        .isna()
        .any(axis=1)
        |
        source[_TTI_EPISODE_KEYS]
        .eq("")
        .any(axis=1)
    )

    if missing_key_mask.any():
        raise ValueError(
            f"{source_name} contains "
            f"{int(missing_key_mask.sum())} rows with "
            "missing patient or encounter identifiers."
        )

    source[_TTI_HEMODYNAMIC_ANCHOR] = (
        _coerce_tti_nullable_boolean(
            source[_TTI_HEMODYNAMIC_ANCHOR],
            source_name=source_name,
        )
    )

    # --------------------------------------------------------
    # Aggregate multiple records to one patient/encounter row.
    # --------------------------------------------------------

    aggregated = (
        source
        .groupby(
            _TTI_EPISODE_KEYS,
            as_index=False,
            sort=False,
            dropna=False,
        )
        .agg(
            row_count=(
                _TTI_HEMODYNAMIC_ANCHOR,
                "size",
            ),
            exposure_true_count=(
                _TTI_HEMODYNAMIC_ANCHOR,
                lambda values: int(
                    values.eq(True).sum()
                ),
            ),
            exposure_false_count=(
                _TTI_HEMODYNAMIC_ANCHOR,
                lambda values: int(
                    values.eq(False).sum()
                ),
            ),
            exposure_missing_count=(
                _TTI_HEMODYNAMIC_ANCHOR,
                lambda values: int(
                    values.isna().sum()
                ),
            ),
        )
    )

    exposure_flag = pd.Series(
        pd.NA,
        index=aggregated.index,
        dtype="boolean",
    )

    exposure_present = (
        aggregated[
            "exposure_true_count"
        ].gt(0)
    )

    exposure_absent = (
        aggregated[
            "exposure_true_count"
        ].eq(0)
        &
        aggregated[
            "exposure_false_count"
        ].gt(0)
    )

    exposure_flag.loc[
        exposure_present
    ] = True

    exposure_flag.loc[
        exposure_absent
    ] = False

    aggregated[
        "exposure_flag"
    ] = exposure_flag

    # --------------------------------------------------------
    # Preserve within-encounter source behavior.
    # --------------------------------------------------------

    aggregated[
        "within_encounter_exposure_state"
    ] = "ALL_EXPOSURE_VALUES_MISSING"

    aggregated.loc[
        (
            aggregated[
                "exposure_true_count"
            ].gt(0)
            &
            aggregated[
                "exposure_false_count"
            ].eq(0)
        ),
        "within_encounter_exposure_state",
    ] = "EXPOSURE_PRESENT_ONLY"

    aggregated.loc[
        (
            aggregated[
                "exposure_true_count"
            ].eq(0)
            &
            aggregated[
                "exposure_false_count"
            ].gt(0)
        ),
        "within_encounter_exposure_state",
    ] = "NO_EXPOSURE_ONLY"

    aggregated.loc[
        (
            aggregated[
                "exposure_true_count"
            ].gt(0)
            &
            aggregated[
                "exposure_false_count"
            ].gt(0)
        ),
        "within_encounter_exposure_state",
    ] = "MIXED_EXPOSURE_OBSERVATIONS"

    aggregated[
        "source_grain_state"
    ] = "SINGLE_ROW_ENCOUNTER"

    aggregated.loc[
        aggregated["row_count"].gt(1),
        "source_grain_state",
    ] = "MULTIROW_ENCOUNTER"

    # Prefix source-level diagnostics before sources are merged.
    prefix = _normalize_tti_source_prefix(
        source_name
    )

    rename_map = {
        col: f"{prefix}_{col}"
        for col in aggregated.columns
        if col not in _TTI_EPISODE_KEYS
    }

    return aggregated.rename(
        columns=rename_map
    )


def build_tti_intervention_episode_contract(
    intervention_df: pd.DataFrame,
    operations_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct the first encounter-level TTI intervention contract.

    Multiple source records per encounter are aggregated without
    discarding evidence.

    Any verified True observation establishes coarse exposure
    during the encounter. Exact intervention timing remains
    unavailable and is not inferred.

    The operations-source copy is used only for consistency
    validation and does not increase evidentiary weight.
    """

    intervention_source = (
        _prepare_tti_episode_source(
            intervention_df,
            source_name="INTERVENTION_SOURCE",
        )
    )

    operations_source = (
        _prepare_tti_episode_source(
            operations_df,
            source_name="OPERATIONS_SOURCE",
        )
    )

    comparison = intervention_source.merge(
        operations_source,
        on=_TTI_EPISODE_KEYS,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    unmatched_source_mask = (
        ~comparison["_merge"].eq("both")
    )

    if unmatched_source_mask.any():
        raise ValueError(
            "Intervention and operations sources do not "
            "contain identical patient/encounter keys after "
            "encounter-level aggregation. "
            f"Unmatched encounters: "
            f"{int(unmatched_source_mask.sum())}"
        )

    intervention_exposure = comparison[
        "intervention_source_exposure_flag"
    ].astype("boolean")

    operations_exposure = comparison[
        "operations_source_exposure_flag"
    ].astype("boolean")

    exposure_mismatch = (
        intervention_exposure
        .fillna(False)
        .ne(
            operations_exposure.fillna(False)
        )
        |
        intervention_exposure
        .isna()
        .ne(
            operations_exposure.isna()
        )
    )

    if exposure_mismatch.any():
        raise ValueError(
            "Aggregated hemodynamic-support exposure "
            "does not match across intervention and "
            "operations sources. "
            f"Mismatch count: "
            f"{int(exposure_mismatch.sum())}"
        )

    contract = comparison[
        _TTI_EPISODE_KEYS
    ].copy()

    exposure_present = (
        intervention_exposure.eq(True)
    )

    exposure_absent = (
        intervention_exposure.eq(False)
    )

    exposure_unknown = (
        intervention_exposure.isna()
    )

    # --------------------------------------------------------
    # Preserve source-grain and exposure diagnostics.
    # --------------------------------------------------------

    contract[
        "tti_intervention_source_row_count"
    ] = comparison[
        "intervention_source_row_count"
    ].astype(int)

    contract[
        "tti_intervention_exposure_observation_count"
    ] = comparison[
        "intervention_source_exposure_true_count"
    ].astype(int)

    contract[
        "tti_intervention_nonexposure_observation_count"
    ] = comparison[
        "intervention_source_exposure_false_count"
    ].astype(int)

    contract[
        "tti_intervention_missing_exposure_count"
    ] = comparison[
        "intervention_source_exposure_missing_count"
    ].astype(int)

    contract[
        "tti_intervention_source_grain_state"
    ] = comparison[
        "intervention_source_source_grain_state"
    ]

    contract[
        "tti_intervention_within_encounter_state"
    ] = comparison[
        "intervention_source_"
        "within_encounter_exposure_state"
    ]

# --------------------------------------------------------
# Canonical encounter-level exposure contract.
# --------------------------------------------------------

# --------------------------------------------------------
# Intervention family being evaluated
# --------------------------------------------------------

    contract[
        "tti_intervention_family_evaluated"
        ] = "HEMODYNAMIC_SUPPORT"


# --------------------------------------------------------
# Exposure flag (unchanged)
# --------------------------------------------------------

    contract[
        "tti_intervention_exposure_flag"
        ] = intervention_exposure


# --------------------------------------------------------
# Family-specific exposure state
# --------------------------------------------------------

    contract[
        "tti_intervention_exposure_state"
        ] = "HEMODYNAMIC_SUPPORT_EXPOSURE_UNKNOWN"

    contract.loc[
        exposure_present,
        "tti_intervention_exposure_state"
        ] = "HEMODYNAMIC_SUPPORT_EXPOSURE_PRESENT"

    contract.loc[
        exposure_absent,
        "tti_intervention_exposure_state"
        ] = "HEMODYNAMIC_SUPPORT_EXPOSURE_NOT_OBSERVED"


# --------------------------------------------------------
# Intervention family (only when exposure exists)
# --------------------------------------------------------

    contract[
        "tti_intervention_family"
        ] = pd.Series(
            pd.NA,
            index=contract.index,
             dtype="string"
             )

    contract.loc[
        exposure_present,
        "tti_intervention_family",
        ] = "HEMODYNAMIC_SUPPORT"


# --------------------------------------------------------
# Episode ID
# --------------------------------------------------------

    contract[
        "tti_intervention_episode_id"
        ] = pd.Series(
            pd.NA,
            index=contract.index,
            dtype="string",
            )

    contract.loc[
        exposure_present,
        "tti_intervention_episode_id",
    ] = (
        contract.loc[
            exposure_present,
            "encounter_id",
    ]
    .astype("string")
    + "::HEMODYNAMIC_SUPPORT"
)


# --------------------------------------------------------
# Anchor metadata
# --------------------------------------------------------

    contract[
        "tti_intervention_anchor_source"
    ] = "INTERVENTION_SOURCE"

    contract[
        "tti_intervention_anchor_column"
    ] = _TTI_HEMODYNAMIC_ANCHOR


# --------------------------------------------------------
# Duplicate source validation
# --------------------------------------------------------

    source_row_counts_match = (
        comparison[
            "intervention_source_row_count"
    ]
        .eq(
            comparison[
                "operations_source_row_count"
        ]
    )
)

    contract[
    "tti_intervention_anchor_consistency_state"
] = "DUPLICATE_SOURCE_EXPOSURE_MATCHED"

    contract.loc[
        source_row_counts_match,
        "tti_intervention_anchor_consistency_state",
    ] = (
        "DUPLICATE_SOURCE_EXPOSURE_AND_"
        "ROW_COUNTS_MATCHED"
)


# --------------------------------------------------------
# Timing state (family-specific)
# --------------------------------------------------------

    contract[
        "tti_intervention_timing_state"
    ] = "HEMODYNAMIC_SUPPORT_TIMING_UNKNOWN"

    contract.loc[
        exposure_present,
        "tti_intervention_timing_state",
    ] = "HEMODYNAMIC_SUPPORT_TIMING_UNAVAILABLE"

    contract.loc[
        exposure_absent,
        "tti_intervention_timing_state",
    ] = (
        "NOT_APPLICABLE_NO_HEMODYNAMIC_SUPPORT_EXPOSURE"
    )


# --------------------------------------------------------
# Contract state (family-specific + variation aware)
# --------------------------------------------------------

    contract[
        "tti_intervention_episode_contract_state"
    ] = "HEMODYNAMIC_SUPPORT_EPISODE_UNVERIFIED"

    contract.loc[
        exposure_present,
        "tti_intervention_episode_contract_state",
    ] = (
        "COARSE_HEMODYNAMIC_SUPPORT_EXPOSURE_"
        "ESTABLISHED_TIMING_INCOMPLETE"
    )

    mixed_exposure_mask = (
        exposure_present
        &
        contract[
            "tti_intervention_within_encounter_state"
        ].eq(
            "MIXED_EXPOSURE_OBSERVATIONS"
        )
    )

    contract.loc[
        mixed_exposure_mask,
        "tti_intervention_episode_contract_state",
    ] = (
        "COARSE_HEMODYNAMIC_SUPPORT_EXPOSURE_"
        "WITHIN_ENCOUNTER_VARIATION_"
        "TIMING_INCOMPLETE"
    )

    contract.loc[
        exposure_absent,
        "tti_intervention_episode_contract_state",
    ] = "NO_HEMODYNAMIC_SUPPORT_EPISODE"

    contract.loc[
        exposure_unknown,
        "tti_intervention_episode_contract_state",
    ] = "HEMODYNAMIC_SUPPORT_EXPOSURE_UNKNOWN"

    return (
        contract
        .sort_values(
            _TTI_EPISODE_KEYS
        )
        .reset_index(drop=True)
    )