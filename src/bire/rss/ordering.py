#============================================================
# Project Sixth Sense — BIRE OS
# Replay Stability Support (RSS)
#
# File:
#     ordering.py
#
# Chapter:
#     68.19 — RSS Canonical Longitudinal Ordering Contract
#
# Purpose:
#     Install a governed patient-level chronology contract for
#     future RSS replay-cycle formation.
#
# Primary Chronology:
#     years_since_sim_start
#
# Secondary Validation:
#     encounter_sequence
#
# Final Deterministic Tie-Breaker:
#     encounter_id
#
# Responsibilities:
#     - Establish canonical patient encounter order
#     - Preserve temporal and ordinal disagreement
#     - Apply deterministic tie-breaking
#     - Quarantine unresolved within-encounter variation
#     - Identify patient histories eligible for replay ordering
#     - Preserve single-encounter non-replayability
#
# Does Not:
#     - Form replay cycles
#     - Assign RSS pattern states
#     - Assign RSS lesson states
#     - Manufacture missing chronology
#     - Allow ordinal sequence to override verified time
#     - Predict future behavior
#
# Governance:
#     Canonical chronology remains subject to NID governance.
#
# Core Doctrine:
#     Chronology follows verified time.
#     Ordinal sequence may support chronology.
#     Ordinal sequence must not rewrite time.
#============================================================

from __future__ import annotations

import pandas as pd


_RSS_PRIMARY_ORDER_FIELD = (
    "years_since_sim_start"
)

_RSS_SECONDARY_ORDER_FIELD = (
    "encounter_sequence"
)

_RSS_FINAL_TIEBREAKER_FIELD = (
    "encounter_id"
)


def build_rss_canonical_longitudinal_order_contract(
    agreement_df: pd.DataFrame,
    patient_column: str = "patient_id",
    encounter_column: str = "encounter_id",
) -> pd.DataFrame:
    """
    Construct the canonical RSS longitudinal ordering contract.

    Ordering hierarchy:

    1. years_since_sim_start
    2. encounter_sequence, only within temporal ties
    3. encounter_id, only as the final deterministic tie-breaker

    Strict disagreement from encounter_sequence is preserved but
    does not override temporal chronology.

    Encounters containing unresolved within-encounter temporal or
    sequence variation are quarantined from replay ordering.

    This function does not form replay cycles or generate RSS
    pattern and lesson states.
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
        "rss_chronology_agreement_state",
        "rss_patient_chronology_agreement_state",
    }

    missing_columns = sorted(
        required_columns
        - set(agreement_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "RSS canonical ordering contract cannot proceed. "
            f"Missing agreement columns: {missing_columns}"
        )

    contract = agreement_df.copy()

    for key in (
        patient_column,
        encounter_column,
    ):
        contract[key] = (
            contract[key]
            .astype("string")
            .str.strip()
        )

    missing_key_mask = (
        contract[
            [
                patient_column,
                encounter_column,
            ]
        ]
        .isna()
        .any(axis=1)
        |
        contract[
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
            "RSS chronology agreement data contains "
            f"{int(missing_key_mask.sum())} records with "
            "missing patient or encounter identifiers."
        )

    duplicate_contract_keys = (
        contract
        .duplicated(
            subset=[
                patient_column,
                encounter_column,
            ],
            keep=False,
        )
    )

    if duplicate_contract_keys.any():
        raise ValueError(
            "RSS chronology agreement data must contain "
            "one row per patient encounter. "
            f"Duplicate rows: "
            f"{int(duplicate_contract_keys.sum())}"
        )

    # --------------------------------------------------------
    # Establish row-level order eligibility.
    # --------------------------------------------------------

    temporal_missing = (
        contract[
            "rss_temporal_start"
        ]
        .isna()
    )

    sequence_missing = (
        contract[
            "rss_sequence_start"
        ]
        .isna()
    )

    temporal_variation = (
        contract[
            "rss_temporal_unique_count"
        ]
        .gt(1)
    )

    sequence_variation = (
        contract[
            "rss_sequence_unique_count"
        ]
        .gt(1)
    )

    within_encounter_variation = (
        temporal_variation
        |
        sequence_variation
    )

    contract[
        "rss_canonical_order_missing_flag"
    ] = (
        temporal_missing
        |
        sequence_missing
    )

    contract[
        "rss_within_encounter_order_variation_flag"
    ] = within_encounter_variation

    contract[
        "rss_canonical_order_row_eligible"
    ] = (
        ~contract[
            "rss_canonical_order_missing_flag"
        ]
        &
        ~contract[
            "rss_within_encounter_order_variation_flag"
        ]
    )

    # --------------------------------------------------------
    # Build deterministic chronology for eligible encounters.
    #
    # The sequence field can affect order only after temporal
    # values have tied.
    # --------------------------------------------------------

    eligible = contract.loc[
        contract[
            "rss_canonical_order_row_eligible"
        ],
        [
            patient_column,
            encounter_column,
            "rss_temporal_start",
            "rss_temporal_end",
            "rss_sequence_start",
            "rss_sequence_end",
        ],
    ].copy()

    eligible[
        "rss_primary_temporal_tie_flag"
    ] = eligible.duplicated(
        subset=[
            patient_column,
            "rss_temporal_start",
        ],
        keep=False,
    )

    eligible[
        "rss_secondary_sequence_tie_flag"
    ] = eligible.duplicated(
        subset=[
            patient_column,
            "rss_temporal_start",
            "rss_sequence_start",
        ],
        keep=False,
    )

    eligible[
        "rss_secondary_sequence_tiebreaker_used"
    ] = (
        eligible[
            "rss_primary_temporal_tie_flag"
        ]
        &
        eligible[
            "rss_sequence_start"
        ]
        .notna()
        &
        ~eligible[
            "rss_secondary_sequence_tie_flag"
        ]
    )

    eligible[
        "rss_final_identifier_tiebreaker_used"
    ] = (
        eligible[
            "rss_primary_temporal_tie_flag"
        ]
        &
        (
            eligible[
                "rss_sequence_start"
            ]
            .isna()
            |
            eligible[
                "rss_secondary_sequence_tie_flag"
            ]
        )
    )

    eligible = (
        eligible
        .sort_values(
            by=[
                patient_column,
                "rss_temporal_start",
                "rss_sequence_start",
                encounter_column,
            ],
            kind="mergesort",
            na_position="last",
        )
        .reset_index(drop=True)
    )

    eligible[
        "rss_canonical_order_position"
    ] = (
        eligible
        .groupby(
            patient_column,
            dropna=False,
        )
        .cumcount()
        .add(1)
        .astype("Int64")
    )

    patient_groups = eligible.groupby(
        patient_column,
        dropna=False,
    )

    eligible[
        "rss_canonical_previous_encounter_id"
    ] = patient_groups[
        encounter_column
    ].shift(1)

    eligible[
        "rss_canonical_next_encounter_id"
    ] = patient_groups[
        encounter_column
    ].shift(-1)

    eligible[
        "rss_canonical_previous_temporal_end"
    ] = patient_groups[
        "rss_temporal_end"
    ].shift(1)

    eligible[
        "rss_canonical_next_temporal_start"
    ] = patient_groups[
        "rss_temporal_start"
    ].shift(-1)

    eligible[
        "rss_canonical_gap_from_previous_end"
    ] = (
        eligible[
            "rss_temporal_start"
        ]
        -
        eligible[
            "rss_canonical_previous_temporal_end"
        ]
    )

    eligible[
        "rss_canonical_temporal_overlap_flag"
    ] = (
        eligible[
            "rss_canonical_previous_temporal_end"
        ]
        .notna()
        &
        eligible[
            "rss_temporal_start"
        ]
        .lt(
            eligible[
                "rss_canonical_previous_temporal_end"
            ]
        )
    )

    ordering_columns = [
        patient_column,
        encounter_column,
        "rss_primary_temporal_tie_flag",
        "rss_secondary_sequence_tie_flag",
        "rss_secondary_sequence_tiebreaker_used",
        "rss_final_identifier_tiebreaker_used",
        "rss_canonical_order_position",
        "rss_canonical_previous_encounter_id",
        "rss_canonical_next_encounter_id",
        "rss_canonical_previous_temporal_end",
        "rss_canonical_next_temporal_start",
        "rss_canonical_gap_from_previous_end",
        "rss_canonical_temporal_overlap_flag",
    ]

    contract = contract.merge(
        eligible[ordering_columns],
        on=[
            patient_column,
            encounter_column,
        ],
        how="left",
        validate="one_to_one",
    )

    boolean_columns = [
        "rss_primary_temporal_tie_flag",
        "rss_secondary_sequence_tie_flag",
        "rss_secondary_sequence_tiebreaker_used",
        "rss_final_identifier_tiebreaker_used",
        "rss_canonical_temporal_overlap_flag",
    ]

    for column in boolean_columns:
        contract[column] = (
            contract[column]
            .fillna(False)
            .astype(bool)
        )

    contract[
        "rss_canonical_order_position"
    ] = (
        contract[
            "rss_canonical_order_position"
        ]
        .astype("Int64")
    )

    # --------------------------------------------------------
    # Preserve the relationship between temporal and ordinal
    # chronology without allowing ordinal evidence to rewrite
    # temporal order.
    # --------------------------------------------------------

    agreement_state = contract[
        "rss_chronology_agreement_state"
    ]

    contract[
        "rss_sequence_validation_state"
    ] = "SEQUENCE_VALIDATION_UNRESOLVED"

    contract.loc[
        agreement_state.eq(
            "ORDER_POSITION_MATCH"
        ),
        "rss_sequence_validation_state",
    ] = "SEQUENCE_SUPPORTS_TEMPORAL_ORDER"

    contract.loc[
        agreement_state.eq(
            "ORDER_DIFFERENCE_TIE_RELATED"
        ),
        "rss_sequence_validation_state",
    ] = (
        "SEQUENCE_TIE_AMBIGUITY_"
        "TEMPORAL_ORDER_PRIMARY"
    )

    contract.loc[
        agreement_state.eq(
            "ORDER_DIFFERENCE_STRICT_"
            "REVIEW_REQUIRED"
        ),
        "rss_sequence_validation_state",
    ] = (
        "SEQUENCE_DISAGREEMENT_PRESERVED_"
        "TEMPORAL_ORDER_PRIMARY"
    )

    contract.loc[
        agreement_state.eq(
            "WITHIN_ENCOUNTER_ORDER_"
            "CONFLICT_REVIEW_REQUIRED"
        ),
        "rss_sequence_validation_state",
    ] = (
        "TEMPORAL_AND_SEQUENCE_VARIATION_"
        "QUARANTINED"
    )

    contract.loc[
        agreement_state.eq(
            "MISSING_ORDER_VALUE"
        ),
        "rss_sequence_validation_state",
    ] = "SEQUENCE_VALIDATION_INCOMPLETE"

    # --------------------------------------------------------
    # Encounter-level chronology contract states.
    # --------------------------------------------------------

    contract[
        "rss_canonical_order_contract_state"
    ] = "CANONICAL_ORDER_UNRESOLVED"

    contract.loc[
        agreement_state.eq(
            "ORDER_POSITION_MATCH"
        ),
        "rss_canonical_order_contract_state",
    ] = "CANONICAL_TEMPORAL_ORDER_ESTABLISHED"

    contract.loc[
        agreement_state.eq(
            "ORDER_DIFFERENCE_TIE_RELATED"
        ),
        "rss_canonical_order_contract_state",
    ] = (
        "CANONICAL_TEMPORAL_ORDER_ESTABLISHED_"
        "SEQUENCE_TIE_PRESERVED"
    )

    contract.loc[
        agreement_state.eq(
            "ORDER_DIFFERENCE_STRICT_"
            "REVIEW_REQUIRED"
        ),
        "rss_canonical_order_contract_state",
    ] = (
        "CANONICAL_TEMPORAL_ORDER_ESTABLISHED_"
        "SEQUENCE_DISAGREEMENT_PRESERVED"
    )

    contract.loc[
        contract[
            "rss_within_encounter_order_variation_flag"
        ],
        "rss_canonical_order_contract_state",
    ] = (
        "CANONICAL_ORDER_QUARANTINED_"
        "WITHIN_ENCOUNTER_VARIATION"
    )

    contract.loc[
        contract[
            "rss_canonical_order_missing_flag"
        ],
        "rss_canonical_order_contract_state",
    ] = (
        "CANONICAL_ORDER_INCOMPLETE_"
        "MISSING_ORDER_VALUE"
    )

    # --------------------------------------------------------
    # Patient-level chronology contract.
    # --------------------------------------------------------

    patient_review = (
        contract
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
            rss_patient_eligible_encounter_count=(
                "rss_canonical_order_row_eligible",
                "sum",
            ),
            rss_patient_has_missing_order=(
                "rss_canonical_order_missing_flag",
                "any",
            ),
            rss_patient_has_within_encounter_variation=(
                "rss_within_encounter_order_variation_flag",
                "any",
            ),
            rss_patient_has_sequence_disagreement=(
                "rss_chronology_agreement_state",
                lambda values: bool(
                    values.eq(
                        "ORDER_DIFFERENCE_STRICT_"
                        "REVIEW_REQUIRED"
                    ).any()
                ),
            ),
            rss_patient_has_sequence_tie_ambiguity=(
                "rss_chronology_agreement_state",
                lambda values: bool(
                    values.eq(
                        "ORDER_DIFFERENCE_TIE_RELATED"
                    ).any()
                ),
            ),
        )
    )

    patient_review[
        "rss_patient_chronology_contract_state"
    ] = "PATIENT_CANONICAL_CHRONOLOGY_ESTABLISHED"

    single_encounter_mask = (
        patient_review[
            "rss_patient_encounter_count"
        ]
        .eq(1)
    )

    patient_review.loc[
        single_encounter_mask,
        "rss_patient_chronology_contract_state",
    ] = (
        "PATIENT_SINGLE_ENCOUNTER_CHRONOLOGY_"
        "ESTABLISHED_NOT_REPLAYABLE"
    )

    patient_review.loc[
        patient_review[
            "rss_patient_has_sequence_tie_ambiguity"
        ],
        "rss_patient_chronology_contract_state",
    ] = (
        "PATIENT_CANONICAL_CHRONOLOGY_ESTABLISHED_"
        "SEQUENCE_TIE_PRESERVED"
    )

    patient_review.loc[
        patient_review[
            "rss_patient_has_sequence_disagreement"
        ],
        "rss_patient_chronology_contract_state",
    ] = (
        "PATIENT_CANONICAL_CHRONOLOGY_ESTABLISHED_"
        "SEQUENCE_DISAGREEMENT_PRESERVED"
    )

    patient_review.loc[
        patient_review[
            "rss_patient_has_within_encounter_variation"
        ],
        "rss_patient_chronology_contract_state",
    ] = (
        "PATIENT_CHRONOLOGY_QUARANTINED_"
        "WITHIN_ENCOUNTER_VARIATION"
    )

    patient_review.loc[
        patient_review[
            "rss_patient_has_missing_order"
        ],
        "rss_patient_chronology_contract_state",
    ] = "PATIENT_CHRONOLOGY_INCOMPLETE"

    patient_review[
        "rss_patient_replay_chronology_eligible"
    ] = (
        patient_review[
            "rss_patient_encounter_count"
        ]
        .ge(2)
        &
        ~patient_review[
            "rss_patient_has_missing_order"
        ]
        &
        ~patient_review[
            "rss_patient_has_within_encounter_variation"
        ]
    )

    contract = contract.merge(
        patient_review[
            [
                patient_column,
                "rss_patient_encounter_count",
                "rss_patient_eligible_encounter_count",
                "rss_patient_chronology_contract_state",
                "rss_patient_replay_chronology_eligible",
            ]
        ],
        on=patient_column,
        how="left",
        validate="many_to_one",
    )

    contract[
        "rss_replay_chronology_eligible"
    ] = (
        contract[
            "rss_canonical_order_row_eligible"
        ]
        &
        contract[
            "rss_patient_replay_chronology_eligible"
        ]
    )

    # --------------------------------------------------------
    # Canonical authority metadata.
    # --------------------------------------------------------

    contract[
        "rss_canonical_order_primary_field"
    ] = _RSS_PRIMARY_ORDER_FIELD

    contract[
        "rss_canonical_order_secondary_field"
    ] = _RSS_SECONDARY_ORDER_FIELD

    contract[
        "rss_canonical_order_final_tiebreaker_field"
    ] = _RSS_FINAL_TIEBREAKER_FIELD

    contract[
        "rss_canonical_order_selection_state"
    ] = (
        "RSS_CANONICAL_LONGITUDINAL_ORDER_INSTALLED"
    )

    contract[
        "rss_canonical_order_selected"
    ] = True

    contract[
        "rss_canonical_order_nid_governance_state"
    ] = "NID_GOVERNED"

    return (
        contract
        .sort_values(
            by=[
                patient_column,
                "rss_temporal_start",
                "rss_sequence_start",
                encounter_column,
            ],
            kind="mergesort",
            na_position="last",
        )
        .reset_index(drop=True)
    )


def build_rss_canonical_longitudinal_order_summary(
    contract_df: pd.DataFrame,
    patient_column: str = "patient_id",
) -> pd.DataFrame:
    """
    Summarize the installed RSS canonical ordering contract.

    Canonical chronology may be installed while replay-cycle
    formation remains unauthorized.
    """

    if not isinstance(
        contract_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "contract_df must be a pandas DataFrame."
        )

    required_columns = {
        patient_column,
        "rss_canonical_order_row_eligible",
        "rss_replay_chronology_eligible",
        "rss_primary_temporal_tie_flag",
        "rss_secondary_sequence_tiebreaker_used",
        "rss_final_identifier_tiebreaker_used",
        "rss_chronology_agreement_state",
        "rss_patient_chronology_contract_state",
        "rss_patient_replay_chronology_eligible",
    }

    missing_columns = sorted(
        required_columns
        - set(contract_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "RSS canonical ordering summary cannot proceed. "
            f"Missing contract columns: {missing_columns}"
        )

    patient_contract = (
        contract_df[
            [
                patient_column,
                "rss_patient_chronology_contract_state",
                "rss_patient_replay_chronology_eligible",
            ]
        ]
        .drop_duplicates(
            subset=patient_column
        )
    )

    patient_states = (
        patient_contract[
            "rss_patient_chronology_contract_state"
        ]
        .value_counts()
    )

    encounter_states = (
        contract_df[
            "rss_chronology_agreement_state"
        ]
        .value_counts()
    )

    return pd.DataFrame([
        {
            "installation_state":
                "RSS_CANONICAL_LONGITUDINAL_ORDER_INSTALLED",

            "primary_order_field":
                _RSS_PRIMARY_ORDER_FIELD,

            "secondary_order_field":
                _RSS_SECONDARY_ORDER_FIELD,

            "final_tiebreaker_field":
                _RSS_FINAL_TIEBREAKER_FIELD,

            "canonical_order_authorized":
                True,

            "replay_cycle_formation_authorized":
                False,

            "total_encounter_count":
                len(contract_df),

            "canonical_order_eligible_encounter_count":
                int(
                    contract_df[
                        "rss_canonical_order_row_eligible"
                    ]
                    .sum()
                ),

            "replay_eligible_encounter_count":
                int(
                    contract_df[
                        "rss_replay_chronology_eligible"
                    ]
                    .sum()
                ),

            "quarantined_or_incomplete_encounter_count":
                int(
                    (
                        ~contract_df[
                            "rss_canonical_order_row_eligible"
                        ]
                    )
                    .sum()
                ),

            "total_patient_count":
                int(
                    contract_df[
                        patient_column
                    ]
                    .nunique()
                ),

            "replay_eligible_patient_count":
                int(
                    patient_contract[
                        "rss_patient_replay_chronology_eligible"
                    ]
                    .sum()
                ),

            "single_encounter_not_replayable_count":
                int(
                    patient_states.get(
                        "PATIENT_SINGLE_ENCOUNTER_"
                        "CHRONOLOGY_ESTABLISHED_"
                        "NOT_REPLAYABLE",
                        0,
                    )
                ),

            "sequence_disagreement_patient_count":
                int(
                    patient_states.get(
                        "PATIENT_CANONICAL_CHRONOLOGY_"
                        "ESTABLISHED_SEQUENCE_"
                        "DISAGREEMENT_PRESERVED",
                        0,
                    )
                ),

            "sequence_tie_patient_count":
                int(
                    patient_states.get(
                        "PATIENT_CANONICAL_CHRONOLOGY_"
                        "ESTABLISHED_SEQUENCE_TIE_PRESERVED",
                        0,
                    )
                ),

            "quarantined_patient_count":
                int(
                    patient_states.get(
                        "PATIENT_CHRONOLOGY_QUARANTINED_"
                        "WITHIN_ENCOUNTER_VARIATION",
                        0,
                    )
                ),

            "incomplete_patient_count":
                int(
                    patient_states.get(
                        "PATIENT_CHRONOLOGY_INCOMPLETE",
                        0,
                    )
                ),

            "strict_sequence_difference_encounter_count":
                int(
                    encounter_states.get(
                        "ORDER_DIFFERENCE_STRICT_"
                        "REVIEW_REQUIRED",
                        0,
                    )
                ),

            "tie_related_difference_encounter_count":
                int(
                    encounter_states.get(
                        "ORDER_DIFFERENCE_TIE_RELATED",
                        0,
                    )
                ),

            "temporal_tie_encounter_count":
                int(
                    contract_df[
                        "rss_primary_temporal_tie_flag"
                    ]
                    .sum()
                ),

            "secondary_sequence_tiebreaker_use_count":
                int(
                    contract_df[
                        "rss_secondary_sequence_tiebreaker_used"
                    ]
                    .sum()
                ),

            "final_identifier_tiebreaker_use_count":
                int(
                    contract_df[
                        "rss_final_identifier_tiebreaker_used"
                    ]
                    .sum()
                ),

            "next_requirement":
                "RSS_REPLAY_CYCLE_BOUNDARY_FORMATION",
        }
    ])


__all__ = [
    "build_rss_canonical_longitudinal_order_contract",
    "build_rss_canonical_longitudinal_order_summary",
]