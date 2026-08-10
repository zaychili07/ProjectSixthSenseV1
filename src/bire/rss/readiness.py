#============================================================
# Project Sixth Sense — BIRE OS
# Chapter 68.29 — RSS Playground Readiness Contract
#============================================================

from __future__ import annotations

from typing import Iterable

import pandas as pd


_RSS_PLAYGROUND_REQUIRED_AUTHORITY_COLUMNS = {
    "boundary_qualification_df": [
        "automatic_boundary_use_authorized",
        "replay_cycle_formation_authorized",
    ],
    "semantic_disposition_df": [
        "automatic_boundary_use_authorized",
        "replay_cycle_formation_authorized",
    ],
    "episode_policy_df": [
        "automatic_boundary_use_authorized",
        "replay_cycle_formation_authorized",
    ],
    "episode_df": [
        "rss_automatic_boundary_use_authorized",
        "rss_replay_cycle_formation_authorized",
    ],
    "governed_alignment_df": [
        "rss_alignment_operationally_authorized",
        "rss_replay_cycle_formation_authorized",
        "rss_replay_lesson_generation_authorized",
        "rss_causal_interpretation_authorized",
        "rss_episode_family_merge_authorized",
    ],
    "recurrence_df": [
        "rss_pattern_maturity_assignment_authorized",
        "rss_replay_cycle_formation_authorized",
        "rss_replay_lesson_generation_authorized",
        "rss_causal_interpretation_authorized",
    ],
}


def _require_dataframe(
    frame: pd.DataFrame,
    name: str,
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"{name} must be a pandas DataFrame."
        )


def _require_columns(
    frame: pd.DataFrame,
    required_columns: Iterable[str],
    name: str,
) -> None:
    missing = sorted(
        set(required_columns)
        - set(frame.columns)
    )

    if missing:
        raise KeyError(
            f"{name} is missing required columns: {missing}"
        )


def _column_contains_true(
    frame: pd.DataFrame,
    column: str,
) -> bool:
    return bool(
        frame[column]
        .fillna(False)
        .astype(bool)
        .any()
    )


def build_rss_playground_readiness_contract(
    ordering_contract_df: pd.DataFrame,
    boundary_qualification_df: pd.DataFrame,
    semantic_disposition_df: pd.DataFrame,
    episode_policy_df: pd.DataFrame,
    episode_df: pd.DataFrame,
    governed_alignment_df: pd.DataFrame,
    recurrence_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the final Chapter 68 RSS Playground readiness contract.

    The contract verifies installed capabilities and confirms
    that prohibited operational authorities remain withheld.

    Playground entry is stress-test authorization only.
    Deployment remains unauthorized pending Chapter 69.
    """

    frames = {
        "ordering_contract_df":
            ordering_contract_df,
        "boundary_qualification_df":
            boundary_qualification_df,
        "semantic_disposition_df":
            semantic_disposition_df,
        "episode_policy_df":
            episode_policy_df,
        "episode_df":
            episode_df,
        "governed_alignment_df":
            governed_alignment_df,
        "recurrence_df":
            recurrence_df,
    }

    for name, frame in frames.items():
        _require_dataframe(
            frame,
            name,
        )

    _require_columns(
        ordering_contract_df,
        {
            "rss_canonical_order_row_eligible",
            "rss_replay_chronology_eligible",
        },
        "ordering_contract_df",
    )

    _require_columns(
        boundary_qualification_df,
        {
            "column",
            "eligible_for_boundary_policy_review",
            "qualification_state",
            "automatic_boundary_use_authorized",
            "replay_cycle_formation_authorized",
        },
        "boundary_qualification_df",
    )

    _require_columns(
        semantic_disposition_df,
        {
            "column",
            "semantic_disposition",
            "semantic_disposition_state",
            "automatic_boundary_use_authorized",
            "replay_cycle_formation_authorized",
        },
        "semantic_disposition_df",
    )

    _require_columns(
        episode_policy_df,
        {
            "column",
            "episode_policy_defined",
            "persistence_policy_defined",
            "episode_construction_review_eligible",
            "automatic_boundary_use_authorized",
            "replay_cycle_formation_authorized",
        },
        "episode_policy_df",
    )

    _require_columns(
        episode_df,
        {
            "rss_episode_id",
            "rss_episode_family",
            "rss_episode_construction_review_eligible",
            "rss_automatic_boundary_use_authorized",
            "rss_replay_cycle_formation_authorized",
        },
        "episode_df",
    )

    _require_columns(
        governed_alignment_df,
        {
            "rss_alignment_id",
            "rss_alignment_governance_policy_defined",
            "rss_cross_signal_conflict_policy_defined",
            "rss_relationship_recurrence_testing_eligible",
            "rss_alignment_operationally_authorized",
            "rss_replay_cycle_formation_authorized",
            "rss_replay_lesson_generation_authorized",
            "rss_causal_interpretation_authorized",
            "rss_episode_family_merge_authorized",
        },
        "governed_alignment_df",
    )

    _require_columns(
        recurrence_df,
        {
            "rss_relationship_recurrence_test_policy_defined",
            "rss_relationship_recurrence_observed",
            "rss_distinct_historical_position_count",
            "rss_pattern_maturity_assignment_authorized",
            "rss_replay_cycle_formation_authorized",
            "rss_replay_lesson_generation_authorized",
            "rss_causal_interpretation_authorized",
        },
        "recurrence_df",
    )

    # --------------------------------------------------------
    # Installed capability checks.
    # --------------------------------------------------------

    chronology_installed = bool(
        len(ordering_contract_df) > 0
        and ordering_contract_df[
            "rss_canonical_order_row_eligible"
        ]
        .fillna(False)
        .any()
    )

    boundary_review_installed = bool(
        len(boundary_qualification_df) > 0
    )

    semantic_roles_defined = bool(
        len(semantic_disposition_df) > 0
        and
        ~semantic_disposition_df[
            "semantic_disposition"
        ]
        .eq(
            "SEMANTIC_DISPOSITION_UNDEFINED"
        )
        .any()
    )

    episode_policy_installed = bool(
        len(episode_policy_df) > 0
        and episode_policy_df[
            "episode_policy_defined"
        ]
        .fillna(False)
        .all()
    )

    provisional_episode_construction_installed = bool(
        len(episode_df) > 0
        and episode_df[
            "rss_episode_id"
        ]
        .notna()
        .all()
    )

    alignment_governance_installed = bool(
        len(governed_alignment_df) > 0
        and governed_alignment_df[
            "rss_alignment_governance_policy_defined"
        ]
        .fillna(False)
        .all()
        and governed_alignment_df[
            "rss_cross_signal_conflict_policy_defined"
        ]
        .fillna(False)
        .all()
    )

    recurrence_test_installed = bool(
        len(recurrence_df) > 0
        and recurrence_df[
            "rss_relationship_recurrence_test_policy_defined"
        ]
        .fillna(False)
        .all()
    )

    # --------------------------------------------------------
    # Readmission must remain deferred.
    # --------------------------------------------------------

    readmission_policy = (
        episode_policy_df.loc[
            episode_policy_df[
                "column"
            ].eq("readmission_flag")
        ]
    )

    readmission_safely_deferred = bool(
        len(readmission_policy) == 1
        and
        not bool(
            readmission_policy[
                "episode_construction_review_eligible"
            ]
            .fillna(False)
            .iloc[0]
        )
        and
        not bool(
            readmission_policy[
                "persistence_policy_defined"
            ]
            .fillna(False)
            .iloc[0]
        )
    )

    # --------------------------------------------------------
    # Prohibited-authority leakage check.
    # --------------------------------------------------------

    authority_leaks: list[str] = []

    for frame_name, columns in (
        _RSS_PLAYGROUND_REQUIRED_AUTHORITY_COLUMNS.items()
    ):
        frame = frames[frame_name]

        _require_columns(
            frame,
            columns,
            frame_name,
        )

        for column in columns:
            if _column_contains_true(
                frame,
                column,
            ):
                authority_leaks.append(
                    f"{frame_name}.{column}"
                )

    no_prohibited_authority_leak = (
        len(authority_leaks) == 0
    )

    installed_capabilities_complete = all(
        [
            chronology_installed,
            boundary_review_installed,
            semantic_roles_defined,
            episode_policy_installed,
            provisional_episode_construction_installed,
            alignment_governance_installed,
            recurrence_test_installed,
            readmission_safely_deferred,
            no_prohibited_authority_leak,
        ]
    )

    playground_entry_authorized = bool(
        installed_capabilities_complete
    )

    readiness_state = (
        "RSS_PLAYGROUND_READY_WITH_GOVERNED_LIMITATIONS"
        if playground_entry_authorized
        else
        "RSS_PLAYGROUND_ENTRY_BLOCKED_REVIEW_REQUIRED"
    )

    return pd.DataFrame(
        [
            {
                "rss_chapter":
                    "CHAPTER_68_REPLAY_STABILITY_SUPPORT",

                "rss_readiness_contract":
                    "RSS_PLAYGROUND_READINESS_CONTRACT",

                "rss_readiness_state":
                    readiness_state,

                "rss_readiness_scope":
                    "CONTROLLED_PLAYGROUND_STRESS_TESTING_ONLY",

                "canonical_chronology_installed":
                    chronology_installed,

                "boundary_review_installed":
                    boundary_review_installed,

                "semantic_roles_defined":
                    semantic_roles_defined,

                "episode_policy_installed":
                    episode_policy_installed,

                "provisional_episode_construction_installed":
                    provisional_episode_construction_installed,

                "alignment_governance_installed":
                    alignment_governance_installed,

                "relationship_recurrence_test_installed":
                    recurrence_test_installed,

                "readmission_safely_deferred":
                    readmission_safely_deferred,

                "no_prohibited_authority_leak":
                    no_prohibited_authority_leak,

                "prohibited_authority_leak_count":
                    len(authority_leaks),

                "prohibited_authority_leak_locations":
                    "NONE"
                    if not authority_leaks
                    else " | ".join(authority_leaks),

                "canonical_order_eligible_encounter_count":
                    int(
                        ordering_contract_df[
                            "rss_canonical_order_row_eligible"
                        ]
                        .fillna(False)
                        .sum()
                    ),

                "replay_chronology_eligible_encounter_count":
                    int(
                        ordering_contract_df[
                            "rss_replay_chronology_eligible"
                        ]
                        .fillna(False)
                        .sum()
                    ),

                "boundary_candidate_count":
                    int(
                        len(boundary_qualification_df)
                    ),

                "boundary_policy_review_eligible_count":
                    int(
                        boundary_qualification_df[
                            "eligible_for_boundary_policy_review"
                        ]
                        .fillna(False)
                        .sum()
                    ),

                "semantic_disposition_count":
                    int(
                        len(semantic_disposition_df)
                    ),

                "episode_policy_count":
                    int(
                        len(episode_policy_df)
                    ),

                "provisional_episode_count":
                    int(
                        episode_df[
                            "rss_episode_id"
                        ]
                        .nunique()
                    ),

                "governed_alignment_count":
                    int(
                        governed_alignment_df[
                            "rss_alignment_id"
                        ]
                        .nunique()
                    ),

                "recurrent_patient_relationship_count":
                    int(
                        recurrence_df[
                            "rss_relationship_recurrence_observed"
                        ]
                        .fillna(False)
                        .sum()
                    ),

                "recurrent_historical_position_count":
                    int(
                        recurrence_df[
                            "rss_distinct_historical_position_count"
                        ]
                        .fillna(0)
                        .sub(1)
                        .clip(lower=0)
                        .sum()
                    ),

                "tti_intervention_replay_state":
                    (
                        "DEFERRED_TTI_RESPONSE_ENGINE_INACTIVE"
                    ),

                "playground_entry_authorized":
                    playground_entry_authorized,

                "operational_deployment_authorized":
                    False,

                "replay_cycle_formation_authorized":
                    False,

                "replay_lesson_generation_authorized":
                    False,

                "causal_interpretation_authorized":
                    False,

                "nid_authorization_state":
                    (
                        "NID_RSS_PLAYGROUND_READINESS_"
                        "CONTRACT_INSTALLED"
                    ),

                "next_required_stage":
                    (
                        "CHAPTER_69_SYSTEM_VALIDATION_"
                        "AND_FINAL_REVIEW"
                    ),

                "rss_doctrine":
                    (
                        "REPLAY_MAY_REVEAL_A_LESSON_"
                        "RECURRENCE_MUST_EARN_IT"
                    ),
            }
        ]
    )


def build_rss_playground_readiness_limitations(
    boundary_qualification_df: pd.DataFrame,
    episode_policy_df: pd.DataFrame,
    recurrence_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce the explicit limitations accompanying RSS Playground
    entry.

    These limitations preserve honesty during stress testing.
    """

    for name, frame in {
        "boundary_qualification_df":
            boundary_qualification_df,
        "episode_policy_df":
            episode_policy_df,
        "recurrence_df":
            recurrence_df,
    }.items():
        _require_dataframe(
            frame,
            name,
        )

    _require_columns(
        boundary_qualification_df,
        {
            "qualification_state",
        },
        "boundary_qualification_df",
    )

    rows: list[dict[str, object]] = []

    pending_states = (
        boundary_qualification_df[
            "qualification_state"
        ]
        .value_counts(dropna=False)
    )

    for state in [
        "NUMERIC_THRESHOLD_POLICY_REQUIRED",
        "CATEGORICAL_STATE_MAPPING_REQUIRED",
        (
            "BINARY_WITHIN_ENCOUNTER_VARIATION_"
            "REVIEW_REQUIRED"
        ),
        (
            "HIGH_CARDINALITY_CONTEXT_"
            "NOT_BOUNDARY_READY"
        ),
        "INSUFFICIENT_SIGNAL_VARIATION",
    ]:
        count = int(
            pending_states.get(
                state,
                0,
            )
        )

        if count:
            rows.append(
                {
                    "rss_limitation":
                        state,

                    "evidence_count":
                        count,

                    "playground_entry_blocked":
                        False,

                    "operational_authority_blocked":
                        True,

                    "required_handling":
                        (
                            "PRESERVE_FOR_PLAYGROUND_REVIEW_"
                            "DO_NOT_MANUFACTURE_BOUNDARY"
                        ),
                }
            )

    readmission_deferred = bool(
        episode_policy_df[
            "column"
        ]
        .eq("readmission_flag")
        .any()
    )

    if readmission_deferred:
        rows.append(
            {
                "rss_limitation":
                    "READMISSION_OUTCOME_GRAIN_UNRESOLVED",

                "evidence_count":
                    1,

                "playground_entry_blocked":
                    False,

                "operational_authority_blocked":
                    True,

                "required_handling":
                    (
                        "DEFER_EPISODE_CONSTRUCTION_UNTIL_"
                        "DISCHARGE_AND_RETURN_TO_CARE_ALIGNMENT"
                    ),
            }
        )

    recurrent_relationship_count = int(
        recurrence_df[
            "rss_relationship_recurrence_observed"
        ]
        .fillna(False)
        .sum()
    )

    rows.append(
        {
            "rss_limitation":
                (
                    "RELATIONSHIP_RECURRENCE_OBSERVED_"
                    "PATTERN_AUTHORITY_WITHHELD"
                ),

            "evidence_count":
                recurrent_relationship_count,

            "playground_entry_blocked":
                False,

            "operational_authority_blocked":
                True,

            "required_handling":
                (
                    "EXPOSE_STRUCTURAL_RECURRENCE_ONLY_"
                    "NO_PATTERN_OR_LESSON_CLAIM"
                ),
        }
    )

    rows.append(
        {
            "rss_limitation":
                "TTI_INTERVENTION_REPLAY_DEFERRED",

            "evidence_count":
                0,

            "playground_entry_blocked":
                False,

            "operational_authority_blocked":
                True,

            "required_handling":
                (
                    "REMAIN_DEFERRED_WHILE_"
                    "TTI_RESPONSE_ENGINE_IS_INACTIVE"
                ),
        }
    )

    rows.append(
        {
            "rss_limitation":
                "SYSTEM_VALIDATION_PENDING",

            "evidence_count":
                1,

            "playground_entry_blocked":
                False,

            "operational_authority_blocked":
                True,

            "required_handling":
                (
                    "COMPLETE_CHAPTER_69_BEFORE_"
                    "ANY_DEPLOYMENT_DECISION"
                ),
        }
    )

    return pd.DataFrame(rows)


__all__ = [
    "build_rss_playground_readiness_contract",
    "build_rss_playground_readiness_limitations",
]