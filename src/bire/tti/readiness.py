#============================================================
# Project Sixth Sense — BIRE OS
# Therapeutic Trajectory Intelligence (TTI)
#
# File:
#     readiness.py
#
# Chapter:
#     67.16 — TTI Activation Readiness Gate
#
# Purpose:
#     Evaluate whether TTI possesses the verified structural,
#     temporal, trajectory, and attribution evidence required
#     to activate therapeutic response interpretation.
#
# Responsibilities:
#     - Validate the installed TTI intervention foundation
#     - Evaluate response-engine activation prerequisites
#     - Preserve deferred requirements without inference
#     - Prevent unsupported therapeutic response generation
#     - Communicate installation and activation readiness
#
# Does Not:
#     - Classify therapeutic response
#     - Infer intervention timing
#     - Manufacture pre/post evidence
#     - Attribute change to a specific intervention
#     - Recommend therapy
#     - Claim causation
#
# Core Doctrine:
#     Therapeutic influence is observed, not assumed.
#============================================================


#============================================================
# Chapter 67.16 — TTI Activation Readiness Gate
#============================================================

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


_TTI_REQUIRED_CONTRACT_COLUMNS = {
    "patient_id",
    "encounter_id",
    "tti_intervention_family_evaluated",
    "tti_intervention_episode_id",
    "tti_intervention_family",
    "tti_intervention_exposure_flag",
    "tti_intervention_exposure_state",
    "tti_intervention_source_grain_state",
    "tti_intervention_within_encounter_state",
    "tti_intervention_anchor_consistency_state",
    "tti_intervention_timing_state",
    "tti_intervention_episode_contract_state",
}


_TTI_ACCEPTED_SOURCE_CONSISTENCY_STATES = {
    "DUPLICATE_SOURCE_EXPOSURE_MATCHED",
    (
        "DUPLICATE_SOURCE_EXPOSURE_AND_"
        "ROW_COUNTS_MATCHED"
    ),
}


_TTI_EXTERNAL_REQUIREMENTS = {
    "intervention_timing_ready": False,
    "pre_post_evidence_ready": False,
    "competing_intervention_handling_ready": False,
}


def _build_tti_readiness_row(
    requirement: str,
    category: str,
    passed: bool,
    observation: str,
) -> dict[str, object]:
    """
    Build one activation-readiness requirement row.
    """

    return {
        "requirement": requirement,
        "category": category,
        "status": (
            "PASS"
            if passed
            else "DEFERRED"
        ),
        "requirement_satisfied": bool(passed),
        "required_for_response_engine": True,
        "observation": observation,
    }


def build_tti_activation_readiness_gate(
    contract_df: pd.DataFrame,
    external_requirements: Mapping[str, bool] | None = None,
) -> pd.DataFrame:
    """
    Evaluate whether TTI may activate therapeutic response
    interpretation.

    Structural requirements are evaluated directly from the
    intervention episode contract.

    Evidence capabilities not yet implemented are supplied through
    explicit external readiness declarations. They default to False
    so that TTI cannot activate through assumption.

    This function does not classify therapeutic response.
    """

    if not isinstance(contract_df, pd.DataFrame):
        raise TypeError(
            "contract_df must be a pandas DataFrame."
        )

    missing_columns = sorted(
        _TTI_REQUIRED_CONTRACT_COLUMNS
        - set(contract_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "TTI activation readiness cannot be evaluated. "
            f"Missing contract columns: {missing_columns}"
        )

    resolved_external_requirements = (
        _TTI_EXTERNAL_REQUIREMENTS.copy()
    )

    if external_requirements is not None:
        unknown_requirements = (
            set(external_requirements)
            - set(resolved_external_requirements)
        )

        if unknown_requirements:
            raise KeyError(
                "Unknown external TTI readiness requirements: "
                f"{sorted(unknown_requirements)}"
            )

        resolved_external_requirements.update(
            {
                key: bool(value)
                for key, value
                in external_requirements.items()
            }
        )

    exposure_present = (
        contract_df[
            "tti_intervention_exposure_flag"
        ]
        .astype("boolean")
        .eq(True)
    )

    missing_keys = (
        contract_df[
            [
                "patient_id",
                "encounter_id",
            ]
        ]
        .isna()
        .any(axis=1)
    )

    duplicate_keys = (
        contract_df
        .duplicated(
            subset=[
                "patient_id",
                "encounter_id",
            ]
        )
    )

    patient_encounter_alignment_ready = (
        not missing_keys.any()
        and not duplicate_keys.any()
    )

    canonical_anchor_ready = (
        exposure_present.any()
        and
        contract_df.loc[
            exposure_present,
            "tti_intervention_episode_id",
        ]
        .notna()
        .all()
        and
        contract_df.loc[
            exposure_present,
            "tti_intervention_family",
        ]
        .eq("HEMODYNAMIC_SUPPORT")
        .all()
    )

    duplicate_source_handling_ready = (
        contract_df[
            "tti_intervention_anchor_consistency_state"
        ]
        .isin(
            _TTI_ACCEPTED_SOURCE_CONSISTENCY_STATES
        )
        .all()
    )

    multirow_aggregation_ready = (
        not duplicate_keys.any()
        and
        contract_df[
            "tti_intervention_source_grain_state"
        ]
        .isin({
            "SINGLE_ROW_ENCOUNTER",
            "MULTIROW_ENCOUNTER",
        })
        .all()
        and
        contract_df[
            "tti_intervention_within_encounter_state"
        ]
        .notna()
        .all()
    )

    family_specificity_ready = (
        contract_df[
            "tti_intervention_family_evaluated"
        ]
        .eq("HEMODYNAMIC_SUPPORT")
        .all()
        and
        contract_df[
            "tti_intervention_exposure_state"
        ]
        .astype("string")
        .str.startswith(
            "HEMODYNAMIC_SUPPORT_",
            na=False,
        )
        .all()
    )

    timing_ready = resolved_external_requirements[
        "intervention_timing_ready"
    ]

    pre_post_ready = resolved_external_requirements[
        "pre_post_evidence_ready"
    ]

    competing_intervention_ready = (
        resolved_external_requirements[
            "competing_intervention_handling_ready"
        ]
    )

    rows = [
        _build_tti_readiness_row(
            requirement="PATIENT_ENCOUNTER_ALIGNMENT",
            category="STRUCTURAL_FOUNDATION",
            passed=patient_encounter_alignment_ready,
            observation=(
                "Patient and encounter identifiers are complete "
                "and unique at contract grain."
                if patient_encounter_alignment_ready
                else
                "Missing or duplicated patient-encounter keys "
                "remain."
            ),
        ),
        _build_tti_readiness_row(
            requirement="CANONICAL_INTERVENTION_EXPOSURE",
            category="INTERVENTION_CONTRACT",
            passed=canonical_anchor_ready,
            observation=(
                "Canonical hemodynamic-support exposure episodes "
                "are established."
                if canonical_anchor_ready
                else
                "Canonical exposure episodes are incomplete."
            ),
        ),
        _build_tti_readiness_row(
            requirement="DUPLICATE_SOURCE_HANDLING",
            category="EVIDENCE_INTEGRITY",
            passed=duplicate_source_handling_ready,
            observation=(
                "Duplicated intervention and operations evidence "
                "is reconciled without double-counting."
                if duplicate_source_handling_ready
                else
                "Source-level exposure inconsistencies remain."
            ),
        ),
        _build_tti_readiness_row(
            requirement="MULTIROW_ENCOUNTER_AGGREGATION",
            category="SOURCE_GRAIN",
            passed=multirow_aggregation_ready,
            observation=(
                "Single-row and multirow encounter evidence is "
                "preserved at one-row-per-encounter contract grain."
                if multirow_aggregation_ready
                else
                "Encounter-level aggregation remains incomplete."
            ),
        ),
        _build_tti_readiness_row(
            requirement="INTERVENTION_FAMILY_SPECIFICITY",
            category="SEMANTIC_INTEGRITY",
            passed=family_specificity_ready,
            observation=(
                "Exposure and absence states remain specific to "
                "hemodynamic support."
                if family_specificity_ready
                else
                "Intervention states overgeneralize beyond the "
                "evaluated family."
            ),
        ),
        _build_tti_readiness_row(
            requirement="VERIFIED_INTERVENTION_TIMING",
            category="TEMPORAL_EVIDENCE",
            passed=timing_ready,
            observation=(
                "Verified intervention timing is available."
                if timing_ready
                else
                "Intervention timing remains unavailable; an "
                "honest before-and-after boundary cannot yet be "
                "established."
            ),
        ),
        _build_tti_readiness_row(
            requirement="VERIFIED_PRE_POST_EVIDENCE",
            category="TRAJECTORY_EVIDENCE",
            passed=pre_post_ready,
            observation=(
                "Leakage-safe pre-intervention and "
                "post-intervention evidence has been verified."
                if pre_post_ready
                else
                "Pre-intervention and post-intervention evidence "
                "pairs have not yet been verified."
            ),
        ),
        _build_tti_readiness_row(
            requirement="COMPETING_INTERVENTION_HANDLING",
            category="ATTRIBUTION_SAFETY",
            passed=competing_intervention_ready,
            observation=(
                "Concurrent and competing interventions can be "
                "identified and preserved during interpretation."
                if competing_intervention_ready
                else
                "Concurrent-intervention attribution handling "
                "has not yet been established."
            ),
        ),
    ]

    return pd.DataFrame(rows)


def build_tti_activation_readiness_summary(
    readiness_gate_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize the TTI activation-readiness gate.

    The response engine is authorized only when every required
    readiness condition passes.
    """

    required_columns = {
        "requirement",
        "status",
        "requirement_satisfied",
        "required_for_response_engine",
    }

    missing_columns = sorted(
        required_columns
        - set(readiness_gate_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "Readiness gate summary cannot be built. "
            f"Missing columns: {missing_columns}"
        )

    required_rows = readiness_gate_df.loc[
        readiness_gate_df[
            "required_for_response_engine"
        ].eq(True)
    ].copy()

    passed_count = int(
        required_rows[
            "requirement_satisfied"
        ].sum()
    )

    total_count = len(required_rows)
    deferred_count = total_count - passed_count

    deferred_requirements = (
        required_rows.loc[
            ~required_rows[
                "requirement_satisfied"
            ],
            "requirement",
        ]
        .astype(str)
        .tolist()
    )

    response_engine_authorized = (
        deferred_count == 0
    )

    if response_engine_authorized:
        activation_state = (
            "TTI_RESPONSE_ENGINE_READY"
        )
        installation_state = (
            "TTI_FOUNDATION_AND_RESPONSE_ENGINE_READY"
        )
    else:
        activation_state = (
            "TTI_RESPONSE_ENGINE_DEFERRED_"
            "PREREQUISITES_INCOMPLETE"
        )
        installation_state = (
            "TTI_FOUNDATION_INSTALLED_"
            "RESPONSE_ENGINE_SAFELY_INACTIVE"
        )

    return pd.DataFrame([
        {
            "installation_state":
                installation_state,
            "activation_state":
                activation_state,
            "response_engine_authorized":
                response_engine_authorized,
            "passed_requirement_count":
                passed_count,
            "deferred_requirement_count":
                deferred_count,
            "total_requirement_count":
                total_count,
            "deferred_requirements":
                " | ".join(
                    deferred_requirements
                ),
        }
    ])


#============================================================
# Chapter 67.17 — TTI Future Activation Contract
#============================================================


import pandas as pd


_TTI_FUTURE_ACTIVATION_CONTRACT = (
    {
        "activation_component":
            "INTERVENTION_EPISODE_CONTRACT",

        "category":
            "INSTALLED_FOUNDATION",

        "current_state":
            "INSTALLED",

        "required_artifact":
            (
                "Encounter-level, family-specific intervention "
                "exposure contract."
            ),

        "activation_blocking":
            False,

        "future_installation_action":
            (
                "Reuse the existing hemodynamic-support episode "
                "contract as the canonical intervention input."
            ),
    },
    {
        "activation_component":
            "VERIFIED_INTERVENTION_TIMING",

        "category":
            "TEMPORAL_EVIDENCE",

        "current_state":
            "DEFERRED",

        "required_artifact":
            (
                "Verified intervention start time, event order, "
                "or equivalent temporal anchor."
            ),

        "activation_blocking":
            True,

        "future_installation_action":
            (
                "Attach verified intervention timing to each "
                "eligible intervention episode."
            ),
    },
    {
        "activation_component":
            "PRE_INTERVENTION_EVIDENCE_WINDOW",

        "category":
            "TRAJECTORY_EVIDENCE",

        "current_state":
            "DEFERRED",

        "required_artifact":
            (
                "Leakage-safe physiology and trajectory evidence "
                "observed before intervention."
            ),

        "activation_blocking":
            True,

        "future_installation_action":
            (
                "Construct a verified pre-intervention evidence "
                "window relative to the intervention anchor."
            ),
    },
    {
        "activation_component":
            "POST_INTERVENTION_EVIDENCE_WINDOW",

        "category":
            "TRAJECTORY_EVIDENCE",

        "current_state":
            "DEFERRED",

        "required_artifact":
            (
                "Leakage-safe physiology and trajectory evidence "
                "observed after intervention."
            ),

        "activation_blocking":
            True,

        "future_installation_action":
            (
                "Construct a verified post-intervention evidence "
                "window relative to the intervention anchor."
            ),
    },
    {
        "activation_component":
            "COMPETING_INTERVENTION_CONTEXT",

        "category":
            "ATTRIBUTION_SAFETY",

        "current_state":
            "DEFERRED",

        "required_artifact":
            (
                "Identification of concurrent, overlapping, or "
                "competing therapeutic interventions."
            ),

        "activation_blocking":
            True,

        "future_installation_action":
            (
                "Preserve simultaneous intervention activity "
                "before therapeutic influence is interpreted."
            ),
    },
    {
        "activation_component":
            "LEAKAGE_SAFE_TRAJECTORY_COMPARISON",

        "category":
            "THERAPEUTIC_COMPARISON",

        "current_state":
            "NOT_INSTALLED",

        "required_artifact":
            (
                "Validated comparison of pre-intervention and "
                "post-intervention trajectory evidence."
            ),

        "activation_blocking":
            True,

        "future_installation_action":
            (
                "Compare observed trajectory modification without "
                "using fields that already encode therapeutic effect."
            ),
    },
    {
        "activation_component":
            "THERAPEUTIC_RESPONSE_STATE_ENGINE",

        "category":
            "RESPONSE_INTERPRETATION",

        "current_state":
            "SAFELY_INACTIVE",

        "required_artifact":
            (
                "Evidence-supported therapeutic response "
                "classification logic."
            ),

        "activation_blocking":
            True,

        "future_installation_action":
            (
                "Activate response-state classification only after "
                "all upstream evidence requirements pass."
            ),
    },
    {
        "activation_component":
            "THERAPEUTIC_RESPONSE_CONFIDENCE",

        "category":
            "CONFIDENCE_INTEGRATION",

        "current_state":
            "NOT_INSTALLED",

        "required_artifact":
            (
                "Accumulated confidence based on evidence agreement, "
                "completeness, stability, and contradiction."
            ),

        "activation_blocking":
            True,

        "future_installation_action":
            (
                "Integrate the Confidence Engine after therapeutic "
                "response evidence becomes valid."
            ),
    },
    {
        "activation_component":
            "ACTIVATION_READINESS_GATE",

        "category":
            "GOVERNANCE",

        "current_state":
            "INSTALLED",

        "required_artifact":
            (
                "Explicit authorization gate preventing premature "
                "therapeutic response interpretation."
            ),

        "activation_blocking":
            False,

        "future_installation_action":
            (
                "Reuse the existing readiness gate to authorize "
                "future response-engine activation."
            ),
    },
)


def build_tti_future_activation_contract() -> pd.DataFrame:
    """
    Build the TTI future activation and installation contract.

    The contract records:
    - what has already been installed
    - what remains deferred
    - which components prevent activation
    - what artifacts must be supplied later
    - how future installation should proceed

    This function does not classify therapeutic response and does
    not modify the current TTI activation state.
    """

    contract = pd.DataFrame(
        _TTI_FUTURE_ACTIVATION_CONTRACT
    )

    required_columns = {
        "activation_component",
        "category",
        "current_state",
        "required_artifact",
        "activation_blocking",
        "future_installation_action",
    }

    missing_columns = sorted(
        required_columns
        - set(contract.columns)
    )

    if missing_columns:
        raise KeyError(
            "TTI future activation contract is incomplete. "
            f"Missing columns: {missing_columns}"
        )

    if contract[
        "activation_component"
    ].duplicated().any():
        duplicates = (
            contract.loc[
                contract[
                    "activation_component"
                ].duplicated(keep=False),
                "activation_component",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "TTI future activation contract contains "
            f"duplicated components: {duplicates}"
        )

    return contract.reset_index(drop=True)