# ============================================================
# BIRE OS — Therapeutic Trajectory Intelligence (TTI)
# File: src/bire/tti/scoring.py
# Chapter: 67
#
# Created: 2026-07-15
# Updated: 2026-07-15
#
# CHANGED:
# - Initialized TTI scoring helper module
# - Added support for therapeutic influence summaries
# - Prepared backend space for therapeutic response evaluation
# - Separated TTI scoring from TTI feature creation
#
# Purpose:
# Backend helper for therapeutic influence summaries,
# intervention response distributions,
# therapeutic response confidence,
# and therapeutic influence investigation outputs.
# ============================================================
from __future__ import annotations

import re

from pandas.api.types import is_bool_dtype, is_numeric_dtype

import pandas as pd
from sympy import python



_TTI_CANDIDATE_TOKENS = {
    "INTERVENTION_IDENTITY": (
        "intervention",
        "treatment",
        "therapy",
        "therapeutic",
        "medication",
        "medicine",
        "drug",
        "procedure",
        "device",
        "oxygen",
        "fluid",
        "insulin",
        "antibiotic",
        "dialysis",
        "ventilation",
    ),
    "TIME_OR_ORDER": (
        "time",
        "timestamp",
        "datetime",
        "date",
        "charttime",
        "start",
        "stop",
        "admin",
        "ordered",
        "sequence",
        "time_step",
        "time_index",
    ),
    "RESPONSE_OR_EFFECT": (
        "response",
        "effect",
        "efficacy",
        "outcome",
        "improvement",
        "worsening",
        "deterioration",
        "change",
        "delta",
        "modification",
    ),
    "PRE_POST_CONTEXT": (
        "pre_",
        "post_",
        "before",
        "after",
        "baseline",
        "followup",
        "follow_up",
    ),
}


def _classify_tti_candidate_column(column_name: str) -> str:
    """
    Classify a column by its possible role in TTI implementation.

    Classification identifies candidates only.
    It does not claim that a column is a valid TTI input.
    """

    normalized = column_name.strip().lower()

    if normalized in {
        "patient_id",
        "encounter_id",
        "intervention_id",
    }:
        return "IDENTIFIER"

    matched_roles = [
        role
        for role, tokens in _TTI_CANDIDATE_TOKENS.items()
        if any(token in normalized for token in tokens)
    ]

    if not matched_roles:
        return ""

    return " | ".join(matched_roles)


def build_tti_input_inventory(
    trajectory_df: pd.DataFrame,
    intervention_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build an inventory of existing Playground columns that may support TTI.

    Reviews the trajectory-capable dataframe and intervention dataframe for:
    - patient / encounter identifiers
    - intervention identity
    - intervention timing or ordering
    - existing response/effect fields
    - before/after context fields

    This function does not score therapeutic response and does not infer
    causation. It establishes the real input contract required for the
    first TTI response implementation.
    """

    if not isinstance(trajectory_df, pd.DataFrame):
        raise TypeError("trajectory_df must be a pandas DataFrame.")

    if not isinstance(intervention_df, pd.DataFrame):
        raise TypeError("intervention_df must be a pandas DataFrame.")

    sources = {
        "TRAJECTORY_SOURCE": trajectory_df,
        "INTERVENTION_SOURCE": intervention_df,
    }

    rows: list[dict[str, object]] = []

    for source_name, source_df in sources.items():
        source_row_count = len(source_df)

        for column in source_df.columns:
            candidate_role = _classify_tti_candidate_column(column)

            if not candidate_role:
                continue

            series = source_df[column]
            non_null_count = int(series.notna().sum())

            if source_row_count:
                non_null_percent = round(
                    (non_null_count / source_row_count) * 100,
                    3,
                )
            else:
                non_null_percent = 0.0

            sample_values = (
                series.dropna()
                .drop_duplicates()
                .astype(str)
                .head(5)
                .tolist()
            )

            rows.append({
                "source": source_name,
                "candidate_role": candidate_role,
                "column": column,
                "dtype": str(series.dtype),
                "non_null_count": non_null_count,
                "non_null_percent": non_null_percent,
                "sample_values": " | ".join(sample_values),
            })

    inventory = pd.DataFrame(rows)

    if inventory.empty:
        return pd.DataFrame(columns=[
            "source",
            "candidate_role",
            "column",
            "dtype",
            "non_null_count",
            "non_null_percent",
            "sample_values",
        ])

    return (
        inventory
        .sort_values(
            by=[
                "candidate_role",
                "source",
                "non_null_count",
                "column",
            ],
            ascending=[
                True,
                True,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

#============================================================
# Chapter 67.12A — TTI Intervention Anchor Qualification
#============================================================

import re
import pandas as pd


INTERVENTION_KEYWORDS = (
    "intervention",
    "treatment",
    "therapy",
    "medication",
    "drug",
    "procedure",
    "oxygen",
    "fluid",
    "antibiotic",
    "insulin",
    "dialysis",
    "ventilation",
)

EXCLUDE_KEYWORDS = (
    "allergy",
    "conflict",
    "risk",
    "pressure",
    "burden",
    "forecast",
    "prediction",
    "score",
    "recommend",
    "candidate",
)

DIAGNOSTIC_KEYWORDS = (
    "lab",
    "imaging",
    "test",
    "diagnosis",
)

ACTION_KEYWORDS = (
    "given",
    "admin",
    "received",
    "started",
    "performed",
    "active",
    "used",
)


def _tokenize(col):
    col = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(col))
    return set(re.findall(r"[a-z0-9]+", col.lower()))


def classify_intervention_column(column):

    tokens = _tokenize(column)

    if any(k in column for k in EXCLUDE_KEYWORDS):
        return "EXCLUDE"

    if any(k in column for k in DIAGNOSTIC_KEYWORDS):
        return "DIAGNOSTIC"

    if any(k in column for k in INTERVENTION_KEYWORDS):

        if any(k in column for k in ACTION_KEYWORDS):
            return "INTERVENTION_EXPOSURE"

        return "INTERVENTION_CONTEXT"

    return None


def build_tti_intervention_candidates(intervention_df):

    rows = []

    for col in intervention_df.columns:

        role = classify_intervention_column(col)

        if role is None:
            continue

        series = intervention_df[col]

        rows.append({
            "column": col,
            "classification": role,
            "dtype": str(series.dtype),
            "non_null_percent":
                round(series.notna().mean() * 100, 3),
            "sample_values":
                " | ".join(series.dropna().astype(str).unique()[:5])
        })

    return pd.DataFrame(rows)

#============================================================
# Chapter 67.13 — TTI Intervention Anchor Qualification
#============================================================



# A column must contain at least one of these terms before it
# can enter the TTI intervention candidate inventory.
_TTI_THERAPEUTIC_TOKENS = {
    "intervention",
    "treatment",
    "therapy",
    "therapeutic",
    "medication",
    "medicine",
    "drug",
    "antibiotic",
    "insulin",
    "oxygen",
    "fluid",
    "dialysis",
    "ventilation",
    "ventilator",
    "vasopressor",
    "pressor",
    "transfusion",
    "procedure",
    "surgery",
    "operation",
    "intubation",
    "extubation",
    "infusion",
    "bolus",
}


# Terms suggesting that an intervention actually occurred.
_TTI_EXPOSURE_TOKENS = {
    "admin",
    "administered",
    "administration",
    "given",
    "received",
    "initiated",
    "started",
    "performed",
    "applied",
    "delivered",
    "active",
    "used",
    "completed",
    "infused",
    "placed",
    "occurred",
    "present",
}


# Terms that may establish intervention timing or ordering.
_TTI_TIMING_TOKENS = {
    "time",
    "timestamp",
    "datetime",
    "date",
    "start",
    "started",
    "stop",
    "stopped",
    "duration",
    "minutes",
    "hours",
    "sequence",
    "index",
}


# An order is not proof that an intervention occurred.
_TTI_ORDER_TOKENS = {
    "order",
    "ordered",
    "planned",
    "requested",
    "recommended",
    "scheduled",
}


# Terms that may identify the intervention itself.
_TTI_IDENTITY_TOKENS = {
    "id",
    "name",
    "type",
    "class",
    "category",
    "code",
    "agent",
    "route",
    "dose",
    "dosage",
}


# Intervention-related columns containing these terms are
# contextual or derived signals, not valid intervention anchors.
_TTI_EXCLUDE_TOKENS = {
    "allergy",
    "conflict",
    "risk",
    "pressure",
    "burden",
    "forecast",
    "prediction",
    "score",
    "candidate",
    "uncertainty",
    "visibility",
    "distortion",
    "confidence",
    "attention",
    "meta",
    "contraindication",
    "eligibility",
    "appropriateness",
    "suppression",
    "consequence",
    "warning",
}


def _tokenize_tti_column(column_name: str) -> set[str]:
    """
    Convert snake_case, camelCase, or mixed column names into tokens.
    """

    normalized = re.sub(
        r"(?<=[a-z0-9])(?=[A-Z])",
        "_",
        str(column_name),
    ).lower()

    return set(re.findall(r"[a-z0-9]+", normalized))


def classify_tti_intervention_column(
    column_name: str,
) -> dict[str, object] | None:
    """
    Classify an intervention-related column.

    A column is ignored unless it first demonstrates therapeutic
    relevance. This prevents generic risk, score, pressure, and
    timing columns from entering the intervention inventory.
    """

    tokens = _tokenize_tti_column(column_name)

    # Critical gate: ignore columns that have no therapeutic meaning.
    if not tokens.intersection(_TTI_THERAPEUTIC_TOKENS):
        return None

    is_excluded = bool(tokens.intersection(_TTI_EXCLUDE_TOKENS))
    is_exposure = bool(tokens.intersection(_TTI_EXPOSURE_TOKENS))
    is_timing = bool(tokens.intersection(_TTI_TIMING_TOKENS))
    is_order_only = bool(tokens.intersection(_TTI_ORDER_TOKENS))
    is_identity = bool(tokens.intersection(_TTI_IDENTITY_TOKENS))

    if is_excluded:
        classification = "EXCLUDE"
        anchor_eligible = False

    elif is_exposure and is_timing:
        classification = "INTERVENTION_EXPOSURE_AND_TIMING"
        anchor_eligible = True

    elif is_exposure:
        classification = "INTERVENTION_EXPOSURE"
        anchor_eligible = True

    elif is_timing:
        classification = "INTERVENTION_TIMING"
        anchor_eligible = True

    elif is_order_only:
        classification = "INTERVENTION_ORDER_ONLY"
        anchor_eligible = False

    elif is_identity:
        classification = "INTERVENTION_IDENTITY"
        anchor_eligible = True

    else:
        classification = "INTERVENTION_CONTEXT"
        anchor_eligible = False

    return {
        "classification": classification,
        "anchor_eligible": anchor_eligible,
        "exposure_candidate": is_exposure and not is_excluded,
        "timing_candidate": is_timing and not is_excluded,
        "identity_candidate": is_identity and not is_excluded,
    }


def _get_tti_active_count(series: pd.Series) -> int | None:
    """
    Return an active/positive count for boolean or binary columns.
    """

    non_null = series.dropna()

    if non_null.empty:
        return 0

    if is_bool_dtype(series):
        return int(non_null.astype(bool).sum())

    if is_numeric_dtype(series):
        unique_values = set(non_null.unique().tolist())

        if unique_values.issubset({0, 1, 0.0, 1.0}):
            return int((non_null == 1).sum())

    return None


def build_tti_intervention_candidates(
    intervention_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify real intervention-anchor candidates in existing
    Playground intervention data.

    This function does not infer therapeutic response and does not
    claim causation. It separates potential intervention identity,
    exposure, timing, order-only, contextual, and excluded columns.
    """

    if not isinstance(intervention_df, pd.DataFrame):
        raise TypeError(
            "intervention_df must be a pandas DataFrame."
        )

    rows: list[dict[str, object]] = []

    for column in intervention_df.columns:
        classification = classify_tti_intervention_column(column)

        if classification is None:
            continue

        series = intervention_df[column]
        non_null_count = int(series.notna().sum())

        rows.append({
            "column": column,
            **classification,
            "dtype": str(series.dtype),
            "non_null_count": non_null_count,
            "non_null_percent": round(
                series.notna().mean() * 100,
                3,
            ),
            "unique_count": int(
                series.nunique(dropna=True)
            ),
            "active_count": _get_tti_active_count(series),
            "sample_values": " | ".join(
                series
                .dropna()
                .drop_duplicates()
                .astype(str)
                .head(5)
                .tolist()
            ),
        })

    columns = [
        "column",
        "classification",
        "anchor_eligible",
        "exposure_candidate",
        "timing_candidate",
        "identity_candidate",
        "dtype",
        "non_null_count",
        "non_null_percent",
        "unique_count",
        "active_count",
        "sample_values",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    result = pd.DataFrame(rows)

    classification_order = {
        "INTERVENTION_EXPOSURE_AND_TIMING": 0,
        "INTERVENTION_EXPOSURE": 1,
        "INTERVENTION_TIMING": 2,
        "INTERVENTION_IDENTITY": 3,
        "INTERVENTION_ORDER_ONLY": 4,
        "INTERVENTION_CONTEXT": 5,
        "EXCLUDE": 6,
    }

    result["_classification_order"] = (
        result["classification"]
        .map(classification_order)
        .fillna(99)
    )

    return (
        result
        .sort_values(
            by=[
                "_classification_order",
                "anchor_eligible",
                "non_null_count",
                "column",
            ],
            ascending=[
                True,
                False,
                False,
                True,
            ],
        )
        .drop(columns="_classification_order")
        .reset_index(drop=True)
    )


## Backend — `src/bire/tti/scoring.py`


#============================================================
# Chapter 67.14 — TTI Intervention Episode Source Discovery
#============================================================



from collections.abc import Mapping
import re

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


# ------------------------------------------------------------
# Therapeutic families currently represented in the Playground.
# These tokens identify possible intervention-bearing columns.
# They do not establish that exposure occurred.
# ------------------------------------------------------------

_TTI_INTERVENTION_FAMILIES = {
    "RESPIRATORY_SUPPORT": {
        "oxygen",
        "respiratory",
        "ventilation",
        "ventilator",
        "intubation",
        "extubation",
        "bronchodilator",
    },
    "HEMODYNAMIC_SUPPORT": {
        "fluid",
        "fluids",
        "vasopressor",
        "pressor",
        "transfusion",
    },
    "GLYCEMIC_THERAPY": {
        "insulin",
        "dextrose",
        "glucose",
    },
    "ANTI_INFECTIVE_THERAPY": {
        "antibiotic",
        "antibiotics",
        "antimicrobial",
    },
    "SEDATION_ANALGESIA": {
        "opioid",
        "sedation",
        "sedative",
        "analgesic",
    },
    "CARDIAC_THERAPY": {
        "antiarrhythmic",
        "anticoagulant",
    },
    "DIURETIC_THERAPY": {
        "diuretic",
        "diuresis",
    },
    "STEROID_THERAPY": {
        "steroid",
        "steroids",
        "corticosteroid",
    },
    "RENAL_REPLACEMENT": {
        "dialysis",
    },
}


_TTI_GENERIC_INTERVENTION_TOKENS = {
    "intervention",
    "treatment",
    "therapy",
    "therapeutic",
    "medication",
    "medicine",
    "drug",
    "procedure",
    "surgery",
    "operation",
}


# Terms indicating that therapeutic exposure may have occurred.
_TTI_EXPOSURE_TOKENS = {
    "admin",
    "administered",
    "administration",
    "given",
    "received",
    "initiated",
    "started",
    "performed",
    "applied",
    "delivered",
    "active",
    "used",
    "completed",
    "infused",
    "placed",
    "support",
}


# Terms capable of establishing temporal alignment.
_TTI_TIMING_TOKENS = {
    "time",
    "timestamp",
    "datetime",
    "date",
    "start",
    "started",
    "stop",
    "stopped",
    "duration",
    "minute",
    "minutes",
    "hour",
    "hours",
    "sequence",
    "index",
}


# Intervention identity metadata.
_TTI_IDENTITY_TOKENS = {
    "id",
    "name",
    "type",
    "class",
    "category",
    "code",
    "agent",
}


# Dose or intervention intensity information.
_TTI_DOSE_TOKENS = {
    "dose",
    "dosage",
    "amount",
    "rate",
    "route",
    "level",
    "intensity",
}


# An order or plan does not prove exposure.
_TTI_ORDER_TOKENS = {
    "order",
    "ordered",
    "planned",
    "requested",
    "recommended",
    "scheduled",
}


# These fields describe a response or apparent effect.
# They must not become intervention anchors or independent
# evidence for the response state they already describe.
_TTI_DERIVED_EFFECT_TOKENS = {
    "effect",
    "response",
    "improved",
    "improvement",
    "worsened",
    "worsening",
    "modified",
    "modification",
    "change",
    "delta",
    "outcome",
    "after",
    "post",
    "recovery",
    "deterioration",
}


# These describe therapeutic context rather than exposure.
_TTI_NON_ANCHOR_CONTEXT_TOKENS = {
    "allergy",
    "conflict",
    "risk",
    "pressure",
    "burden",
    "stress",
    "mismatch",
    "uncertainty",
    "visibility",
    "distortion",
    "confidence",
    "masking",
    "dependency",
    "attention",
    "meta",
    "contraindication",
    "eligibility",
    "appropriateness",
    "warning",
    "notes",
    "quality",
    "complexity",
    "suppression",
    "consequence",
    "failure",
    "loss",
    "charting",
    "lag",
    "diagnosis",
    "induced",
    "silent",
    "strain",
    "gap",
    "reassessment",
    "reliability",

}

_TTI_EXPLICIT_THERAPY_AGENT_TOKENS = {
    "antibiotic",
    "antibiotics",
    "antimicrobial",
    "insulin",
    "dextrose",
    "vasopressor",
    "pressor",
    "bronchodilator",
    "opioid",
    "sedative",
    "sedation",
    "antiarrhythmic",
    "anticoagulant",
    "diuretic",
    "steroid",
    "steroids",
    "corticosteroid",
    "dialysis",
    "transfusion",
    "ventilation",
    "ventilator",
    "intubation",
    "extubation",
}

# These may describe interruption or withholding but do not
# independently establish the initial intervention exposure.
_TTI_STATUS_CONTEXT_TOKENS = {
    "interrupted",
    "held",
    "withheld",
    "refused",
    "missed",
    "discontinued",
    "cancelled",
    "canceled",
}


def _tokenize_tti_column(column_name: str) -> set[str]:
    """
    Convert snake_case, camelCase, and mixed names into tokens.
    """

    normalized = re.sub(
        r"(?<=[a-z0-9])(?=[A-Z])",
        "_",
        str(column_name),
    ).lower()

    return set(re.findall(r"[a-z0-9]+", normalized))


def _match_tti_intervention_families(
    tokens: set[str],
) -> list[str]:
    """
    Identify therapeutic families represented by column tokens.
    """

    return [
        family
        for family, family_tokens
        in _TTI_INTERVENTION_FAMILIES.items()
        if tokens.intersection(family_tokens)
    ]


def _is_binary_series(series: pd.Series) -> bool:
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


def _get_tti_active_count(
    series: pd.Series,
) -> int | None:
    """
    Return the active count for boolean or binary fields.
    """

    if not _is_binary_series(series):
        return None

    non_null = series.dropna()

    if is_bool_dtype(series):
        return int(non_null.astype(bool).sum())

    numeric = pd.to_numeric(
        non_null,
        errors="coerce",
    )

    return int((numeric == 1).sum())


def _classify_tti_episode_source_column(
    column_name: str,
    series: pd.Series,
) -> dict[str, object] | None:
    """
    Classify a possible intervention-event source column.

    This function identifies possible event components only.
    Every candidate still requires verification before use.
    """

    tokens = _tokenize_tti_column(column_name)

    intervention_families = (
        _match_tti_intervention_families(tokens)
    )

    has_generic_intervention_language = bool(
        tokens.intersection(
            _TTI_GENERIC_INTERVENTION_TOKENS
        )
    )

    has_explicit_intervention_namespace = bool(
    tokens.intersection({
        "intervention",
        "treatment",
        "therapy",
        "therapeutic",
        })

    )

    has_explicit_therapy_agent = bool(
        tokens.intersection(
            _TTI_EXPLICIT_THERAPY_AGENT_TOKENS
        )
    )

    if (
        not intervention_families
        and not has_generic_intervention_language
    ):
        return None

    has_derived_effect_language = bool(
        tokens.intersection(
            _TTI_DERIVED_EFFECT_TOKENS
        )
    )

    has_non_anchor_context = bool(
        tokens.intersection(
            _TTI_NON_ANCHOR_CONTEXT_TOKENS
        )
    )

    has_status_context = bool(
        tokens.intersection(
            _TTI_STATUS_CONTEXT_TOKENS
        )
    )

    has_exposure_language = bool(
        tokens.intersection(
            _TTI_EXPOSURE_TOKENS
        )
    )

    has_timing_language = bool(
        tokens.intersection(
            _TTI_TIMING_TOKENS
        )
    )

    has_identity_language = bool(
        tokens.intersection(
            _TTI_IDENTITY_TOKENS
        )
    )

    has_dose_language = bool(
        tokens.intersection(
            _TTI_DOSE_TOKENS
        )
    )

    has_order_language = bool(
        tokens.intersection(
            _TTI_ORDER_TOKENS
        )
    )

    if has_derived_effect_language:
        candidate_role = "DERIVED_EFFECT_CONTEXT"
        anchor_component_eligible = False
        evidence_leakage_risk = True

    elif has_non_anchor_context:
        candidate_role = "NON_ANCHOR_CONTEXT"
        anchor_component_eligible = False
        evidence_leakage_risk = False

    elif has_status_context:
        candidate_role = "THERAPY_STATUS_CONTEXT"
        anchor_component_eligible = False
        evidence_leakage_risk = False

    elif has_exposure_language and has_timing_language:
        candidate_role = "POSSIBLE_EXPOSURE_AND_TIMING"
        anchor_component_eligible = True
        evidence_leakage_risk = False

    elif has_timing_language:
        candidate_role = "POSSIBLE_TIMING"
        anchor_component_eligible = True
        evidence_leakage_risk = False

    elif has_order_language:
        candidate_role = "ORDER_ONLY"
        anchor_component_eligible = False
        evidence_leakage_risk = False

    elif has_identity_language:
        candidate_role = "POSSIBLE_IDENTITY"
        anchor_component_eligible = True
        evidence_leakage_risk = False

    elif has_dose_language:
        candidate_role = "POSSIBLE_DOSE_OR_INTENSITY"
        anchor_component_eligible = True
        evidence_leakage_risk = False

    elif has_exposure_language:
        candidate_role = "POSSIBLE_EXPOSURE"
        anchor_component_eligible = True
        evidence_leakage_risk = False

    elif (
        _is_binary_series(series)
        and (
            has_explicit_intervention_namespace
            or has_explicit_therapy_agent
        )
    ):
        candidate_role = "POSSIBLE_EXPOSURE_FLAG"
        anchor_component_eligible = True
        evidence_leakage_risk = False

    else:
        candidate_role = "REVIEW_CONTEXT"
        anchor_component_eligible = False
        evidence_leakage_risk = False

    return {
        "intervention_families": (
            " | ".join(intervention_families)
            if intervention_families
            else "GENERIC_INTERVENTION"
        ),
        "candidate_role": candidate_role,
        "anchor_component_eligible":
            anchor_component_eligible,
        "evidence_leakage_risk":
            evidence_leakage_risk,
        "manual_verification_required": True,
    }


def build_tti_intervention_episode_source_discovery(
    source_frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Search available BIRE OS sources for intervention identity,
    exposure, timing, dose, and related contextual fields.

    No candidate is automatically accepted as a verified TTI
    intervention anchor.
    """

    if not isinstance(source_frames, Mapping):
        raise TypeError(
            "source_frames must be a mapping of "
            "source names to pandas DataFrames."
        )

    rows: list[dict[str, object]] = []

    for source_name, source_df in source_frames.items():

        if not isinstance(source_df, pd.DataFrame):
            raise TypeError(
                f"{source_name} must be a pandas DataFrame."
            )

        for column in source_df.columns:

            series = source_df[column]

            classification = (
                _classify_tti_episode_source_column(
                    column_name=column,
                    series=series,
                )
            )

            if classification is None:
                continue

            rows.append({
                "source": source_name,
                "column": column,
                **classification,
                "dtype": str(series.dtype),
                "non_null_count":
                    int(series.notna().sum()),
                "non_null_percent":
                    round(series.notna().mean() * 100, 3),
                "unique_count":
                    int(series.nunique(dropna=True)),
                "active_count":
                    _get_tti_active_count(series),
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
        "intervention_families",
        "candidate_role",
        "anchor_component_eligible",
        "evidence_leakage_risk",
        "manual_verification_required",
        "dtype",
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

    candidate_order = {
        "POSSIBLE_EXPOSURE_AND_TIMING": 0,
        "POSSIBLE_EXPOSURE": 1,
        "POSSIBLE_EXPOSURE_FLAG": 2,
        "POSSIBLE_TIMING": 3,
        "POSSIBLE_IDENTITY": 4,
        "POSSIBLE_DOSE_OR_INTENSITY": 5,
        "ORDER_ONLY": 6,
        "THERAPY_STATUS_CONTEXT": 7,
        "REVIEW_CONTEXT": 8,
        "NON_ANCHOR_CONTEXT": 9,
        "DERIVED_EFFECT_CONTEXT": 10,
    }

    result["_candidate_order"] = (
        result["candidate_role"]
        .map(candidate_order)
        .fillna(99)
    )

    return (
        result
        .sort_values(
            by=[
                "_candidate_order",
                "anchor_component_eligible",
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
        .drop(columns="_candidate_order")
        .reset_index(drop=True)
    )