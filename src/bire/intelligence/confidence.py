from __future__ import annotations

import numpy as np
import pandas as pd


def add_horizon_agreement_features(
    df: pd.DataFrame,
    risk_cols: tuple[str, str, str] = ("risk_15", "risk_30", "risk_60"),
) -> pd.DataFrame:
    """
    Add multi-horizon agreement / disagreement features.

    These features measure whether the 15, 30, and 60 minute
    forecasts are aligned or conflicting.
    """

    out = df.copy()
    existing = [c for c in risk_cols if c in out.columns]

    if len(existing) < 2:
        out["horizon_spread"] = np.nan
        out["horizon_agreement_score"] = np.nan
        return out

    out["horizon_spread"] = out[existing].max(axis=1) - out[existing].min(axis=1)
    out["horizon_agreement_score"] = 1 - out["horizon_spread"].clip(0, 1)

    return out


def add_prediction_stability_features(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    risk_col: str = "risk_60",
    window: int = 6,
) -> pd.DataFrame:
    """
    Add prediction stability features over time.

    High rolling volatility means the risk estimate is unstable.
    """

    out = df.copy()

    if risk_col not in out.columns:
        out["prediction_volatility"] = np.nan
        out["prediction_stability_score"] = np.nan
        return out

    out = out.sort_values([patient_col, time_col]).copy()

    shifted_risk = out.groupby(patient_col)[risk_col].shift(1)

    out["prediction_volatility"] = (
        shifted_risk.groupby(out[patient_col])
        .rolling(window=window, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
    )

    out["prediction_stability_score"] = 1 - out["prediction_volatility"].fillna(0).clip(0, 1)

    return out


def add_evidence_quality_features(
    df: pd.DataFrame,
    signal_cols: tuple[str, ...] = (
        "heart_rate",
        "resp_rate",
        "spo2",
        "temperature",
        "sbp",
        "dbp",
    ),
) -> pd.DataFrame:
    """
    Add evidence completeness features based on available signal columns.
    """

    out = df.copy()
    existing = [c for c in signal_cols if c in out.columns]

    if not existing:
        out["evidence_completeness_score"] = np.nan
        out["missing_signal_count"] = np.nan
        return out

    out["missing_signal_count"] = out[existing].isna().sum(axis=1)
    out["evidence_completeness_score"] = 1 - (
        out["missing_signal_count"] / len(existing)
    )

    return out


def add_bire_confidence_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine engineered confidence features into a transparent
    BIRE confidence and uncertainty score.
    """

    out = df.copy()

    required_scores = [
        "horizon_agreement_score",
        "prediction_stability_score",
        "evidence_completeness_score",
    ]

    existing = [c for c in required_scores if c in out.columns]

    if not existing:
        out["confidence_score"] = np.nan
        out["uncertainty_score"] = np.nan
        out["confidence_band"] = "UNKNOWN"
        out["uncertainty_band"] = "UNKNOWN"
        return out

    out["confidence_score"] = out[existing].mean(axis=1).clip(0, 1)
    out["uncertainty_score"] = (1 - out["confidence_score"]).clip(0, 1)

    out["confidence_band"] = pd.cut(
        out["confidence_score"],
        bins=[-0.01, 0.80, 0.95, 1.01],
        labels=["LOW", "MODERATE", "HIGH"],
    ).astype(str)

    out["uncertainty_band"] = pd.cut(
        out["uncertainty_score"],
        bins=[-0.01, 0.80, 0.95, 1.01],
        labels=["LOW", "MODERATE", "HIGH"],
    ).astype(str)

    return out


def add_confidence_stress_factors(
    df: pd.DataFrame,
    horizon_spread_col: str = "horizon_spread",
    volatility_col: str = "prediction_volatility",
    acceleration_col: str = "risk_acceleration",
    evidence_col: str = "evidence_completeness_score",
) -> pd.DataFrame:
    """
    Add soft confidence stress factors.

    These factors reduce confidence when signals are unstable,
    conflicting, incomplete, or rapidly changing.

    They are NOT governance gates.
    """

    out = df.copy()

    out["horizon_stress"] = (
        out[horizon_spread_col].fillna(0).clip(0, 1)
        if horizon_spread_col in out.columns
        else 0
    )

    out["volatility_stress"] = (
        out[volatility_col].fillna(0).clip(0, 1)
        if volatility_col in out.columns
        else 0
    )

    out["acceleration_stress"] = (
        out[acceleration_col].abs().fillna(0).clip(0, 1)
        if acceleration_col in out.columns
        else 0
    )

    out["evidence_stress"] = (
        (1 - out[evidence_col].fillna(0)).clip(0, 1)
        if evidence_col in out.columns
        else 0
    )

    stress_cols = [
        "horizon_stress",
        "volatility_stress",
        "acceleration_stress",
        "evidence_stress",
    ]

    out["confidence_stress_score"] = out[stress_cols].mean(axis=1).clip(0, 1)

    out["adjusted_confidence_score"] = (
        out["confidence_score"] * (1 - out["confidence_stress_score"])
        if "confidence_score" in out.columns
        else np.nan
    ).clip(0, 1)

    out["adjusted_uncertainty_score"] = (
        1 - out["adjusted_confidence_score"]
    ).clip(0, 1)

    out["adjusted_confidence_band"] = pd.cut(
        out["adjusted_confidence_score"],
        bins=[-0.01, 0.80, 0.95, 1.01],
        labels=["LOW", "MODERATE", "HIGH"],
    ).astype(str)

    out["adjusted_uncertainty_band"] = pd.cut(
        out["adjusted_uncertainty_score"],
        bins=[-0.01, 0.80, 0.95, 1.01],
        labels=["LOW", "MODERATE", "HIGH"],
    ).astype(str)

    return out

# The function below applies weighted stress tuning to confidence scores,
# allowing for customizable emphasis on different stress factors.
# This keeps confidence separate from governance while adjusting trust interpretation.
def add_weighted_confidence_stress(
    df: pd.DataFrame,
    weights: dict | None = None,
    power: float = 1.0,
) -> pd.DataFrame:
    """
    Apply weighted stress tuning to confidence scores.

    This keeps confidence separate from governance.
    It adjusts trust interpretation without suppressing alerts.
    """

    out = df.copy()

    weights = weights or {
        "horizon_stress": 0.40,
        "volatility_stress": 0.25,
        "acceleration_stress": 0.25,
        "evidence_stress": 0.10,
    }

    for col in weights:
        if col not in out.columns:
            out[col] = 0.0

    total_weight = sum(weights.values())

    out["weighted_confidence_stress_score"] = sum(
        out[col].fillna(0).clip(0, 1) * weight
        for col, weight in weights.items()
    ) / total_weight

    if "confidence_score" in out.columns:
        out["weighted_adjusted_confidence_score"] = (
            out["confidence_score"]
            * ((1 - out["weighted_confidence_stress_score"]).clip(0, 1) ** power)
        ).clip(0, 1)
    else:
        out["weighted_adjusted_confidence_score"] = np.nan

    out["weighted_adjusted_uncertainty_score"] = (
        1 - out["weighted_adjusted_confidence_score"]
    ).clip(0, 1)

    out["weighted_confidence_band"] = pd.cut(
        out["weighted_adjusted_confidence_score"],
        bins=[-0.01, 0.80, 0.95, 1.01],
        labels=["LOW", "MODERATE", "HIGH"],
    ).astype(str)

    out["weighted_uncertainty_band"] = pd.cut(
        out["weighted_adjusted_uncertainty_score"],
        bins=[-0.01, 0.80, 0.95, 1.01],
        labels=["LOW", "MODERATE", "HIGH"],
    ).astype(str)

    return out
# The functions below add confidence context to escalation states,
#  which can be used for interpretation and monitoring.
def add_pre_escalation_confidence_context(
    df: pd.DataFrame,
    state_col: str = "bire_final_tier",
    confidence_band_col: str = "weighted_confidence_band",
) -> pd.DataFrame:
    """
    Add confidence context to pre-event escalation states.

    This does NOT change governance.
    It only describes how strongly BIRE backs the current escalation tier.
    """

    out = df.copy()

    pre_escalation_states = ["WATCH", "ESCALATE", "URGENT"]

    if state_col not in out.columns:
        out["pre_escalation_confidence_context"] = "NO_STATE_AVAILABLE"
        return out

    if confidence_band_col not in out.columns:
        out["pre_escalation_confidence_context"] = "NO_CONFIDENCE_AVAILABLE"
        return out

    state = out[state_col].astype(str).str.upper()
    conf = out[confidence_band_col].astype(str).str.upper()

    out["pre_escalation_confidence_context"] = "NOT_PRE_ESCALATION"

    mask = state.isin(pre_escalation_states)

    out.loc[mask, "pre_escalation_confidence_context"] = (
        conf[mask] + "_CONFIDENCE_" + state[mask]
    )

    return out

# This function can be extended in the future to add post-event confidence context as well,
# for example by looking at "CRITICAL", "POST_EVENT_ESCALATE", and
# "POST_EVENT_URGENT" states in a similar way.
def add_post_event_confidence_context(
    df: pd.DataFrame,
    state_col: str = "gss_v3_state",
    confidence_band_col: str = "weighted_confidence_band",
) -> pd.DataFrame:
    """
    Add confidence context to post-event escalation states.

    This remains observational only and does not modify governance.
    """

    out = df.copy()

    post_event_states = [
        "MONITOR",
        "RE-ESCALATE",
        "CRITICAL",
        "POST_EVENT_ESCALATE",
        "POST_EVENT_URGENT",
    ]

    if state_col not in out.columns:
        out["post_event_confidence_context"] = "NO_STATE_AVAILABLE"
        return out

    if confidence_band_col not in out.columns:
        out["post_event_confidence_context"] = "NO_CONFIDENCE_AVAILABLE"
        return out

    state = out[state_col].astype(str).str.upper()
    conf = out[confidence_band_col].astype(str).str.upper()

    out["post_event_confidence_context"] = "NOT_POST_EVENT"

    mask = state.isin(post_event_states)

    out.loc[mask, "post_event_confidence_context"] = (
        conf[mask] + "_CONFIDENCE_" + state[mask]
    )

    return out