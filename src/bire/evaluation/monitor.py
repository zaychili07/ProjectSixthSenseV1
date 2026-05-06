import numpy as np
import pandas as pd


def _resolve_tier_column(df):
    """
    Resolve which column represents the final BIRE tier/state.

    Preference order matters:
    - bire_final_tier is the final system decision
    - earlier episode tier columns are fallbacks
    """
    candidates = [
        "tier",
        "bire_final_tier",
        "episode_tier_acuity",
        "episode_tier_final",
        "episode_tier_gated",
        "episode_tier",
        "bire_state",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise KeyError(
        "No tier column found. Expected one of: "
        "tier, bire_final_tier, episode_tier_acuity, "
        "episode_tier_final, episode_tier_gated, episode_tier, bire_state"
    )


def compute_monitor_states(
    df,
    window=3,
    risk_col="pred_proba",
    patient_col="patient_id",
):
    """
    Compute post-event MONITOR trajectory states.

    Output states:
    - STABLE_MONITOR
    - DECLINING_MONITOR
    - VOLATILE_MONITOR

    Non-MONITOR rows receive None.
    """
    df = df.copy()

    required_cols = [patient_col, risk_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    tier_col = _resolve_tier_column(df)

    # Standardized tier column for downstream logic
    df["tier"] = df[tier_col]
    df["is_monitor"] = df["tier"] == "MONITOR"

    # --- Risk trend ---
    df["risk_delta"] = df.groupby(patient_col)[risk_col].diff()

    df["risk_trend"] = (
        df.groupby(patient_col)["risk_delta"]
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # --- Risk stability / volatility ---
    df["risk_std"] = (
        df.groupby(patient_col)[risk_col]
        .rolling(window=window, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
    ).fillna(0)

    # --- Physiological instability ---
    abnormal_flags = []

    if "heart_rate" in df.columns:
        abnormal_flags.append(df["heart_rate"] > 120)

    if "spo2" in df.columns:
        abnormal_flags.append(df["spo2"] < 92)

    if "resp_rate" in df.columns:
        abnormal_flags.append(df["resp_rate"] > 24)

    if "sbp" in df.columns:
        abnormal_flags.append(df["sbp"] < 100)

    if "temperature" in df.columns:
        abnormal_flags.append((df["temperature"] > 38.5) | (df["temperature"] < 36))

    if abnormal_flags:
        df["abnormal_count"] = np.sum(abnormal_flags, axis=0)
    elif "ibpip_n_abnormal_signals" in df.columns:
        df["abnormal_count"] = df["ibpip_n_abnormal_signals"]
    else:
        df["abnormal_count"] = 0

    # --- MONITOR state classification ---
    def classify(row):
        if not row["is_monitor"]:
            return None

        if row["risk_trend"] < -0.01 and row["abnormal_count"] <= 1:
            return "STABLE_MONITOR"

        if row["risk_trend"] > 0.01 and row["abnormal_count"] >= 2:
            return "DECLINING_MONITOR"

        return "VOLATILE_MONITOR"

    df["monitor_state"] = df.apply(classify, axis=1)

    return df


def summarize_monitor_states(df):
    """
    Summarize MONITOR trajectory behavior.
    """
    df = df.copy()

    tier_col = _resolve_tier_column(df)
    df["tier"] = df[tier_col]

    monitor_df = df[df["tier"] == "MONITOR"].copy()

    summary = {
        "total_monitor_rows": int(len(monitor_df)),
        "state_distribution": {},
        "avg_risk_in_monitor": None,
        "avg_abnormal_signals": None,
    }

    if monitor_df.empty:
        return summary

    if "monitor_state" in monitor_df.columns:
        summary["state_distribution"] = (
            monitor_df["monitor_state"]
            .value_counts(normalize=True, dropna=False)
            .to_dict()
        )

    if "pred_proba" in monitor_df.columns:
        summary["avg_risk_in_monitor"] = float(monitor_df["pred_proba"].mean())

    if "abnormal_count" in monitor_df.columns:
        summary["avg_abnormal_signals"] = float(monitor_df["abnormal_count"].mean())

    return summary

def apply_re_escalation_logic(
    df,
    risk_trend_threshold=0.01,
    abnormal_threshold=2,
    ibpip_threshold=2,
):
    df = df.copy()

    # Ensure required columns exist
    required_cols = ["monitor_state", "risk_trend", "abnormal_count"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    # IBPIP optional
    has_ibpip = "ibpip_n_abnormal_signals" in df.columns

    # Initialize
    df["re_escalate_flag"] = False
    df["re_escalate_reason"] = None

    def decide(row):
        if row["monitor_state"] != "DECLINING_MONITOR":
            return False, None

        reasons = []

        if row["risk_trend"] > risk_trend_threshold:
            reasons.append("rising_risk")

        if row["abnormal_count"] >= abnormal_threshold:
            reasons.append("multi_signal_instability")

        if has_ibpip and row["ibpip_n_abnormal_signals"] >= ibpip_threshold:
            reasons.append("ibpip_instability")

        if len(reasons) == 0:
            return False, None

        return True, "|".join(reasons)

    results = df.apply(lambda row: decide(row), axis=1)

    df["re_escalate_flag"] = [r[0] for r in results]
    df["re_escalate_reason"] = [r[1] for r in results]

    # Update final tier
    df.loc[df["re_escalate_flag"], "bire_final_tier"] = "RE-ESCALATE"

    return df

def summarize_re_escalation(df):
    df = df.copy()

    re_df = df[df["bire_final_tier"] == "RE-ESCALATE"]

    summary = {
        "total_re_escalations": int(len(re_df)),
        "reason_distribution": {},
        "avg_risk": None,
    }

    if re_df.empty:
        return summary

    summary["reason_distribution"] = (
        re_df["re_escalate_reason"]
        .value_counts(normalize=True)
        .to_dict()
    )

    if "pred_proba" in re_df.columns:
        summary["avg_risk"] = float(re_df["pred_proba"].mean())

    return summary

def apply_critical_logic(
    df,
    high_risk_threshold=0.95,
    risk_trend_threshold=0.05,
    abnormal_threshold=3,
    ibpip_threshold=3,
):
    """
    Apply CRITICAL post-event logic.

    CRITICAL represents severe failure to stabilize after an event.
    It should only trigger when RE-ESCALATE is already active and
    strong evidence supports continued deterioration.
    """
    df = df.copy()

    required_cols = [
        "bire_final_tier",
        "pred_proba",
        "risk_trend",
        "abnormal_count",
        "re_escalate_flag",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for CRITICAL logic: {missing}")

    has_ibpip = "ibpip_n_abnormal_signals" in df.columns

    df["critical_flag"] = False
    df["critical_reason"] = None

    def decide(row):
        if not row["re_escalate_flag"]:
            return False, None

        reasons = []

        if row["pred_proba"] >= high_risk_threshold:
            reasons.append("very_high_risk")

        if row["risk_trend"] >= risk_trend_threshold:
            reasons.append("continued_risk_acceleration")

        if row["abnormal_count"] >= abnormal_threshold:
            reasons.append("severe_multi_signal_instability")

        if has_ibpip and row["ibpip_n_abnormal_signals"] >= ibpip_threshold:
            reasons.append("ibpip_severe_instability")

        if len(reasons) >= 2:
            return True, "|".join(reasons)

        return False, None

    results = df.apply(lambda row: decide(row), axis=1)

    df["critical_flag"] = [r[0] for r in results]
    df["critical_reason"] = [r[1] for r in results]

    df.loc[df["critical_flag"], "bire_final_tier"] = "CRITICAL"

    return df
