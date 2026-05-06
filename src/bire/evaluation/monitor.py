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
