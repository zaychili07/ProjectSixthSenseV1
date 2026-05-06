import numpy as np
import pandas as pd


def compute_monitor_states(df, window=3):
    df = df.copy()

    # --- Identify MONITOR rows ---
    df["is_monitor"] = df["tier"] == "MONITOR"

    # --- Risk trend ---
    df["risk_delta"] = df.groupby("patient_id")["pred_proba"].diff()

    df["risk_trend"] = (
        df.groupby("patient_id")["risk_delta"]
        .rolling(window)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # --- Stability (variance) ---
    df["risk_std"] = (
        df.groupby("patient_id")["pred_proba"]
        .rolling(window)
        .std()
        .reset_index(level=0, drop=True)
    )

    # --- Physiological instability (IBPIP-lite proxy) ---
    abnormal_flags = [
        df["heart_rate"] > 120,
        df["spo2"] < 92,
        df["resp_rate"] > 24,
        df["sbp"] < 100,
        (df["temperature"] > 38.5) | (df["temperature"] < 36),
    ]

    df["abnormal_count"] = np.sum(abnormal_flags, axis=0)

    # --- Classification ---
    def classify(row):
        if not row["is_monitor"]:
            return None

        # Stable
        if row["risk_trend"] < -0.01 and row["abnormal_count"] <= 1:
            return "STABLE_MONITOR"

        # Declining
        if row["risk_trend"] > 0.01 and row["abnormal_count"] >= 2:
            return "DECLINING_MONITOR"

        # Otherwise
        return "VOLATILE_MONITOR"

    df["monitor_state"] = df.apply(classify, axis=1)

    return df

def summarize_monitor_states(df):
    summary = {}

    monitor_df = df[df["tier"] == "MONITOR"]

    summary["total_monitor_rows"] = len(monitor_df)

    summary["state_distribution"] = (
        monitor_df["monitor_state"]
        .value_counts(normalize=True)
        .to_dict()
    )

    summary["avg_risk_in_monitor"] = monitor_df["pred_proba"].mean()

    summary["avg_abnormal_signals"] = monitor_df["abnormal_count"].mean()

    return summary
