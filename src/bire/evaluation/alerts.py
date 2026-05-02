import numpy as np
import pandas as pd



def apply_alert_logic(df: pd.DataFrame, threshold: float = 0.5, window: int = 3) -> pd.DataFrame:
    """
    Apply persistence-based alert logic.

    An alert is triggered when predicted risk exceeds threshold
    for 'window' consecutive time steps.
    """
    df = df.copy().sort_values(["patient_id", "timestamp"])

    # Step 1: High-risk flag
    df["high_risk"] = (df["pred_proba"] >= threshold).astype(int)

    # Step 2: Rolling persistence check
    df["alert"] = (
        df.groupby("patient_id")["high_risk"]
        .rolling(window=window, min_periods=window)
        .sum()
        .reset_index(level=0, drop=True)
        .ge(window)
        .astype(int)
    )

    return df


def summarize_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize alerts per patient.
    """
    summary = (
        df.groupby("patient_id")
        .agg(
            n_rows=("patient_id", "size"),
            n_alerts=("alert", "sum"),
            max_risk=("pred_proba", "max"),
            mean_risk=("pred_proba", "mean"),
        )
        .reset_index()
    )

    return summary

# =========================================================
# BMS v2 — Clinical Mode / ESI-Based Thresholding
# =========================================================
# 
BMS_CONFIG = { # Dont forget to move form notebook to backed and make call to it. We are going to make BMS a thing FINALLY!
    "ICU": {
        "threshold": 0.970,
        "persistence": 2,
        "sensitivity": "very_high",
    },

    "ER_ESI_1": {
        "threshold": 0.965,
        "persistence": 1,
        "sensitivity": "maximum",
    },
    "ER_ESI_2": {
        "threshold": 0.975,
        "persistence": 2,
        "sensitivity": "very_high",
    },
    "ER_ESI_3": {
        "threshold": 0.985,
        "persistence": 2,
        "sensitivity": "high",
    },
    "ER_ESI_4": {
        "threshold": 0.992,
        "persistence": 3,
        "sensitivity": "moderate",
    },
    "ER_ESI_5": {
        "threshold": 0.996,
        "persistence": 3,
        "sensitivity": "low",
    },

    "Inpatient": {
        "threshold": 0.990,
        "persistence": 3,
        "sensitivity": "moderate_high",
    },

    "Outpatient": {
        "threshold": 0.995,
        "persistence": 4,
        "sensitivity": "conservative",
    },
}


BMS_MODES = list(BMS_CONFIG.keys())


def assign_bms_modes_per_patient(
    df,
    patient_col="patient_id",
    mode_probs=None,
    random_state=42,
):
    """
    Assign one BMS mode per patient.

    This is intended for research simulation only. In a real clinical
    system, mode would come from encounter context, care setting,
    clinician triage, or validated operational metadata.
    """
    out = df.copy()

    if mode_probs is None:
        mode_probs = {
            "ICU": 0.15,
            "ER_ESI_1": 0.05,
            "ER_ESI_2": 0.10,
            "ER_ESI_3": 0.20,
            "ER_ESI_4": 0.15,
            "ER_ESI_5": 0.10,
            "Inpatient": 0.15,
            "Outpatient": 0.10,
        }

    modes = list(mode_probs.keys())
    probs = list(mode_probs.values())

    if not np.isclose(sum(probs), 1.0):
        raise ValueError("mode_probs must sum to 1.0")

    rng = np.random.default_rng(random_state)

    unique_patients = out[patient_col].drop_duplicates()

    patient_mode_df = pd.DataFrame({
        patient_col: unique_patients,
        "bms_mode": rng.choice(
            modes,
            size=len(unique_patients),
            p=probs,
        ),
    })

    out = out.drop(columns=["bms_mode"], errors="ignore")
    out = out.merge(patient_mode_df, on=patient_col, how="left")

    return out


def apply_bms_thresholds(
    df,
    risk_col="risk_60min",
    mode_col="bms_mode",
    output_alert_col="bms_alert",
):
    """
    Apply BMS mode-specific thresholds to generate alerts.
    """
    out = df.copy()

    out["bms_threshold"] = out[mode_col].map(
        lambda mode: BMS_CONFIG[mode]["threshold"]
    )

    out["bms_persistence"] = out[mode_col].map(
        lambda mode: BMS_CONFIG[mode]["persistence"]
    )

    out["bms_sensitivity"] = out[mode_col].map(
        lambda mode: BMS_CONFIG[mode]["sensitivity"]
    )

    out[output_alert_col] = (
        out[risk_col] >= out["bms_threshold"]
    ).astype(int)

    return out


def summarize_bms_alerts(
    df,
    mode_col="bms_mode",
    alert_col="bms_alert",
    risk_col="risk_60min",
    patient_col="patient_id",
):
    """
    Summarize BMS alert behavior by mode.
    """
    return (
        df
        .groupby(mode_col)
        .agg(
            rows=(patient_col, "count"),
            patients=(patient_col, "nunique"),
            alerts=(alert_col, "sum"),
            alert_rate=(alert_col, "mean"),
            mean_risk=(risk_col, "mean"),
            max_risk=(risk_col, "max"),
        )
        .reset_index()
        .sort_values(mode_col)
    )


def compare_bms_to_global_threshold(
    df,
    global_threshold=0.990,
    risk_col="risk_60min",
    bms_alert_col="bms_alert",
    patient_col="patient_id",
):
    """
    Compare global-threshold alerting against BMS mode-based alerting.
    """
    out = df.copy()

    out["global_alert"] = (
        out[risk_col] >= global_threshold
    ).astype(int)

    n_patients = out[patient_col].nunique()

    return pd.DataFrame([
        {
            "policy": "global_threshold",
            "threshold_basis": global_threshold,
            "alerts": int(out["global_alert"].sum()),
            "alert_rate": float(out["global_alert"].mean()),
            "alerts_per_patient": float(out["global_alert"].sum() / n_patients),
        },
        {
            "policy": "bms_mode_threshold",
            "threshold_basis": "mode-based",
            "alerts": int(out[bms_alert_col].sum()),
            "alert_rate": float(out[bms_alert_col].mean()),
            "alerts_per_patient": float(out[bms_alert_col].sum() / n_patients),
        },
    ])
