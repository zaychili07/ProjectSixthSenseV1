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
    risk_col="pred_proba",
    event_col="event_now",
    output_col="bms_mode",
):
    """
    Assign BMS clinical modes at the patient level using signal-driven severity logic.

    This replaces random/placeholding assignment with patient-level acuity stratification
    based on risk burden and deterioration signals.

    Notes
    -----
    This is ESI-inspired for research/prototype use only.
    It is NOT clinical triage and should not be treated as a medical decision tool.
    """

    import numpy as np
    import pandas as pd

    out = df.copy()

    if patient_col not in out.columns:
        raise ValueError(f"Missing required patient column: {patient_col}")

    if risk_col not in out.columns:
        raise ValueError(f"Missing required risk column: {risk_col}")

    # Event column is optional.
    has_event_col = event_col in out.columns

    def _safe_count_threshold(x, threshold):
        return int((x >= threshold).sum())

    agg_dict = {
        "rows": (risk_col, "size"),
        "max_risk": (risk_col, "max"),
        "mean_risk": (risk_col, "mean"),
        "high_risk_rows": (risk_col, lambda x: _safe_count_threshold(x, 0.30)),
        "very_high_risk_rows": (risk_col, lambda x: _safe_count_threshold(x, 0.70)),
        "extreme_risk_rows": (risk_col, lambda x: _safe_count_threshold(x, 0.90)),
    }

    if has_event_col:
        agg_dict["event_rows"] = (event_col, "sum")

    profile = (
        out.groupby(patient_col)
        .agg(**agg_dict)
        .reset_index()
    )

    if not has_event_col:
        profile["event_rows"] = 0

    profile["high_risk_ratio"] = profile["high_risk_rows"] / profile["rows"].clip(lower=1)
    profile["very_high_risk_ratio"] = profile["very_high_risk_rows"] / profile["rows"].clip(lower=1)
    profile["extreme_risk_ratio"] = profile["extreme_risk_rows"] / profile["rows"].clip(lower=1)
    profile["event_ratio"] = profile["event_rows"] / profile["rows"].clip(lower=1)

    def _assign_mode(row):
        max_risk = row["max_risk"]
        mean_risk = row["mean_risk"]
        high_ratio = row["high_risk_ratio"]
        very_high_ratio = row["very_high_risk_ratio"]
        extreme_ratio = row["extreme_risk_ratio"]
        event_rows = row["event_rows"]

        # Highest acuity: sustained extreme risk or actual deterioration burden.
        if (
            mean_risk >= 0.60
            or extreme_ratio >= 0.35
            or very_high_ratio >= 0.45
            or event_rows >= 5
        ):
            return "ICU"

        # ER_ESI_1: critical / immediate concern.
        if (
            max_risk >= 0.98
            and (
                mean_risk >= 0.35
                or extreme_ratio >= 0.15
                or very_high_ratio >= 0.25
                or high_ratio >= 0.45
            )
        ):
            return "ER_ESI_1"

        # ER_ESI_2: high risk but less sustained than ICU/ESI-1.
        if (
            max_risk >= 0.95
            or mean_risk >= 0.25
            or very_high_ratio >= 0.10
            or high_ratio >= 0.30
        ):
            return "ER_ESI_2"

        # ER_ESI_3: moderate risk / needs workup.
        if (
            max_risk >= 0.75
            or mean_risk >= 0.12
            or high_ratio >= 0.12
        ):
            return "ER_ESI_3"

        # ER_ESI_4: low-to-moderate episodic risk.
        if (
            max_risk >= 0.45
            or mean_risk >= 0.06
            or high_ratio >= 0.03
        ):
            return "ER_ESI_4"

        # ER_ESI_5: very low risk, walk-in-like ER pattern.
        if max_risk >= 0.20 or mean_risk >= 0.03:
            return "ER_ESI_5"

        # Outpatient: stable / very low risk.
        return "Outpatient"

    profile[output_col] = profile.apply(_assign_mode, axis=1)

    mode_map = profile[[patient_col, output_col]]

    out = out.drop(columns=[output_col], errors="ignore")
    out = out.merge(mode_map, on=patient_col, how="left")

    if out[output_col].isna().any():
        raise ValueError("BMS mode assignment produced null modes.")

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

###############################################
# IBPIP-Aware GSS Override Logic — Balanced v2
###############################################

def apply_ibpip_gss_logic( #IBPIP upgrade to v2
    df,
    base_alert_col="bms_alert",
    output_alert_col="final_ibpip_alert",
):
    df = df.copy()

    df["ibpip_override_alert"] = False
    df["ibpip_suppress_alert"] = False
    df["ibpip_override_reason"] = None

    warmup_mask = df["ibpip_state"] == "WARMUP"

    # Stronger escalation rule: require meaningful deviation
    high_deviation = (
        (~warmup_mask) &
        (
            ((df["ibpip_score"] >= 2.5) & (df["ibpip_n_abnormal_signals"] >= 2)) |
            (df["ibpip_max_abs_z"] >= 4.0)
        )
    )

    df.loc[high_deviation, "ibpip_override_alert"] = True
    df.loc[high_deviation, "ibpip_override_reason"] = "HIGH_DEVIATION_ESCALATION"

    # Suppression rule: stable relative to patient baseline
    stable_suppression = (
        (~warmup_mask) &
        (df["ibpip_state"] == "CAUTIOUS_BASELINE") &
        (df["ibpip_score"] < 0.75) &
        (df["ibpip_n_abnormal_signals"] == 0)
    )

    df.loc[stable_suppression, "ibpip_suppress_alert"] = True
    df.loc[stable_suppression, "ibpip_override_reason"] = "SUPPRESSED_STABLE_BASELINE"

    # Recovery-aware suppression
    recovery_suppression = (
        (~warmup_mask) &
        (df["ibpip_state"] == "REASSESSING") &
        (df["ibpip_score"] < 0.90) &
        (df["ibpip_n_abnormal_signals"] == 0)
    )

    df.loc[recovery_suppression, "ibpip_suppress_alert"] = True
    df.loc[recovery_suppression, "ibpip_override_reason"] = "SUPPRESSED_RECOVERY_TREND"

    # Final alert decision (force boolean type)
    df[output_alert_col] = df[base_alert_col].astype(bool)

    # Suppress first
    df.loc[df["ibpip_suppress_alert"], output_alert_col] = False

    # Then allow escalation to override suppression
    df.loc[df["ibpip_override_alert"], output_alert_col] = True

return df
