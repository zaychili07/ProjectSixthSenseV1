import pandas as pd

def build_bire_output_from_patient(
    patient_df: pd.DataFrame,
    feature_cols: list,
    bire_model,
    patient_id_col: str = "patient_id",
    time_col: str = "timestamp",
    alert_threshold: float = 0.5,
):
    """
    Build a structured BIRE output dictionary from one patient's latest row.

    Assumes:
    - patient_df contains one patient's time-ordered feature rows
    - feature_cols are the columns used by the trained model
    - bire_model supports predict_proba()
    """

    patient_df = patient_df.sort_values(time_col).copy()
    latest_row = patient_df.iloc[-1]

    X_latest = latest_row[feature_cols].to_frame().T
    risk_score = float(bire_model.predict_proba(X_latest)[0, 1])
    
    if risk_score >= alert_threshold:
        risk_band = "high"
        alert = True
    elif risk_score >= 0.25:
        risk_band = "moderate"
        alert = False
    else:
        risk_band = "low"
        alert = False

    top_drivers = []

    driver_candidates = [
        ("spo2_delta", "worsening"),
        ("resp_rate_delta", "worsening"),
        ("sbp_delta", "worsening"),
        ("temperature_delta", "worsening"),
        ("heart_rate_delta", "worsening"),
    ]

    for feature_name, direction in driver_candidates:
        if feature_name in latest_row.index and pd.notna(latest_row[feature_name]):
            value = float(latest_row[feature_name])
            if abs(value) > 0:
                top_drivers.append({
                    "feature": feature_name,
                    "direction": direction,
                    "value": round(value, 3)
                })

    top_drivers = sorted(
        top_drivers,
        key=lambda x: abs(x["value"]),
        reverse=True
    )[:3]

    trend_summary = {}

    if "spo2_delta" in latest_row.index and pd.notna(latest_row["spo2_delta"]):
        trend_summary["spo2"] = (
            "down over last 10 minutes"
            if latest_row["spo2_delta"] < 0
            else "up over last 10 minutes"
        )

    if "resp_rate_delta" in latest_row.index and pd.notna(latest_row["resp_rate_delta"]):
        trend_summary["resp_rate"] = (
            "up over last 10 minutes"
            if latest_row["resp_rate_delta"] > 0
            else "down over last 10 minutes"
        )

    if "sbp_delta" in latest_row.index and pd.notna(latest_row["sbp_delta"]):
        trend_summary["sbp"] = (
            "down slightly"
            if latest_row["sbp_delta"] < 0
            else "up slightly"
        )

    if "temperature_delta" in latest_row.index and pd.notna(latest_row["temperature_delta"]):
        trend_summary["temperature"] = (
            "rising"
            if latest_row["temperature_delta"] > 0
            else "stable or falling"
        )

    data_quality = "adequate"
    if X_latest.isna().sum().sum() > 0:
        data_quality = "limited"

    return {
        "patient_id": str(latest_row[patient_id_col]),
        "timestamp": str(latest_row[time_col]),
        "risk_score": round(risk_score, 3),
        "risk_band": risk_band,
        "alert": alert,
        "prediction_horizon_minutes": 60,
        "top_drivers": top_drivers,
        "trend_summary": trend_summary,
        "data_quality": data_quality,
    }
    
def summarize_deterioration_strength(patient_df):
    patient_df = patient_df.sort_values("timestamp").copy()

    summary = {}

    for col in ["spo2", "resp_rate", "sbp", "heart_rate", "temperature"]:
        if col in patient_df.columns and len(patient_df) >= 2:
            start_val = patient_df[col].iloc[0]
            end_val = patient_df[col].iloc[-1]

            if pd.notna(start_val) and pd.notna(end_val):
                delta = end_val - start_val
                summary[f"{col}_delta_total"] = float(delta)

    spo2_drop = abs(min(summary.get("spo2_delta_total", 0), 0))
    rr_rise = max(summary.get("resp_rate_delta_total", 0), 0)
    sbp_drop = abs(min(summary.get("sbp_delta_total", 0), 0))
    hr_rise = max(summary.get("heart_rate_delta_total", 0), 0)
    temp_rise = max(summary.get("temperature_delta_total", 0), 0)

    deterioration_score = spo2_drop + rr_rise + sbp_drop + hr_rise + temp_rise
    summary["deterioration_score"] = deterioration_score

    return summary

def build_bire_dashboard_markdown(bire_output: dict) -> str:
    return f"""
# BIRE Clinical Risk Dashboard

**Patient ID:** {bire_output['patient_id']}  
**Timestamp:** {bire_output['timestamp']}  

**Risk Score:** {bire_output['risk_score']:.3f}  
**Risk Band:** {bire_output['risk_band']}  
**Alert:** {bire_output['alert']}  
**Prediction Horizon:** {bire_output['prediction_horizon_minutes']} minutes  
**Data Quality:** {bire_output['data_quality']}
""".strip()


def build_top_drivers_table(bire_output: dict) -> pd.DataFrame:
    top_drivers = bire_output.get("top_drivers", [])
    if not top_drivers:
        return pd.DataFrame(columns=["feature", "direction", "value"])
    return pd.DataFrame(top_drivers)


def build_trend_summary_table(bire_output: dict) -> pd.DataFrame:
    trend_summary = bire_output.get("trend_summary", {})
    if not trend_summary:
        return pd.DataFrame(columns=["vital", "trend"])

    return pd.DataFrame(
        [{"vital": vital, "trend": trend} for vital, trend in trend_summary.items()]
    )


def select_best_demo_patient(
    scored_df: pd.DataFrame,
    patient_id_col: str = "patient_id",
    score_col: str = "pred_proba",
) -> str:
    if score_col not in scored_df.columns:
        raise ValueError(f"Expected '{score_col}' column in scored_df.")

    best_row = scored_df.sort_values(score_col, ascending=False).iloc[0]
    return str(best_row[patient_id_col])
