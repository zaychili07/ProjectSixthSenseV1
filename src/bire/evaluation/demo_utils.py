import pandas as pd
import numpy as np


def build_bire_output_from_patient(
    patient_df,
    feature_cols,
    bire_model,
    alert_threshold=0.40,
):
    if patient_df.empty:
        raise ValueError("patient_df is empty")

    missing = [c for c in feature_cols if c not in patient_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns for BIRE output: {missing}")

    patient_df = patient_df.sort_values("timestamp").copy()
    latest_row = patient_df.iloc[-1]
   
    X_latest = latest_row[feature_cols].to_frame().T.copy()
    
    X_latest = X_latest.apply(pd.to_numeric, errors="coerce")
    risk_score = float(bire_model.predict_proba(X_latest)[0, 1])

    # -----------------------------
    # Risk band aligned to BIRE alerting
    # -----------------------------
    if risk_score >= alert_threshold:
        risk_band = "high"
        alert = True
    elif risk_score >= 0.25:
        risk_band = "moderate"
        alert = False
    else:
        risk_band = "low"
        alert = False

    # -----------------------------
    # Driver extraction
    # -----------------------------
    top_drivers = []

    # Prefer model-based feature importance when available
    model_importance = None
    if hasattr(bire_model, "feature_importances_"):
        model_importance = dict(
            zip(feature_cols, bire_model.feature_importances_)
        )
    elif hasattr(bire_model, "coef_"):
        model_importance = dict(
            zip(feature_cols, np.abs(bire_model.coef_[0]))
        )

    if model_importance is not None:
        driver_rows = []
        for col in feature_cols:
            val = latest_row[col]
            if pd.notna(val):
                driver_rows.append({
                    "feature": col,
                    "value": float(val),
                    "importance": float(model_importance.get(col, 0.0)),
                    "direction": infer_driver_direction(col, val),
                })

        top_drivers = sorted(
            driver_rows,
            key=lambda x: x["importance"],
            reverse=True,
        )[:5]

    else:
        # fallback
        for col in feature_cols:
            if "delta" in col:
                val = latest_row[col]
                if pd.notna(val) and abs(val) > 0:
                    top_drivers.append({
                        "feature": col,
                        "direction": infer_driver_direction(col, val),
                        "value": float(val),
                        "importance": abs(float(val)),
                    })

        top_drivers = sorted(
            top_drivers,
            key=lambda x: x["importance"],
            reverse=True,
        )[:5]

    # -----------------------------
    # Trend summary
    # -----------------------------
    trend_summary = build_trend_summary(patient_df)

    return {
        "patient_id": str(latest_row["patient_id"]),
        "risk_score": round(risk_score, 3),
        "risk_band": risk_band,
        "alert": alert,
        "prediction_horizon_minutes": 60,
        "top_drivers": top_drivers,
        "trend_summary": trend_summary,
        "data_quality": "adequate",
    }


def infer_driver_direction(feature_name, value):
    feature_lower = feature_name.lower()

    if "spo2" in feature_lower or "sbp" in feature_lower or "dbp" in feature_lower:
        if value < 0:
            return "worsening"
        elif value > 0:
            return "improving"
        return "stable"

    if "heart_rate" in feature_lower or "resp_rate" in feature_lower or "temp" in feature_lower:
        if value > 0:
            return "worsening"
        elif value < 0:
            return "improving"
        return "stable"

    if "std" in feature_lower:
        if value > 0:
            return "instability rising"
        return "stable"

    return "changing"


def build_trend_summary(patient_df):
    df = patient_df.sort_values("timestamp").copy()
    trend_summary = {}

    signal_map = {
        "spo2": "SpO2",
        "resp_rate": "Resp Rate",
        "sbp": "SBP",
        "heart_rate": "Heart Rate",
        "temperature": "Temperature",
    }

    for col in signal_map:
        if col in df.columns and len(df) >= 2:
            start_val = df[col].iloc[0]
            end_val = df[col].iloc[-1]
            delta = end_val - start_val
            trend_summary[col] = interpret_change(signal_map[col], delta)

    return trend_summary


import pandas as pd


def summarize_deterioration_strength(patient_df: pd.DataFrame) -> list[dict]:
    """
    Summarize directional change in key vitals across a patient's trajectory.
    """
    if patient_df is None or patient_df.empty:
        raise ValueError("patient_df is empty")

    signal_map = {
        "SpO2": "spo2",
        "Resp Rate": "resp_rate",
        "SBP": "sbp",
        "Heart Rate": "heart_rate",
        "Temp": "temperature",
    }

    summary_rows = []

    for signal_name, col in signal_map.items():
        if col not in patient_df.columns:
            summary_rows.append({
                "Signal": signal_name,
                "Start": pd.NA,
                "End": pd.NA,
                "Total Change": pd.NA,
                "Clinical Direction": "Missing column",
            })
            continue

        signal_series = patient_df[col].dropna()

        if signal_series.empty:
            summary_rows.append({
                "Signal": signal_name,
                "Start": pd.NA,
                "End": pd.NA,
                "Total Change": pd.NA,
                "Clinical Direction": "No data",
            })
            continue

        start_val = signal_series.iloc[0]
        end_val = signal_series.iloc[-1]
        delta = end_val - start_val

        if col == "spo2":
            direction = "Worsening ↓" if delta < 0 else "Improving ↑" if delta > 0 else "Stable →"
        elif col in ["resp_rate", "heart_rate", "temperature"]:
            direction = "Worsening ↑" if delta > 0 else "Improving ↓" if delta < 0 else "Stable →"
        elif col == "sbp":
            direction = "Worsening ↓" if delta < 0 else "Improving ↑" if delta > 0 else "Stable →"
        else:
            direction = "Stable →"

        summary_rows.append({
            "Signal": signal_name,
            "Start": round(float(start_val), 2),
            "End": round(float(end_val), 2),
            "Total Change": round(float(delta), 2),
            "Clinical Direction": direction,
        })

    return summary_rows


def interpret_change(signal, delta):
    if signal in ["SpO2", "SBP"]:
        if delta < 0:
            return "Worsening ↓"
        elif delta > 0:
            return "Improving ↑"
        return "Stable →"
    else:
        if delta > 0:
            return "Worsening ↑"
        elif delta < 0:
            return "Improving ↓"
        return "Stable →"


def build_deterioration_table(det_summary):
    """
    Convert deterioration summary output into a display-ready DataFrame.

    Expected input:
        det_summary = list[dict]
    """
    if det_summary is None:
        return pd.DataFrame()

    if isinstance(det_summary, list):
        return pd.DataFrame(det_summary)

    raise ValueError(
        "build_deterioration_table expected det_summary as list[dict]. "
        f"Got type: {type(det_summary)}"
    )


def style_direction(val):
    if "Worsening" in val:
        return "color: white; background-color: #dc2626; font-weight: bold;"
    elif "Improving" in val:
        return "color: white; background-color: #16a34a; font-weight: bold;"
    return "color: black; background-color: #e5e7eb; font-weight: bold;"


def build_bire_dashboard_markdown(bire_output):
    risk_score = bire_output["risk_score"]
    risk_band_raw = bire_output["risk_band"]
    risk_band = str(risk_band_raw).upper()
    patient_id = bire_output["patient_id"]
    alert_status = "YES" if bire_output["alert"] else "NO"
    horizon = bire_output["prediction_horizon_minutes"]
    data_quality = bire_output["data_quality"]

    risk_color_map = {
        "LOW": "#16a34a",
        "MODERATE": "#f59e0b",
        "HIGH": "#dc2626",
    }
    risk_color = risk_color_map.get(risk_band, "#6b7280")
    alert_color = "#dc2626" if bire_output["alert"] else "#16a34a"

    dashboard_md = f"""
# 🏥 BIRE Clinical Risk Dashboard

<div style="padding:14px 16px; border:1px solid #333; border-radius:12px; margin:10px 0 14px 0;">
  <div style="font-size:16px; margin-bottom:10px;"><b>Patient Overview</b></div>
  <div><b>Patient ID:</b> <code>{patient_id}</code></div>
  <div><b>Risk Score:</b> <code>{risk_score:.3f}</code></div>
  <div>
    <b>Risk Band:</b>
    <span style="
      display:inline-block;
      padding:4px 10px;
      border-radius:999px;
      color:white;
      background:{risk_color};
      font-weight:700;
      margin-left:6px;
    ">{risk_band}</span>
  </div>
  <div>
    <b>Alert Triggered:</b>
    <span style="
      display:inline-block;
      padding:4px 10px;
      border-radius:999px;
      color:white;
      background:{alert_color};
      font-weight:700;
      margin-left:6px;
    ">{alert_status}</span>
  </div>
  <div><b>Prediction Horizon:</b> <b>{horizon} minutes</b></div>
  <div><b>Data Quality:</b> <b>{data_quality}</b></div>
</div>
"""
    return dashboard_md


def select_best_demo_patient(trajectory_summary_df):
    if trajectory_summary_df.empty:
        raise ValueError("trajectory_summary_df is empty")

    best_row = trajectory_summary_df.sort_values(
        ["n_alerts", "max_risk", "n_rows"],
        ascending=[False, False, False]
    ).iloc[0]

    return best_row["patient_id"]

def build_top_drivers_table(bire_output):
    df = pd.DataFrame(bire_output.get("top_drivers", []))
    if not df.empty and "importance" in df.columns:
        df["importance"] = df["importance"].round(4)
    return df


def build_trend_summary_table(bire_output):
    trend_summary = bire_output.get("trend_summary", {})
    df = pd.DataFrame(
        [{"vital": k, "trend": v} for k, v in trend_summary.items()]
    )
    return df
