# ----------------------------
# BIRE output builder
# ----------------------------
import pandas as pd

def build_bire_output_from_patient(
    patient_df,
    feature_cols,
    bire_model
):
    patient_df = patient_df.sort_values("timestamp").copy()
    latest_row = patient_df.iloc[-1]

    X_latest = latest_row[feature_cols].to_frame().T
    risk_score = float(bire_model.predict_proba(X_latest)[0, 1])

   if risk_score >= 0.5:
    risk_band = "high"
    alert = True
elif risk_score >= 0.25:
    risk_band = "moderate"
    alert = False
else:
    risk_band = "low"
    alert = False

    top_drivers = []
    for col in feature_cols:
        if "delta" in col and col in latest_row:
            val = latest_row[col]
            if pd.notna(val) and abs(val) > 0:
                top_drivers.append({
                    "feature": col,
                    "direction": "worsening",
                    "value": float(val)
                })

    top_drivers = sorted(top_drivers, key=lambda x: abs(x["value"]), reverse=True)[:3]

    trend_summary = {
        k: "changing"
        for k in ["spo2", "resp_rate", "sbp", "temperature"]
        if k in patient_df.columns
    }

    return {
        "patient_id": str(latest_row["patient_id"]),
        "risk_score": round(risk_score, 3),
        "risk_band": risk_band,
        "alert": alert,
        "prediction_horizon_minutes": 60,
        "top_drivers": top_drivers,
        "trend_summary": trend_summary,
        "data_quality": "adequate"
    }

def summarize_deterioration_strength(patient_df):
    """
    Summarize total directional change across key vitals.
    Negative SpO2 / SBP is usually worse.
    Positive Resp Rate / Heart Rate / Temperature is usually worse.
    """
    df = patient_df.sort_values("timestamp").copy()

    summary = {}
    signals = ["spo2", "resp_rate", "sbp", "heart_rate", "temperature"]

    for col in signals:
        if col in df.columns and len(df) >= 2:
            start_val = df[col].iloc[0]
            end_val = df[col].iloc[-1]
            summary[f"{col}_start"] = float(start_val)
            summary[f"{col}_end"] = float(end_val)
            summary[f"{col}_delta_total"] = float(end_val - start_val)

    return summary

det_summary = summarize_deterioration_strength(patient_example_df)

display(Markdown("## 🚨 Deterioration Snapshot"))

def interpret_change(signal, delta):
    if signal in ["SpO2", "SBP"]:
        if delta < 0:
            return "Worsening ↓"
        elif delta > 0:
            return "Improving ↑"
        return "Stable →"
    else:  # Resp Rate, Heart Rate, Temp
        if delta > 0:
            return "Worsening ↑"
        elif delta < 0:
            return "Improving ↓"
        return "Stable →"

det_df = pd.DataFrame({
    "Signal": ["SpO2", "Resp Rate", "SBP", "Heart Rate", "Temp"],
    "Start": [
        round(det_summary.get("spo2_start", 0), 2),
        round(det_summary.get("resp_rate_start", 0), 2),
        round(det_summary.get("sbp_start", 0), 2),
        round(det_summary.get("heart_rate_start", 0), 2),
        round(det_summary.get("temperature_start", 0), 2),
    ],
    "End": [
        round(det_summary.get("spo2_end", 0), 2),
        round(det_summary.get("resp_rate_end", 0), 2),
        round(det_summary.get("sbp_end", 0), 2),
        round(det_summary.get("heart_rate_end", 0), 2),
        round(det_summary.get("temperature_end", 0), 2),
    ],
    "Total Change": [
        round(det_summary.get("spo2_delta_total", 0), 2),
        round(det_summary.get("resp_rate_delta_total", 0), 2),
        round(det_summary.get("sbp_delta_total", 0), 2),
        round(det_summary.get("heart_rate_delta_total", 0), 2),
        round(det_summary.get("temperature_delta_total", 0), 2),
    ]
})

det_df["Clinical Direction"] = [
    interpret_change("SpO2", det_df.loc[0, "Total Change"]),
    interpret_change("Resp Rate", det_df.loc[1, "Total Change"]),
    interpret_change("SBP", det_df.loc[2, "Total Change"]),
    interpret_change("Heart Rate", det_df.loc[3, "Total Change"]),
    interpret_change("Temp", det_df.loc[4, "Total Change"]),
]

def style_direction(val):
    if "Worsening" in val:
        return "color: white; background-color: #dc2626; font-weight: bold;"
    elif "Improving" in val:
        return "color: white; background-color: #16a34a; font-weight: bold;"
    return "color: black; background-color: #e5e7eb; font-weight: bold;"

display(det_df.style.applymap(style_direction, subset=["Clinical Direction"]))

best_demo_patient = trajectory_summary_df.sort_values(
    ["n_alerts", "max_risk", "n_rows"],
    ascending=[False, False, False]
).iloc[0]["patient_id"]

patient_example_df = df[df["patient_id"] == best_demo_patient].copy()
patient_example_df = patient_example_df.sort_values("timestamp")

patient_traj = trajectory_summary_df[
    trajectory_summary_df["patient_id"] == best_demo_patient
]

patient_rankings = []

for pid in trajectory_summary_df["patient_id"].unique():
    this_patient = df[df["patient_id"] == pid].copy().sort_values("timestamp")
    det_summary = summarize_deterioration_strength(this_patient)

    row = trajectory_summary_df[trajectory_summary_df["patient_id"] == pid].iloc[0].to_dict()
    row.update(det_summary)
    patient_rankings.append(row)

patient_rankings_df = pd.DataFrame(patient_rankings)

patient_rankings_df = patient_rankings_df.sort_values(
    ["n_alerts", "max_risk", "deterioration_score", "n_rows"],
    ascending=[False, False, False, False]
)

display(patient_rankings_df.head(10))

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
