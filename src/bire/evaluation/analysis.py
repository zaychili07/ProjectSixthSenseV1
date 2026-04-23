import pandas as pd


def compute_lead_time_summary(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    alert_col: str = "alert",
    event_col: str = "target",
) -> pd.DataFrame:
    """
    Compute first alert time, first event time, and lead time per patient.
    """

    working_df = df.copy()
    working_df[time_col] = pd.to_datetime(working_df[time_col])
    working_df = working_df.sort_values([patient_col, time_col])

    rows = []

    for patient_id, patient_df in working_df.groupby(patient_col, sort=False):
        alert_rows = patient_df[patient_df[alert_col] == 1]
        event_rows = patient_df[patient_df[event_col] == 1]

        first_alert_time = alert_rows[time_col].min() if not alert_rows.empty else pd.NaT
        first_event_time = event_rows[time_col].min() if not event_rows.empty else pd.NaT

        if pd.isna(first_alert_time) or pd.isna(first_event_time):
            lead_time_minutes = float("nan")
        else:
            lead_time_minutes = (
                first_event_time - first_alert_time
            ).total_seconds() / 60.0

        rows.append({
            "patient_id": patient_id,
            "first_alert_time": first_alert_time,
            "first_event_time": first_event_time,
            "lead_time_minutes": lead_time_minutes,
            "has_event": not event_rows.empty,
        })

    return pd.DataFrame(rows)

import pandas as pd

def build_trajectory_summary_df(
    df,
    patient_col="patient_id",
    risk_col="pred_proba",
    alert_col="alert",
    event_col="target",
):
    if df.empty:
        return pd.DataFrame(
            columns=[
                patient_col,
                "n_rows",
                "max_risk",
                "mean_risk",
                "n_alerts",
                "n_events",
            ]
        )

    summary_df = (
        df.groupby(patient_col)
        .agg(
            n_rows=(patient_col, "size"),
            max_risk=(risk_col, "max"),
            mean_risk=(risk_col, "mean"),
            n_alerts=(alert_col, "sum"),
            n_events=(event_col, "sum"),
        )
        .reset_index()
    )

    return summary_df

import numpy as np
import pandas as pd


def build_event_leadtime_table(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    event_col: str = "event_now",
    alert_col: str = "alert_episode_flag",
) -> pd.DataFrame:
    work_df = df.copy()
    work_df[time_col] = pd.to_datetime(work_df[time_col])
    work_df = work_df.sort_values([patient_col, time_col]).copy()

    required_cols = {patient_col, time_col, alert_col}
    missing = required_cols - set(work_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if event_col not in work_df.columns:
        raise ValueError(
            f"'{event_col}' not found in dataframe. "
            "Use event_now if available; otherwise temporarily use target."
        )

    event_rows = []

    for patient_id, patient_df in work_df.groupby(patient_col, sort=False):
        patient_df = patient_df.sort_values(time_col).copy()

        # true event starts only: 0 -> 1 transition
        patient_df["prev_event"] = (
            patient_df[event_col].shift(1).fillna(0).astype(int)
        )
        patient_df["event_start_flag"] = (
            (patient_df[event_col] == 1) &
            (patient_df["prev_event"] == 0)
        ).astype(int)

        event_times = patient_df.loc[
            patient_df["event_start_flag"] == 1, time_col
        ].tolist()

        alert_times = patient_df.loc[
            patient_df[alert_col] == 1, time_col
        ].tolist()

        for idx, event_time in enumerate(event_times, start=1):
            prior_alerts = [t for t in alert_times if t < event_time]

            if prior_alerts:
                first_alert_time = prior_alerts[0]
                last_alert_time = prior_alerts[-1]
                first_lead_min = (
                    event_time - first_alert_time
                ).total_seconds() / 60.0
                last_lead_min = (
                    event_time - last_alert_time
                ).total_seconds() / 60.0
                detected = 1
            else:
                first_alert_time = pd.NaT
                last_alert_time = pd.NaT
                first_lead_min = np.nan
                last_lead_min = np.nan
                detected = 0

            event_rows.append({
                "patient_id": patient_id,
                "event_id": f"{patient_id}_event_{idx}",
                "event_time": event_time,
                "detected_before_event": detected,
                "first_alert_time": first_alert_time,
                "last_alert_time_before_event": last_alert_time,
                "first_alert_lead_minutes": first_lead_min,
                "last_alert_lead_minutes": last_lead_min,
                "detected_ge_15m": int(detected == 1 and first_lead_min >= 15),
                "detected_ge_30m": int(detected == 1 and first_lead_min >= 30),
                "detected_ge_60m": int(detected == 1 and first_lead_min >= 60),
            })

    return pd.DataFrame(event_rows)


def summarize_event_leadtime(event_lead_df: pd.DataFrame) -> pd.DataFrame:
    if event_lead_df.empty:
        return pd.DataFrame([{
            "n_events": 0,
            "events_detected_before_event": 0,
            "event_detection_rate": np.nan,
            "median_first_alert_lead_min": np.nan,
            "mean_first_alert_lead_min": np.nan,
            "pct_detected_ge_15m": np.nan,
            "pct_detected_ge_30m": np.nan,
            "pct_detected_ge_60m": np.nan,
        }])

    detected_df = event_lead_df.loc[
        event_lead_df["detected_before_event"] == 1
    ].copy()

    return pd.DataFrame([{
        "n_events": len(event_lead_df),
        "events_detected_before_event": int(
            event_lead_df["detected_before_event"].sum()
        ),
        "event_detection_rate": round(
            event_lead_df["detected_before_event"].mean(), 4
        ),
        "median_first_alert_lead_min": round(
            detected_df["first_alert_lead_minutes"].median(), 2
        ) if not detected_df.empty else np.nan,
        "mean_first_alert_lead_min": round(
            detected_df["first_alert_lead_minutes"].mean(), 2
        ) if not detected_df.empty else np.nan,
        "pct_detected_ge_15m": round(
            event_lead_df["detected_ge_15m"].mean(), 4
        ),
        "pct_detected_ge_30m": round(
            event_lead_df["detected_ge_30m"].mean(), 4
        ),
        "pct_detected_ge_60m": round(
            event_lead_df["detected_ge_60m"].mean(), 4
        ),
    }])


def compute_event_leadtime_outputs(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    event_col: str = "event_now",
    alert_col: str = "alert_episode_flag",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_lead_df = build_event_leadtime_table(
        df=df,
        patient_col=patient_col,
        time_col=time_col,
        event_col=event_col,
        alert_col=alert_col,
    )
    lead_time_summary_df = summarize_event_leadtime(event_lead_df)
    return event_lead_df, lead_time_summary_df
