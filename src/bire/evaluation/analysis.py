
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
