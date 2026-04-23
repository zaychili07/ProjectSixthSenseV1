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


import numpy as np
import pandas as pd


def build_bire_evaluation_df(
    scored_df: pd.DataFrame,
    alert_df: pd.DataFrame,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    alert_col: str = "alert_episode_flag",
) -> pd.DataFrame:
    """
    Build the canonical BIRE evaluation dataframe by merging model scores
    with alert episode flags.
    """
    if scored_df.empty:
        raise ValueError("scored_df is empty")

    required_scored = {patient_col, time_col, "pred_proba"}
    missing_scored = required_scored - set(scored_df.columns)
    if missing_scored:
        raise ValueError(f"scored_df missing required columns: {missing_scored}")

    required_alert = {patient_col, time_col, alert_col}
    missing_alert = required_alert - set(alert_df.columns)
    if missing_alert:
        raise ValueError(f"alert_df missing required columns: {missing_alert}")

    bire_df = scored_df.copy()

    if alert_col not in bire_df.columns:
        bire_df = bire_df.merge(
            alert_df[[patient_col, time_col, alert_col]],
            on=[patient_col, time_col],
            how="left",
        )

    bire_df[alert_col] = bire_df[alert_col].fillna(0).astype(int)
    bire_df[time_col] = pd.to_datetime(bire_df[time_col])
    bire_df = bire_df.sort_values([patient_col, time_col]).copy()

    return bire_df


def build_event_leadtime_table(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    event_col: str = "event_now",
    alert_col: str = "alert_episode_flag",
) -> pd.DataFrame:
    """
    Build one row per event with first-alert lead time metrics.
    Event is anchored on 0->1 transitions of event_col.
    """
    work_df = df.copy()
    work_df[time_col] = pd.to_datetime(work_df[time_col])
    work_df = work_df.sort_values([patient_col, time_col]).copy()

    required_cols = {patient_col, time_col, alert_col, event_col}
    missing = required_cols - set(work_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    event_rows = []

    for patient_id, patient_df in work_df.groupby(patient_col, sort=False):
        patient_df = patient_df.sort_values(time_col).copy()

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
        ].sort_values().tolist()

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
    """
    Summarize event-level lead-time performance into headline metrics.
    """
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
    scored_df: pd.DataFrame,
    alert_df: pd.DataFrame,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    event_col: str = "event_now",
    alert_col: str = "alert_episode_flag",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    End-to-end helper:
    1) builds canonical bire_df
    2) builds event_lead_df
    3) builds lead_time_summary_df
    """
    bire_df = build_bire_evaluation_df(
        scored_df=scored_df,
        alert_df=alert_df,
        patient_col=patient_col,
        time_col=time_col,
        alert_col=alert_col,
    )

    event_lead_df = build_event_leadtime_table(
        df=bire_df,
        patient_col=patient_col,
        time_col=time_col,
        event_col=event_col,
        alert_col=alert_col,
    )

    lead_time_summary_df = summarize_event_leadtime(event_lead_df)

    return bire_df, event_lead_df, lead_time_summary_df

import pandas as pd
import numpy as np


def build_alert_episode_evaluation_df(
    scored_df: pd.DataFrame,
    alert_df: pd.DataFrame,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    alert_col: str = "alert_episode_flag",
    target_col: str = "target",
    interval_minutes: int = 5,
) -> pd.DataFrame:
    """
    Build canonical alert evaluation dataframe by merging scored rows
    with episode-level alerts.
    """
    required_scored = {patient_col, time_col, target_col}
    required_alert = {patient_col, time_col, alert_col}

    missing_scored = required_scored - set(scored_df.columns)
    missing_alert = required_alert - set(alert_df.columns)

    if missing_scored:
        raise ValueError(f"scored_df missing required columns: {missing_scored}")
    if missing_alert:
        raise ValueError(f"alert_df missing required columns: {missing_alert}")

    eval_df = scored_df.copy()
    eval_df[time_col] = pd.to_datetime(eval_df[time_col])

    alert_slice = alert_df[[patient_col, time_col, alert_col]].copy()
    alert_slice[time_col] = pd.to_datetime(alert_slice[time_col])

    eval_df = eval_df.merge(
        alert_slice,
        on=[patient_col, time_col],
        how="left",
    )

    eval_df[alert_col] = eval_df[alert_col].fillna(0).astype(int)
    eval_df["row_hours"] = interval_minutes / 60.0

    return eval_df


def summarize_alert_burden(
    eval_df: pd.DataFrame,
    patient_col: str = "patient_id",
    alert_col: str = "alert_episode_flag",
) -> pd.DataFrame:
    """
    Per-patient alert burden summary.
    """
    required_cols = {patient_col, alert_col, "row_hours"}
    missing = required_cols - set(eval_df.columns)
    if missing:
        raise ValueError(f"eval_df missing required columns: {missing}")

    patient_summary = (
        eval_df.groupby(patient_col, as_index=False)
        .agg(
            n_rows=("row_hours", "size"),
            patient_hours=("row_hours", "sum"),
            alert_episodes=(alert_col, "sum"),
        )
    )

    patient_summary["alerts_per_patient_hour"] = np.where(
        patient_summary["patient_hours"] > 0,
        patient_summary["alert_episodes"] / patient_summary["patient_hours"],
        np.nan,
    )

    return patient_summary.sort_values(
        "alerts_per_patient_hour", ascending=False
    ).reset_index(drop=True)


def summarize_false_alert_episodes(
    eval_df: pd.DataFrame,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    alert_col: str = "alert_episode_flag",
    target_col: str = "target",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate alert episodes as true vs false using forward-looking target.
    target == 1 at alert time => true/actionable alert
    target == 0 at alert time => false alert
    """
    required_cols = {patient_col, time_col, alert_col, target_col}
    missing = required_cols - set(eval_df.columns)
    if missing:
        raise ValueError(f"eval_df missing required columns: {missing}")

    alert_events_df = (
        eval_df.loc[eval_df[alert_col] == 1, [patient_col, time_col, alert_col, target_col]]
        .copy()
        .sort_values([patient_col, time_col])
        .reset_index(drop=True)
    )

    if alert_events_df.empty:
        summary_df = pd.DataFrame([{
            "total_alert_episodes": 0,
            "true_alert_episodes": 0,
            "false_alert_episodes": 0,
            "false_alert_episode_rate": np.nan,
            "true_alert_episode_rate": np.nan,
        }])
        return alert_events_df, summary_df

    alert_events_df["is_true_alert"] = (alert_events_df[target_col] == 1).astype(int)
    alert_events_df["is_false_alert"] = (alert_events_df[target_col] == 0).astype(int)

    total_alerts = int(alert_events_df.shape[0])
    true_alerts = int(alert_events_df["is_true_alert"].sum())
    false_alerts = int(alert_events_df["is_false_alert"].sum())

    summary_df = pd.DataFrame([{
        "total_alert_episodes": total_alerts,
        "true_alert_episodes": true_alerts,
        "false_alert_episodes": false_alerts,
        "false_alert_episode_rate": false_alerts / total_alerts if total_alerts > 0 else np.nan,
        "true_alert_episode_rate": true_alerts / total_alerts if total_alerts > 0 else np.nan,
    }])

    return alert_events_df, summary_df


def compute_alert_burden_outputs(
    scored_df: pd.DataFrame,
    alert_df: pd.DataFrame,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    alert_col: str = "alert_episode_flag",
    target_col: str = "target",
    interval_minutes: int = 5,
):
    """
    One-stop wrapper for alert burden and false alert episode evaluation.
    """
    eval_df = build_alert_episode_evaluation_df(
        scored_df=scored_df,
        alert_df=alert_df,
        patient_col=patient_col,
        time_col=time_col,
        alert_col=alert_col,
        target_col=target_col,
        interval_minutes=interval_minutes,
    )

    patient_alert_burden_df = summarize_alert_burden(
        eval_df=eval_df,
        patient_col=patient_col,
        alert_col=alert_col,
    )

    alert_events_df, false_alert_summary_df = summarize_false_alert_episodes(
        eval_df=eval_df,
        patient_col=patient_col,
        time_col=time_col,
        alert_col=alert_col,
        target_col=target_col,
    )

    return eval_df, patient_alert_burden_df, alert_events_df, false_alert_summary_df
