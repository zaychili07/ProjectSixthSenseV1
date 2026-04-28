import pandas as pd
import numpy as np


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

    # Prevent merge suffix collisions like alert_episode_flag_x / _y
    cols_to_drop = [
        c for c in eval_df.columns
        if c == alert_col or c.startswith(f"{alert_col}_")
    ]
    if cols_to_drop:
        eval_df = eval_df.drop(columns=cols_to_drop)

    eval_df = eval_df.merge(
        alert_slice,
        on=[patient_col, time_col],
        how="left",
    )

    if alert_col not in eval_df.columns:
        raise ValueError(
            f"Expected merged alert column '{alert_col}' not found. "
            f"Available alert-like columns: {[c for c in eval_df.columns if 'alert' in c.lower()]}"
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

# this function will prevent a false alert by sending it as true besides just false using foreward looking targets.
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

# This is going to change
def add_alert_episode_flags(
    df: pd.DataFrame,
    threshold: float,
    prob_col: str = "pred_proba",
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    persistence_steps: int = 2,
) -> pd.DataFrame:

    out = df.copy()
    out = out.sort_values([patient_col, time_col])

    out["alert_raw"] = out[prob_col] >= threshold

    out["alert_persistent"] = (
        out.groupby(patient_col)["alert_raw"]
        .rolling(window=persistence_steps, min_periods=persistence_steps)
        .sum()
        .reset_index(level=0, drop=True)
        >= persistence_steps
    )

    previous_alert = (
        out.groupby(patient_col)["alert_persistent"]
        .shift(1)
    )

    previous_alert = previous_alert.fillna(False).astype(bool)

    out["alert_episode_flag"] = (
        out["alert_persistent"] & ~previous_alert
    ).astype(int)

    return out

def run_threshold_sweep(
    scored_df: pd.DataFrame,
    thresholds: list[float],
    event_col: str = "event_now",
    target_col: str = "target",
    prob_col: str = "pred_proba",
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    interval_minutes: int = 5,
    prediction_horizon_minutes: int = 60,
    persistence_steps: int = 2,
) -> pd.DataFrame:
    """
    Runs threshold sweep for BIRE alert tuning.

    Compares:
    - detection rate
    - median lead time
    - alerts per patient-hour
    - false alert episode rate
    - missed events
    """

    rows = []

    if event_col not in scored_df.columns:
        event_col = target_col

    for threshold in thresholds:
        temp_df = add_alert_episode_flags(
            scored_df,
            threshold=threshold,
            prob_col=prob_col,
            patient_col=patient_col,
            time_col=time_col,
            persistence_steps=persistence_steps,
        )

        total_patients = temp_df[patient_col].nunique()
        total_rows = len(temp_df)

        total_patient_hours = (
            total_rows * interval_minutes
        ) / 60

        total_alerts = temp_df["alert_episode_flag"].sum()

        alerts_per_patient_hour = (
            total_alerts / total_patient_hours
            if total_patient_hours > 0
            else np.nan
        )

        event_rows = temp_df[temp_df[event_col] == 1].copy()
        total_events = len(event_rows)

        detected_events = 0
        lead_times = []

        for _, event in event_rows.iterrows():
            pid = event[patient_col]
            event_time = event[time_col]

            lookback_start = event_time - pd.Timedelta(
                minutes=prediction_horizon_minutes
            )

            prior_alerts = temp_df[
                (temp_df[patient_col] == pid)
                & (temp_df[time_col] >= lookback_start)
                & (temp_df[time_col] < event_time)
                & (temp_df["alert_episode_flag"] == 1)
            ]

            if len(prior_alerts) > 0:
                detected_events += 1

                first_alert_time = prior_alerts[time_col].min()
                lead_time = (
                    event_time - first_alert_time
                ).total_seconds() / 60

                lead_times.append(lead_time)

        missed_events = total_events - detected_events

        detection_rate = (
            detected_events / total_events
            if total_events > 0
            else np.nan
        )

        median_lead_time = (
            np.median(lead_times)
            if len(lead_times) > 0
            else np.nan
        )

        false_alerts = 0

        alert_rows = temp_df[temp_df["alert_episode_flag"] == 1].copy()

        for _, alert in alert_rows.iterrows():
            pid = alert[patient_col]
            alert_time = alert[time_col]

            future_window_end = alert_time + pd.Timedelta(
                minutes=prediction_horizon_minutes
            )

            future_events = temp_df[
                (temp_df[patient_col] == pid)
                & (temp_df[time_col] > alert_time)
                & (temp_df[time_col] <= future_window_end)
                & (temp_df[event_col] == 1)
            ]

            if len(future_events) == 0:
                false_alerts += 1

        false_alert_rate = (
            false_alerts / total_alerts
            if total_alerts > 0
            else np.nan
        )

        rows.append(
            {
                "threshold": threshold,
                "total_events": total_events,
                "detected_events": detected_events,
                "missed_events": missed_events,
                "detection_rate": detection_rate,
                "median_lead_time_min": median_lead_time,
                "total_alert_episodes": total_alerts,
                "alerts_per_patient_hour": alerts_per_patient_hour,
                "false_alert_episodes": false_alerts,
                "false_alert_rate": false_alert_rate,
            }
        )

    return pd.DataFrame(rows)

def add_adaptive_patient_alerts(
    df: pd.DataFrame,
    base_threshold: float = 0.992,
    patient_quantile: float = 0.90,
    prob_col: str = "pred_proba",
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    persistence_steps: int = 3,
) -> pd.DataFrame:
    """
    Patient-adaptive alerting.

    Uses the larger of:
    1. global base threshold
    2. patient-specific risk quantile

    This prevents patients with chronically high risk from repeatedly alerting
    unless they exceed their own baseline risk pattern.
    """

    out = df.copy()
    out = out.sort_values([patient_col, time_col])

    patient_thresholds = (
        out.groupby(patient_col)[prob_col]
        .quantile(patient_quantile)
        .rename("patient_adaptive_threshold")
        .reset_index()
    )

    out = out.merge(patient_thresholds, on=patient_col, how="left")

    out["effective_threshold"] = out[
        ["patient_adaptive_threshold"]
    ].max(axis=1)

    out["effective_threshold"] = out["effective_threshold"].clip(
        lower=base_threshold
    )

    out["alert_raw"] = out[prob_col] >= out["effective_threshold"]

    out["alert_persistent"] = (
        out.groupby(patient_col)["alert_raw"]
        .rolling(window=persistence_steps, min_periods=persistence_steps)
        .sum()
        .reset_index(level=0, drop=True)
        >= persistence_steps
    )

    previous_alert = (
        out.groupby(patient_col)["alert_persistent"]
        .shift(1)
    )

    previous_alert = previous_alert.fillna(False).astype(bool)

    out["alert_episode_flag"] = (
        out["alert_persistent"] & ~previous_alert
    ).astype(int)

    return out
BIRE_MODE_CONFIGS = {
    "icu": {
        "label": "ICU Mode",
        "threshold": 0.990,
        "persistence_steps": 3,
        "description": "Highest sensitivity for high-acuity continuous monitoring.",
    },
    "er": {
        "label": "ER Mode",
        "threshold": 0.992,
        "persistence_steps": 3,
        "description": "High sensitivity with slightly stronger alert control for high-throughput care.",
    },
    "inpatient": {
        "label": "Inpatient Mode",
        "threshold": 0.994,
        "persistence_steps": 3,
        "description": "Balanced alerting for admitted patients outside critical care.",
    },
    "walk_in": {
        "label": "Walk-in / Check-up Mode",
        "threshold": 0.995,
        "persistence_steps": 3,
        "description": "Conservative monitoring for lower-acuity surveillance.",
    },
}


def get_bire_mode_config(mode: str) -> dict:
    """
    Return alert-policy configuration for a BIRE clinical mode.
    """
    mode_key = mode.lower().strip()

    if mode_key not in BIRE_MODE_CONFIGS:
        valid_modes = list(BIRE_MODE_CONFIGS.keys())
        raise ValueError(
            f"Unknown BIRE mode: {mode}. Valid modes are: {valid_modes}"
        )

    return BIRE_MODE_CONFIGS[mode_key]


def apply_bire_alert_mode(
    df: pd.DataFrame,
    mode: str,
    prob_col: str = "pred_proba",
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Apply a predefined BIRE alert mode to scored patient data.
    """
    config = get_bire_mode_config(mode)

    out = add_alert_episode_flags(
        df=df,
        threshold=config["threshold"],
        prob_col=prob_col,
        patient_col=patient_col,
        time_col=time_col,
        persistence_steps=config["persistence_steps"],
    )

    out["bire_mode"] = mode
    out["bire_mode_label"] = config["label"]
    out["bire_threshold"] = config["threshold"]
    out["bire_persistence_steps"] = config["persistence_steps"]

    return out

def compare_bire_alert_modes(
    df: pd.DataFrame,
    modes: list[str] | None = None,
    event_col: str = "event_now",
    target_col: str = "target",
    prob_col: str = "pred_proba",
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    interval_minutes: int = 5,
    prediction_horizon_minutes: int = 60,
) -> pd.DataFrame:
    """
    Compare predefined BIRE alert modes side-by-side.
    """
    if modes is None:
        modes = ["icu", "er", "inpatient", "walk_in"]

    if event_col not in df.columns:
        event_col = target_col

    rows = []

    for mode in modes:
        config = get_bire_mode_config(mode)

        mode_df = apply_bire_alert_mode(
            df=df,
            mode=mode,
            prob_col=prob_col,
            patient_col=patient_col,
            time_col=time_col,
        )

        total_rows = len(mode_df)
        total_patient_hours = total_rows * interval_minutes / 60
        total_alerts = int(mode_df["alert_episode_flag"].sum())

        event_rows = mode_df[mode_df[event_col] == 1].copy()
        total_events = len(event_rows)

        detected_events = 0
        lead_times = []

        for _, event in event_rows.iterrows():
            pid = event[patient_col]
            event_time = event[time_col]
            lookback_start = event_time - pd.Timedelta(
                minutes=prediction_horizon_minutes
            )

            prior_alerts = mode_df[
                (mode_df[patient_col] == pid)
                & (mode_df[time_col] >= lookback_start)
                & (mode_df[time_col] < event_time)
                & (mode_df["alert_episode_flag"] == 1)
            ]

            if not prior_alerts.empty:
                detected_events += 1
                first_alert_time = prior_alerts[time_col].min()
                lead_times.append(
                    (event_time - first_alert_time).total_seconds() / 60
                )

        false_alerts = 0
        alert_rows = mode_df[mode_df["alert_episode_flag"] == 1].copy()

        for _, alert in alert_rows.iterrows():
            pid = alert[patient_col]
            alert_time = alert[time_col]
            future_end = alert_time + pd.Timedelta(
                minutes=prediction_horizon_minutes
            )

            future_events = mode_df[
                (mode_df[patient_col] == pid)
                & (mode_df[time_col] > alert_time)
                & (mode_df[time_col] <= future_end)
                & (mode_df[event_col] == 1)
            ]

            if future_events.empty:
                false_alerts += 1

        rows.append({
            "mode": mode,
            "mode_label": config["label"],
            "threshold": config["threshold"],
            "persistence_steps": config["persistence_steps"],
            "total_events": total_events,
            "detected_events": detected_events,
            "missed_events": total_events - detected_events,
            "detection_rate": detected_events / total_events if total_events > 0 else np.nan,
            "median_lead_time_min": np.median(lead_times) if lead_times else np.nan,
            "total_alert_episodes": total_alerts,
            "alerts_per_patient_hour": total_alerts / total_patient_hours if total_patient_hours > 0 else np.nan,
            "false_alert_episodes": false_alerts,
            "false_alert_rate": false_alerts / total_alerts if total_alerts > 0 else np.nan,
            "description": config["description"],
        })

    return pd.DataFrame(rows)

def add_event_episode_flags(
    df: pd.DataFrame,
    event_col: str = "event_now",
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    output_col: str = "event_episode_flag",
) -> pd.DataFrame:
    """
    Marks the start of each deterioration event episode.

    Instead of counting every event row, this counts only 0 -> 1 transitions.
    """
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    out = out.sort_values([patient_col, time_col]).copy()

    out[event_col] = out[event_col].fillna(0).astype(int)

    previous_event = (
        out.groupby(patient_col)[event_col]
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    out[output_col] = (
        (out[event_col] == 1) & (previous_event == 0)
    ).astype(int)

    return out

def apply_gss(
    df,
    prob_col="pred_proba",
    alert_col="alert_episode_flag",
    patient_col="patient_id",
    time_col="timestamp",
    gss_alert_col="gss_alert",
    suppressed_col="gss_suppressed",
    escalation_col="gss_escalation",
    risk_delta=0.02,
):
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    out = out.sort_values([patient_col, time_col]).copy()

    out[gss_alert_col] = 0
    out[suppressed_col] = 0
    out[escalation_col] = 0

    for _, group in out.groupby(patient_col, sort=False):
        suppressing = False
        last_alert_risk = None

        for idx, row in group.iterrows():
            risk = row[prob_col]
            raw_alert = row[alert_col] == 1

            if raw_alert and not suppressing:
                out.loc[idx, gss_alert_col] = 1
                suppressing = True
                last_alert_risk = risk

            elif raw_alert and suppressing:
                if last_alert_risk is not None and risk >= last_alert_risk + risk_delta:
                    out.loc[idx, gss_alert_col] = 1
                    out.loc[idx, escalation_col] = 1
                    last_alert_risk = risk
                else:
                    out.loc[idx, suppressed_col] = 1

            if suppressing and last_alert_risk is not None:
                if risk < last_alert_risk - risk_delta:
                    suppressing = False

    return out

def apply_gss_with_vital_override(
    df,
    prob_col="pred_proba",
    alert_col="alert_episode_flag",
    patient_col="patient_id",
    time_col="timestamp",
    event_col="event_now",
    gss_alert_col="gss_v2_alert",
    suppressed_col="gss_v2_suppressed",
    escalation_col="gss_v2_escalation",
    escalation_reason_col="gss_v2_escalation_reason",
    risk_delta=0.013,
    escalation_threshold=0.995,
    spo2_drop = 1.0,
    sbp_drop = 5.0,
    resp_rate_rise = 2.0,
    heart_rate_rise = 5.0,
    temp_rise = 0.3,
    
):
    """
    GSS v2: Gateway Suppression System with vital-based override.

    Purpose:
    - Suppress redundant repeat alerts when risk/vitals are stable.
    - Break suppression when risk escalates or vitals worsen.
    - Designed as a prototype alert-control layer for synthetic-data research.

    Important:
    These vital override thresholds are prototype engineering rules,
    not validated clinical thresholds.
    """

    import numpy as np
    import pandas as pd

    out = df.copy()
    out = out.sort_values([patient_col, time_col]).reset_index(drop=True)

    required_cols = [patient_col, time_col, prob_col, alert_col]
    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns for GSS v2: {missing}")

    # Initialize output columns
    out[gss_alert_col] = False
    out[suppressed_col] = False
    out[escalation_col] = False
    out[escalation_reason_col] = "no_alert"

    # Work patient-by-patient
    for patient_id, group in out.groupby(patient_col, sort=False):
        last_alert_risk = None
        last_alert_vitals = None
        suppression_active = False

        for idx in group.index:
            is_alert_episode = bool(out.at[idx, alert_col])

            if not is_alert_episode:
                out.at[idx, escalation_reason_col] = "no_alert"
                continue

            current_risk = out.at[idx, prob_col]

            current_vitals = {
                "spo2": out.at[idx, "spo2"] if "spo2" in out.columns else np.nan,
                "sbp": out.at[idx, "sbp"] if "sbp" in out.columns else np.nan,
                "resp_rate": out.at[idx, "resp_rate"] if "resp_rate" in out.columns else np.nan,
                "heart_rate": out.at[idx, "heart_rate"] if "heart_rate" in out.columns else np.nan,
                "temperature": out.at[idx, "temperature"] if "temperature" in out.columns else np.nan,
            }

            # First alert always fires
            if last_alert_risk is None:
                out.at[idx, gss_alert_col] = True
                out.at[idx, escalation_col] = True
                out.at[idx, escalation_reason_col] = "initial_alert"

                last_alert_risk = current_risk
                last_alert_vitals = current_vitals
                suppression_active = True
                continue

            reasons = []

            # Risk-based override
            if current_risk >= escalation_threshold:
                reasons.append("escalation_threshold")

            if current_risk - last_alert_risk >= risk_delta:
                reasons.append("risk_delta_break")

            # Event override
            if event_col in out.columns and bool(out.at[idx, event_col]):
                reasons.append("event_now_override")

            # Vital-based overrides
            if last_alert_vitals is not None:
                if (
                    not pd.isna(current_vitals["spo2"])
                    and not pd.isna(last_alert_vitals["spo2"])
                    and last_alert_vitals["spo2"] - current_vitals["spo2"] >= spo2_drop
                ):
                    reasons.append("spo2_drop")

                if (
                    not pd.isna(current_vitals["sbp"])
                    and not pd.isna(last_alert_vitals["sbp"])
                    and last_alert_vitals["sbp"] - current_vitals["sbp"] >= sbp_drop
                ):
                    reasons.append("sbp_drop")

                if (
                    not pd.isna(current_vitals["resp_rate"])
                    and not pd.isna(last_alert_vitals["resp_rate"])
                    and current_vitals["resp_rate"] - last_alert_vitals["resp_rate"] >= resp_rate_rise
                ):
                    reasons.append("resp_rate_rise")

                if (
                    not pd.isna(current_vitals["heart_rate"])
                    and not pd.isna(last_alert_vitals["heart_rate"])
                    and current_vitals["heart_rate"] - last_alert_vitals["heart_rate"] >= heart_rate_rise
                ):
                    reasons.append("heart_rate_rise")

                if (
                    not pd.isna(current_vitals["temperature"])
                    and not pd.isna(last_alert_vitals["temperature"])
                    and abs(current_vitals["temperature"] - 37.0)
                    - abs(last_alert_vitals["temperature"] - 37.0)
                    >= temp_rise
                ):
                    reasons.append("temp_more_abnormal")

            # Fire or suppress
            if reasons:
                out.at[idx, gss_alert_col] = True
                out.at[idx, escalation_col] = True
                out.at[idx, escalation_reason_col] = "+".join(reasons)

                last_alert_risk = current_risk
                last_alert_vitals = current_vitals
                suppression_active = True
            else:
                out.at[idx, gss_alert_col] = False
                out.at[idx, suppressed_col] = True
                out.at[idx, escalation_col] = False
                out.at[idx, escalation_reason_col] = "suppressed_stable"

    return out

# ============================================================
# Temporal Persistence Helper (GSS v2.3)
# ============================================================

def is_persistent(df, idx, col, threshold, direction, steps=2):
    """
    Check if a signal persists over N timesteps.
    direction: "rise" or "drop"
    """
    if idx - (steps - 1) < 0:
        return False

    values = df.loc[idx - (steps - 1):idx, col]

    if direction == "rise":
        return all(values >= threshold)
    elif direction == "drop":
        return all(values <= threshold)

    return False


def apply_gss_with_delta_override(
    df,
    prob_col="pred_proba",
    alert_col="alert_episode_flag",
    patient_col="patient_id",
    time_col="timestamp",
    event_col="event_now",
    gss_alert_col="gss_v22_alert",
    suppressed_col="gss_v22_suppressed",
    escalation_col="gss_v22_escalation",
    escalation_reason_col="gss_v22_escalation_reason",
    risk_delta=0.013,
    escalation_threshold=0.997,
    spo2_delta_drop=-0.5,
    sbp_delta_drop=-2.0,
    resp_rate_delta_rise=1.0,
    heart_rate_delta_rise=2.0,
    temp_delta_worsen=0.2,
    min_delta_signals=2,
    delta_persistence_steps = 1,
):
    """
    GSS v2.2 — Delta-Based Override with Multi-Signal Confirmation.

    Fires when:
    - first alert occurs
    - risk meaningfully escalates
    - escalation threshold is crossed
    - at least min_delta_signals physiologic delta signals worsen
    - event_now occurs as a fallback only
    """

    import pandas as pd

    out = df.copy()
    out = out.sort_values([patient_col, time_col]).reset_index(drop=True)

    required_cols = [patient_col, time_col, prob_col, alert_col]
    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out[gss_alert_col] = False
    out[suppressed_col] = False
    out[escalation_col] = False
    out[escalation_reason_col] = "no_alert"

    delta_signal_names = [
        "spo2_delta_drop",
        "sbp_delta_drop",
        "resp_rate_delta_rise",
        "heart_rate_delta_rise",
        "temp_delta_worsen",
    ]

    risk_signal_names = [
        "risk_delta_break",
        "escalation_threshold",
    ]

    for patient_id, group in out.groupby(patient_col, sort=False):
        last_alert_risk = None

        for idx in group.index:
            is_alert = bool(out.at[idx, alert_col])

            if not is_alert:
                continue

            current_risk = out.at[idx, prob_col]
            reasons = []

            if last_alert_risk is None:
                reasons.append("initial_alert")

            else:
                if current_risk >= escalation_threshold:
                    reasons.append("escalation_threshold")

                if current_risk - last_alert_risk >= risk_delta:
                    reasons.append("risk_delta_break")

                if "spo2_delta" in out.columns:
                    if is_persistent(out, idx, "spo2_delta", spo2_delta_drop, "drop", steps = delta_persistence_steps):
                        reasons.append("spo2_delta_drop")

                if "sbp_delta" in out.columns:
                    if is_persistent(out, idx, "sbp_delta", sbp_delta_drop, "drop", steps = delta_persistence_steps):
                        reasons.append("sbp_delta_drop")

                if "resp_rate_delta" in out.columns:
                   if is_persistent(out, idx, "resp_rate_delta", resp_rate_delta_rise, "rise", steps = delta_persistence_steps):
                       reasons.append("resp_rate_delta_rise")
                       
                if "heart_rate_delta" in out.columns:
                    if is_persistent(out, idx, "heart_rate_delta", heart_rate_delta_rise, "rise", steps = delta_persistence_steps):
                        reasons.append("heart_rate_delta_rise")

                if "temperature_delta" in out.columns:
                    if is_persistent(out, idx, "temperature_delta", temp_delta_worsen, "rise", steps = delta_persistence_steps):
                        reasons.append("temp_delta_worsen")

            

                if event_col in out.columns and bool(out.at[idx, event_col]):
                    if not reasons:
                        reasons.append("event_now_override")

            delta_signals = [r for r in reasons if r in delta_signal_names]
            risk_signals = [r for r in reasons if r in risk_signal_names]

            should_fire = (
                "initial_alert" in reasons
                or len(risk_signals) > 0
                or len(delta_signals) >= min_delta_signals
            )

            if should_fire:
                out.at[idx, gss_alert_col] = True
                out.at[idx, escalation_col] = True

                if "initial_alert" in reasons:
                    out.at[idx, escalation_reason_col] = "initial_alert"
                else:
                    out.at[idx, escalation_reason_col] = "+".join(risk_signals + delta_signals)

                last_alert_risk = current_risk

            elif "event_now_override" in reasons:
                out.at[idx, gss_alert_col] = True
                out.at[idx, escalation_col] = True
                out.at[idx, escalation_reason_col] = "event_now_fallback"
                last_alert_risk = current_risk

            else:
                out.at[idx, gss_alert_col] = False
                out.at[idx, suppressed_col] = True
                out.at[idx, escalation_reason_col] = "suppressed_stable"

    return out
