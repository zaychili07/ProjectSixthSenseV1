import pandas as pd


def evaluate_episode_timing(
    df,
    patient_col="patient_id",
    time_col="timestamp",
    episode_alert_col="episode_alert",
    event_col="event_now",
    horizon_minutes=60,
):
    """
    Evaluate when alert episodes occur relative to deterioration events.

    Classifies episode starts as:
    - true_predictive_episode
    - early_beyond_horizon_episode
    - post_event_episode
    - no_event_episode
    """

    required_cols = [patient_col, time_col, episode_alert_col, event_col]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.sort_values([patient_col, time_col]).copy()

    # Episode starts only
    episode_starts = df[df[episode_alert_col] == True].copy()

    # First deterioration event per patient
    event_times = (
        df[df[event_col] == 1]
        .groupby(patient_col)[time_col]
        .min()
        .reset_index()
        .rename(columns={time_col: "event_time"})
    )

    # Attach event time
    episode_starts = episode_starts.merge(
        event_times,
        on=patient_col,
        how="left",
    )

    # Lead time: positive = episode before event
    episode_starts["lead_time_min"] = (
        (episode_starts["event_time"] - episode_starts[time_col])
        .dt.total_seconds()
        / 60
    )

    def classify_timing(row):
        if pd.isna(row["event_time"]):
            return "no_event_episode"

        if row["lead_time_min"] >= 0:
            if row["lead_time_min"] <= horizon_minutes:
                return "true_predictive_episode"
            return "early_beyond_horizon_episode"

        return "post_event_episode"

    episode_starts["timing_category"] = episode_starts.apply(
        classify_timing,
        axis=1,
    )

    return episode_starts
