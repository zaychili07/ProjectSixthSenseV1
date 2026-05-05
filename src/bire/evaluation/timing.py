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

def assign_episode_intelligence_type(
    df,
    timing_col="timing_category",
    output_col="episode_intelligence_type",
):
    """
    Assign higher-level BIRE intelligence type based on episode timing category.

    This separates early instability detection from actionable deterioration alerts.

    Mapping:
    - early_beyond_horizon_episode -> INSTABILITY_WARNING
    - true_predictive_episode -> DETERIORATION_ALERT
    - post_event_episode -> POST_EVENT_ALERT
    - no_event_episode -> NO_EVENT_WARNING
    """

    df = df.copy()

    if timing_col not in df.columns:
        raise ValueError(f"Missing required column: {timing_col}")

    mapping = {
        "early_beyond_horizon_episode": "INSTABILITY_WARNING",
        "true_predictive_episode": "DETERIORATION_ALERT",
        "post_event_episode": "POST_EVENT_ALERT",
        "no_event_episode": "NO_EVENT_WARNING",
    }

    df[output_col] = (
        df[timing_col]
        .map(mapping)
        .fillna("UNKNOWN_EPISODE_TYPE")
    )

    return df

def assign_episode_tier(
    df,
    intelligence_col="episode_intelligence_type",
    output_col="episode_tier",
):
    """
    Assign episode tiers with post-event monitoring policy.

    Mapping:
    - INSTABILITY_WARNING → WATCH
    - NO_EVENT_WARNING → WATCH
    - DETERIORATION_ALERT → ESCALATE
    - POST_EVENT_ALERT → MONITOR
    """

    df = df.copy()

    if intelligence_col not in df.columns:
        raise ValueError(f"Missing required column: {intelligence_col}")

    tier_map = {
        "INSTABILITY_WARNING": "WATCH",
        "NO_EVENT_WARNING": "WATCH",
        "DETERIORATION_ALERT": "ESCALATE",
        "POST_EVENT_ALERT": "MONITOR",
    }

    df[output_col] = (
        df[intelligence_col]
        .map(tier_map)
        .fillna("REVIEW")
    )

    return df
