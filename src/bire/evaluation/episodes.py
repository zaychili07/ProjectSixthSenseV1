import pandas as pd
# Used for chapter 29 - episode based alerting
def prepare_episode_dataframe(
    df,
    patient_col="patient_id",
    time_col="timestamp",
    alert_col="final_alert_with_velocity"
):
    """
    Prepares dataframe for episode construction.

    Ensures:
    - Sorting by patient and time
    - Boolean alert column
    - No missing critical columns
    """

    required_cols = [patient_col, time_col, alert_col]

    # Check required columns
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Sort for temporal consistency
    df = df.sort_values([patient_col, time_col]).copy()

    # Ensure alert column is boolean
    df[alert_col] = df[alert_col].astype(bool)

    return df

def build_alert_episodes(
    df,
    alert_col="final_alert_with_velocity",
    patient_col="patient_id",
    time_col="timestamp",
    cooldown_steps=12
):
    """
    Converts row-level alerts into episode-based alerts.

    Parameters
    ----------
    df : pd.DataFrame
    alert_col : str
    patient_col : str
    time_col : str
    cooldown_steps : int

    Returns
    -------
    pd.DataFrame with episode columns added
    """

    df = df.copy()

    # Initialize columns
    df["alert_episode_id"] = -1
    df["alert_episode_start"] = False
    df["alert_episode_active"] = False
    df["episode_alert"] = False
    df["episode_start_time"] = pd.NaT
    df["episode_end_time"] = pd.NaT

    episode_counter = 0

    # Process per patient
    for patient_id, group in df.groupby(patient_col):

        group = group.sort_values(time_col)

        current_episode = None
        cooldown_counter = 0

        for idx in group.index:

            alert = df.loc[idx, alert_col]

           
            # START NEW EPISODE
            if alert:
                if current_episode is None:
                    episode_counter += 1
                    current_episode = episode_counter

                    df.loc[idx, "alert_episode_start"] = True
                    df.loc[idx, "episode_alert"] = True
                    df.loc[idx, "episode_start_time"] = df.loc[idx, time_col]

                cooldown_counter = 0

            
           
            # CONTINUATION OR COOLDOWN
            else:
                if current_episode is not None:
                    cooldown_counter += 1

                    if cooldown_counter >= cooldown_steps:
                        # End episode
                        df.loc[idx, "episode_end_time"] = df.loc[idx, time_col]
                        current_episode = None
                        cooldown_counter = 0
             # MARK ACTIVE STATE
            if current_episode is not None:
                df.loc[idx, "alert_episode_id"] = current_episode
                df.loc[idx, "alert_episode_active"] = True

   
    # Forward fill episode_start_time
    df["episode_start_time"] = df.groupby("alert_episode_id")["episode_start_time"].ffill()

    
    # Compute duration (in minutes)
    df["episode_duration_minutes"] = (
        (df[time_col] - df["episode_start_time"])
        .dt.total_seconds() / 60
    )

    return df

def build_bms_aware_episodes(
    df,
    bms_mode_col="bms_mode",
    alert_col="final_alert_with_velocity",
    patient_col="patient_id",
    time_col="timestamp",
    cooldown_policy=None,
    default_cooldown=12
):
    """
    Build alert episodes using BMS-aware cooldown policy.

    Parameters
    ----------
    df : pd.DataFrame
    bms_mode_col : str
    alert_col : str
    patient_col : str
    time_col : str
    cooldown_policy : dict
        Mapping of BMS mode → cooldown steps
    default_cooldown : int

    Returns
    -------
    pd.DataFrame
    """

    import pandas as pd

    if cooldown_policy is None:
        cooldown_policy = {}

    if bms_mode_col not in df.columns:
        raise ValueError(f"Missing required column: {bms_mode_col}")

    parts = []

    for mode, mode_df in df.groupby(bms_mode_col):

        cooldown_steps = cooldown_policy.get(mode, default_cooldown)

        temp = build_alert_episodes(
            mode_df.copy(),
            alert_col=alert_col,
            patient_col=patient_col,
            time_col=time_col,
            cooldown_steps=cooldown_steps
        )

        temp["bms_episode_cooldown_steps"] = cooldown_steps
        temp["bms_episode_cooldown_minutes"] = cooldown_steps * 5

        parts.append(temp)

    out = (
        pd.concat(parts, axis=0)
        .sort_values([patient_col, time_col])
        .reset_index(drop=True)
    )

    return out
