import pandas as pd


def add_event_and_target_labels(
    df: pd.DataFrame,
    lookahead_steps: int = 12,
) -> pd.DataFrame:
    """
    Add current deterioration flag (`event_now`) and forward-looking target label (`target`).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing patient_id, timestamp, and vital columns.
    lookahead_steps : int, default=12
        Number of future 5-minute steps to scan for deterioration.
        12 steps = 60 minutes.

    Returns
    -------
    pd.DataFrame
        DataFrame with added `event_now` and `target` columns.
    """
    required_cols = {
        "patient_id",
        "timestamp",
        "spo2",
        "sbp",
        "heart_rate",
        "resp_rate",
        "temperature",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for labeling: {missing}")

    df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True).copy()

    df["event_now"] = (
        (df["spo2"] < 90) |
        (df["sbp"] < 90) |
        (df["heart_rate"] > 130) |
        (df["resp_rate"] > 30) |
        (df["temperature"] > 39) |
        (df["temperature"] < 35)
    ).astype(int)

    future_flags = [
        df.groupby("patient_id")["event_now"].shift(-k)
        for k in range(1, lookahead_steps + 1)
    ]

    df["target"] = (
        pd.concat(future_flags, axis=1)
        .fillna(0)
        .max(axis=1)
        .astype(int)
    )

    return df
