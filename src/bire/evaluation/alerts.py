import pandas as pd


def apply_alert_logic(df: pd.DataFrame, threshold: float = 0.5, window: int = 3) -> pd.DataFrame:
    """
    Apply persistence-based alert logic.

    An alert is triggered when predicted risk exceeds threshold
    for 'window' consecutive time steps.
    """
    df = df.copy().sort_values(["patient_id", "timestamp"])

    # Step 1: High-risk flag
    df["high_risk"] = (df["pred_proba"] >= threshold).astype(int)

    # Step 2: Rolling persistence check
    df["alert"] = (
        df.groupby("patient_id")["high_risk"]
        .rolling(window=window, min_periods=window)
        .sum()
        .reset_index(level=0, drop=True)
        .ge(window)
        .astype(int)
    )

    return df


def summarize_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize alerts per patient.
    """
    summary = (
        df.groupby("patient_id")
        .agg(
            n_rows=("patient_id", "size"),
            n_alerts=("alert", "sum"),
            max_risk=("pred_proba", "max"),
            mean_risk=("pred_proba", "mean"),
        )
        .reset_index()
    )

    return summary
