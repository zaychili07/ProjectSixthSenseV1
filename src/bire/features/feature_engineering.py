import pandas as pd


def add_temporal_features(
    patient_df: pd.DataFrame,
    signal_cols: list[str],
    window_size: int = 6,
) -> pd.DataFrame:
    """
    Add simple rolling time-series features for a single patient.

    Parameters
    ----------
    patient_df : pd.DataFrame
        Data for one patient only. Must include 'timestamp' and signal columns.
    signal_cols : list[str]
        Physiological signal columns to transform.
    window_size : int
        Rolling window length in rows. Default 6 = 30 minutes for 5-minute bins.

    Returns
    -------
    pd.DataFrame
        Original patient dataframe with added temporal features.
    """
    patient_df = patient_df.sort_values("timestamp").copy()
    present_signals = [c for c in signal_cols if c in patient_df.columns]

    for col in present_signals:
        patient_df[f"{col}_delta"] = patient_df[col].diff()
        patient_df[f"{col}_rolling_mean_{window_size}"] = (
            patient_df[col].rolling(window=window_size, min_periods=1).mean()
        )
        patient_df[f"{col}_rolling_std_{window_size}"] = (
            patient_df[col].rolling(window=window_size, min_periods=1).std()
        )
        patient_df[f"{col}_rolling_min_{window_size}"] = (
            patient_df[col].rolling(window=window_size, min_periods=1).min()
        )
        patient_df[f"{col}_rolling_max_{window_size}"] = (
            patient_df[col].rolling(window=window_size, min_periods=1).max()
        )

    return patient_df


def add_features_all_patients(
    df: pd.DataFrame,
    signal_cols: list[str],
    window_size: int = 6,
) -> pd.DataFrame:
    """
    Apply temporal feature engineering patient-by-patient.
    """
    return (
        df.groupby("patient_id", group_keys=False)
        .apply(
            add_temporal_features,
            signal_cols=signal_cols,
            window_size=window_size,
            include_groups=False,
        )
        .reset_index(drop=True)
    )


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return modeling feature columns, excluding identifiers and metadata.
    """
    excluded = {"patient_id", "timestamp", "clinical_state"}
    return [c for c in df.columns if c not in excluded]
