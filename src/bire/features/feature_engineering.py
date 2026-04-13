import pandas as pd


def add_temporal_features(
    patient_df: pd.DataFrame,
    signal_cols: list[str],
    window_size: int = 6,
) -> pd.DataFrame:
    patient_df = patient_df.sort_values("timestamp").copy()
    present_signals = [c for c in signal_cols if c in patient_df.columns]

def add_features_all_patients(df, signal_cols, window_size=6):
    return (
        df.groupby("patient_id", group_keys=False)
          .apply(lambda x: add_temporal_features(x, signal_cols, window_size))
          .reset_index(drop=True)
    )
    for col in present_signals:
        patient_df[f"{col}_lag1"] = patient_df[col].shift(1)
        patient_df[f"{col}_lag2"] = patient_df[col].shift(2)

        patient_df[f"{col}_delta"] = patient_df[col].diff()

        shifted = patient_df[col].shift(1)

        patient_df[f"{col}_rolling_mean_{window_size}"] = (
            shifted.rolling(window=window_size, min_periods=1).mean()
        )
        patient_df[f"{col}_rolling_std_{window_size}"] = (
            shifted.rolling(window=window_size, min_periods=1).std()
        )
        patient_df[f"{col}_rolling_min_{window_size}"] = (
            shifted.rolling(window=window_size, min_periods=1).min()
        )
        patient_df[f"{col}_rolling_max_{window_size}"] = (
            shifted.rolling(window=window_size, min_periods=1).max()
        )

    return patient_df
