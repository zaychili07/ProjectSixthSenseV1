import pandas as pd


def impute_patient_time_series(
    patient_df: pd.DataFrame,
    signal_cols: list[str],
) -> pd.DataFrame:
    """
    Impute one patient's time series using forward fill then backward fill
    within that patient only.
    """
    out = patient_df.copy()

    required_cols = ["patient_id", "timestamp"]
    missing_required = [col for col in required_cols if col not in out.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    out = out.sort_values("timestamp").reset_index(drop=True)

    cols_to_impute = [col for col in signal_cols if col in out.columns]
    out[cols_to_impute] = out[cols_to_impute].ffill().bfill()

    return out


def impute_all_patients(
    df: pd.DataFrame,
    signal_cols: list[str],
) -> pd.DataFrame:
    """
    Apply imputation to all patients separately and concatenate results.
    """
    required_cols = ["patient_id", "timestamp"]
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    imputed = []

    for _, patient_df in df.groupby("patient_id", sort=False):
        if patient_df.empty:
            continue

        patient_imputed = impute_patient_time_series(
            patient_df=patient_df,
            signal_cols=signal_cols,
        )
        imputed.append(patient_imputed)

    if not imputed:
        return pd.DataFrame(columns=df.columns)

    out = pd.concat(imputed, ignore_index=True)
    out = out.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    return out
