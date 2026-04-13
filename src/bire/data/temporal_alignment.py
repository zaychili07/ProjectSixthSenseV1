import pandas as pd


def align_patient_time_series(
    patient_df: pd.DataFrame,
    signal_cols: list[str],
    resample_freq: str,
) -> pd.DataFrame:
    """
    Resample one patient's time series onto a regular grid.
    """
    out = patient_df.copy()

    required_cols = ["patient_id", "timestamp"]
    missing_required = [col for col in required_cols if col not in out.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")

    patient_id = out["patient_id"].iloc[0]

    keep_cols = [col for col in signal_cols if col in out.columns]

    out = (
        out.set_index("timestamp")[keep_cols]
        .resample(resample_freq)
        .mean()
        .reset_index()
    )

    out.insert(0, "patient_id", patient_id)
    return out


def align_all_patients(
    df: pd.DataFrame,
    signal_cols: list[str],
    resample_freq: str,
) -> pd.DataFrame:
    """
    Apply temporal alignment to all patients and concatenate results.
    """
    required_cols = ["patient_id", "timestamp"]
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    aligned = []

    for patient_id, patient_df in df.groupby("patient_id", sort=False):
        if patient_df.empty:
            continue

        patient_aligned = align_patient_time_series(
            patient_df=patient_df,
            signal_cols=signal_cols,
            resample_freq=resample_freq,
        )
        aligned.append(patient_aligned)

    if not aligned:
        return pd.DataFrame(columns=["patient_id", "timestamp"] + signal_cols)

    out = pd.concat(aligned, ignore_index=True)
    out = out.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    return out
