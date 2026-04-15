import pandas as pd


def drop_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop structurally invalid rows (missing key identifiers).
    """
    out = df.copy()

    required_cols = ["patient_id", "timestamp"]
    missing_required = [col for col in required_cols if col not in out.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    out = out.dropna(subset=required_cols)
    return out


def validate_ranges(df: pd.DataFrame, valid_ranges: dict) -> pd.DataFrame:
    """
    Replace out-of-range physiological values with NaN.
    """
    out = df.copy()

    for col, (low, high) in valid_ranges.items():
        if col in out.columns:
            invalid_mask = ~out[col].isna() & ~out[col].between(low, high)
            out.loc[invalid_mask, col] = pd.NA

    return out


def deduplicate_patient_timestamps(df: pd.DataFrame, signal_cols: list[str]) -> pd.DataFrame:
    """
    Resolve duplicate (patient_id, timestamp) rows.
    """
    out = df.copy()

    agg_dict = {}
    for col in out.columns:
        if col in ["patient_id", "timestamp"]:
            continue
        if col in signal_cols:
            agg_dict[col] = "mean"
        else:
            agg_dict[col] = "first"

    out = (
        out.groupby(["patient_id", "timestamp"], as_index=False)
        .agg(agg_dict)
        .sort_values(["patient_id", "timestamp"])
        .reset_index(drop=True)
    )

    return out

# =========================
# TEMPORAL VALIDATION (CHAPTER 13)
# =========================

def time_aware_patient_split(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    train_frac: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split each patient's timeline into earlier (train) and later (test) segments.

    This ensures we train on past data and evaluate on future data,
    mimicking real-world deployment.
    """
    if patient_col not in df.columns:
        raise ValueError(f"Missing column: {patient_col}")
    if time_col not in df.columns:
        raise ValueError(f"Missing column: {time_col}")

    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be between 0 and 1.")

    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    out = out.sort_values([patient_col, time_col])

    train_parts = []
    test_parts = []

    for patient_id, patient_df in out.groupby(patient_col):
        patient_df = patient_df.sort_values(time_col)
        n = len(patient_df)

        if n < 2:
            train_parts.append(patient_df)
            continue

        split_idx = max(1, int(n * train_frac))
        split_idx = min(split_idx, n - 1)

        train_parts.append(patient_df.iloc[:split_idx])
        test_parts.append(patient_df.iloc[split_idx:])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()

    return train_df, test_df


def summarize_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "target",
    patient_col: str = "patient_id",
) -> dict:
    """
    Provide a quick summary of the temporal split.
    """
    return {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_patients": train_df[patient_col].nunique() if patient_col in train_df else 0,
        "test_patients": test_df[patient_col].nunique() if patient_col in test_df else 0,
        "train_positive_rows": int(train_df[target_col].sum()) if target_col in train_df else 0,
        "test_positive_rows": int(test_df[target_col].sum()) if target_col in test_df else 0,
    }
