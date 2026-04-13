<<<<<<< HEAD
import pandas as pd


def drop_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop structurally invalid rows (missing key identifiers).
    Does NOT handle signal validity — that belongs to validate_ranges.
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
    This preserves rows while marking bad measurements for imputation.
    """
    out = df.copy()

    for col, (low, high) in valid_ranges.items():
        if col in out.columns:
            invalid_mask = ~out[col].isna() & ~out[col].between(low, high)
            out.loc[invalid_mask, col] = pd.NA

    return out


def deduplicate_patient_timestamps(
    df: pd.DataFrame,
    signal_cols: list[str],
) -> pd.DataFrame:
    """
    Resolve duplicate (patient_id, timestamp) rows.
    Signal columns are averaged; others take first value.
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
=======
import pandas as pd


def drop_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop structurally invalid rows (missing key identifiers).
    Does NOT handle signal validity — that belongs to validate_ranges.
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
    This preserves rows while marking bad measurements for imputation.
    """
    out = df.copy()

    for col, (low, high) in valid_ranges.items():
        if col in out.columns:
            invalid_mask = ~out[col].isna() & ~out[col].between(low, high)
            out.loc[invalid_mask, col] = pd.NA

    return out


def deduplicate_patient_timestamps(
    df: pd.DataFrame,
    signal_cols: list[str],
) -> pd.DataFrame:
    """
    Resolve duplicate (patient_id, timestamp) rows.
    Signal columns are averaged; others take first value.
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
>>>>>>> 75f3642 (Implement validator module for pipeline consistency)
