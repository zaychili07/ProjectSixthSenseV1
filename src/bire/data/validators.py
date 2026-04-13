import pandas as pd


def drop_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    required_cols = ["patient_id", "timestamp"]
    missing_required = [col for col in required_cols if col not in out.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    out = out.dropna(subset=["patient_id", "timestamp"])
    return out


def validate_ranges(df: pd.DataFrame, valid_ranges: dict) -> pd.DataFrame:
    out = df.copy()

    for col, (low, high) in valid_ranges.items():
        if col in out.columns:
            out.loc[~out[col].isna() & ~out[col].between(low, high), col] = pd.NA

    return out


def deduplicate_patient_timestamps(
    df: pd.DataFrame,
    signal_cols: list[str],
) -> pd.DataFrame:
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
