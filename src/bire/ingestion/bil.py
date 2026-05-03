#====================================
# BioSignal Intake Layer (BIL) v2
#====================================

from __future__ import annotations

from typing import Dict, Optional, Tuple
import pandas as pd


CANONICAL_BIRE_COLUMNS = [
    "patient_id",
    "timestamp",
    "heart_rate",
    "resp_rate",
    "spo2",
    "temperature",
    "sbp",
    "dbp",
]

REQUIRED_COLUMNS = CANONICAL_BIRE_COLUMNS.copy()


COMMON_COLUMN_ALIASES = {
    # patient/time
    "subject_id": "patient_id",
    "stay_id": "patient_id",
    "hadm_id": "patient_id",
    "charttime": "timestamp",
    "time": "timestamp",
    "datetime": "timestamp",

    # vitals
    "hr": "heart_rate",
    "heartrate": "heart_rate",
    "heart rate": "heart_rate",
    "rr": "resp_rate",
    "respiratory_rate": "resp_rate",
    "resp rate": "resp_rate",
    "respiration": "resp_rate",
    "spo2": "spo2",
    "o2sat": "spo2",
    "oxygen_saturation": "spo2",
    "oxygen saturation": "spo2",
    "temp": "temperature",
    "temperature_c": "temperature",
    "sbp": "sbp",
    "systolic_bp": "sbp",
    "systolic blood pressure": "sbp",
    "dbp": "dbp",
    "diastolic_bp": "dbp",
    "diastolic blood pressure": "dbp",
}


def normalize_column_name(col: str) -> str:
    """
    Normalize raw column names into a predictable lowercase snake-like form.
    """
    return (
        str(col)
        .strip()
        .lower()
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def standardize_columns(
    df: pd.DataFrame,
    column_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Standardize external column names into canonical BIRE names.

    column_map should map source columns to BIRE columns.
    Example:
        {"HR": "heart_rate", "SpO2": "spo2"}
    """
    out = df.copy()

    out.columns = [normalize_column_name(c) for c in out.columns]

    normalized_aliases = {
        normalize_column_name(k): v for k, v in COMMON_COLUMN_ALIASES.items()
    }

    rename_map = {}
    for col in out.columns:
        if col in normalized_aliases:
            rename_map[col] = normalized_aliases[col]

    if column_map:
        user_map = {
            normalize_column_name(k): v for k, v in column_map.items()
        }
        rename_map.update(user_map)

    out = out.rename(columns=rename_map)

    return out


def validate_required_columns(df: pd.DataFrame) -> None:
    """
    Ensure canonical BIRE columns exist after standardization.
    """
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing_cols:
        raise ValueError(
            "BIL validation failed. Missing required canonical columns: "
            f"{missing_cols}. Available columns: {list(df.columns)}"
        )


def enforce_temporal_integrity(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Enforce leakage-safe temporal ordering per patient.
    """
    out = df.copy()

    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")

    if out[time_col].isna().any():
        bad_count = int(out[time_col].isna().sum())
        raise ValueError(f"BIL found {bad_count} invalid timestamps.")

    out = out.sort_values([patient_col, time_col]).reset_index(drop=True)

    time_diff = out.groupby(patient_col)[time_col].diff()

    if (time_diff < pd.Timedelta(0)).any():
        raise ValueError(
            "Temporal integrity failure: backward timestamp movement detected."
        )

    return out


def handle_duplicate_patient_timestamps(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    strategy: str = "mean",
) -> pd.DataFrame:
    """
    Handle duplicate patient/timestamp rows.

    strategy:
        - "error": raise error if duplicates exist
        - "first": keep first row
        - "mean": average numeric vitals per duplicate timestamp
    """
    out = df.copy()

    dup_count = int(out.duplicated(subset=[patient_col, time_col]).sum())

    if dup_count == 0:
        return out

    if strategy == "error":
        raise ValueError(
            f"BIL found {dup_count} duplicate patient/timestamp rows."
        )

    if strategy == "first":
        return (
            out
            .drop_duplicates(subset=[patient_col, time_col], keep="first")
            .reset_index(drop=True)
        )

    if strategy == "mean":
        numeric_cols = [
            c for c in CANONICAL_BIRE_COLUMNS
            if c not in [patient_col, time_col]
        ]

        other_cols = [
            c for c in out.columns
            if c not in numeric_cols + [patient_col, time_col]
        ]

        agg_dict = {c: "mean" for c in numeric_cols if c in out.columns}
        agg_dict.update({c: "first" for c in other_cols})

        return (
            out
            .groupby([patient_col, time_col], as_index=False)
            .agg(agg_dict)
            .reset_index(drop=True)
        )

    raise ValueError(f"Unsupported duplicate strategy: {strategy}")


def build_bil_metadata(
    df: pd.DataFrame,
    source_name: str,
    schema_version: str = "BIL_v2",
) -> dict:
    """
    Build lightweight ingestion metadata for lineage/debugging.
    """
    return {
        "source_name": source_name,
        "schema_version": schema_version,
        "rows": int(len(df)),
        "patients": int(df["patient_id"].nunique()),
        "start_time": str(df["timestamp"].min()),
        "end_time": str(df["timestamp"].max()),
        "missing_values_total": int(df[CANONICAL_BIRE_COLUMNS].isna().sum().sum()),
        "columns": list(df.columns),
    }


def load_external_to_bire_format(
    file_path: str,
    column_map: Optional[Dict[str, str]] = None,
    source_name: str = "external_csv",
    duplicate_strategy: str = "mean",
    return_metadata: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, dict]:
    """
    Load an external CSV-like dataset into canonical BIRE format.

    BIL v2 performs:
    - column normalization
    - source-to-canonical column mapping
    - required schema validation
    - timestamp parsing
    - duplicate patient/timestamp handling
    - temporal integrity enforcement
    - lightweight metadata capture

    Notes
    -----
    BIL does NOT create lag, delta, rolling, target, or future-aware features.
    Those remain in the leakage-safe Cycle 1 pipeline.
    """
    df = pd.read_csv(file_path)

    df = standardize_columns(df, column_map=column_map)

    validate_required_columns(df)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = handle_duplicate_patient_timestamps(
        df,
        patient_col="patient_id",
        time_col="timestamp",
        strategy=duplicate_strategy,
    )

    df = enforce_temporal_integrity(
        df,
        patient_col="patient_id",
        time_col="timestamp",
    )

    # Keep canonical columns first, then preserve any extras.
    extra_cols = [c for c in df.columns if c not in CANONICAL_BIRE_COLUMNS]
    df = df[CANONICAL_BIRE_COLUMNS + extra_cols]

    metadata = build_bil_metadata(df, source_name=source_name)

    if return_metadata:
        return df, metadata

    return df


# Backward-compatible wrapper
def load_csv_to_bire_format(file_path: str) -> pd.DataFrame:
    return load_external_to_bire_format(file_path)
