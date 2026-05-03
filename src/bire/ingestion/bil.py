#====================================
# BioSignal Intake Layer (BIL)
#====================================

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


def load_csv_to_bire_format(file_path: str) -> pd.DataFrame:
    """
    Load external CSV and convert to BIRE-compatible format.
    """

    df = pd.read_csv(file_path)

    # Basic column normalization (lowercase)
    df.columns = [c.lower().strip() for c in df.columns]

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Timestamp conversion
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Sort for temporal integrity
    df = df.sort_values(["patient_id", "timestamp"])

    # leakage enforcement!!! 
    df = enforce_temporal_integrity(df)
    
    # Reset index
    df = df.reset_index(drop=True)

    return df

# this wil enforce data leakage and give it guide lines
def enforce_temporal_integrity(df, patient_col="patient_id", time_col="timestamp"):
    """
    Ensures strict temporal ordering per patient.
    """

    df = df.sort_values([patient_col, time_col]).reset_index(drop=True)

    # Check for backward time jumps (leakage indicator)
    time_diff = df.groupby(patient_col)[time_col].diff()

    if (time_diff < pd.Timedelta(0)).any():
        raise ValueError("Temporal leakage detected: timestamps not strictly increasing")

    return df

def assert_no_duplicate_timestamps(df, patient_col="patient_id", time_col="timestamp"):
    dupes = df.duplicated(subset=[patient_col, time_col]).sum()
    if dupes > 0:
        raise ValueError(f"Duplicate timestamps detected: {dupes}")
