#====================================
# BioSignal Intake Layer (BIL)
#====================================

import pandas as pd

REQUIRED_COLUMNS = [
    "patient_id",
    "timestamp",
    "heart_rate",
    "resp_rate",
    "spo2",
    "temperature",
    "sbp",
    "dbp",
]


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

    # Reset index
    df = df.reset_index(drop=True)

    return df
