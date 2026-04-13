import pandas as pd

from bire.config import SIGNAL_COLS, VALID_RANGES, RESAMPLE_FREQ, WINDOW_SIZE
from bire.data.validators import (
    drop_invalid_rows,
    validate_ranges,
    deduplicate_patient_timestamps,
)
from bire.data.temporal_alignment import align_all_patients
from bire.data.imputers import impute_all_patients
from bire.features.feature_engineering import (
    add_features_all_patients,
    get_feature_columns,
)


def run_cycle1(input_path: str, output_path: str = None):
    print("Loading data...")
    df = pd.read_csv(input_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    print("Cleaning data...")
    df = drop_invalid_rows(df)
    df = validate_ranges(df, VALID_RANGES)
    df = deduplicate_patient_timestamps(df, SIGNAL_COLS)

    print("Aligning time series...")
    df = align_all_patients(df, SIGNAL_COLS, RESAMPLE_FREQ)

    print("Imputing missing values...")
    df = impute_all_patients(df, SIGNAL_COLS)

    print("Engineering features...")
    df = add_features_all_patients(df, SIGNAL_COLS, WINDOW_SIZE)

    feature_cols = get_feature_columns(df)
    print(f"Generated {len(feature_cols)} feature columns.")

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved processed feature dataset to {output_path}")

    print("Cycle I complete ✅")
    return df


if __name__ == "__main__":
    import os

    input_file = os.getenv("BIRE_INPUT_PATH", "data/raw/bire_mock_vitals.csv")
    output_file = os.getenv("BIRE_OUTPUT_PATH", "data/processed/bire_cycle1_features.csv")

    print(f"Using input: {input_file}")
    print(f"Saving output to: {output_file}")

    run_cycle1(input_file, output_file)
