import pandas as pd

from bire.config import SIGNAL_COLS, VALID_RANGES, RESAMPLE_FREQ, WINDOW_SIZE
from bire.data.validators import (
    drop_invalid_rows,
    validate_ranges,
    deduplicate_patient_timestamps,
    time_aware_patient_split,
)
from bire.data.temporal_alignment import align_all_patients
from bire.data.imputers import impute_all_patients
from bire.features.feature_engineering import (
    add_features_all_patients,
    get_feature_columns,
)
from bire.models.logistic import build_logistic_model
from bire.evaluation.alerts import apply_alert_logic


REQUIRED_PIPELINE_COLS = ["patient_id", "timestamp"]


def _validate_required_columns(df, required_cols, stage):
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns after {stage}: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )


def _debug_cols(stage, df, enabled=False):
    if not enabled:
        return

    print(f"\n[{stage}]")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head(2))


def run_cycle1(
    input_path: str,
    output_path: str | None = None,
    debug: bool = False,
):
    """
    Run Cycle I preprocessing and feature engineering.

    This pipeline:
    - loads raw vitals
    - validates rows/ranges
    - deduplicates patient timestamps
    - aligns patient time series
    - imputes missing values
    - engineers temporal features
    - preserves patient_id and timestamp for downstream labeling/modeling
    """

    print("Loading data...")
    df = pd.read_csv(input_path)

    _validate_required_columns(df, REQUIRED_PIPELINE_COLS, "load")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    _debug_cols("After load", df, enabled=debug)

    print("Cleaning data...")
    df = drop_invalid_rows(df)
    _validate_required_columns(df, REQUIRED_PIPELINE_COLS, "drop_invalid_rows")
    _debug_cols("After drop_invalid_rows", df, enabled=debug)

    df = validate_ranges(df, VALID_RANGES)
    _validate_required_columns(df, REQUIRED_PIPELINE_COLS, "validate_ranges")
    _debug_cols("After validate_ranges", df, enabled=debug)

    df = deduplicate_patient_timestamps(df, SIGNAL_COLS)
    _validate_required_columns(
        df,
        REQUIRED_PIPELINE_COLS,
        "deduplicate_patient_timestamps",
    )
    _debug_cols("After deduplicate_patient_timestamps", df, enabled=debug)

    print("Aligning time series...")
    df = align_all_patients(df, SIGNAL_COLS, RESAMPLE_FREQ)
    _validate_required_columns(df, REQUIRED_PIPELINE_COLS, "align_all_patients")
    _debug_cols("After align_all_patients", df, enabled=debug)

    print("Imputing missing values...")
    df = impute_all_patients(df, SIGNAL_COLS)
    _validate_required_columns(df, REQUIRED_PIPELINE_COLS, "impute_all_patients")
    _debug_cols("After impute_all_patients", df, enabled=debug)

    print("Engineering features...")
    df = add_features_all_patients(df, SIGNAL_COLS, WINDOW_SIZE)
    _validate_required_columns(
        df,
        REQUIRED_PIPELINE_COLS,
        "add_features_all_patients",
    )
    _debug_cols("After add_features_all_patients", df, enabled=debug)

    feature_cols = get_feature_columns(df)
    print(f"Generated {len(feature_cols)} feature columns.")

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved processed feature dataset to {output_path}")

    print("Cycle I complete ✅")

    return df


def run_bire_modeling(df, feature_cols, threshold=0.5, window=3):
    """
    Train and evaluate the final BIRE logistic model on a time-aware split.
    """

    required_cols = ["patient_id", "timestamp", "target"]
    _validate_required_columns(df, required_cols, "run_bire_modeling input")

    missing_features = [col for col in feature_cols if col not in df.columns]

    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    train_df, test_df = time_aware_patient_split(df)

    X_train = train_df[feature_cols]
    y_train = train_df["target"]

    X_test = test_df[feature_cols]

    model = build_logistic_model()
    model.fit(X_train, y_train)

    test_df = test_df.copy()
    test_df["pred_proba"] = model.predict_proba(X_test)[:, 1]
    test_df = apply_alert_logic(test_df, threshold=threshold, window=window)

    return model, train_df, test_df


if __name__ == "__main__":
    import os

    input_file = os.getenv(
        "BIRE_INPUT_PATH",
        "data/raw/bire_mock_vitals.csv",
    )

    output_file = os.getenv(
        "BIRE_OUTPUT_PATH",
        "data/processed/bire_cycle1_features.csv",
    )

    print(f"Using input: {input_file}")
    print(f"Saving output to: {output_file}")

    run_cycle1(input_file, output_file, debug=True)