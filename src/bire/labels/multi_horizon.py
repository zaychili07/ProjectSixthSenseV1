import pandas as pd
import numpy as np


def build_multi_horizon_targets(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    event_col: str = "event_now",
    horizons: dict | None = None,
) -> pd.DataFrame:

    if horizons is None:
        horizons = {
            "target_15min": 3,
            "target_30min": 6,
            "target_60min": 12,
        }

    required_cols = [patient_col, time_col, event_col]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    out = out.sort_values([patient_col, time_col]).reset_index(drop=True)

    out[event_col] = out[event_col].fillna(0).astype(int)

    grouped = out.groupby(patient_col, group_keys=False)

    for target_name, steps in horizons.items():
        future_flags = []

        for step in range(1, steps + 1):
            future_flags.append(grouped[event_col].shift(-step))

        out[target_name] = (
            pd.concat(future_flags, axis=1)
            .max(axis=1)
            .fillna(0)
            .astype(int)
        )

    return out
