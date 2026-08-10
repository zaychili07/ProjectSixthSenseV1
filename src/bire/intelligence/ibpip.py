###############################################
# IBPIP — Individual Baseline Intelligence
###############################################

import numpy as np
import pandas as pd


SIGNAL_COLS = [
    "heart_rate",
    "resp_rate",
    "spo2",
    "sbp",
    "dbp",
    "temperature"
]


IBPIP_MODE_POLICY = {
    "ICU": {"warmup_steps": 12},
    "ER_ESI_1": {"warmup_steps": 12},
    "ER_ESI_2": {"warmup_steps": 9},
    "ER_ESI_3": {"warmup_steps": 6},
    "INPATIENT": {"warmup_steps": 6},
    "ER_ESI_4": {"warmup_steps": 4},
    "ER_ESI_5": {"warmup_steps": 3},
    "OUTPATIENT": {"warmup_steps": 3},
}

def _validate_ibpip_inputs(df, mode_col):
    required = ["patient_id", "timestamp", mode_col] + SIGNAL_COLS
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns for IBPIP: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )


def add_ibpip_features(df, mode_col="bms_mode"):
    df = df.copy()

    _validate_ibpip_inputs(df, mode_col)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

    ###############################################
    # Step 1 — Row index per patient
    ###############################################
    df["ibpip_step"] = df.groupby("patient_id").cumcount()

    ###############################################
    # Step 2 — Assign warmup steps per row
    ###############################################
    def get_warmup(mode):
        return IBPIP_MODE_POLICY.get(mode, {"warmup_steps": 6})["warmup_steps"]

    df["ibpip_warmup_steps"] = df[mode_col].apply(get_warmup)

    ###############################################
    # Step 3 — Warmup flag
    ###############################################
    df["ibpip_warmup"] = df["ibpip_step"] < df["ibpip_warmup_steps"]

    ###############################################
    # Step 4 — Rolling baseline (LEAKAGE SAFE)
    ###############################################
    for col in SIGNAL_COLS:

        shifted = df.groupby("patient_id")[col].shift(1)

        rolling_mean = shifted.groupby(df["patient_id"]).rolling(
            window=6, min_periods=3
        ).mean().reset_index(level=0, drop=True)

        rolling_std = shifted.groupby(df["patient_id"]).rolling(
            window=6, min_periods=3
        ).std().reset_index(level=0, drop=True)

        df[f"{col}_ibpip_mean"] = rolling_mean
        df[f"{col}_ibpip_std"] = rolling_std

        ###############################################
        # Step 5 — Deviation + Z-score
        ###############################################
        df[f"{col}_ibpip_dev"] = df[col] - df[f"{col}_ibpip_mean"]

        df[f"{col}_ibpip_z"] = df[f"{col}_ibpip_dev"] / (
            df[f"{col}_ibpip_std"] + 1e-6
        )

    ###############################################
    # Step 6 — IBPIP Score
    ###############################################
    z_cols = [f"{c}_ibpip_z" for c in SIGNAL_COLS]

    df["ibpip_score"] = df[z_cols].abs().mean(axis=1)
    df["ibpip_max_abs_z"] = df[z_cols].abs().max(axis=1)
    df["ibpip_n_abnormal_signals"] = (df[z_cols].abs() > 2).sum(axis=1)

    ###############################################
    # Step 7 — IBPIP State Logic
    ###############################################
    df["ibpip_state"] = "CAUTIOUS_BASELINE"

    # Warmup override
    df.loc[df["ibpip_warmup"], "ibpip_state"] = "WARMUP"

    ###############################################
    # Step 8 — Reassessment Logic (Recovery)
    ###############################################
    # If deviation is shrinking → reassessing

    df["ibpip_reassessing"] = False

    for col in SIGNAL_COLS:
        dev = df[f"{col}_ibpip_dev"]

        improving = dev.abs() < dev.abs().groupby(df["patient_id"]).shift(1)

        df["ibpip_reassessing"] = df["ibpip_reassessing"] | improving.fillna(False)

    df.loc[
        (~df["ibpip_warmup"]) & (df["ibpip_reassessing"]),
        "ibpip_state"
    ] = "REASSESSING"

    ###############################################
    # Step 9 — Baseline Ready Flag
    ###############################################
    df["ibpip_baseline_ready"] = ~df["ibpip_warmup"]

    return df
