from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_SIGNAL_COLS = (
    "heart_rate",
    "resp_rate",
    "spo2",
    "temperature",
    "sbp",
    "dbp",
)


def inject_random_missingness(
    df: pd.DataFrame,
    signal_cols: tuple[str, ...] = DEFAULT_SIGNAL_COLS,
    missing_rate: float = 0.08,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Randomly inject missing values into physiologic signal columns.
    """
    out = df.copy()
    rng = np.random.default_rng(random_state)

    existing = [c for c in signal_cols if c in out.columns]

    for col in existing:
        mask = rng.random(len(out)) < missing_rate
        out.loc[mask, col] = np.nan

    return out


def inject_block_dropouts(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    signal_cols: tuple[str, ...] = DEFAULT_SIGNAL_COLS,
    dropout_rate: float = 0.05,
    max_block_size: int = 6,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Simulate monitor/device dropout by removing consecutive blocks of signal data.
    """
    out = df.copy()
    rng = np.random.default_rng(random_state)

    existing = [c for c in signal_cols if c in out.columns]

    for patient_id, group in out.groupby(patient_col):
        idx = group.index.to_list()

        if len(idx) < 2:
            continue

        n_blocks = max(1, int(len(idx) * dropout_rate / max_block_size))

        for _ in range(n_blocks):
            start_pos = rng.integers(0, len(idx))
            block_size = rng.integers(1, max_block_size + 1)
            block_idx = idx[start_pos : start_pos + block_size]

            if not block_idx:
                continue

            selected_signal = rng.choice(existing)
            out.loc[block_idx, selected_signal] = np.nan

    return out


def inject_sensor_noise(
    df: pd.DataFrame,
    signal_noise_scale: dict | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Add realistic random sensor noise to vitals.
    """
    out = df.copy()
    rng = np.random.default_rng(random_state)

    signal_noise_scale = signal_noise_scale or {
        "heart_rate": 6.0,
        "resp_rate": 2.5,
        "spo2": 1.5,
        "temperature": 0.4,
        "sbp": 8.0,
        "dbp": 5.0,
    }

    for col, scale in signal_noise_scale.items():
        if col in out.columns:
            noise = rng.normal(0, scale, size=len(out))
            out[col] = out[col] + noise

    return out


def inject_typo_outliers(
    df: pd.DataFrame,
    signal_cols: tuple[str, ...] = DEFAULT_SIGNAL_COLS,
    typo_rate: float = 0.01,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Simulate human-entry errors or impossible device readings.
    """
    out = df.copy()
    rng = np.random.default_rng(random_state)

    existing = [c for c in signal_cols if c in out.columns]

    for col in existing:
        mask = rng.random(len(out)) < typo_rate

        if not mask.any():
            continue

        if col == "heart_rate":
            out.loc[mask, col] = rng.choice([0, 25, 250, 820], size=mask.sum())
        elif col == "resp_rate":
            out.loc[mask, col] = rng.choice([0, 3, 70, 120], size=mask.sum())
        elif col == "spo2":
            out.loc[mask, col] = rng.choice([20, 45, 101, 180], size=mask.sum())
        elif col == "temperature":
            out.loc[mask, col] = rng.choice([25, 30, 43, 73], size=mask.sum())
        elif col == "sbp":
            out.loc[mask, col] = rng.choice([0, 35, 260, 999], size=mask.sum())
        elif col == "dbp":
            out.loc[mask, col] = rng.choice([0, 20, 160, 500], size=mask.sum())

    return out


def inject_contradictory_signals(
    df: pd.DataFrame,
    contradiction_rate: float = 0.03,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Simulate physiologically confusing patterns.

    Example:
    - respiratory rate rises while SpO2 appears stable
    - blood pressure appears normal while heart rate spikes
    """
    out = df.copy()
    rng = np.random.default_rng(random_state)

    mask = rng.random(len(out)) < contradiction_rate

    if "resp_rate" in out.columns and "spo2" in out.columns:
        out.loc[mask, "resp_rate"] = out.loc[mask, "resp_rate"] + rng.normal(10, 3, mask.sum())
        out.loc[mask, "spo2"] = out.loc[mask, "spo2"] + rng.normal(1, 0.5, mask.sum())

    if "heart_rate" in out.columns and "sbp" in out.columns:
        out.loc[mask, "heart_rate"] = out.loc[mask, "heart_rate"] + rng.normal(35, 8, mask.sum())
        out.loc[mask, "sbp"] = out.loc[mask, "sbp"] + rng.normal(5, 4, mask.sum())

    return out


def inject_false_recovery_pattern(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    recovery_rate: float = 0.05,
    recovery_window: int = 4,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Simulate temporary improvement followed by renewed instability.
    """
    out = df.copy()
    rng = np.random.default_rng(random_state)

    out = out.sort_values([patient_col, time_col]).copy()

    for patient_id, group in out.groupby(patient_col):
        idx = group.index.to_list()

        if len(idx) < recovery_window * 3:
            continue

        if rng.random() > recovery_rate:
            continue

        start = rng.integers(0, len(idx) - recovery_window)
        block = idx[start : start + recovery_window]

        if "spo2" in out.columns:
            out.loc[block, "spo2"] = out.loc[block, "spo2"] + rng.normal(4, 1, len(block))

        if "sbp" in out.columns:
            out.loc[block, "sbp"] = out.loc[block, "sbp"] + rng.normal(10, 3, len(block))

        if "heart_rate" in out.columns:
            out.loc[block, "heart_rate"] = out.loc[block, "heart_rate"] - rng.normal(10, 3, len(block))

    return out


def add_messiness_flags(
    original_df: pd.DataFrame,
    messy_df: pd.DataFrame,
    signal_cols: tuple[str, ...] = DEFAULT_SIGNAL_COLS,
) -> pd.DataFrame:
    """
    Add lightweight flags describing messiness after corruption.
    """
    out = messy_df.copy()
    existing = [c for c in signal_cols if c in out.columns]

    out["messy_missing_signal_count"] = out[existing].isna().sum(axis=1)
    out["messy_any_missing_signal"] = out["messy_missing_signal_count"] > 0

    out["messy_outlier_flag"] = False

    if "heart_rate" in out.columns:
        out["messy_outlier_flag"] |= (out["heart_rate"] < 30) | (out["heart_rate"] > 220)

    if "resp_rate" in out.columns:
        out["messy_outlier_flag"] |= (out["resp_rate"] < 5) | (out["resp_rate"] > 60)

    if "spo2" in out.columns:
        out["messy_outlier_flag"] |= (out["spo2"] < 50) | (out["spo2"] > 100)

    if "temperature" in out.columns:
        out["messy_outlier_flag"] |= (out["temperature"] < 30) | (out["temperature"] > 43)

    if "sbp" in out.columns:
        out["messy_outlier_flag"] |= (out["sbp"] < 50) | (out["sbp"] > 250)

    if "dbp" in out.columns:
        out["messy_outlier_flag"] |= (out["dbp"] < 25) | (out["dbp"] > 150)

    out["messy_data_flag"] = out["messy_any_missing_signal"] | out["messy_outlier_flag"]

    return out


def generate_incomplete_messy_stress_data(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    signal_cols: tuple[str, ...] = DEFAULT_SIGNAL_COLS,
    missing_rate: float = 0.08,
    dropout_rate: float = 0.05,
    typo_rate: float = 0.01,
    contradiction_rate: float = 0.03,
    recovery_rate: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a messy, incomplete, more realistic stress-test dataset.

    This is designed to challenge BIRE confidence and uncertainty logic.
    """

    messy = df.copy()

    messy = inject_random_missingness(
        messy,
        signal_cols=signal_cols,
        missing_rate=missing_rate,
        random_state=random_state,
    )

    messy = inject_block_dropouts(
        messy,
        patient_col=patient_col,
        signal_cols=signal_cols,
        dropout_rate=dropout_rate,
        random_state=random_state + 1,
    )

    messy = inject_sensor_noise(
        messy,
        random_state=random_state + 2,
    )

    messy = inject_typo_outliers(
        messy,
        signal_cols=signal_cols,
        typo_rate=typo_rate,
        random_state=random_state + 3,
    )

    messy = inject_contradictory_signals(
        messy,
        contradiction_rate=contradiction_rate,
        random_state=random_state + 4,
    )

    messy = inject_false_recovery_pattern(
        messy,
        patient_col=patient_col,
        time_col=time_col,
        recovery_rate=recovery_rate,
        random_state=random_state + 5,
    )

    messy = add_messiness_flags(
        original_df=df,
        messy_df=messy,
        signal_cols=signal_cols,
    )

    messy["stress_dataset"] = True
    messy["stress_random_state"] = random_state

    return messy