# THIS IS A SYNTHETIC GENERATOR I HAD TO ASK GEMMA 4 TO HELP GENERATE
"""
Synthetic ICU-Grade Data Generator for Project Sixth Sense / BIRE.

Purpose:
- Generate realistic synthetic time-series vital signs.
- Support BIRE Mode Selection:
    ICU, ER, Inpatient, Walk-in
- Simulate multiple deterioration modes:
    respiratory failure, septic shock, cardiac instability,
    hypotensive shock, fever/infection, mixed deterioration.

Important:
This is synthetic prototype data for ML system testing.
It is NOT validated clinical physiology and should not be used for real care.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SIGNAL_COLS = [
    "heart_rate",
    "resp_rate",
    "spo2",
    "temperature",
    "sbp",
    "dbp",
]


CARE_MODE_CONFIGS = {
    "icu": {
        "deterioration_prob": 0.45,
        "min_hours": 12,
        "max_hours": 24,
        "noise_scale": 1.20,
        "deterioration_speed": 1.20,
    },
    "er": {
        "deterioration_prob": 0.35,
        "min_hours": 4,
        "max_hours": 10,
        "noise_scale": 1.10,
        "deterioration_speed": 1.50,
    },
    "inpatient": {
        "deterioration_prob": 0.25,
        "min_hours": 12,
        "max_hours": 36,
        "noise_scale": 0.90,
        "deterioration_speed": 0.80,
    },
    "walk_in": {
        "deterioration_prob": 0.08,
        "min_hours": 2,
        "max_hours": 6,
        "noise_scale": 0.60,
        "deterioration_speed": 0.50,
    },
}


DETERIORATION_MODES = [
    "respiratory_failure",
    "septic_shock",
    "cardiac_instability",
    "hypotensive_shock",
    "fever_infection",
    "mixed_deterioration",
]


def _sample_baseline_vitals(rng: np.random.Generator) -> dict:
    """Sample patient-specific baseline vitals."""
    return {
        "heart_rate": rng.normal(82, 10),
        "resp_rate": rng.normal(18, 3),
        "spo2": rng.normal(97, 1.5),
        "temperature": rng.normal(37.0, 0.3),
        "sbp": rng.normal(122, 12),
        "dbp": rng.normal(76, 8),
    }


def _event_now(row: pd.Series) -> int:
    """
    Clinical threshold-style deterioration label.

    These are prototype rules for synthetic data.
    """
    return int(
        (row["spo2"] < 90)
        or (row["sbp"] < 90)
        or (row["heart_rate"] > 130)
        or (row["resp_rate"] > 30)
        or (row["temperature"] > 39)
        or (row["temperature"] < 35)
    )


def _mode_effects(
    deterioration_mode: str,
    severity: float,
    rng: np.random.Generator,
) -> dict:
    """
    Convert deterioration severity into correlated vital-sign effects.

    severity ranges roughly from 0 to 1+.
    """

    effects = {
        "heart_rate": 0.0,
        "resp_rate": 0.0,
        "spo2": 0.0,
        "temperature": 0.0,
        "sbp": 0.0,
        "dbp": 0.0,
    }

    if deterioration_mode == "respiratory_failure":
        effects["spo2"] = -10.0 * severity
        effects["resp_rate"] = 10.0 * severity
        effects["heart_rate"] = 12.0 * severity
        effects["sbp"] = -4.0 * severity

    elif deterioration_mode == "septic_shock":
        effects["heart_rate"] = 28.0 * severity
        effects["resp_rate"] = 8.0 * severity
        effects["temperature"] = 2.2 * severity
        effects["sbp"] = -28.0 * severity
        effects["dbp"] = -14.0 * severity
        effects["spo2"] = -3.0 * severity

    elif deterioration_mode == "cardiac_instability":
        effects["heart_rate"] = 35.0 * severity
        effects["sbp"] = -24.0 * severity
        effects["dbp"] = -12.0 * severity
        effects["spo2"] = -4.0 * severity
        effects["resp_rate"] = 4.0 * severity

    elif deterioration_mode == "hypotensive_shock":
        effects["sbp"] = -36.0 * severity
        effects["dbp"] = -18.0 * severity
        effects["heart_rate"] = 24.0 * severity
        effects["resp_rate"] = 5.0 * severity
        effects["spo2"] = -2.0 * severity

    elif deterioration_mode == "fever_infection":
        effects["temperature"] = 2.4 * severity
        effects["heart_rate"] = 22.0 * severity
        effects["resp_rate"] = 6.0 * severity
        effects["sbp"] = -8.0 * severity

    elif deterioration_mode == "mixed_deterioration":
        effects["heart_rate"] = 30.0 * severity
        effects["resp_rate"] = 10.0 * severity
        effects["spo2"] = -8.0 * severity
        effects["temperature"] = 1.5 * severity
        effects["sbp"] = -25.0 * severity
        effects["dbp"] = -12.0 * severity

    # Small random patient-to-patient variation
    for col in effects:
        effects[col] *= rng.normal(1.0, 0.08)

    return effects


def _build_forward_target(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    event_col: str = "event_now",
    horizon_steps: int = 12,
) -> pd.Series:
    """
    Forward-looking target:
    1 if an event occurs within the next horizon_steps.
    """

    future_events = []

    for k in range(1, horizon_steps + 1):
        future_events.append(
            df.groupby(patient_col)[event_col].shift(-k).fillna(0)
        )

    return pd.concat(future_events, axis=1).max(axis=1).astype(int)


def generate_synthetic_icu_data(
    n_patients: int = 500,
    start_time: str = "2026-01-01 00:00:00",
    freq: str = "5min",
    random_state: int = 42,
    horizon_steps: int = 12,
) -> pd.DataFrame:
    """
    Generate synthetic ICU-grade time-series vitals.

    Returns a dataframe compatible with BIRE.
    """

    rng = np.random.default_rng(random_state)
    start_ts = pd.Timestamp(start_time)

    rows = []

    care_modes = list(CARE_MODE_CONFIGS.keys())
    care_mode_probs = [0.35, 0.25, 0.30, 0.10]

    for patient_idx in range(n_patients):
        patient_id = f"P{patient_idx + 1:05d}"

        care_mode = rng.choice(care_modes, p=care_mode_probs)
        cfg = CARE_MODE_CONFIGS[care_mode]

        hours = int(rng.integers(cfg["min_hours"], cfg["max_hours"] + 1))
        n_steps = int((hours * 60) / 5)

        baseline = _sample_baseline_vitals(rng)

        will_deteriorate = rng.random() < cfg["deterioration_prob"]
        deterioration_mode = "stable"

        deterioration_start_step = None
        event_anchor_step = None

        if will_deteriorate and n_steps > 36:
            deterioration_mode = rng.choice(DETERIORATION_MODES)

            # Deterioration begins before event.
            # ICU/ER can decline faster; inpatient slower.
            lead_steps = int(
                rng.integers(
                    low=max(12, int(18 / cfg["deterioration_speed"])),
                    high=max(24, int(36 / cfg["deterioration_speed"])),
                )
            )
            
            min_event_step = min(lead_steps + 6, max(1, n_steps // 2))
            max_event_step = max(min_event_step + 1, n_steps - 3)
            
            
            if max_event_step > min_event_step:
                
            event_anchor_step = int(rng.integers(min_event_step, max_event_step))
            deterioration_start_step = max(0, event_anchor_step - lead_steps)
        
        else:
            will_deteriorate = False
            deterioration_mode = "stable"
            deterioration_start_step = None
            event_anchor_step = None
    
    
 for step in range(n_steps):
    timestamp = start_ts + pd.Timedelta(minutes=5 * step)

            vitals = baseline.copy()

            # Natural circadian-ish variation / drift
            vitals["heart_rate"] += 3 * np.sin(step / 24)
            vitals["resp_rate"] += 1.2 * np.sin(step / 18)
            vitals["temperature"] += 0.15 * np.sin(step / 48)
            vitals["sbp"] += 4 * np.sin(step / 36)
            vitals["dbp"] += 2 * np.sin(step / 36)

            is_deteriorating = 0
            severity = 0.0

            if deterioration_start_step is not None and step >= deterioration_start_step:
                is_deteriorating = 1

                progress = (step - deterioration_start_step) / max(
                    1, event_anchor_step - deterioration_start_step
                )

                # Smooth nonlinear deterioration curve
                severity = min(1.35, max(0.0, progress ** 1.7))

                effects = _mode_effects(deterioration_mode, severity, rng)

                for col in SIGNAL_COLS:
                    vitals[col] += effects[col]

            # Measurement noise
            noise_scale = cfg["noise_scale"]
            vitals["heart_rate"] += rng.normal(0, 3.0 * noise_scale)
            vitals["resp_rate"] += rng.normal(0, 1.2 * noise_scale)
            vitals["spo2"] += rng.normal(0, 0.8 * noise_scale)
            vitals["temperature"] += rng.normal(0, 0.12 * noise_scale)
            vitals["sbp"] += rng.normal(0, 4.0 * noise_scale)
            vitals["dbp"] += rng.normal(0, 2.5 * noise_scale)

            # Keep values within plausible synthetic bounds
            vitals["heart_rate"] = float(np.clip(vitals["heart_rate"], 40, 170))
            vitals["resp_rate"] = float(np.clip(vitals["resp_rate"], 8, 45))
            vitals["spo2"] = float(np.clip(vitals["spo2"], 70, 100))
            vitals["temperature"] = float(np.clip(vitals["temperature"], 34, 41))
            vitals["sbp"] = float(np.clip(vitals["sbp"], 60, 190))
            vitals["dbp"] = float(np.clip(vitals["dbp"], 35, 120))

            row = {
                "patient_id": patient_id,
                "timestamp": timestamp,
                "care_mode": care_mode,
                "deterioration_mode": deterioration_mode,
                "is_deteriorating": is_deteriorating,
                "deterioration_start_step": deterioration_start_step,
                "event_anchor_step": event_anchor_step,
                **vitals,
            }

            row["event_now"] = _event_now(pd.Series(row))
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

    df["target"] = _build_forward_target(
        df,
        patient_col="patient_id",
        event_col="event_now",
        horizon_steps=horizon_steps,
    )

    return df


def summarize_synthetic_data(df: pd.DataFrame) -> pd.DataFrame:
    """Quick summary for notebook inspection."""

    summary = {
        "n_rows": len(df),
        "n_patients": df["patient_id"].nunique(),
        "event_rows": int(df["event_now"].sum()),
        "target_rows": int(df["target"].sum()),
        "event_row_rate": float(df["event_now"].mean()),
        "target_row_rate": float(df["target"].mean()),
    }

    return pd.DataFrame([summary])


def summarize_by_care_mode(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize synthetic data by BMS care mode."""

    return (
        df.groupby("care_mode")
        .agg(
            n_rows=("patient_id", "size"),
            n_patients=("patient_id", "nunique"),
            event_rows=("event_now", "sum"),
            target_rows=("target", "sum"),
            event_rate=("event_now", "mean"),
            target_rate=("target", "mean"),
        )
        .reset_index()
    )


def summarize_by_deterioration_mode(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize synthetic data by deterioration mode."""

    return (
        df.groupby("deterioration_mode")
        .agg(
            n_rows=("patient_id", "size"),
            n_patients=("patient_id", "nunique"),
            event_rows=("event_now", "sum"),
            target_rows=("target", "sum"),
            event_rate=("event_now", "mean"),
            target_rate=("target", "mean"),
        )
        .reset_index()
        .sort_values("event_rate", ascending=False)
    )
