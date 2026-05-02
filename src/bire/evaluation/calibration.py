"""
Calibration utilities for BIRE.

This module handles:
- Probability calibration (Platt scaling, isotonic regression)
- Application of calibration models
- Brier score evaluation

No plotting functions should exist in this module.
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss


# =========================================================
# Calibration Model Fitting
# =========================================================

HORIZONS = {
    "15min": {
        "risk_col": "risk_15min",
        "target_col": "target_15min",
    },
    "30min": {
        "risk_col": "risk_30min",
        "target_col": "target_30min",
    },
    "60min": {
        "risk_col": "risk_60min",
        "target_col": "target_60min",
    },
}

def fit_platt_calibrator(y_true, y_prob):
    """
    Fit Platt scaling using logistic regression.

    Parameters
    ----------
    y_true : array-like
        Binary target labels.
    y_prob : array-like
        Raw predicted probabilities.

    Returns
    -------
    LogisticRegression
        Fitted Platt calibration model.
    """
    model = LogisticRegression(solver="lbfgs")
    model.fit(np.asarray(y_prob).reshape(-1, 1), y_true)
    return model


def fit_isotonic_calibrator(y_true, y_prob):
    """
    Fit isotonic regression calibration.

    Parameters
    ----------
    y_true : array-like
        Binary target labels.
    y_prob : array-like
        Raw predicted probabilities.

    Returns
    -------
    IsotonicRegression
        Fitted isotonic calibration model.
    """
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(y_prob, y_true)
    return model


# =========================================================
# Apply Calibration
# =========================================================

def apply_calibrators(df, risk_col, platt_model, isotonic_model):
    """
    Apply Platt and isotonic calibrators to a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing raw probabilities.
    risk_col : str
        Column name of raw probabilities.
    platt_model : LogisticRegression
        Fitted Platt model.
    isotonic_model : IsotonicRegression
        Fitted isotonic model.

    Returns
    -------
    pd.DataFrame
        Dataframe with calibrated probability columns added.
    """
    out = df.copy()

    raw_prob = out[risk_col].astype(float).values

    out[f"{risk_col}_platt"] = platt_model.predict_proba(
        raw_prob.reshape(-1, 1)
    )[:, 1]

    out[f"{risk_col}_isotonic"] = isotonic_model.predict(raw_prob)

    return out


# =========================================================
# Multi-Horizon Calibration Pipeline
# =========================================================

def fit_and_apply_horizon_calibrators(
    calibration_df,
    evaluation_df,
    horizons,
):
    """
    Fit and apply calibration models across multiple horizons.

    Parameters
    ----------
    calibration_df : pd.DataFrame
        Data used to fit calibration models.
    evaluation_df : pd.DataFrame
        Data used to evaluate calibrated probabilities.
    horizons : dict
        Dictionary mapping horizon names to config:
        {
            "15min": {"risk_col": ..., "target_col": ...},
            ...
        }

    Returns
    -------
    tuple
        evaluation_calibrated_df : pd.DataFrame
        calibrators : dict
    """
    calibrators = {}
    evaluation_calibrated_df = evaluation_df.copy()

    for horizon_name, cfg in horizons.items():
        risk_col = cfg["risk_col"]
        target_col = cfg["target_col"]

        y_cal = calibration_df[target_col].astype(int)
        p_cal = calibration_df[risk_col].astype(float)

        platt_model = fit_platt_calibrator(y_cal, p_cal)
        isotonic_model = fit_isotonic_calibrator(y_cal, p_cal)

        calibrators[horizon_name] = {
            "platt": platt_model,
            "isotonic": isotonic_model,
        }

        evaluation_calibrated_df = apply_calibrators(
            evaluation_calibrated_df,
            risk_col=risk_col,
            platt_model=platt_model,
            isotonic_model=isotonic_model,
        )

    return evaluation_calibrated_df, calibrators


# =========================================================
# Brier Score Evaluation
# =========================================================

def compute_brier_summary(df, horizon_name, target_col, risk_col):
    """
    Compute Brier scores for raw and calibrated probabilities.

    Parameters
    ----------
    df : pd.DataFrame
        Evaluation dataframe.
    horizon_name : str
        Horizon label.
    target_col : str
        Ground truth column.
    risk_col : str
        Raw probability column.

    Returns
    -------
    list[dict]
        Brier score results.
    """
    y_true = df[target_col].astype(int)

    rows = []

    for label, col in [
        ("raw", risk_col),
        ("platt", f"{risk_col}_platt"),
        ("isotonic", f"{risk_col}_isotonic"),
    ]:
        rows.append({
            "horizon": horizon_name,
            "version": label,
            "brier_score": brier_score_loss(y_true, df[col]),
        })

    return rows


def build_brier_summary(evaluation_df, horizons):
    """
    Build Brier score summary across all horizons.

    Parameters
    ----------
    evaluation_df : pd.DataFrame
        Dataframe with calibrated probabilities.
    horizons : dict
        Horizon configuration.

    Returns
    -------
    pd.DataFrame
        Sorted Brier score summary.
    """
    rows = []

    for horizon_name, cfg in horizons.items():
        rows.extend(
            compute_brier_summary(
                evaluation_df,
                horizon_name=horizon_name,
                target_col=cfg["target_col"],
                risk_col=cfg["risk_col"],
            )
        )

    return pd.DataFrame(rows).sort_values(["horizon", "brier_score"])
