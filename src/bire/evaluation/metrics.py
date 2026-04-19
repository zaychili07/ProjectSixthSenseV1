import pandas as pd
from sklearn.metrics import roc_auc_score


def compute_auc(y_true, y_proba):
    """
    Safe AUC computation (handles single-class edge case).
    """
    if len(set(y_true)) < 2:
        return None
    return roc_auc_score(y_true, y_proba)


def summarize_split(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Summarize time-aware split.
    """
    return {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_patients": train_df["patient_id"].nunique(),
        "test_patients": test_df["patient_id"].nunique(),
        "train_positive_rows": int(train_df["target"].sum()),
        "test_positive_rows": int(test_df["target"].sum()),
    }


def compare_models(log_df: pd.DataFrame, xgb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare model outputs (alerts + risk stats).
    """
    log_df = log_df.copy()
    log_df["model"] = "logistic_regression"

    xgb_df = xgb_df.copy()
    xgb_df["model"] = "xgboost"

    combined = pd.concat([log_df, xgb_df], ignore_index=True)

    summary = (
        combined.groupby(["model", "patient_id"])
        .agg(
            n_rows=("patient_id", "size"),
            n_alerts=("alert", "sum"),
            max_risk=("pred_proba", "max"),
            mean_risk=("pred_proba", "mean"),
        )
        .reset_index()
    )

    return summary

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate_binary_model(model, X, y, split_name="split"):
    y_proba = model.predict_proba(X)[:, 1]

    results = {
        "split": split_name,
        "n_rows": len(y),
        "positive_rate": float(np.mean(y)),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
    }

    if len(np.unique(y)) >= 2:
        results["roc_auc"] = roc_auc_score(y, y_proba)
        results["pr_auc"] = average_precision_score(y, y_proba)
    else:
        results["warning"] = f"{split_name} has only one class"

    return results, y_proba
