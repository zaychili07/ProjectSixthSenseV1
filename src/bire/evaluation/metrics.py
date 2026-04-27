import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_auc(y_true, y_proba):
    """
    Safe ROC-AUC computation.
    Returns np.nan when only one class is present.
    """
    if len(set(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_proba)


def summarize_split(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Summarize a time-aware train/test split.
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
    Compare alert and risk behavior across two model outputs.
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


def evaluate_binary_model(model, X, y, split_name="split"):
    """
    Evaluate binary classification performance on one split.
    Returns a results dictionary and predicted probabilities.
    """
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


def evaluate_multiple_splits(model, splits_dict):
    """
    Evaluate a model across multiple named splits.

    Parameters
    ----------
    model : fitted classifier
        Must support predict_proba().
    splits_dict : dict
        Example:
        {
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }

    Returns
    -------
    results_df : pd.DataFrame
        One row per split.
    all_probas : dict
        Mapping of split name -> predicted probabilities.
    """
    all_results = []
    all_probas = {}

    for split_name, (X, y) in splits_dict.items():
        results, proba = evaluate_binary_model(model, X, y, split_name)
        all_results.append(results)
        all_probas[split_name] = proba

    return pd.DataFrame(all_results), all_probas

# used for Risk_delta tuning sweeps
def compute_alert_metrics(
    df,
    alert_col,
    event_col="event_episode_flag",
    patient_col="patient_id",
    time_col="timestamp",
    interval_minutes=5,
    prediction_horizon_minutes=60,
):
    total_rows = len(df)
    total_patient_hours = (total_rows * interval_minutes) / 60
    total_alerts = int(df[alert_col].sum())

    event_rows = df[df[event_col] == 1]
    total_events = len(event_rows)

    detected_events = 0
    lead_times = []

    for _, event in event_rows.iterrows():
        pid = event[patient_col]
        event_time = event[time_col]

        lookback_start = event_time - pd.Timedelta(
            minutes=prediction_horizon_minutes
        )

        prior_alerts = df[
            (df[patient_col] == pid)
            & (df[time_col] >= lookback_start)
            & (df[time_col] < event_time)
            & (df[alert_col] == 1)
        ]

        if not prior_alerts.empty:
            detected_events += 1
            first_alert_time = prior_alerts[time_col].min()
            lead_times.append(
                (event_time - first_alert_time).total_seconds() / 60
            )

    false_alerts = 0
    alert_rows = df[df[alert_col] == 1]

    for _, alert in alert_rows.iterrows():
        pid = alert[patient_col]
        alert_time = alert[time_col]

        future_end = alert_time + pd.Timedelta(
            minutes=prediction_horizon_minutes
        )

        future_events = df[
            (df[patient_col] == pid)
            & (df[time_col] > alert_time)
            & (df[time_col] <= future_end)
            & (df[event_col] == 1)
        ]

        if future_events.empty:
            false_alerts += 1

    return {
        "alerts_per_hour": total_alerts / total_patient_hours if total_patient_hours > 0 else None,
        "detection_rate": detected_events / total_events if total_events > 0 else None,
        "false_alert_rate": false_alerts / total_alerts if total_alerts > 0 else None,
        "median_lead_time": np.median(lead_times) if lead_times else None,
    }
