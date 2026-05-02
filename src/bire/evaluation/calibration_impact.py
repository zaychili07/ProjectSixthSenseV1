"""
Calibration impact utilities for BIRE.

Compares downstream GSS behavior when using raw vs calibrated probabilities.
"""

import pandas as pd


def summarize_alerts(df, version_name, alert_col="gss_v33_alert"):
    """
    Summarize total alert burden for a GSS output dataframe.
    """
    return {
        "version": version_name,
        "alerts": int(df[alert_col].sum()),
        "patients": int(df["patient_id"].nunique()),
        "alerts_per_patient": float(df[alert_col].sum() / df["patient_id"].nunique()),
        "alert_rate": float(df[alert_col].mean()),
    }


def build_alert_summary(raw_gss_df, platt_gss_df, isotonic_gss_df, alert_col="gss_v33_alert"):
    """
    Build alert burden summary for raw, Platt, and isotonic GSS outputs.
    """
    return pd.DataFrame([
        summarize_alerts(raw_gss_df, "raw", alert_col=alert_col),
        summarize_alerts(platt_gss_df, "platt", alert_col=alert_col),
        summarize_alerts(isotonic_gss_df, "isotonic", alert_col=alert_col),
    ])


def build_timing_counts(timing_results):
    """
    Count timing categories by version.
    """
    return (
        timing_results
        .groupby(["version", "timing_category"])
        .size()
        .reset_index(name="count")
        .sort_values(["version", "count"], ascending=[True, False])
    )


def build_timing_pivot(timing_results):
    """
    Build side-by-side timing category comparison table.
    """
    return (
        timing_results
        .groupby(["version", "timing_category"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )


def compute_detection_metrics(
    df,
    version_name,
    audit_fn,
    alert_col="gss_v33_alert",
    event_col="event_now",
):
    """
    Compute patient-level detection metrics from timing-aware audit output.
    """
    timing_df = audit_fn(
        df,
        alert_col=alert_col,
        event_col=event_col,
    )

    total_events = int(timing_df["event_flag"].sum())
    true_predictive = int((timing_df["timing_category"] == "true_predictive_alert").sum())
    post_event = int((timing_df["timing_category"] == "post_event_alert").sum())
    missed = int((timing_df["timing_category"] == "no_alert").sum())
    no_event_alerts = int((timing_df["timing_category"] == "no_event_alert").sum())

    return {
        "version": version_name,
        "total_event_patients": total_events,
        "true_predictive": true_predictive,
        "post_event_only": post_event,
        "missed_events": missed,
        "no_event_alerts": no_event_alerts,
        "pre_event_detection_rate": true_predictive / total_events if total_events > 0 else 0.0,
        "post_event_rate": post_event / total_events if total_events > 0 else 0.0,
        "miss_rate": missed / total_events if total_events > 0 else 0.0,
    }


def build_detection_metrics(
    raw_gss_df,
    platt_gss_df,
    isotonic_gss_df,
    audit_fn,
    alert_col="gss_v33_alert",
    event_col="event_now",
):
    """
    Build detection metrics table for raw, Platt, and isotonic GSS outputs.
    """
    return pd.DataFrame([
        compute_detection_metrics(
            raw_gss_df,
            "raw",
            audit_fn=audit_fn,
            alert_col=alert_col,
            event_col=event_col,
        ),
        compute_detection_metrics(
            platt_gss_df,
            "platt",
            audit_fn=audit_fn,
            alert_col=alert_col,
            event_col=event_col,
        ),
        compute_detection_metrics(
            isotonic_gss_df,
            "isotonic",
            audit_fn=audit_fn,
            alert_col=alert_col,
            event_col=event_col,
        ),
    ])


def build_calibration_impact_report(
    raw_gss_df,
    platt_gss_df,
    isotonic_gss_df,
    timing_results,
    audit_fn,
    alert_col="gss_v33_alert",
    event_col="event_now",
):
    """
    Build all Chapter 19 comparison outputs.

    Returns
    -------
    dict
        Dictionary containing:
        - alert_summary
        - timing_counts
        - timing_pivot
        - detection_metrics
    """
    return {
        "alert_summary": build_alert_summary(
            raw_gss_df,
            platt_gss_df,
            isotonic_gss_df,
            alert_col=alert_col,
        ),
        "timing_counts": build_timing_counts(timing_results),
        "timing_pivot": build_timing_pivot(timing_results),
        "detection_metrics": build_detection_metrics(
            raw_gss_df,
            platt_gss_df,
            isotonic_gss_df,
            audit_fn=audit_fn,
            alert_col=alert_col,
            event_col=event_col,
        ),
    }


def build_timing_results(
    raw_gss_df,
    platt_gss_df,
    isotonic_gss_df,
    audit_fn,
    alert_col="gss_v33_alert",
    event_col="event_now",
):
    """
    Build timing-aware results across raw, Platt, and isotonic GSS outputs.
    """

    def run_timing(df, name):
        timing_df = audit_fn(
            df,
            alert_col=alert_col,
            event_col=event_col,
        )
        timing_df["version"] = name
        return timing_df

    return pd.concat([
        run_timing(raw_gss_df, "raw"),
        run_timing(platt_gss_df, "platt"),
        run_timing(isotonic_gss_df, "isotonic"),
    ])
