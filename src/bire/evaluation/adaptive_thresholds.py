"""
Adaptive threshold evaluation utilities for BIRE.
"""

import pandas as pd


def summarize_policy_detection(timing_df, policy_name):
    """
    Summarize timing-aware detection performance for one alert policy.
    """
    total_event_patients = int(timing_df["has_event"].sum())
    true_predictive = int((timing_df["timing_category"] == "true_predictive_alert").sum())
    post_event = int((timing_df["timing_category"] == "post_event_alert").sum())
    missed = int(
        (
            (timing_df["has_event"] == 1)
            & (timing_df["has_alert"] == 0)
        ).sum()
    )
    no_event_alerts = int((timing_df["timing_category"] == "no_event_alert").sum())

    return {
        "policy": policy_name,
        "total_event_patients": total_event_patients,
        "true_predictive": true_predictive,
        "post_event_only": post_event,
        "missed_events": missed,
        "no_event_alerts": no_event_alerts,
        "pre_event_detection_rate": (
            true_predictive / total_event_patients if total_event_patients > 0 else 0.0
        ),
        "post_event_rate": (
            post_event / total_event_patients if total_event_patients > 0 else 0.0
        ),
        "miss_rate": (
            missed / total_event_patients if total_event_patients > 0 else 0.0
        ),
    }


def build_policy_detection_metrics(global_timing_df, adaptive_timing_df):
    """
    Compare global threshold vs adaptive threshold timing performance.
    """
    return pd.DataFrame([
        summarize_policy_detection(global_timing_df, "global_threshold"),
        summarize_policy_detection(adaptive_timing_df, "adaptive_threshold"),
    ])
