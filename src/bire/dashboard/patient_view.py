import pandas as pd
import matplotlib.pyplot as plt


def plot_patient_lifecycle_timeline(
    df,
    patient_id,
    state_col="bire_final_tier",
    timestamp_col="timestamp",
    event_col="event_now",
):
    """
    Plot BIRE lifecycle state progression for a single patient.
    """

    patient_df = (
        df[df["patient_id"] == patient_id]
        .sort_values(timestamp_col)
        .copy()
    )

    state_map = {
        "SUPPRESSED_WATCH": 0,
        "WATCH": 1,
        "ESCALATE": 2,
        "URGENT": 3,
        "MONITOR": 4,
        "RE-ESCALATE": 5,
        "CRITICAL": 6,
    }

    patient_df["state_y"] = patient_df[state_col].map(state_map)

    plt.figure(figsize=(16, 6))

    plt.plot(
        patient_df[timestamp_col],
        patient_df["state_y"],
        marker="o",
        linewidth=3,
    )

    if event_col in patient_df.columns:
        event_rows = patient_df[patient_df[event_col] == 1]

        if not event_rows.empty:
            plt.scatter(
                event_rows[timestamp_col],
                event_rows["state_y"],
                s=200,
                marker="X",
                label="Clinical Event",
            )

    plt.yticks(
        list(state_map.values()),
        list(state_map.keys()),
    )

    plt.title(f"BIRE Lifecycle Timeline — {patient_id}")
    plt.xlabel("Timestamp")
    plt.ylabel("Lifecycle State")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return patient_df


def plot_patient_risk_trajectory(
    patient_df,
    patient_id=None,
    risk_col="pred_proba",
    timestamp_col="timestamp",
    event_col="event_now",
    tier_col="bire_final_tier",
):
    """
    Plot predicted risk trajectory for a single patient.
    """

    plot_df = patient_df.sort_values(timestamp_col).copy()

    if patient_id is None and "patient_id" in plot_df.columns:
        patient_id = plot_df["patient_id"].iloc[0]

    plt.figure(figsize=(16, 6))

    plt.plot(
        plot_df[timestamp_col],
        plot_df[risk_col],
        marker="o",
        linewidth=3,
        label="Predicted Risk",
    )

    if event_col in plot_df.columns:
        event_rows = plot_df[plot_df[event_col] == 1]

        if not event_rows.empty:
            plt.scatter(
                event_rows[timestamp_col],
                event_rows[risk_col],
                s=250,
                marker="X",
                label="Clinical Event",
            )

    plt.axhline(
        y=0.50,
        linestyle="--",
        linewidth=2,
        label="Moderate Risk",
    )

    plt.axhline(
        y=0.80,
        linestyle="--",
        linewidth=2,
        label="High Risk",
    )

    if tier_col in plot_df.columns:
        for _, row in plot_df.iterrows():
            plt.text(
                row[timestamp_col],
                row[risk_col] + 0.02,
                str(row[tier_col]),
                fontsize=8,
                rotation=45,
            )

    plt.title(f"BIRE Risk Trajectory — {patient_id}")
    plt.xlabel("Timestamp")
    plt.ylabel("Predicted Risk")
    plt.ylim(0, 1.05)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return plot_df


def plot_patient_vital_trends(
    patient_df,
    patient_id=None,
    timestamp_col="timestamp",
    vitals=None,
):
    """
    Plot vital sign trends for a single patient.
    """

    if vitals is None:
        vitals = [
            "heart_rate",
            "resp_rate",
            "spo2",
            "sbp",
        ]

    plot_df = patient_df.sort_values(timestamp_col).copy()

    if patient_id is None and "patient_id" in plot_df.columns:
        patient_id = plot_df["patient_id"].iloc[0]

    available_vitals = [
        vital for vital in vitals
        if vital in plot_df.columns
    ]

    plt.figure(figsize=(18, 10))

    for vital in available_vitals:
        plt.plot(
            plot_df[timestamp_col],
            plot_df[vital],
            marker="o",
            linewidth=2,
            label=vital,
        )

    plt.title(f"Patient Vital Trends — {patient_id}")
    plt.xlabel("Timestamp")
    plt.ylabel("Vital Measurement")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return plot_df


def build_episode_timing_view(patient_df):
    """
    Build a patient-level episode timing summary table.
    """

    episode_cols = [
        "patient_id",
        "alert_episode_id",
        "episode_start_time",
        "episode_end_time",
        "episode_duration_minutes",
        "event_time",
        "lead_time_min",
        "timing_category",
        "episode_intelligence_type",
        "episode_tier",
        "episode_tier_final",
        "bire_final_tier",
        "bire_state",
        "bire_timing",
        "bire_decision_reason",
    ]

    available_episode_cols = [
        col for col in episode_cols
        if col in patient_df.columns
    ]

    episode_view = (
        patient_df[available_episode_cols]
        .drop_duplicates()
        .sort_values(
            ["episode_start_time", "lead_time_min"],
            ascending=[True, False],
        )
    )

    return episode_view


def plot_patient_lead_time_timeline(
    patient_df,
    patient_id=None,
    timestamp_col="timestamp",
    lead_time_col="lead_time_min",
    event_col="event_now",
    tier_col="bire_final_tier",
):
    """
    Plot lead-time behavior across a patient timeline.
    """

    plot_df = patient_df.sort_values(timestamp_col).copy()

    if patient_id is None and "patient_id" in plot_df.columns:
        patient_id = plot_df["patient_id"].iloc[0]

    plt.figure(figsize=(16, 5))

    plt.plot(
        plot_df[timestamp_col],
        plot_df[lead_time_col],
        marker="o",
        linewidth=3,
        label="Lead Time Before Event",
    )

    if event_col in plot_df.columns:
        event_rows = plot_df[plot_df[event_col] == 1]

        if not event_rows.empty:
            for event_time in event_rows[timestamp_col]:
                plt.axvline(
                    x=event_time,
                    linestyle="--",
                    linewidth=2,
                    label="Clinical Event",
                )

    if tier_col in plot_df.columns:
        for _, row in plot_df.iterrows():
            plt.text(
                row[timestamp_col],
                row[lead_time_col],
                str(row[tier_col]),
                fontsize=8,
                rotation=45,
            )

    plt.title(f"BIRE Lead-Time Timeline — {patient_id}")
    plt.xlabel("Timestamp")
    plt.ylabel("Lead Time Before Event (minutes)")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return plot_df


def build_decision_reasoning_view(patient_df):
    """
    Build a patient-level decision reasoning table for dashboard review.
    """

    reasoning_cols = [
        "timestamp",
        "patient_id",
        "pred_proba",
        "risk_band",
        "bire_final_tier",
        "bire_state",
        "bire_timing",
        "timing_category",
        "episode_intelligence_type",
        "bire_decision_reason",
        "watch_override_reason",
        "ibpip_override_reason",
        "gss_ve_reason",
        "monitor_state",
        "re_escalate_reason",
        "critical_reason",
    ]

    available_reasoning_cols = [
        col for col in reasoning_cols
        if col in patient_df.columns
    ]

    reasoning_view = (
        patient_df[available_reasoning_cols]
        .sort_values("timestamp")
        .copy()
    )

    return reasoning_view


def build_clinician_summary_snapshot(patient_df):
    """
    Build a concise latest-state clinician summary snapshot for one patient.
    """

    latest_row = (
        patient_df
        .sort_values("timestamp")
        .iloc[-1]
    )

    summary_items = {
        "Patient ID": latest_row.get("patient_id"),
        "Timestamp": latest_row.get("timestamp"),
        "Predicted Risk": round(float(latest_row.get("pred_proba", 0)), 4),
        "Risk Band": latest_row.get("risk_band"),
        "Final Tier": latest_row.get("bire_final_tier"),
        "BIRE State": latest_row.get("bire_state"),
        "BIRE Timing": latest_row.get("bire_timing"),
        "Timing Category": latest_row.get("timing_category"),
        "Episode Intelligence Type": latest_row.get("episode_intelligence_type"),
        "Lead Time (min)": latest_row.get("lead_time_min"),
        "Abnormal Count": latest_row.get("abnormal_count"),
        "Decision Reason": latest_row.get("bire_decision_reason"),
        "Watch Override Reason": latest_row.get("watch_override_reason"),
        "IBPIP Override Reason": latest_row.get("ibpip_override_reason"),
        "GSS-VE Reason": latest_row.get("gss_ve_reason"),
        "Monitor State": latest_row.get("monitor_state"),
        "Re-Escalate Reason": latest_row.get("re_escalate_reason"),
        "Critical Reason": latest_row.get("critical_reason"),
    }

    return pd.DataFrame({
        "Field": list(summary_items.keys()),
        "Value": list(summary_items.values()),
    })

def plot_patient_risk_trajectory_with_thresholds(
    patient_df,
    patient_id=None,
    risk_col="pred_proba",
    timestamp_col="timestamp",
    event_col="event_now",
    tier_col="bire_final_tier",
    watch_threshold=0.40,
    escalate_threshold=0.60,
    urgent_threshold=0.80,
    critical_threshold=0.95,
):
    """
    Plot patient risk trajectory with operational threshold overlays.
    """

    plot_df = patient_df.sort_values(timestamp_col).copy()

    if patient_id is None and "patient_id" in plot_df.columns:
        patient_id = plot_df["patient_id"].iloc[0]

    plt.figure(figsize=(16, 6))


    # Risk Curve
    plt.plot(
        plot_df[timestamp_col],
        plot_df[risk_col],
        marker="o",
        linewidth=3,
        label="Predicted Risk",
    )

    # Event Markers
    if event_col in plot_df.columns:

        event_rows = plot_df[
            plot_df[event_col] == 1
        ]

        if not event_rows.empty:

            plt.scatter(
                event_rows[timestamp_col],
                event_rows[risk_col],
                s=250,
                marker="X",
                label="Clinical Event",
            )

    # Threshold Overlays
    plt.axhline(
        y=watch_threshold,
        linestyle="--",
        linewidth=2,
        label="WATCH Threshold",
    )

    plt.axhline(
        y=escalate_threshold,
        linestyle="--",
        linewidth=2,
        label="ESCALATE Threshold",
    )

    plt.axhline(
        y=urgent_threshold,
        linestyle="--",
        linewidth=2,
        label="URGENT Threshold",
    )

    plt.axhline(
        y=critical_threshold,
        linestyle="--",
        linewidth=2,
        label="CRITICAL Threshold",
    )

   
    # Tier Labels
    if tier_col in plot_df.columns:

        for _, row in plot_df.iterrows():

            plt.text(
                row[timestamp_col],
                row[risk_col] + 0.02,
                str(row[tier_col]),
                fontsize=8,
                rotation=45,
            )

   
    # Plot Formatting
    plt.title(
        f"BIRE Operational Risk Trajectory — {patient_id}"
    )

    plt.xlabel("Timestamp")
    plt.ylabel("Predicted Risk")

    plt.ylim(0, 1.05)

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    return plot_df
