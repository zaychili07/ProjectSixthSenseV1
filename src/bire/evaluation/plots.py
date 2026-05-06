import matplotlib.pyplot as plt

# Core trajectory plot (simple)
def plot_patient_risk_trajectory(
    df,
    patient_id,
    time_col="timestamp",
    risk_col="pred_proba",
    alert_col="alert_episode_flag",
    event_col="target",
    threshold=0.5,
):
    patient_df = df[df["patient_id"] == patient_id].sort_values(time_col)

    if patient_df.empty:
        print(f"No data found for patient {patient_id}")
        return

    plt.figure(figsize=(10, 5))

    # Main risk line
    plt.plot(
        patient_df[time_col],
        patient_df[risk_col],
        marker="o",
        label="Predicted Risk",
    )

    # Threshold line
    plt.axhline(
        y=threshold,
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold}",
    )

    # Alert points
    alert_df = patient_df[patient_df[alert_col] == 1]
    if not alert_df.empty:
        plt.scatter(
            alert_df[time_col],
            alert_df[risk_col],
            s=80,
            label="Alerts",
            zorder=5,
        )

    # Event points
    event_df = patient_df[patient_df[event_col] == 1]
    if not event_df.empty:
        plt.scatter(
            event_df[time_col],
            event_df[risk_col],
            s=100,
            marker="x",
            label="Deterioration Event",
            zorder=6,
        )

    plt.title(f"Risk Trajectory – Patient {patient_id}")
    plt.xlabel("Timestamp")
    plt.ylabel("Predicted Risk")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -------------------------------
# Demo / presentation plot (enhanced)
# -------------------------------
def plot_demo_trajectory(df, patient_id, threshold=0.5):
    patient_df = df[df["patient_id"] == patient_id].sort_values("timestamp")

    if patient_df.empty:
        print(f"No data found for patient {patient_id}")
        return

    max_idx = patient_df["pred_proba"].idxmax()
    max_row = patient_df.loc[max_idx]
    final_risk = patient_df["pred_proba"].iloc[-1]
    n_alerts = (patient_df["pred_proba"] >= threshold).sum()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Main risk line
    ax.plot(
        patient_df["timestamp"],
        patient_df["pred_proba"],
        linewidth=2.5,
        marker="o",
        markersize=6,
        label="Predicted Risk",
    )

    # Threshold line
    ax.axhline(
        y=threshold,
        linestyle="--",
        linewidth=2,
        label=f"Alert Threshold ({threshold:.2f})",
    )

    # High-risk shading
    ax.fill_between(
        patient_df["timestamp"],
        threshold,
        patient_df["pred_proba"],
        where=patient_df["pred_proba"] >= threshold,
        alpha=0.25,
        interpolate=True,
        label="High-Risk Zone",
    )

    # Alert points
    alert_points = patient_df[patient_df["pred_proba"] >= threshold]
    if not alert_points.empty:
        ax.scatter(
            alert_points["timestamp"],
            alert_points["pred_proba"],
            s=90,
            zorder=5,
            label="Triggered Alerts",
        )

    # Peak risk point
    ax.scatter(
        max_row["timestamp"],
        max_row["pred_proba"],
        s=140,
        zorder=6,
        label="Peak Risk",
    )

    ax.annotate(
        f"Peak Risk: {max_row['pred_proba']:.3f}",
        xy=(max_row["timestamp"], max_row["pred_proba"]),
        xytext=(10, 15),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.85),
    )

    ax.set_title(
        f"BIRE Risk Trajectory – Patient {patient_id}\n"
        f"Final Risk: {final_risk:.3f} | Alerts Triggered: {n_alerts}",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Predicted Deterioration Risk")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    plt.xticks(rotation=45)

    ax.legend(frameon=True, fancybox=True)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt


def plot_alert_bar_summary(demo_summary_df):
    df = demo_summary_df.copy()

    if "alerts" not in df.columns:
        if "n_alerts" in df.columns:
            df["alerts"] = df["n_alerts"]
        else:
            raise ValueError("demo_summary_df must contain 'alerts' or 'n_alerts'")

    required_cols = {"patient_id", "alerts"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"demo_summary_df is missing required columns: {missing}")

    # rest of plotting code uses df["alerts"]
    # THIS MUST EXIST
    df_plot = demo_summary_df.sort_values("alerts", ascending=False).copy()

    colors = ["#d62728" if x > 0 else "#2ca02c" for x in df_plot["alerts"]]

    plt.figure(figsize=(9, 5))

    bars = plt.bar(
        df_plot["patient_id"],
        df_plot["alerts"],
        color=colors,
        edgecolor="black",
        linewidth=1.2,
    )

    plt.axhline(
        y=1,
        linestyle="--",
        linewidth=2.5,
        alpha=0.9,
        zorder=3,
    )

    plt.axhspan(1, plt.ylim()[1], alpha=0.2)

    plt.text(
        len(df_plot) - 0.5,
        1.15,
        "⚠️ Alert Threshold",
        fontsize=11,
        fontweight="bold",
        ha="right",
        va="bottom",
    )

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.05,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.title("BIRE Alerts by Patient", fontsize=14, fontweight="bold")
    plt.xlabel("Patient ID", fontsize=11)
    plt.ylabel("Alert Count", fontsize=11)

    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.ylim(0, max(df_plot["alerts"]) + 1)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()
def plot_vital_trajectories(   # multi signal overview
    patient_df,
    patient_id=None,
    time_col="timestamp",
    vital_cols=None,
):
    import matplotlib.pyplot as plt

    if patient_df.empty:
        print("patient_df is empty.")
        return

    if time_col not in patient_df.columns:
        raise ValueError(f"Missing time column: {time_col}")

    patient_df = patient_df.sort_values(time_col).copy()

    if vital_cols is None:
        vital_cols = ["spo2", "resp_rate", "sbp", "heart_rate", "temperature"]

    plot_cols = [c for c in vital_cols if c in patient_df.columns]

    if len(plot_cols) == 0:
        print("No valid vital columns found.")
        return

    for col in plot_cols:
        plt.figure(figsize=(10, 4))

        plt.plot(
            patient_df[time_col],
            patient_df[col],
            marker="o"
        )

        plt.grid(True, alpha=0.3)

        title_id = patient_id if patient_id is not None else "Unknown"
        plt.title(f"{col.upper()} Trajectory – Patient {title_id}")

        plt.xlabel("Timestamp")
        plt.ylabel(col.replace("_", " ").title())
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

def plot_top_feature_coefficients(
    coef_df,
    top_n=10,
    feature_col="feature",
    coef_col="coefficient",
):
    import matplotlib.pyplot as plt

    if coef_df.empty:
        print("Coefficient DataFrame is empty.")
        return

    top_features = (
        coef_df.head(top_n)
        .sort_values(coef_col)
    )

    plt.figure(figsize=(8, 5))

    plt.barh(
        top_features[feature_col],
        top_features[coef_col]
    )

    plt.axvline(0)
    plt.grid(axis="x", linestyle="--", alpha=0.3) ### added line #
    plt.title("Top Feature Effects on Risk (Logistic Regression)")
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")

    plt.tight_layout()
    plt.show()

def plot_single_vital_with_threshold(     # focused clinical view
    patient_df,
    signal,
    threshold=None,
    patient_id=None,
    time_col="timestamp",
):
    import matplotlib.pyplot as plt

    if patient_df.empty:
        print("patient_df is empty.")
        return

    if signal not in patient_df.columns:
        raise ValueError(f"Signal '{signal}' not found in DataFrame")

    if time_col not in patient_df.columns:
        raise ValueError(f"Missing time column: {time_col}")

    patient_df = patient_df.sort_values(time_col).copy()

    plt.figure(figsize=(10, 4))

    plt.plot(
        patient_df[time_col],
        patient_df[signal],
        label=signal.replace("_", " ").title()
    )

    # Optional threshold line
    if threshold is not None:
        plt.axhline(threshold, linestyle="--", alpha=0.8, label="Threshold")

    title_id = patient_id if patient_id is not None else "Unknown"
    plt.title(f"{signal.upper()} Over Time – Patient {title_id}")

    plt.xlabel("Time")
    plt.ylabel(signal.replace("_", " ").title())
    plt.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_patient_event_timeline(patient_df, patient_id, save_path=None):
    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(patient_df["timestamp"], patient_df["spo2"], label="spo2")
    ax1.plot(patient_df["timestamp"], patient_df["sbp"], label="sbp")
    ax1.plot(patient_df["timestamp"], patient_df["heart_rate"], label="heart_rate")
    ax1.set_title(f"Vitals and Labels for Patient {patient_id}")
    ax1.set_ylabel("Signal Value")
    ax1.legend()

    for ts in patient_df.loc[patient_df["event_now"] == 1, "timestamp"]:
        ax1.axvline(ts, linestyle="--", alpha=0.5)

    for ts in patient_df.loc[patient_df["target"] == 1, "timestamp"]:
        ax1.axvline(ts, linestyle=":", alpha=0.7)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(patient_df["timestamp"], patient_df["event_now"], label="event_now")
    ax2.plot(patient_df["timestamp"], patient_df["target"], label="target")
    ax2.set_xlabel("Timestamp")
    ax2.set_ylabel("Flag")
    ax2.set_yticks([0, 1])
    ax2.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

import pandas as pd
import matplotlib.pyplot as plt


def plot_leadtime_distribution(
    event_lead_df: pd.DataFrame,
    lead_col: str = "first_alert_lead_minutes",
    detected_col: str = "detected_before_event",
    bins: int = 20,
) -> None:
    """
    Plot distribution of lead times for events detected before deterioration.
    """
    if event_lead_df.empty:
        raise ValueError("event_lead_df is empty")

    required_cols = {lead_col, detected_col}
    missing = required_cols - set(event_lead_df.columns)
    if missing:
        raise ValueError(f"event_lead_df missing required columns: {missing}")

    detected_events_df = event_lead_df.loc[
        event_lead_df[detected_col] == 1
    ].copy()

    if detected_events_df.empty:
        raise ValueError("No detected events available for lead-time distribution")

    plt.figure(figsize=(8, 5))
    plt.hist(detected_events_df[lead_col].dropna(), bins=bins)
    plt.title("Distribution of First-Alert Lead Time")
    plt.xlabel("Lead Time (minutes)")
    plt.ylabel("Number of Events")
    plt.tight_layout()
    plt.show()


def plot_event_detection_summary(
    event_lead_df: pd.DataFrame,
    detected_col: str = "detected_before_event",
) -> None:
    """
    Plot event-level detected vs missed counts.
    """
    if event_lead_df.empty:
        raise ValueError("event_lead_df is empty")

    if detected_col not in event_lead_df.columns:
        raise ValueError(f"event_lead_df missing required column: {detected_col}")

    event_status_counts = event_lead_df[detected_col].value_counts().sort_index()

    status_labels = ["Missed", "Detected Before Event"]
    status_values = [
        event_status_counts.get(0, 0),
        event_status_counts.get(1, 0),
    ]

    plt.figure(figsize=(6, 4))
    plt.bar(status_labels, status_values)
    plt.title("Event-Level Detection Summary")
    plt.ylabel("Number of Events")
    plt.tight_layout()
    plt.show()


def build_event_leadtime_display_table(
    event_lead_df: pd.DataFrame,
    sort_cols: list[str] | None = None,
    ascending: list[bool] | None = None,
    round_cols: list[str] | None = None,
    top_n: int = 25,
) -> pd.DataFrame:
    """
    Build a presentation-ready event lead-time table.
    """
    if event_lead_df.empty:
        return pd.DataFrame()

    display_df = event_lead_df.copy()

    if round_cols is None:
        round_cols = ["first_alert_lead_minutes", "last_alert_lead_minutes"]

    for col in round_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(1)

    if sort_cols is None:
        sort_cols = ["detected_before_event", "first_alert_lead_minutes"]

    if ascending is None:
        ascending = [False, False]

    display_df = display_df.sort_values(sort_cols, ascending=ascending)

    if top_n is not None:
        display_df = display_df.head(top_n)

    return display_df


def plot_event_leadtime_suite(
    event_lead_df: pd.DataFrame,
    top_n: int = 25,
) -> pd.DataFrame:
    """
    Convenience helper:
    1) lead-time histogram
    2) missed vs detected bar chart
    3) returns display-ready table
    """
    plot_leadtime_distribution(event_lead_df)
    plot_event_detection_summary(event_lead_df)
    return build_event_leadtime_display_table(event_lead_df, top_n=top_n)

import matplotlib.pyplot as plt
import pandas as pd


def plot_alert_burden_distribution(
    patient_alert_burden_df: pd.DataFrame,
    burden_col: str = "alerts_per_patient_hour",
    bins: int = 15,
) -> None:
    if patient_alert_burden_df.empty:
        raise ValueError("patient_alert_burden_df is empty")
    if burden_col not in patient_alert_burden_df.columns:
        raise ValueError(f"Missing burden column: {burden_col}")

    plt.figure(figsize=(8, 5))
    plt.hist(patient_alert_burden_df[burden_col].dropna(), bins=bins)
    plt.title("Distribution of Alert Burden per Patient-Hour")
    plt.xlabel("Alerts per Patient-Hour")
    plt.ylabel("Number of Patients")
    plt.tight_layout()
    plt.show()


def plot_top_alert_burden_patients(
    patient_alert_burden_df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    if patient_alert_burden_df.empty:
        raise ValueError("patient_alert_burden_df is empty")

    display_df = (
        patient_alert_burden_df
        .sort_values("alerts_per_patient_hour", ascending=False)
        .head(top_n)
        .copy()
    )

    plt.figure(figsize=(10, 5))
    plt.bar(display_df["patient_id"].astype(str), display_df["alerts_per_patient_hour"])
    plt.title(f"Top {top_n} Patients by Alert Burden")
    plt.xlabel("Patient ID")
    plt.ylabel("Alerts per Patient-Hour")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return display_df


def plot_false_alert_episode_summary(
    false_alert_summary_df: pd.DataFrame,
) -> None:
    if false_alert_summary_df.empty:
        raise ValueError("false_alert_summary_df is empty")

    row = false_alert_summary_df.iloc[0]
    labels = ["True Alert Episodes", "False Alert Episodes"]
    values = [row["true_alert_episodes"], row["false_alert_episodes"]]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.title("True vs False Alert Episodes")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def plot_reliability_curves(
    df,
    horizon_name,
    target_col,
    risk_col,
    n_bins=10,
):
    """
    Plot calibration (reliability) curves for raw and calibrated probabilities.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing raw and calibrated probabilities.
    horizon_name : str
        Horizon label (e.g., "15min", "30min", "60min").
    target_col : str
        Ground truth binary target column.
    risk_col : str
        Raw probability column.
    n_bins : int
        Number of bins for calibration curve.
    """

    y_true = df[target_col].astype(int)

    curve_specs = [
        ("Raw XGBoost", risk_col),
        ("Platt Calibrated", f"{risk_col}_platt"),
        ("Isotonic Calibrated", f"{risk_col}_isotonic"),
    ]

    plt.figure(figsize=(7, 6))

    for label, col in curve_specs:
        prob_true, prob_pred = calibration_curve(
            y_true,
            df[col],
            n_bins=n_bins,
            strategy="quantile",
        )

        plt.plot(prob_pred, prob_true, marker="o", label=label)

    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")

    plt.title(f"Calibration Curve — {horizon_name}")
    plt.xlabel("Predicted Risk")
    plt.ylabel("Observed Event Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_calibrated_risk_distributions(
    df,
    horizon_name,
    risk_col,
    bins=30,
):
    """
    Plot distributions of raw and calibrated probabilities.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing probability columns.
    horizon_name : str
        Horizon label.
    risk_col : str
        Raw probability column.
    bins : int
        Number of histogram bins.
    """

    plt.figure(figsize=(8, 5))

    df[risk_col].hist(alpha=0.5, bins=bins, label="Raw")
    df[f"{risk_col}_platt"].hist(alpha=0.5, bins=bins, label="Platt")
    df[f"{risk_col}_isotonic"].hist(alpha=0.5, bins=bins, label="Isotonic")

    plt.title(f"Risk Score Distribution — {horizon_name}")
    plt.xlabel("Predicted Risk")
    plt.ylabel("Row Count")
    plt.legend()
    plt.show()

import matplotlib.pyplot as plt
import pandas as pd


def plot_bire_patient_timeline(
    df,
    patient_id=None,
    tier_col="bire_final_tier",
    score_col="bire_fi_score",
    event_time_col="event_time",
):
    """
    Plot BIRE tier progression and risk score for a single patient.

    This visual confirms:
    - ESCALATE / URGENT occur before event
    - MONITOR occurs after event
    """

    plot_df = df.copy()
    plot_df["timestamp"] = pd.to_datetime(plot_df["timestamp"])

    # Auto-select patient with interesting signal
    if patient_id is None:
        priority_patients = (
            plot_df[plot_df[tier_col].isin(["URGENT", "ESCALATE"])]
            ["patient_id"]
            .dropna()
            .unique()
        )

        if len(priority_patients) > 0:
            patient_id = priority_patients[0]
        else:
            patient_id = plot_df["patient_id"].iloc[0]

    patient_df = (
        plot_df[plot_df["patient_id"] == patient_id]
        .sort_values("timestamp")
        .copy()
    )

    tier_y_map = {
        "SUPPRESSED_WATCH": 0,
        "WATCH": 1,
        "ESCALATE": 2,
        "URGENT": 3,
        "MONITOR": 4,
    }

    patient_df["tier_y"] = patient_df[tier_col].map(tier_y_map)

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Tier step plot
    ax1.step(
        patient_df["timestamp"],
        patient_df["tier_y"],
        where="post",
        linewidth=2,
        label="BIRE Tier",
    )

    ax1.scatter(
        patient_df["timestamp"],
        patient_df["tier_y"],
        s=45,
        label="Episodes",
    )

    ax1.set_yticks(list(tier_y_map.values()))
    ax1.set_yticklabels(list(tier_y_map.keys()))
    ax1.set_xlabel("Time")
    ax1.set_ylabel("BIRE Tier")
    ax1.set_title(f"BIRE Timeline — {patient_id}")
    ax1.grid(True, alpha=0.3)

    # Risk score line
    ax2 = ax1.twinx()
    ax2.plot(
        patient_df["timestamp"],
        patient_df[score_col],
        linestyle="--",
        linewidth=2,
        label="BIRE-FI Score",
    )
    ax2.set_ylabel("BIRE-FI Score")
    ax2.set_ylim(0, 1.05)

    # Event markers
    if event_time_col in patient_df.columns:
        event_times = (
            pd.to_datetime(patient_df[event_time_col])
            .dropna()
            .unique()
        )

        for event_time in event_times:
            ax1.axvline(
                event_time,
                linestyle=":",
                linewidth=2,
                label="Deterioration Event",
            )

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return patient_id

def plot_bire_tier_transition_heatmap(
    df,
    patient_col="patient_id",
    time_col="timestamp",
    tier_col="bire_final_tier",
):
    """
    Plot a heatmap showing BIRE tier transitions across patients over time.

    Rows = patients
    Columns = ordered episode timestamps
    Values = BIRE tier severity level
    """

    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    plot_df = df.copy()
    plot_df[time_col] = pd.to_datetime(plot_df[time_col])

    tier_map = {
        "SUPPRESSED_WATCH": 0,
        "WATCH": 1,
        "ESCALATE": 2,
        "URGENT": 3,
        "MONITOR": 4,
    }

    plot_df["tier_numeric"] = plot_df[tier_col].map(tier_map)

    plot_df = plot_df.sort_values([patient_col, time_col])

    plot_df["episode_order"] = plot_df.groupby(patient_col).cumcount()

    heatmap_df = plot_df.pivot_table(
        index=patient_col,
        columns="episode_order",
        values="tier_numeric",
        aggfunc="first",
    )

    fig, ax = plt.subplots(figsize=(14, 8))

    im = ax.imshow(
        heatmap_df,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_title("BIRE Tier Transition Heatmap")
    ax.set_xlabel("Episode Order")
    ax.set_ylabel("Patient ID")

    ax.set_yticks(range(len(heatmap_df.index)))
    ax.set_yticklabels(heatmap_df.index)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_ticks(list(tier_map.values()))
    cbar.set_ticklabels(list(tier_map.keys()))
    cbar.set_label("BIRE Tier")

    plt.tight_layout()
    plt.show()

    return heatmap_df

import matplotlib.pyplot as plt
import pandas as pd


def plot_bire_lifecycle_timeline(
    df,
    patient_id,
    time_col="timestamp",
    risk_col="pred_proba",
    tier_col="bire_final_tier",
    event_col="event_now",
    title=None,
):
    """
    Plot full BIRE lifecycle timeline for one patient.

    Shows:
    - Risk trajectory
    - Final BIRE tier/state
    - Event markers
    - Post-event escalation path
    """

    p = df[df["patient_id"] == patient_id].copy()

    if p.empty:
        raise ValueError(f"No rows found for patient_id={patient_id}")

    required_cols = [time_col, risk_col, tier_col]
    missing = [col for col in required_cols if col not in p.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    p = p.sort_values(time_col)

    tier_styles = {
        "SUPPRESSED_WATCH": {"marker": "o", "label": "Suppressed Watch"},
        "WATCH": {"marker": "o", "label": "Watch"},
        "ESCALATE": {"marker": "^", "label": "Escalate"},
        "URGENT": {"marker": "X", "label": "Urgent"},
        "MONITOR": {"marker": "s", "label": "Monitor"},
        "RE-ESCALATE": {"marker": "D", "label": "Re-escalate"},
        "CRITICAL": {"marker": "*", "label": "Critical"},
    }

    plt.figure(figsize=(14, 5))

    # Main risk line
    plt.plot(
        p[time_col],
        p[risk_col],
        linewidth=2,
        label="BIRE Risk Score",
    )

    # Plot tiers
    for tier, style in tier_styles.items():
        subset = p[p[tier_col] == tier]

        if not subset.empty:
            plt.scatter(
                subset[time_col],
                subset[risk_col],
                marker=style["marker"],
                s=80 if tier != "CRITICAL" else 180,
                label=style["label"],
            )

    # Event markers
    if event_col in p.columns:
        event_rows = p[p[event_col] == 1]

        for event_time in event_rows[time_col]:
            plt.axvline(
                event_time,
                linestyle="--",
                linewidth=1,
                alpha=0.6,
            )

    plt.title(title or f"BIRE Full Lifecycle Timeline — Patient {patient_id}")
    plt.xlabel("Time")
    plt.ylabel("Risk Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import pandas as pd


def plot_bire_lifecycle_timeline(
    df,
    patient_id,
    time_col="timestamp",
    risk_col="pred_proba",
    tier_col="bire_final_tier",
    event_col="event_now",
    lead_time_col="lead_time_min",
    title=None,
):
    """
    Plot full BIRE lifecycle timeline for one patient.

    Shows:
    - Risk trajectory
    - Final BIRE tier/state
    - Event marker
    - Lead-time annotation
    - Pre-event alert timing
    """

    p = df[df["patient_id"] == patient_id].copy()

    if p.empty:
        raise ValueError(f"No rows found for patient_id={patient_id}")

    required_cols = [time_col, risk_col, tier_col]
    missing = [col for col in required_cols if col not in p.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    p[time_col] = pd.to_datetime(p[time_col])
    p = p.sort_values(time_col)

    tier_styles = {
        "SUPPRESSED_WATCH": {"marker": "o", "label": "Suppressed Watch"},
        "WATCH": {"marker": "o", "label": "Watch"},
        "ESCALATE": {"marker": "^", "label": "Escalate"},
        "URGENT": {"marker": "X", "label": "Urgent"},
        "MONITOR": {"marker": "s", "label": "Monitor"},
        "RE-ESCALATE": {"marker": "D", "label": "Re-escalate"},
        "CRITICAL": {"marker": "*", "label": "Critical"},
    }

    plt.figure(figsize=(15, 5))

    # Main risk trajectory
    plt.plot(
        p[time_col],
        p[risk_col],
        linewidth=2,
        label="BIRE Risk Score",
    )

    # Tier markers
    for tier, style in tier_styles.items():
        subset = p[p[tier_col] == tier]

        if not subset.empty:
            plt.scatter(
                subset[time_col],
                subset[risk_col],
                marker=style["marker"],
                s=90 if tier != "CRITICAL" else 220,
                label=style["label"],
            )

    # Event marker
    event_time = None

    if event_col in p.columns and p[event_col].eq(1).any():
        event_time = p.loc[p[event_col] == 1, time_col].min()

        plt.axvline(
            event_time,
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label="Event",
        )

        plt.annotate(
            "EVENT",
            xy=(event_time, 1.0),
            xytext=(event_time, 0.88),
            arrowprops=dict(arrowstyle="->", linewidth=1.5),
            ha="center",
        )

    # Lead-time annotation
    if event_time is not None:
        pre_event_alerts = p[
            (p[time_col] < event_time)
            & (p[risk_col] >= 0.4)
        
        ]
        
        if not pre_event_alerts.empty:
            first_alert_time = pre_event_alerts[time_col].min()
            first_alert_risk = pre_event_alerts.loc[
                pre_event_alerts[time_col] == first_alert_time,
                risk_col
            ].iloc[0]

            lead_time_minutes = int(
                (event_time - first_alert_time).total_seconds() / 60
            )

            plt.axvspan(
                first_alert_time,
                event_time,
                alpha=0.12,
                label=f"Lead Time Window ({lead_time_minutes} min)",
            )

            plt.annotate(
                f"First pre-event signal\nLead time: {lead_time_minutes} min",
                xy=(first_alert_time, first_alert_risk),
                xytext=(first_alert_time, min(first_alert_risk + 0.25, 0.95)),
                arrowprops=dict(arrowstyle="->", linewidth=1.5),
                ha="left",
            )

    plt.title(title or f"BIRE Full Lifecycle Timeline — Patient {patient_id}")
    plt.xlabel("Time")
    plt.ylabel("Risk Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_bire_lifecycle_heatmap(
    df,
    patient_ids=None,
    patient_col="patient_id",
    time_col="timestamp",
    tier_col="bire_final_tier",
    max_patients=25,
    title="BIRE Lifecycle State Heatmap",
):
    """
    Plot a multi-patient lifecycle heatmap.

    Shows final BIRE tier/state progression across time.
    """

    data = df.copy()

    required_cols = [patient_col, time_col, tier_col]
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    data[time_col] = pd.to_datetime(data[time_col])
    data = data.sort_values([patient_col, time_col])

    if patient_ids is not None:
        data = data[data[patient_col].isin(patient_ids)]
    else:
        patient_ids = (
            data[patient_col]
            .drop_duplicates()
            .head(max_patients)
            .tolist()
        )
        data = data[data[patient_col].isin(patient_ids)]

    tier_map = {
        "SUPPRESSED_WATCH": 0,
        "WATCH": 1,
        "ESCALATE": 2,
        "URGENT": 3,
        "MONITOR": 4,
        "RE-ESCALATE": 5,
        "CRITICAL": 6,
    }

    data["tier_numeric"] = data[tier_col].map(tier_map)

    # Create relative time index per patient
    data["step"] = data.groupby(patient_col).cumcount()

    heatmap_df = data.pivot_table(
        index=patient_col,
        columns="step",
        values="tier_numeric",
        aggfunc="max"
    )

    plt.figure(figsize=(16, max(5, len(heatmap_df) * 0.35)))

    im = plt.imshow(
        heatmap_df,
        aspect="auto",
        interpolation="nearest",
    )

    plt.title(title)
    plt.xlabel("Patient Timeline Step")
    plt.ylabel("Patient ID")

    plt.yticks(
        ticks=np.arange(len(heatmap_df.index)),
        labels=heatmap_df.index
    )

    cbar = plt.colorbar(im)
    cbar.set_ticks(list(tier_map.values()))
    cbar.set_ticklabels(list(tier_map.keys()))

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_bire_lifecycle_heatmap_with_timing(
    df,
    patient_ids=None,
    patient_col="patient_id",
    time_col="timestamp",
    tier_col="bire_final_tier",
    timing_col="timing_category",
    max_patients=25,
    title="BIRE Lifecycle Heatmap with Timing Categories",
):
    """
    Plot BIRE lifecycle heatmap with timing category overlays.

    Base color = final BIRE tier
    Overlay marker = timing category
    """

    data = df.copy()

    required_cols = [patient_col, time_col, tier_col, timing_col]
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    data[time_col] = pd.to_datetime(data[time_col])
    data = data.sort_values([patient_col, time_col])

    if patient_ids is not None:
        data = data[data[patient_col].isin(patient_ids)]
    else:
        patient_ids = (
            data[patient_col]
            .drop_duplicates()
            .head(max_patients)
            .tolist()
        )
        data = data[data[patient_col].isin(patient_ids)]

    tier_map = {
        "SUPPRESSED_WATCH": 0,
        "WATCH": 1,
        "ESCALATE": 2,
        "URGENT": 3,
        "MONITOR": 4,
        "RE-ESCALATE": 5,
        "CRITICAL": 6,
    }

    data["tier_numeric"] = data[tier_col].map(tier_map)
    data["step"] = data.groupby(patient_col).cumcount()

    heatmap_df = data.pivot_table(
        index=patient_col,
        columns="step",
        values="tier_numeric",
        aggfunc="max",
    )

    plt.figure(figsize=(17, max(5, len(heatmap_df) * 0.4)))

    im = plt.imshow(
        heatmap_df,
        aspect="auto",
        interpolation="nearest",
    )

    # Overlay timing category markers
    timing_markers = {
        "true_predictive_alert": "o",
        "post_event_alert": "x",
        "early_beyond_60_alert": "^",
    }

    timing_labels_used = set()

    patient_to_y = {
        patient_id: i for i, patient_id in enumerate(heatmap_df.index)
    }

    overlay_df = data[
        data[timing_col].isin(timing_markers.keys())
    ].copy()

    for _, row in overlay_df.iterrows():
        patient = row[patient_col]

        if patient not in patient_to_y:
            continue

        x = row["step"]
        y = patient_to_y[patient]
        timing = row[timing_col]
        marker = timing_markers[timing]

        label = timing if timing not in timing_labels_used else None
        timing_labels_used.add(timing)

        plt.scatter(
            x,
            y,
            marker=marker,
            s=60,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            label=label,
        )

    plt.title(title)
    plt.xlabel("Patient Timeline Step")
    plt.ylabel("Patient ID")

    plt.yticks(
        ticks=np.arange(len(heatmap_df.index)),
        labels=heatmap_df.index,
    )

    cbar = plt.colorbar(im)
    cbar.set_ticks(list(tier_map.values()))
    cbar.set_ticklabels(list(tier_map.keys()))

    plt.legend(
        title="Timing Category Overlay",
        loc="upper right",
        bbox_to_anchor=(1.25, 1),
    )

    plt.tight_layout()
    plt.show()
