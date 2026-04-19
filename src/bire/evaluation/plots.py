import matplotlib.pyplot as plt


# -------------------------------
# Core trajectory plot (simple)
# -------------------------------
def plot_patient_risk_trajectory(
    df,
    patient_id,
    time_col="timestamp",
    risk_col="pred_proba",
    alert_col="alert",
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
    """
    Plot alert counts by patient for demo summary output.

    Parameters
    ----------
    demo_summary_df : pd.DataFrame
        DataFrame expected to contain:
        - patient_id
        - alerts
    """
    required_cols = {"patient_id", "alerts"}
    missing = required_cols - set(demo_summary_df.columns)
    if missing:
        raise ValueError(f"demo_summary_df is missing required columns: {missing}")

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
