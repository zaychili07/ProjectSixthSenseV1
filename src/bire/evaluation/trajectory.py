import pandas as pd
import matplotlib.pyplot as plt


def plot_patient_trajectory(df: pd.DataFrame, patient_id: str, threshold: float = 0.5):
    """
    Plot risk trajectory for a single patient.
    """
    patient_df = df[df["patient_id"] == patient_id].copy()

    plt.figure(figsize=(10, 4))

    plt.plot(
        patient_df["timestamp"],
        patient_df["pred_proba"],
        marker="o",
        label="Predicted risk"
    )

    plt.axhline(threshold, linestyle="--", label=f"Threshold = {threshold}")

    # Plot alerts if they exist
    if "alert" in patient_df.columns:
        alert_points = patient_df[patient_df["alert"] == 1]
        plt.scatter(
            alert_points["timestamp"],
            alert_points["pred_proba"],
            s=80,
            label="Alert"
        )

    plt.title(f"BIRE Risk Trajectory — Patient {patient_id}")
    plt.xlabel("Timestamp")
    plt.ylabel("Predicted risk")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def compute_lead_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute lead time between first alert and first event.
    """
    results = []

    for patient_id, group in df.groupby("patient_id"):
        group = group.sort_values("timestamp")

        first_alert_time = (
            group.loc[group["alert"] == 1, "timestamp"].min()
            if "alert" in group.columns else pd.NaT
        )

        first_event_time = group.loc[group["target"] == 1, "timestamp"].min()

        if pd.notna(first_alert_time) and pd.notna(first_event_time):
            lead_time = (first_event_time - first_alert_time).total_seconds() / 60
        else:
            lead_time = None

        results.append({
            "patient_id": patient_id,
            "first_alert_time": first_alert_time,
            "first_event_time": first_event_time,
            "lead_time_minutes": lead_time,
        })

    return pd.DataFrame(results)
