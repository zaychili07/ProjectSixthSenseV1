from __future__ import annotations

import pandas as pd

# this module provides functions to build and summarize continuous surveillance frames for operational evaluation of the BIRE model.
def build_continuous_surveillance_frames(
    df: pd.DataFrame,
    cycle_col: str = "movement_cycle",
    queue_col: str = "queue_status",
    patient_col: str = "patient_id",
    risk_col: str = "max_risk_60min",
    psr_col: str = "attention_score",
) -> dict[int, pd.DataFrame]:
    """
    Build lightweight continuous surveillance frames.

    Each movement cycle becomes one simulated operational refresh frame.
    """

    frames = {}

    for cycle, frame_df in df.groupby(cycle_col):
        frame = frame_df.copy()

        frame["refresh_cycle"] = int(cycle)

        frame["refresh_rank"] = (
            frame[psr_col]
            .rank(ascending=False, method="dense")
            .astype(int)
        )

        frame = frame.sort_values(
            [queue_col, "refresh_rank"],
            ascending=[True, True],
        )

        frames[int(cycle)] = frame

    return frames


def summarize_continuous_surveillance_frames(
    frames: dict[int, pd.DataFrame],
    queue_col: str = "queue_status",
    risk_col: str = "max_risk_60min",
    psr_col: str = "attention_score",
) -> pd.DataFrame:
    """
    Summarize each simulated refresh frame.
    """

    rows = []

    for cycle, frame in frames.items():
        queue_counts = frame[queue_col].value_counts().to_dict()

        rows.append({
            "refresh_cycle": cycle,
            "patients": frame.shape[0],
            "active_queue": queue_counts.get("ACTIVE_QUEUE", 0),
            "background_surveillance": queue_counts.get(
                "BACKGROUND_SURVEILLANCE",
                0,
            ),
            "avg_risk": frame[risk_col].mean(),
            "avg_psr": frame[psr_col].mean(),
            "max_psr": frame[psr_col].max(),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("refresh_cycle")
        .reset_index(drop=True)
    )


def refresh_active_queue(
    active_df: pd.DataFrame,
    background_df: pd.DataFrame,
    max_active: int = 20
) -> pd.DataFrame:
    """
    Refresh the active patient queue by elevating background surveillance patients
    based on PSR operational priority scores. Allows swaps if background PSR is higher.

    Parameters:
        active_df (pd.DataFrame): Current active queue
        background_df (pd.DataFrame): Background surveillance patients
        max_active (int): Maximum allowed active patients in queue

    Returns:
        pd.DataFrame: Updated active queue
    """

    # Step 1: Keep only active ESCALATE / URGENT patients
    active_df = active_df[active_df['latest_state'].isin(['ESCALATE', 'URGENT'])].copy()

    # Step 2: Rank background patients by PSR
    ranked_background = background_df.sort_values(
        by='psr_operational_priority_score', ascending=False
    ).copy()

    # Step 3: Check for available slots
    slots_available = max_active - len(active_df)

    if slots_available > 0:
        # Fill empty slots directly
        to_activate = ranked_background.head(slots_available).copy()
        to_activate['movement_cycle'] += 1
        to_activate['simulated_operational_state'] = 'ACTIVE_QUEUE'
        updated_active_df = pd.concat([active_df, to_activate], ignore_index=True)
        return updated_active_df

    # Step 4: If queue full, check for swaps
    active_df = active_df.sort_values(by='psr_operational_priority_score')
    for idx, bg_row in ranked_background.iterrows():
        lowest_active_psr = active_df.iloc[0]['psr_operational_priority_score']
        if bg_row['psr_operational_priority_score'] > lowest_active_psr:
            # Swap background patient in
            to_activate = bg_row.copy()
            to_activate['movement_cycle'] += 1
            to_activate['simulated_operational_state'] = 'ACTIVE_QUEUE'

            # Replace lowest PSR active patient
            active_df.iloc[0] = to_activate
            active_df = active_df.sort_values(by='psr_operational_priority_score')
        else:
            # No more swaps possible
            break

    return active_df.reset_index(drop=True)


# This function simulates the operational refresh of the active patient queue by promoting background surveillance patients based on their PSR scores.
# It first fills any available slots with the highest PSR background patients, then checks for potential swaps if the active queue is full,
# ensuring that only patients with higher PSR scores can replace those in the active queue.
# The function returns an updated active queue DataFrame reflecting these changes.
def build_queue_refresh_audit_log(
    previous_df: pd.DataFrame,
    updated_df: pd.DataFrame,
    refresh_cycle: int,
) -> pd.DataFrame:

    previous = previous_df.copy()
    updated = updated_df.copy()

    previous = previous.rename(columns={
        "queue_status": "previous_queue_status"
    })

    updated = updated.rename(columns={
        "queue_status": "new_queue_status"
    })

    audit = previous.merge(
        updated[
            [
                "patient_id",
                "new_queue_status",
                "psr_operational_priority_score",
                "trajectory_context",
                "baseline_context",
            ]
        ],
        on="patient_id",
        how="outer",
    )

    audit["refresh_cycle"] = refresh_cycle

    audit["movement_detected"] = (
    audit["previous_queue_status"].notna()
    &
    audit["new_queue_status"].notna()
    &
    (
        audit["previous_queue_status"]
        != audit["new_queue_status"]
    )
)


    def determine_reason(row):

        if row["movement_detected"]:

            if row["new_queue_status"] == "ACTIVE_QUEUE":
                return "PROMOTED_TO_ACTIVE_QUEUE"

            if row["new_queue_status"] == "BACKGROUND_SURVEILLANCE":
                return "RETURNED_TO_BACKGROUND"

        return "NO_QUEUE_CHANGE"

    audit["movement_reason"] = audit.apply(
        determine_reason,
        axis=1,
    )

    return audit