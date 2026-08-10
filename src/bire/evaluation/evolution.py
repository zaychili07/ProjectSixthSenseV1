import numpy as np
import pandas as pd

# this function simulates operational state evolution within the BIRE OS, based on current state, trajectory context, and PSR score.
def evolve_operational_states(
    df: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:

    """
    Simulate operational state evolution within BIRE OS.
    """

    np.random.seed(random_state)

    evolved_df = df.copy()

    next_states = []

    for _, row in evolved_df.iterrows():

        current_state = row["latest_state"]
        trajectory = row["trajectory_context"]
        psr = row["psr_operational_priority_score"]

        # ----------------------------------------
        # ESCALATE logic
        # ----------------------------------------

        if current_state == "ESCALATE":

            if trajectory == "WORSENING_ACCELERATING":
                next_state = np.random.choice(
                    ["URGENT", "ESCALATE"],
                    p=[0.65, 0.35],
                )

            elif trajectory == "IMPROVING_STABILIZING":
                next_state = np.random.choice(
                    ["STABILIZING", "ESCALATE"],
                    p = [0.70, 0.30],
                )

            elif trajectory == "UNSTABLE_VOLATILE":
                next_state = np.random.choice(
                    ["ESCALATE", "URGENT"],
                    p =[0.60, 0.40],
                )

            else:
                next_state = "ESCALATE"

        # ----------------------------------------
        # URGENT logic
        # ----------------------------------------

        elif current_state == "URGENT":

            if psr >= 320:
                next_state = "CRITICAL"

            else:
                next_state = np.random.choice(
                    ["URGENT", "RECOVERING"],
                    p =[0.70, 0.30],
                )

        # ----------------------------------------
        # SUPPRESS logic
        # ----------------------------------------

        elif current_state == "SUPPRESS":

            if trajectory == "WORSENING_ACCELERATING":
                next_state = "WATCH"

            else:
                next_state = "SUPPRESS"

        else:
            next_state = current_state

        next_states.append(next_state)

    evolved_df ["next_operational_state"] = next_states

    return evolved_df

# this function computes a composite temporal operational pressure score for each case,
# based on queue status, trajectory context, and latest state.
def compute_temporal_operational_pressure(
    df: pd.DataFrame,
) -> pd.DataFrame:

    """
    Compute ecosystem-wide operational pressure metrics.
    """

    pressure_df = df.copy()

    # ----------------------------------------
    # Pressure components
    # ----------------------------------------

    active_queue_pressure = (
        pressure_df["queue_status"]
        == "ACTIVE_QUEUE"
    ).astype(int)

    worsening_pressure = (
        pressure_df["trajectory_context"]
        == "WORSENING_ACCELERATING"
    ).astype(int)

    volatile_pressure = (
        pressure_df["trajectory_context"]
        == "UNSTABLE_VOLATILE"
    ).astype(int)

    urgent_pressure = (
        pressure_df["latest_state"]
        == "URGENT"
    ).astype(int)

    # ----------------------------------------
    # Aggregate operational pressure
    # ----------------------------------------

    pressure_df["temporal_operational_pressure"] = (
        active_queue_pressure * 2
        + worsening_pressure * 3
        + volatile_pressure * 2
        + urgent_pressure * 4
    )

    return pressure_df


# this function applies de-escalation reintegration logic for patients in WATCH and SUPPRESSED_WATCH states,
# based on trajectory and operational pressure.
def apply_deescalation_reintegration(
    df: pd.DataFrame,
) -> pd.DataFrame:

    """
    Apply WATCH and SUPPRESSED_WATCH reintegration logic.
    """

    reintegration_df = df.copy()

    reintegrated_states = []

    for _, row in reintegration_df.iterrows():

        current_state = row["next_operational_state"]
        trajectory = row["trajectory_context"]
        pressure = row["temporal_operational_pressure"]

        # ----------------------------------------
        # Stabilizing ESCALATE patients
        # ----------------------------------------

        if (
            current_state == "STABILIZING"
            and trajectory == "IMPROVING_STABILIZING"
            and pressure <= 2
        ):

            next_state = "WATCH"

        # ----------------------------------------
        # WATCH decompression
        # ----------------------------------------

        elif (
            current_state == "WATCH"
            and trajectory == "IMPROVING_STABILIZING"
            and pressure <= 1
        ):

            next_state = "SUPPRESSED_WATCH"

        # ----------------------------------------
        # Volatile worsening remains elevated
        # ----------------------------------------

        elif (
            trajectory == "WORSENING_ACCELERATING"
            and pressure >= 4
        ):

            next_state = current_state

        else:
            next_state = current_state

        reintegrated_states.append(next_state)

    reintegration_df[
        "reintegrated_operational_state"
    ] = reintegrated_states

    return reintegration_df

# this function simulates continuous operational queue evolution over multiple cycles, applying state evolution,
# pressure computation, and reintegration logic iteratively.
def compute_temporal_operational_pressure(
    df: pd.DataFrame,
) -> pd.DataFrame:

    """
    Compute ecosystem-wide operational pressure metrics.
    """

    pressure_df = df.copy()

    # ----------------------------------------
    # Pressure Components
    # ----------------------------------------

    active_queue_pressure = (
        pressure_df["queue_status"] == "ACTIVE_QUEUE"
    ).astype(int)

    worsening_pressure = (
        pressure_df["trajectory_context"] == "WORSENING_ACCELERATING"
    ).astype(int)

    volatile_pressure = (
        pressure_df["trajectory_context"] == "UNSTABLE_VOLATILE"
    ).astype(int)

    urgent_pressure = (
        pressure_df["latest_state"] == "URGENT"
    ).astype(int)

    # ----------------------------------------
    # Aggregate Operational Pressure
    # ----------------------------------------

    pressure_df["temporal_operational_pressure"] = (
        active_queue_pressure * 2
        + worsening_pressure * 3
        + volatile_pressure * 2
        + urgent_pressure * 4
    )

    return pressure_df



#this function simulates continuous operational queue evolution over multiple cycles, applying state evolution,
#  pressure computation, and reintegration logic iteratively.
def simulate_continuous_operational_queue(
    df: pd.DataFrame,
    cycles: int = 5,
) -> pd.DataFrame:

    """
    Simulate continuous operational queue evolution.
    """

    simulation_df = df.copy()

    cycle_history = []

    current_df = simulation_df.copy()

    for cycle in range(cycles):

        # ----------------------------------------
        # Step 1: Evolve operational states
        # ----------------------------------------

        evolved_df = evolve_operational_states(
            current_df
        )

        # ----------------------------------------
        # Step 2: Compute temporal pressure
        # ----------------------------------------

        pressure_df = compute_temporal_operational_pressure(
            evolved_df
        )

        # ----------------------------------------
        # Step 3: Apply reintegration logic
        # ----------------------------------------

        reintegration_df = apply_deescalation_reintegration(
            pressure_df
        )

        reintegration_df["simulation_cycle"] = cycle

        cycle_history.append(
            reintegration_df.copy()
        )

        # ----------------------------------------
        # Step 4: Prepare next cycle
        # ----------------------------------------

        current_df = reintegration_df.copy()

        # Remove old latest_state
        current_df = current_df.drop(
            columns=["latest_state"]
        )

        # Promote reintegrated state into latest_state
        current_df = current_df.rename(
            columns={
                "reintegrated_operational_state": "latest_state"
            }
        )

    # ----------------------------------------
    # Final simulation output
    # ----------------------------------------

    final_df = pd.concat(
        cycle_history,
        ignore_index=True,
    )

    return final_df

# this function builds a clinician-facing live simulation view, removing internal orchestration fields and
#  adding human-readable interpretations of state transitions.
def build_clinician_live_simulation_view(
    simulation_df: pd.DataFrame,
    cycle_col: str = "simulation_cycle",
    patient_col: str = "patient_id",
    psr_col: str = "psr_operational_priority_score",
) -> pd.DataFrame:
    """
    Build a clinician-facing live simulation view.

    This removes internal orchestration-only fields and keeps
    clinician-facing operational interpretation fields.
    """

    out = simulation_df.copy()

    # Simulated clock: each cycle = 5 minutes
    out["simulated_minutes_elapsed"] = out[cycle_col] * 5

    # Human-readable movement label
    out["state_transition"] = (
        out["latest_state"].astype(str)
        + " → "
        + out["reintegrated_operational_state"].astype(str)
    )

    # Clinician-safe movement interpretation
    def describe_movement(row):
        previous_state = str(row["latest_state"])
        new_state = str(row["reintegrated_operational_state"])

        if previous_state == new_state:
            return "No major operational change"

        if new_state in ["URGENT", "CRITICAL"]:
            return "Patient moved into higher-priority review"

        if new_state in ["WATCH", "SUPPRESSED_WATCH"]:
            return "Patient appears to be de-escalating into lower-intensity surveillance"

        if new_state in ["STABILIZING", "RECOVERING"]:
            return "Patient appears to be stabilizing"

        if new_state == "ESCALATE":
            return "Patient remains in elevated surveillance"

        return "Operational state updated"

    out["clinician_movement_summary"] = out.apply(
        describe_movement,
        axis=1,
    )

    keep_cols = [
        patient_col,
        cycle_col,
        "simulated_minutes_elapsed",
        "latest_state",
        "next_operational_state",
        "reintegrated_operational_state",
        "state_transition",
        "queue_status",
        psr_col,
        "trajectory_context",
        "baseline_context",
        "clinician_movement_summary",
    ]

    keep_cols = [col for col in keep_cols if col in out.columns]

    clinician_view = (
        out[keep_cols]
        .sort_values(
            [cycle_col, psr_col],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )

    return clinician_view