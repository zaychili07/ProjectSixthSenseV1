from __future__ import annotations

import pandas as pd
import numpy as np


# =========================================================
# MULTI-PATIENT SURVEILLANCE
# =========================================================

# this function builds a summary of the operational surveillance data by state,
# showing the number of unique patients, total rows observed, average risk, and average confidence for each state.
# It can be used to understand the distribution of patients across different states and the associated risk and confidence levels.
def build_operational_surveillance_summary(
    df: pd.DataFrame,
    state_col: str = "gss_v3_state",
    patient_col: str = "patient_id",
    risk_col: str = "risk_60min",
    confidence_col: str = (
        "weighted_adjusted_confidence_score"
    ),
):
    """
    Build multi-patient operational surveillance summary.

    Purpose:
    - summarize operational burden
    - summarize escalation distribution
    - summarize environment-level instability
    - support surveillance board architecture
    """

    surveillance_df = (
        df.groupby(state_col)
        .agg(
            patients=(patient_col, "nunique"),
            rows=(patient_col, "size"),
            avg_risk_60min=(risk_col, "mean"),
            avg_confidence=(
                confidence_col,
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "avg_risk_60min",
            ascending=False,
        )
    )

    return surveillance_df

# this function is a bit rough but it captures the idea of building a patient-level priority board based on the most
# concerning state, risk, confidence, and uncertainty observed for each patient. It can be further refined and expanded
# with additional features as needed.
def build_patient_priority_board(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    state_col: str = "gss_v3_state",
    risk_col: str = "risk_60min",
    confidence_col: str = "weighted_adjusted_confidence_score",
    uncertainty_col: str = "weighted_adjusted_uncertainty_score",
) -> pd.DataFrame:
    """
    Build a patient-level operational priority board.

    This summarizes each patient's highest current operational concern
    using risk, state severity, confidence, and uncertainty.
    """

    out = df.copy()

    state_priority = {
        "CRITICAL": 6,
        "URGENT": 5,
        "ESCALATE": 4,
        "RE-ESCALATE": 4,
        "WATCH": 3,
        "MONITOR": 2,
        "SUPPRESS": 1,
    }

    out["state_priority_score"] = (
        out[state_col]
        .astype(str)
        .str.upper()
        .map(state_priority)
        .fillna(0)
    )

    patient_board = (
        out.groupby(patient_col)
        .agg(
            latest_state=(state_col, "last"),
            max_state_priority=("state_priority_score", "max"),
            max_risk_60min=(risk_col, "max"),
            avg_risk_60min=(risk_col, "mean"),
            avg_confidence=(confidence_col, "mean"),
            max_uncertainty=(uncertainty_col, "max"),
            rows_observed=(patient_col, "size"),
        )
        .reset_index()
    )

    patient_board["operational_priority_score"] = (
        (patient_board["max_state_priority"] * 10)
        + (patient_board["max_risk_60min"] * 100)
        + (patient_board["max_uncertainty"] * 25)
        - (patient_board["avg_confidence"] * 10)
    )

    patient_board = patient_board.sort_values(
        "operational_priority_score",
        ascending=False,
    )

    return patient_board

def add_psr_operational_priority(
    patient_board_df: pd.DataFrame,
    psr_col: str = "attention_score",
) -> pd.DataFrame:
    """
    Add PSR-aware operational prioritization.

    Purpose:
    - integrate surveillance burden
    - integrate operational attention scoring
    - strengthen command-center prioritization
    """

    out = patient_board_df.copy()

    if psr_col not in out.columns:
        out[psr_col] = 0

    out["psr_operational_priority_score"] = (
        out["operational_priority_score"]
        + out[psr_col]
    )

    out = out.sort_values(
        "psr_operational_priority_score",
        ascending=False,
    )

    return out

def add_patient_surveillance_score(
    patient_board_df: pd.DataFrame,
    state_priority_col: str = "max_state_priority",
    risk_col: str = "max_risk_60min",
    avg_risk_col: str = "avg_risk_60min",
    uncertainty_col: str = "max_uncertainty",
    confidence_col: str = "avg_confidence",
    rows_col: str = "rows_observed",
) -> pd.DataFrame:
    """
    Add Patient Surveillance Ranking score.

    PSR estimates operational attention burden.
    """

    out = patient_board_df.copy()

    out["attention_score"] = (
        (out[state_priority_col] * 20)
        + (out[risk_col] * 50)
        + (out[avg_risk_col] * 25)
        + (out[uncertainty_col] * 20)
        + (out[rows_col] / out[rows_col].max() * 10)
        - (out[confidence_col] * 10)
    ).round(3)

    return out

# this function adds a queue status based on the PSR attention score, categorizing patients into "ACTIVE_QUEUE" or
# "BACKGROUND_SURVEILLANCE" based on whether they meet a specified threshold. It also optionally excludes patients in a
# "WATCH" state from being classified as active, ensuring that the queue status reflects both the PSR score and the patient's
# current state.
def add_psr_queue_status(
    df: pd.DataFrame,
    psr_col: str = "attention_score",
    active_threshold: float = 150.0,
    watch_excluded: bool = True,
    state_col: str = "latest_state",
) -> pd.DataFrame:
    """
    Add PSR-based queue status.

    This does not suppress patients.
    It separates active command-center visibility from background surveillance.
    """

    out = df.copy()

    out["queue_status"] = "BACKGROUND_SURVEILLANCE"

    active_mask = out[psr_col] >= active_threshold

    if watch_excluded and state_col in out.columns:
        active_mask = active_mask & (
            out[state_col].astype(str).str.upper() != "WATCH"
        )

    out.loc[active_mask, "queue_status"] = "ACTIVE_QUEUE"

    out["queue_reason"] = out["queue_status"].map(
        {
            "ACTIVE_QUEUE": "PSR threshold met for active operational queue visibility.",
            "BACKGROUND_SURVEILLANCE": "Patient remains monitored but below active queue threshold.",
        }
    )

    return out

# This function simulates operational movement by creating a new column that indicates the cycle of movement
# based on the patient's state and a specified window size.

# It also creates a simulated operational state that combines the original state with the movement
# cycle, and assigns a movement priority based on the PSR attention score. This can be used to model
# how patients might move through different operational states over time and
# how their priority might change accordingly.
def simulate_operational_movement(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    state_col: str = "latest_state",
    risk_col: str = "max_risk_60min",
    movement_window: int = 5,
):
    """
    Simulate lightweight operational movement.

    Purpose:
    - simulate evolving operational burden
    - simulate changing surveillance environments
    - support future dynamic operational intelligence
    """

    out = df.copy()

    out["movement_cycle"] = (
        out.groupby(state_col)
        .cumcount() // movement_window
    )

    out["simulated_operational_state"] = (
        out[state_col].astype(str)
        + "_CYCLE_"
        + out["movement_cycle"].astype(str)
    )

    out["movement_priority"] = (
        out[risk_col]
        .rank(
            ascending=False,
            method="dense",
        )
    )

    return out

# This function adds a trajectory context to the dataframe based on the risk and uncertainty columns.
# It categorizes patients into different trajectory contexts such as "WORSENING_ACCELERATING", "WORSENING_STABLE",
# "UNSTABLE_VOLATILE", or "IMPROVING_STABILIZING" based on specified conditions. This can help provide additional
# context for understanding the patient's trajectory and potential changes in their condition over time.
def add_bire_fi_queue_context(
    df: pd.DataFrame,
    risk_col: str = "avg_risk_60min",
    uncertainty_col: str = "max_uncertainty",
):
    """
    Add lightweight BIRE-FI trajectory context
    into queue orchestration.
    """

    out = df.copy()

    conditions = [
        (
            (out[risk_col] >= 0.60)
            & (out[uncertainty_col] >= 0.75)
        ),

        (
            (out[risk_col] >= 0.45)
            & (out[uncertainty_col] < 0.75)
        ),

        (
            (out[risk_col] < 0.45)
            & (out[uncertainty_col] >= 0.75)
        ),
    ]

    values = [
        "WORSENING_ACCELERATING",
        "WORSENING_STABLE",
        "UNSTABLE_VOLATILE",
    ]

    out["trajectory_context"] = np.select(
        conditions,
        values,
        default="IMPROVING_STABILIZING",
    )

    return out

# This function adds priority-aligned queue language to the dataframe based on the trajectory context and queue status.
def add_priority_aligned_queue_language(
    df: pd.DataFrame,
    trajectory_col: str = "trajectory_context",
    psr_col: str = "attention_score",
    queue_col: str = "queue_status",
) -> pd.DataFrame:
    """
    Align trajectory language with PSR queue authority.

    BIRE-FI provides trajectory context.
    PSR remains the operational prioritization authority.
    """

    out = df.copy()

    def build_context(row):
        trajectory = str(row[trajectory_col])
        psr = row[psr_col]
        queue = str(row[queue_col])

        if queue == "ACTIVE_QUEUE":
            if trajectory == "WORSENING_ACCELERATING":
                return "ACTIVE_HIGH_PRIORITY_WORSENING"
            if trajectory == "UNSTABLE_VOLATILE":
                return "ACTIVE_VOLATILE_PRIORITY"
            if trajectory == "WORSENING_STABLE":
                return "ACTIVE_STABLE_WORSENING"
            if trajectory == "IMPROVING_STABILIZING":
                return "ACTIVE_BUT_STABILIZING"

        if queue == "BACKGROUND_SURVEILLANCE":
            if trajectory == "UNSTABLE_VOLATILE":
                return "BACKGROUND_VOLATILE_MONITORING"
            if trajectory == "WORSENING_STABLE":
                return "BACKGROUND_WORSENING_MONITORING"
            if trajectory == "IMPROVING_STABILIZING":
                return "BACKGROUND_STABILIZING"
            if trajectory == "WORSENING_ACCELERATING":
                return "BACKGROUND_HIGH_RISK_REVIEW"

        return "QUEUE_CONTEXT_REVIEW"

    out["priority_aligned_context"] = out.apply(
        build_context,
        axis=1,
    )

    out["priority_language_note"] = (
        "PSR remains final queue authority; trajectory context provides supporting interpretation."
    )

    return out

# This function builds a structured BIRE output dictionary for a single patient based on their latest data row.
# It calculates the risk score using the provided model, determines the risk band and alert status,
# identifies the top drivers of risk based on feature deltas, summarizes recent trends in key vitals, and assesses data quality.
def add_ibpip_queue_context(
    df: pd.DataFrame,
    uncertainty_col: str = "max_uncertainty",
    confidence_col: str = "avg_confidence",
):
    """
    Add lightweight IBPIP baseline interpretation
    into operational queue orchestration.
    """

    out = df.copy()

    conditions = [

        (
            (out[uncertainty_col] >= 0.80)
            & (out[confidence_col] >= 0.85)
        ),

        (
            (out[uncertainty_col] >= 0.65)
            & (out[confidence_col] >= 0.80)
        ),

        (
            (out[uncertainty_col] < 0.65)
            & (out[confidence_col] >= 0.90)
        ),
    ]

    values = [
        "PATIENT_SPECIFIC_ESCALATION",
        "BASELINE_DRIFT_MONITORING",
        "NEAR_BASELINE_STABLE",
    ]

    out["baseline_context"] = np.select(
        conditions,
        values,
        default="BASELINE_CONTEXT_REVIEW",
    )

    return out