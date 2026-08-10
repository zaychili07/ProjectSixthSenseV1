import pandas as pd


def _validate_bridge_inputs(df):
    required = [
        "pred_proba",
        "bire_fi_trajectory_state",
        "ibpip_state",
        "ibpip_baseline_ready",
        "bms_mode",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required escalation bridge columns: {missing}"
        )


def add_forecast_escalation_bridge(df):
    """
    Forecast-aware escalation bridge layer.

    This layer does NOT replace GSS.
    It provides operational guidance signals based on:

    - BIRE-FI trajectory intelligence
    - IBPIP baseline readiness
    - BMS operational context
    """

    _validate_bridge_inputs(df)

    out = df.copy()

    out["forecast_escalation_signal"] = "NO_ACTION"
    out["forecast_escalation_reason"] = "NONE"
    out["forecast_escalation_confidence"] = "LOW"

    ###############################################
    # Rule 1 — Insufficient baseline maturity
    ###############################################

    insufficient_mask = (
        (out["ibpip_baseline_ready"] == False)
        &
        (
            out["bire_fi_trajectory_state"]
            == "INSUFFICIENT_HISTORY"
        )
    )

    out.loc[
        insufficient_mask,
        "forecast_escalation_signal"
    ] = "DEFER_INSUFFICIENT_CONTEXT"

    out.loc[
        insufficient_mask,
        "forecast_escalation_reason"
    ] = "BASELINE_WARMUP"

    ###############################################
    # Rule 2 — Accelerating deterioration
    ###############################################

    accel_mask = (
        out["bire_fi_trajectory_state"]
        == "WORSENING_ACCELERATING"
    )

    out.loc[
        accel_mask,
        "forecast_escalation_signal"
    ] = "CONSIDER_ESCALATION"

    out.loc[
        accel_mask,
        "forecast_escalation_reason"
    ] = "ACCELERATING_TRAJECTORY"

    out.loc[
        accel_mask,
        "forecast_escalation_confidence"
    ] = "HIGH"

    ###############################################
    # Rule 3 — Volatile instability
    ###############################################

    volatile_mask = (
        out["bire_fi_trajectory_state"]
        == "UNSTABLE_VOLATILE"
    )

    out.loc[
        volatile_mask,
        "forecast_escalation_signal"
    ] = "WATCH_CLOSELY"

    out.loc[
        volatile_mask,
        "forecast_escalation_reason"
    ] = "VOLATILE_TRAJECTORY"

    out.loc[
        volatile_mask,
        "forecast_escalation_confidence"
    ] = "MEDIUM"

    ###############################################
    # Rule 4 — Stabilizing trajectory
    ###############################################

    stabilizing_mask = (
        out["bire_fi_trajectory_state"]
        == "STABILIZING"
    )

    out.loc[
        stabilizing_mask,
        "forecast_escalation_signal"
    ] = "STABILIZING_NO_ACTION"

    out.loc[
        stabilizing_mask,
        "forecast_escalation_reason"
    ] = "TRAJECTORY_IMPROVING"

    ###############################################
    # Rule 5 — Low stable
    ###############################################

    low_stable_mask = (
        out["bire_fi_trajectory_state"]
        == "LOW_STABLE"
    )

    out.loc[
        low_stable_mask,
        "forecast_escalation_signal"
    ] = "CONTINUE_SUPPRESSION"

    out.loc[
        low_stable_mask,
        "forecast_escalation_reason"
    ] = "LOW_STABLE_TRAJECTORY"

    return out