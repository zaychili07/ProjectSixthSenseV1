import pandas as pd


def gss_v3_decision(
    row,
    high_short_risk: float = 0.85,
    high_mid_risk: float = 0.75,
    high_long_risk: float = 0.70,
    watch_long_risk: float = 0.45,
    convergence_threshold: float = 0.03,
    spread_threshold: float = 0.12,
    velocity_escalate: float = 0.02,
    acceleration_escalate: float = 0.01,
    velocity_watch: float = 0.0075,
):
    """
    GSS V3.1 trajectory-aware decision logic.

    Design goal:
    - Avoid instant alerts from isolated spikes.
    - Escalate only when high risk is supported by horizon agreement
      or meaningful worsening over time.
    """

    r15 = row["risk_15min"]
    r30 = row["risk_30min"]
    r60 = row["risk_60min"]

    spread = row["risk_spread"]
    convergence = row["risk_convergence"]
    velocity = row["risk_velocity"]
    acceleration = row["risk_acceleration"]

    velocity = 0 if pd.isna(velocity) else velocity
    acceleration = 0 if pd.isna(acceleration) else acceleration

    horizon_confirmed_high_risk = (
        r15 >= high_short_risk
        and r30 >= high_mid_risk
    )

    converged_high_risk = (
        r60 >= high_long_risk
        and r30 >= high_mid_risk
        and convergence <= convergence_threshold
    )

    rapid_worsening_confirmed = (
        r60 >= watch_long_risk
        and velocity >= velocity_escalate
        and acceleration >= acceleration_escalate
    )

    if (
        horizon_confirmed_high_risk
        or converged_high_risk
        or rapid_worsening_confirmed
    ):
        return "ESCALATE"

    if (
        r60 >= watch_long_risk
        or spread >= spread_threshold
        or velocity >= velocity_watch
    ):
        return "WATCH"

    return "SUPPRESS"


def apply_gss_v3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply GSS V3.1 to an entire dataframe.
    """

    out = df.copy()

    out["gss_v3_state"] = out.apply(gss_v3_decision, axis=1)
    out["gss_v3_alert"] = (out["gss_v3_state"] == "ESCALATE").astype(int)

    return out
