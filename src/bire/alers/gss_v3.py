import pandas as pd

## THis is going to be bire/gss risk intelligence system.
def gss_v3_decision(
    row,
    high_short_risk: float = 0.70,
    high_long_risk: float = 0.60,
    watch_long_risk: float = 0.40,
    convergence_threshold: float = 0.05,
    spread_threshold: float = 0.10,
    velocity_escalate: float = 0.01,
    acceleration_escalate: float = 0.005,
    velocity_watch: float = 0.005,
):
    """
    Assign GSS V3 decision state based on multi-horizon risk + trajectory.
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

    # ESCALATE
    if (
        r15 >= high_short_risk
        or (r60 >= high_long_risk and convergence <= convergence_threshold)
        or (velocity >= velocity_escalate and acceleration >= acceleration_escalate)
    ):
        return "ESCALATE"

    # WATCH
    if (
        r60 >= watch_long_risk
        or spread >= spread_threshold
        or velocity >= velocity_watch
    ):
        return "WATCH"

    return "SUPPRESS"
