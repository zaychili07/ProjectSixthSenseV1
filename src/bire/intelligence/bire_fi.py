import numpy as np
import pandas as pd


def compute_bire_fi_score(
    df,
    risk_col="pred_proba",
    ibpip_col="ibpip_score",
    output_col="bire_fi_score",
    ibpip_norm_col="ibpip_score_norm",
    risk_weight=0.7,
    ibpip_weight=0.3,
):
    """
    Compute BIRE-FI V1 score.

    Combines model risk with normalized patient baseline deviation.

    Notes
    -----
    pred_proba is already bounded between 0 and 1.
    ibpip_score can be much larger, so it is compressed into [0, 1)
    before combining.
    """

    df = df.copy()

    if risk_col not in df.columns:
        raise ValueError(f"Missing column: {risk_col}")

    if ibpip_col not in df.columns:
        raise ValueError(f"Missing column: {ibpip_col}")

    df[ibpip_norm_col] = df[ibpip_col] / (1 + df[ibpip_col])

    df[output_col] = (
        risk_weight * df[risk_col]
        + ibpip_weight * df[ibpip_norm_col]
    )

    return df

## Adding a layer of Forecasting Intelligence to BIRE-FI
def add_bire_fi_features(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    risk_col: str = "pred_proba",
    window: int = 6,
) -> pd.DataFrame:
    """
    Add BIRE-FI trajectory features.

    BIRE-FI evaluates short-term risk movement using:
    - risk velocity
    - risk acceleration
    - rolling risk mean
    - rolling risk volatility
    - rolling risk slope proxy

    window=6 means 30 minutes if rows are spaced every 5 minutes.
    """

    required = [patient_col, time_col, risk_col]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns for BIRE-FI: {missing}")

    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    out = out.sort_values([patient_col, time_col]).reset_index(drop=True)

    grouped = out.groupby(patient_col, group_keys=False)

    out["risk_velocity"] = grouped[risk_col].diff()
    out["risk_acceleration"] = grouped["risk_velocity"].diff()

    out[f"risk_rolling_mean_{window}"] = grouped[risk_col].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=2).mean()
    )

    out[f"risk_volatility_{window}"] = grouped[risk_col].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=2).std()
    )

    out[f"risk_slope_proxy_{window}"] = grouped[risk_col].transform(
        lambda x: x.shift(1).diff(periods=window)
    )

    return out


def assign_bire_fi_trajectory_states(
    df: pd.DataFrame,
    risk_col: str = "pred_proba",
    velocity_col: str = "risk_velocity",
    acceleration_col: str = "risk_acceleration",
    volatility_col: str = "risk_volatility_6",
    low_risk_threshold: float = 0.20,
    elevated_risk_threshold: float = 0.40,
    high_risk_threshold: float = 0.70,
    velocity_threshold: float = 0.03,
    acceleration_threshold: float = 0.02,
    volatility_threshold: float = 0.08,
) -> pd.DataFrame:
    """
    Assign BIRE-FI trajectory states.

    These states describe directional risk behavior.
    They do not replace BIRE tiers.
    """

    out = df.copy()

    required = [risk_col, velocity_col, acceleration_col, volatility_col]
    missing = [col for col in required if col not in out.columns]

    if missing:
        raise ValueError(f"Missing required BIRE-FI state columns: {missing}")

    def classify(row):
        risk = row[risk_col]
        velocity = row[velocity_col]
        acceleration = row[acceleration_col]
        volatility = row[volatility_col]

        if pd.isna(velocity) or pd.isna(acceleration):
            return "INSUFFICIENT_HISTORY"

        if risk < low_risk_threshold and abs(velocity) < velocity_threshold:
            return "LOW_STABLE"

        if velocity >= velocity_threshold and acceleration >= acceleration_threshold:
            return "WORSENING_ACCELERATING"

        if velocity >= velocity_threshold:
            return "WORSENING"

        if volatility >= volatility_threshold:
            return "UNSTABLE_VOLATILE"

        if risk >= elevated_risk_threshold and abs(velocity) < velocity_threshold:
            return "ELEVATED_PLATEAU"

        if velocity <= -velocity_threshold:
            return "STABILIZING"

        if risk >= high_risk_threshold:
            return "HIGH_RISK_STABLE"

        return "STABLE_UNCLEAR"

    out["bire_fi_trajectory_state"] = out.apply(classify, axis=1)

    return out


def add_bire_fi_forecasting_layer(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    risk_col: str = "pred_proba",
    window: int = 6,
) -> pd.DataFrame:
    """
    Full BIRE-FI forecasting layer.

    Adds trajectory features and assigns BIRE-FI trajectory states.
    """

    out = add_bire_fi_features(
        df=df,
        patient_col=patient_col,
        time_col=time_col,
        risk_col=risk_col,
        window=window,
    )

    out = assign_bire_fi_trajectory_states(
        df=out,
        risk_col=risk_col,
        velocity_col="risk_velocity",
        acceleration_col="risk_acceleration",
        volatility_col=f"risk_volatility_{window}",
    )

    return out
