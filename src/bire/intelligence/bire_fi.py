import pandas as pd


def compute_bire_fi_score(
    df,
    risk_col="pred_proba",
    ibpip_col="ibpip_score",
    output_col="bire_fi_score",
):
    """
    Compute a simple forward-looking BIRE-FI score.

    Combines:
    - current model risk (pred_proba)
    - baseline deviation (IBPIP)

    This is a V1 implementation and will evolve into multi-horizon forecasting.
    """

    df = df.copy()

    if risk_col not in df.columns:
        raise ValueError(f"Missing column: {risk_col}")

    if ibpip_col not in df.columns:
        raise ValueError(f"Missing column: {ibpip_col}")

    # Simple weighted combination
    df[output_col] = (
        0.7 * df[risk_col]
        + 0.3 * df[ibpip_col]
    )

    return df
