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
