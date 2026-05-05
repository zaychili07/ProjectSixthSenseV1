"""
WATCH Safety Override

This module applies the final WATCH gating and safety override policy for BIRE.

Purpose:
- Suppress low-value WATCH episodes
- Restore meaningful instability signals
- Block unsafe overrides when supporting evidence is missing
"""

from dataclasses import dataclass
import pandas as pd


@dataclass
class WatchOverrideConfig:
    """
    Configuration for WATCH gating and safety override behavior.
    """

    watch_gate_threshold: float = 0.42
    bire_fi_override_threshold: float = 0.45
    ibpip_score_override_threshold: float = 2.5
    ibpip_abnormal_signal_threshold: int = 3
    require_evidence_for_gss_ve: bool = True


def apply_watch_safety_override(
    df: pd.DataFrame,
    config: WatchOverrideConfig | None = None,
) -> pd.DataFrame:
    """
    Apply WATCH gating and safety override logic.

    Parameters
    ----------
    df : pd.DataFrame
        Episode-level dataframe containing WATCH tiers and supporting risk signals.

    config : WatchOverrideConfig, optional
        Policy configuration. If None, default thresholds are used.

    Returns
    -------
    pd.DataFrame
        Dataframe with final WATCH tier decisions and override audit columns.
    """

    if config is None:
        config = WatchOverrideConfig()

    out = df.copy()

    required_cols = [
        "episode_tier",
        "bire_fi_score",
        "ibpip_score",
        "ibpip_n_abnormal_signals",
        "gss_ve_override",
    ]

    missing_cols = [col for col in required_cols if col not in out.columns]

    if missing_cols:
        raise ValueError(
            f"Missing required columns for WATCH override: {missing_cols}"
        )

    # --------------------------------------------------
    # Step 1 — Apply WATCH gate
    # --------------------------------------------------
    out["episode_tier_gated"] = out["episode_tier"]

    out.loc[
        (out["episode_tier"] == "WATCH")
        & (out["bire_fi_score"] < config.watch_gate_threshold),
        "episode_tier_gated",
    ] = "SUPPRESSED_WATCH"

    # --------------------------------------------------
    # Step 2 — Evidence completeness rule
    # --------------------------------------------------
    has_enough_evidence = (
        out["bire_fi_score"].notna()
        & out["ibpip_score"].notna()
    )

    out["watch_safety_override"] = False
    out["watch_override_reason"] = "none"

    out["watch_override_blocked_missing_evidence"] = (
        (out["episode_tier_gated"] == "SUPPRESSED_WATCH")
        & (out["gss_ve_override"] == True)
        & ~has_enough_evidence
    )

    # --------------------------------------------------
    # Step 3 — Override conditions
    # --------------------------------------------------
    mask_bire_fi = (
        (out["episode_tier_gated"] == "SUPPRESSED_WATCH")
        & (out["bire_fi_score"] >= config.bire_fi_override_threshold)
    )

    mask_ibpip_score = (
        (out["episode_tier_gated"] == "SUPPRESSED_WATCH")
        & (out["ibpip_score"] >= config.ibpip_score_override_threshold)
    )

    mask_ibpip_multi_signal = (
        (out["episode_tier_gated"] == "SUPPRESSED_WATCH")
        & (
            out["ibpip_n_abnormal_signals"]
            >= config.ibpip_abnormal_signal_threshold
        )
    )

    if config.require_evidence_for_gss_ve:
        mask_gss_ve = (
            (out["episode_tier_gated"] == "SUPPRESSED_WATCH")
            & has_enough_evidence
            & (out["gss_ve_override"] == True)
        )
    else:
        mask_gss_ve = (
            (out["episode_tier_gated"] == "SUPPRESSED_WATCH")
            & (out["gss_ve_override"] == True)
        )

    # --------------------------------------------------
    # Step 4 — Apply override reasons
    # --------------------------------------------------
    out.loc[mask_bire_fi, "watch_safety_override"] = True
    out.loc[mask_bire_fi, "watch_override_reason"] = "high_bire_fi"

    out.loc[mask_ibpip_score, "watch_safety_override"] = True
    out.loc[mask_ibpip_score, "watch_override_reason"] = "severe_ibpip_deviation"

    out.loc[mask_ibpip_multi_signal, "watch_safety_override"] = True
    out.loc[
        mask_ibpip_multi_signal,
        "watch_override_reason",
    ] = "strong_multi_signal_ibpip"

    out.loc[mask_gss_ve, "watch_safety_override"] = True
    out.loc[mask_gss_ve, "watch_override_reason"] = "gss_velocity_override"

    # --------------------------------------------------
    # Step 5 — Final tier decision
    # --------------------------------------------------
    out["episode_tier_final"] = out["episode_tier_gated"]

    out.loc[
        out["watch_safety_override"],
        "episode_tier_final",
    ] = "WATCH"

    return out
