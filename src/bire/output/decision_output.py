import pandas as pd


def build_bire_decision_reason(row):
    tier = row["bire_final_tier"]

    if row.get("bire_safe_block"):
        return "Suppressed because supporting evidence was incomplete."

    if tier == "SUPPRESSED_WATCH":
        return "Suppressed due to low forecast risk and no qualifying safety override."

    if tier == "WATCH":
        if row.get("watch_safety_override"):
            return f"Restored to WATCH by safety override: {row.get('watch_override_reason')}."
        return "Retained as WATCH for early instability awareness."

    if tier == "ESCALATE":
        return (
            "Assigned ESCALATE because the episode occurred within the 60-minute "
            "prediction window with moderate-to-high BIRE-FI risk."
        )

    if tier == "URGENT":
        return (
            "Assigned URGENT because the episode occurred within the 60-minute "
            "prediction window with high BIRE-FI risk."
        )

    if tier == "MONITOR":
        return "Assigned MONITOR because the episode occurred after deterioration onset."

    return "No decision reason available."


def build_bire_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build final BIRE system output with clean, clinician-facing fields.
    """

    out = df.copy()

    # Final tier
    out["bire_final_tier"] = out["episode_tier_acuity"]

    # State descriptions
    tier_state_map = {
        "SUPPRESSED_WATCH": "Suppressed low-value signal",
        "WATCH": "Early instability awareness",
        "ESCALATE": "Predictive clinical concern",
        "URGENT": "High-confidence predictive concern",
        "MONITOR": "Post-event monitoring",
    }

    out["bire_state"] = out["bire_final_tier"].map(tier_state_map)

    # Timing descriptions
    timing_map = {
        "no_event_episode": "No associated deterioration event",
        "early_beyond_horizon_episode": "Outside 60-minute prediction window",
        "true_predictive_episode": "Within 60-minute prediction window",
        "post_event_episode": "After deterioration onset",
    }

    out["bire_timing"] = out["timing_category"].map(timing_map)

    # Safety flags
    out["bire_safe_block"] = out[
        "watch_override_blocked_missing_evidence"
    ].fillna(False)

    # Decision reason
    out["bire_decision_reason"] = out.apply(
        build_bire_decision_reason,
        axis=1,
    )

    return out
