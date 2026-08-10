# ============================================================
# BIRE OS — Noise-Aware Escalation Governance
# File: src/bire/noise/noise_aware_escalation.py
# Chapter: 61.6
#
# Created: 2026-05-26
# Updated: 2026-05-26
#
# Purpose:
# Backend helper for noise-aware escalation governance,
# NCL + GSS + GSS-VE coordination,
# and reliability-aware alert burden moderation doctrine.
# ============================================================

import pandas as pd


def build_noise_aware_escalation_framework() -> pd.DataFrame:
    """
    Build the noise-aware escalation governance framework.

    This table defines how NCL reliability states should influence
    escalation moderation, GSS suppression behavior, and GSS-VE
    velocity override behavior.

    Returns
    -------
    pd.DataFrame
        Noise-aware escalation governance table.
    """

    return pd.DataFrame(
        {
            "escalation_state": [
                "STABLE_ESCALATION",
                "MODERATED_ESCALATION",
                "SUPPRESSED_NOISE_ESCALATION",
                "FRAGMENTED_ESCALATION",
                "VELOCITY_OVERRIDE_ESCALATION",
                "UNCERTAIN_ESCALATION",
                "RESTRICTED_ESCALATION",
            ],

            "escalation_state_description": [
                "Reliable signal conditions support stable escalation logic.",
                "Partial reliability requires confidence-aware escalation moderation.",
                "Noisy signal condition likely should not interrupt clinicians.",
                "Fragmented continuity weakens escalation confidence.",
                "Rapid worsening may justify escalation despite suppression.",
                "Unclear signal reliability requires cautious operational interpretation.",
                "Severe signal degradation requires suppression or strong moderation.",
            ],

            "ncl_behavior": [
                "Preserve normal reliability weighting.",
                "Apply moderate signal-confidence reduction.",
                "Flag signal as likely noise-driven.",
                "Flag longitudinal continuity as degraded.",
                "Allow velocity concern to remain visible despite noise.",
                "Preserve uncertainty and avoid false certainty.",
                "Strongly restrict confidence propagation.",
            ],

            "gss_behavior": [
                "Allow standard GSS rules to operate.",
                "Increase suppression caution but preserve meaningful concern.",
                "Suppress unless multi-layer support or persistence exists.",
                "Require stronger continuity or multi-layer support.",
                "Permit suppression break if velocity criteria are met.",
                "Hold or moderate escalation until better support emerges.",
                "Suppress or heavily moderate unless critical override exists.",
            ],

            "gss_ve_behavior": [
                "No override required unless deterioration accelerates.",
                "Monitor for acceleration before override.",
                "Override only if rapid worsening becomes meaningful.",
                "Override only if fragmentation coincides with dangerous acceleration.",
                "Break suppression when velocity or acceleration risk is high.",
                "Escalate only with strong velocity or multi-layer confirmation.",
                "Override only for critical velocity-supported deterioration.",
            ],

            "core_escalation_principle": [
                "Reliable concern may pass normally",
                "Partial reliability requires moderated escalation",
                "Noise should not become alert burden",
                "Fragmentation weakens escalation continuity",
                "Velocity can justify suppression override",
                "Uncertainty should remain operationally visible",
                "Severe degradation requires cautious escalation governance",
            ],
        }
    )