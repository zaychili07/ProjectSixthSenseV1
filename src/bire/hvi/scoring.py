# ============================================================
# BIRE OS — Hidden Vitals Intelligence (HVI)
# File: src/bire/hvi/scoring.py
# Created: 2026-06-11
# Updated: 2026-06-11
#
# CHANGED:
# - Added build_recovery_family_comparison_review()
# - Supports Chapter 63.15 Recovery Family Comparison Review
#
# Purpose:
# Review, evaluation, comparison, ranking, and scoring utilities
# for Hidden Vitals Intelligence investigations.
# Chapter Origin: Chapter 63 — Hidden Vitals Intelligence (HVI)
# NOTES:
# This file primarily contains review-oriented and comparison-oriented
# helper functions used to support HVI investigations.
# HVI creates physiologic measurements.
# HVI scoring utilities help organize, compare, and evaluate
# existing HVI signal families.
# Notebook cells should remain lightweight frontend calls.
# Backend review logic should be implemented here whenever practical.
# ============================================================

def build_recovery_family_comparison_review():
    import pandas as pd

    return pd.DataFrame({
        "family": [
            "Recovery Authenticity",
            "Recovery Resilience",
            "Recovery Stability",
            "Recovery Legitimacy",
            "Recovery Contradiction",
            "Recovery Re-Escalation",
        ],

        "observed_activity": [
            "Mixed",
            "Low-Mixed",
            "Moderate",
            "Moderate",
            "High",
            "High",
        ],

        "notable_observation": [
            "Active and inactive signals coexisted.",
            "Context pressure dominated family behavior.",
            "Recovery instability was consistently active.",
            "Trust appeared more active than quality.",
            "Multiple contradiction signals demonstrated activity.",
            "Pressure signals appeared active before escalation flags.",
        ],

        "initial_information_contribution": [
            "Moderate",
            "Low",
            "Moderate",
            "Moderate",
            "High",
            "High",
        ],

        "status": [
            "Observation",
            "Observation",
            "Observation",
            "Observation",
            "Observation",
            "Observation",
        ]
    })