# ============================================================
# BIRE OS — Forecast Authority Hierarchy
# File: src/bire/hvi/policy.py
# Chapter: 63.1
#
# Created: 2026-06-03
# Updated: 2026-06-03
#
# CHANGED:
# - Added get_hvi_governance_contract function
#
# Purpose:
# Defines the governance contract for the Hidden Vitals Intelligence (HVI) component of BIRE OS,
# outlining its responsibilities, limitations, and role within the overall system.
# ============================================================


def get_hvi_governance_contract() -> dict:
    return {
        "owns": [
            "hidden_physiologic_relationships",
            "hidden_physiologic_features",
            "hidden_burden_measurements",
            "physiologic_relationship_calculations",
        ],
        "does_not_own": [
            "diagnosis",
            "treatment_recommendations",
            "forecasting",
            "memory",
            "governance",
            "escalation_authority",
            "care_decisions",
            "intervention_selection",
        ],
        "role": "shared_physiologic_intelligence_service",
        "authority": "advisory_only",
    }