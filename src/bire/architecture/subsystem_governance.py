# ============================================================
# BIRE OS — Subsystem Responsibility Governance
# File: src/bire/architecture/subsystem_governance.py
# Updated: 2026-05-24
# Purpose:
# Backend helper for Chapter 60 subsystem responsibility mapping.
# ============================================================

import pandas as pd


def build_subsystem_responsibility_map() -> pd.DataFrame:
    """
    Build the BIRE OS subsystem responsibility governance table.

    This table defines each subsystem's primary responsibility,
    core output, coordination partners, and boundary limitations.

    Returns
    -------
    pd.DataFrame
        Subsystem responsibility governance table.
    """

    return pd.DataFrame(
        {
            "subsystem": [
                "BIL",
                "NCL",
                "IBPIP",
                "LPMR",
                "HVI",
                "ATI",
                "TTI",
                "BMS",
                "HIT",
                "BIRE-FI",
                "PSR-v2",
            ],
            "primary_responsibility": [
                "Ingest, normalize, and align incoming patient data to the BIRE OS schema.",
                "Clean, qualify, flag, or suppress noisy signals before downstream interpretation.",
                "Personalize interpretation using patient-specific baselines and individualized context.",
                "Retain longitudinal patient history, instability cycles, rebound patterns, and stabilization memory.",
                "Interpret hidden physiology, compensated instability, deceptive stability, and concealed burden.",
                "Analyze trajectory behavior, instability cycling, rebound pathways, and longitudinal movement patterns.",
                "Analyze how medications, interventions, and treatments influence longitudinal deterioration and recovery trajectories.",
                "Adapt thresholds and operational logic based on care mode and patient acuity setting.",
                "Tailor BIRE OS behavior to the specific hospital, unit, workflow, and operational setting.",
                "Coordinate forecasting using historical memory, current state, hidden burden, trajectory behavior, and confidence logic.",
                "Translate complex BIRE OS intelligence into readable patient prioritization and operational summaries.",
            ],
            "core_output": [
                "Normalized patient signal stream",
                "Cleaned and reliability-qualified signal layer",
                "Patient-specific baseline interpretation",
                "Longitudinal memory context",
                "Hidden physiological burden interpretation",
                "Trajectory-aware instability interpretation",
                "Intervention-aware deterioration analysis",
                "Care-mode adapted operational context",
                "Hospital-specific operational context",
                "Forecast-ready trajectory prediction context",
                "Readable clinician-facing operational summary",
            ],
            "should_not_do": [
                "Interpret downstream clinical risk independently.",
                "Delete difficult data without preserving uncertainty or traceability.",
                "Override ecosystem-level forecasting independently.",
                "Escalate patients independently without downstream forecasting and operational context.",
                "Overrule care-mode logic independently.",
                "Act as the final forecasting authority alone.",
                "Prescribe medications, recommend treatments, or replace clinical judgment.",
                "Ignore hidden burden, memory, or forecasting conflict.",
                "Replace BMS or patient-specific intelligence.",
                "Forecast without confidence, noise, memory, and conflict awareness.",
                "Expose raw subsystem complexity in a way that overloads clinicians.",
            ],
            "coordination_partners": [
                "NCL, IBPIP, HVI, ATI, BIRE-FI",
                "BIL, HVI, ATI, LPMR, BIRE-FI, BMS",
                "LPMR, HVI, ATI, BMS, BIRE-FI",
                "IBPIP, HVI, ATI, BIRE-FI, BMS",
                "IBPIP, LPMR, ATI, TTI, BIRE-FI",
                "HVI, LPMR, TTI, BIRE-FI, BMS",
                "HVI, ATI, BIRE-FI, IBPIP, LPMR",
                "HVI, ATI, LPMR, HIT, BIRE-FI",
                "BMS, PSR-v2, BIRE-FI, operational leadership context",
                "LPMR, HVI, ATI, TTI, BMS, NCL",
                "BIRE-FI, BMS, HIT, HVI, ATI, LPMR",
            ],
        }
    )