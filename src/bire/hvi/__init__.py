# ============================================================
# BIRE OS — Hidden Vitals Intelligence
# File: src/bire/hvi/__init__.py
# Chapter: 63.5A
#
# Created: 2026-06-02
# Updated: 2026-06-07
#
# CHANGED:
# - Exposed calculate_shock_index
# - Exposed build_recovery_hidden_vitals_family_review
#
# Purpose:
# Package exports for Hidden Vitals Intelligence.
# ============================================================

from bire.hvi.features import calculate_shock_index

__all__ = [
    "calculate_shock_index",
    "calculate_pulse_pressure",
    "calculate_map_estimate",
    "calculate_oxygen_gap",
    "calculate_temperature_deviation",
    "calculate_fever_stress",
    "calculate_hypothermia_stress",
    "calculate_hr_rr_ratio",
    "calculate_resp_spo2_stress",
    "calculate_cardiorespiratory_stress",
    "calculate_pressure_rate_product",
    "calculate_bp_narrowing_index",
    "calculate_diastolic_pressure_ratio",
    "calculate_hemodynamic_ratio",
    "calculate_respiratory_load_index",
    "calculate_oxygenation_pressure_stress",
    "calculate_shock_respiratory_combo",
    "calculate_shock_oxygen_combo",
    "calculate_shock_temperature_combo",
    "calculate_cardio_pressure_stress",
    "calculate_hemodynamic_oxygen_stress",
    "calculate_respiratory_oxygen_temperature_stress",
    "calculate_narrow_pressure_shock",
    "calculate_map_oxygen_stress",
    "calculate_cardiorespiratory_hemodynamic_stress",
    "calculate_circulatory_respiratory_burden",
    "calculate_oxygen_compensation_burden",
    "calculate_thermal_respiratory_burden",
    "calculate_pressure_oxygen_burden",
    "calculate_multi_system_burden_index",
    "calculate_normal_vitals_hidden_burden_flag",
    "calculate_high_shock_acceptable_bp_flag",
    "calculate_hidden_respiratory_strain_flag",
    "calculate_false_recovery_signal",
    "calculate_normal_range_deception_signal",
    "review_trajectory_hvi_candidates",
    "build_recovery_hidden_vitals_family_review",
]
