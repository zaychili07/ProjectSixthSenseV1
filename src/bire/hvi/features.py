# ============================================================
# BIRE OS — Hidden Vitals Intelligence
# File: src/bire/hvi/features.py
# Chapter: 63.5A
#
# Created: 2026-06-02
# Updated: 2026-06-03
#
# CHANGED:
# - Added initial Shock Index calculation
# - Added Pulse Pressure calculation
# - Added MAP Estimate calculation
# - Added Oxygen Gap calculation
# - Added Temperature Deviation calculation
# - Added Fever Stress calculation
# - Added Hypothermia Stress calculation
# - Added Respiratory SpO2 Stress calculation
# - Added HR/RR Ratio calculation
# - Added Cardiorespiratory Stress calculation
# - Added Pressure Rate Product calculation
# - Added BP Narrowing Index calculation
# - Added Diastolic Pressure Ratio calculation
# - Added Hemodynamic Ratio calculation
# - Added Respiratory Load Index calculation
# - Added Oxygenation Pressure Stress calculation
# - Added Shock Respiratory Combo calculation
# - Added Shock Oxygen Combo calculation
# - Added Shock Temperature Combo calculation
# - Added Cardio Pressure Stress calculation
# - line 178 - changed baseline to fahrenheit to match temperature units in dataset
# - Added Hemodynamic Oxygen Stress calculation
# - Added Respiratory Oxygen Temperature Stress calculation
# - Added Narrow Pressure Shock calculation
# - Added MAP Oxygen Stress calculation
# - Added Cardiorespiratory Hemodynamic Stress calculation
# - Added Circulatory Respiratory Burden calculation
# - Added Oxygen Compensation Burden calculation
# - Added Thermal Respiratory Burden calculation
# - Added Pressure Oxygen Burden calculation
# - Added Multi-System Burden Index calculation
# - Added Normal Vitals Hidden Burden Flag calculation
# - Added High Shock With Acceptable Blood Pressure Flag calculation
# - Added Hidden Respiratory Strain Flag calculation
# - Added Hidden Oxygenation Strain Flag calculation
# - Added Hidden Thermal Strain Flag calculation
# - Added Hidden Cardio-Respiratory Strain Flag calculation
# - Added Hidden Multi-System Strain Flag calculation
# - Added Flagged Observation Review calculation
# - Added comprehensive error handling for missing columns in all feature functions
#
#
# Purpose:
# Backend feature functions for Hidden Vitals Intelligence.
# ============================================================

import pandas as pd
import numpy as np

# This function calculates the Shock Index based on heart rate and systolic blood pressure values.
def calculate_shock_index(
    df: pd.DataFrame,
    hr_col: str = "heart_rate",
    sbp_col: str = "sbp",
    output_col: str = "shock_index",
) -> pd.DataFrame:
    """
    Calculate Shock Index.

    Shock Index = heart_rate / systolic blood pressure

    Purpose:
    Detect hemodynamic strain before systolic blood pressure
    becomes critically low.

    HVI Doctrine:
    A hidden vital is not another vital sign.
    A hidden vital is a physiologic relationship.
    """

    result_df = df.copy()

    required_cols = [hr_col, sbp_col]
    missing_cols = [col for col in required_cols if col not in result_df.columns]

    if missing_cols:
        raise ValueError(
            f"Cannot calculate shock index. Missing required columns: {missing_cols}"
        )

    result_df[output_col] = np.where(
        result_df[sbp_col] > 0,
        result_df[hr_col] / result_df[sbp_col],
        np.nan,
    )

    return result_df

# This function calculates the Pulse Pressure based on systolic and diastolic blood pressure values.
def calculate_pulse_pressure(
    df: pd.DataFrame,
    sbp_col: str = "sbp",
    dbp_col: str = "dbp",
    output_col: str = "pulse_pressure",
) -> pd.DataFrame:
    """
    Calculate Pulse Pressure.

    Pulse Pressure = SBP - DBP

    Purpose:
    Detect narrowing or widening pressure dynamics.

    HVI Doctrine:
    A hidden vital is a physiologic relationship.
    """

    result_df = df.copy()

    required_cols = [sbp_col, dbp_col]
    missing_cols = [col for col in required_cols if col not in result_df.columns]

    if missing_cols:
        raise ValueError(
            f"Cannot calculate pulse pressure. Missing required columns: {missing_cols}"
        )

    result_df[output_col] = (
        result_df[sbp_col] - result_df[dbp_col]
    )

    return result_df

# This function calculates the Mean Arterial Pressure (MAP) estimate based on systolic and diastolic blood pressure values.
#  It is useful for assessing average arterial pressure and supporting hidden perfusion interpretation.
def calculate_map_estimate(
    df: pd.DataFrame,
    sbp_col: str = "sbp",
    dbp_col: str = "dbp",
    output_col: str = "map_estimate",
) -> pd.DataFrame:
    """
    Calculate MAP Estimate.

    MAP Estimate = (SBP + 2 * DBP) / 3

    Purpose:
    Estimate average arterial pressure and support
    hidden perfusion interpretation.
    """

    result_df = df.copy()

    required_cols = [sbp_col, dbp_col]
    missing_cols = [col for col in required_cols if col not in result_df.columns]

    if missing_cols:
        raise ValueError(
            f"Cannot calculate MAP estimate. Missing required columns: {missing_cols}"
        )

    result_df[output_col] = (
        result_df[sbp_col] + 2 * result_df[dbp_col]
    ) / 3

    return result_df

# This function calculates the Oxygen Gap based on SpO2 values.
def calculate_oxygen_gap(
    df: pd.DataFrame,
    spo2_col: str = "spo2",
    output_col: str = "oxygen_gap",
) -> pd.DataFrame:
    """
    Calculate Oxygen Gap.

    Oxygen Gap = 100 - SpO2

    Purpose:
    Represent distance from ideal oxygen saturation.

    HVI Doctrine:
    Oxygen saturation alone does not fully describe
    oxygenation reserve.
    """

    result_df = df.copy()

    if spo2_col not in result_df.columns:
        raise ValueError(
            f"Cannot calculate oxygen gap. Missing required column: {spo2_col}"
        )

    result_df[output_col] = 100 - result_df[spo2_col]

    return result_df

# This function calculates the Temperature Deviation based on temperature values.
def calculate_temperature_deviation(
    df: pd.DataFrame,
    temp_col: str = "temperature",
    output_col: str = "temperature_deviation",
    baseline_temp: float = 98.6,
) -> pd.DataFrame:
    """
    Calculate Temperature Deviation.

    Temperature Deviation =
    abs(temperature - baseline_temp)

    Purpose:
    Measure distance from normal physiologic temperature.

    HVI Doctrine:
    Temperature Deviation represents physiologic drift
    rather than threshold crossing.
    """

    result_df = df.copy()

    if temp_col not in result_df.columns:
        raise ValueError(
            f"Cannot calculate temperature deviation. Missing required column: {temp_col}"
        )

    result_df[output_col] = (
        result_df[temp_col] - baseline_temp
    ).abs()

    return result_df


# This function calculates the Fever Stress based on temperature values.
def calculate_fever_stress(
    df: pd.DataFrame,
    temp_col: str = "temperature",
    output_col: str = "fever_stress",
) -> pd.DataFrame:
    """
    Calculate Fever Stress.

    Fever Stress = max(temperature - 37.5, 0)

    Purpose:
    Capture thermal burden above normal physiologic range.
    """

    result_df = df.copy()

    if temp_col not in result_df.columns:
        raise ValueError(
            f"Cannot calculate fever stress. Missing required column: {temp_col}"
        )

    result_df[output_col] = (
        result_df[temp_col] - 37.5
    ).clip(lower=0)

    return result_df

# This function calculates the Hypothermia Stress based on temperature values.
def calculate_hypothermia_stress(
    df: pd.DataFrame,
    temp_col: str = "temperature",
    output_col: str = "hypothermia_stress",
) -> pd.DataFrame:
    """
    Calculate Hypothermia Stress.

    Hypothermia Stress = max(36.0 - temperature, 0)

    Purpose:
    Capture physiologic burden associated with
    low body temperature.
    """

    result_df = df.copy()

    if temp_col not in result_df.columns:
        raise ValueError(
            f"Cannot calculate hypothermia stress. Missing required column: {temp_col}"
        )

    result_df[output_col] = (
        36.0 - result_df[temp_col]
    ).clip(lower=0)

    return result_df

# This function calculates the Heart Rate to Respiratory Rate (HR/RR) Ratio based on heart rate and respiratory rate values.
def calculate_hr_rr_ratio(
    df: pd.DataFrame,
    hr_col: str = "heart_rate",
    rr_col: str = "resp_rate",
    output_col: str = "hr_rr_ratio",
) -> pd.DataFrame:
    """
    Calculate HR/RR Ratio.

    HR/RR Ratio = heart_rate / resp_rate

    Purpose:
    Evaluate the relationship between cardiac
    and respiratory activity.
    """

    result_df = df.copy()

    required_cols = [hr_col, rr_col]
    missing_cols = [col for col in required_cols if col not in result_df.columns]

    if missing_cols:
        raise ValueError(
            f"Cannot calculate HR/RR Ratio. Missing required columns: {missing_cols}"
        )

    result_df[output_col] = (
        result_df[hr_col] / result_df[rr_col]
    )

    return result_df

# This function calculates the Respiratory SpO2 Stress based on respiratory rate and SpO2 values.
def calculate_resp_spo2_stress(
    df: pd.DataFrame,
    rr_col: str = "resp_rate",
    spo2_col: str = "spo2",
    output_col: str = "resp_spo2_stress",
) -> pd.DataFrame:
    """
    Calculate Respiratory SpO2 Stress.

    Respiratory SpO2 Stress = resp_rate / spo2

    Purpose:
    Evaluate respiratory workload relative to oxygenation.
    """

    result_df = df.copy()

    required_cols = [rr_col, spo2_col]
    missing_cols = [col for col in required_cols if col not in result_df.columns]

    if missing_cols:
        raise ValueError(
            f"Cannot calculate Respiratory SpO2 Stress. Missing required columns: {missing_cols}"
        )

    result_df[output_col] = (
        result_df[rr_col] / result_df[spo2_col]
    )

    return result_df

# This function calculates the Cardiorespiratory Stress based on heart rate, respiratory rate, and SpO2 values.
def calculate_cardiorespiratory_stress(
    df: pd.DataFrame,
    hr_col: str = "heart_rate",
    rr_col: str = "resp_rate",
    spo2_col: str = "spo2",
    output_col: str = "cardiorespiratory_stress",
) -> pd.DataFrame:
    """
    Calculate Cardiorespiratory Stress.

    Cardiorespiratory Stress =
    (heart_rate * resp_rate) / spo2

    Purpose:
    Estimate combined cardiac and respiratory
    workload relative to oxygenation.
    """

    result_df = df.copy()

    required_cols = [hr_col, rr_col, spo2_col]
    missing_cols = [col for col in required_cols if col not in result_df.columns]

    if missing_cols:
        raise ValueError(
            f"Cannot calculate Cardiorespiratory Stress. Missing required columns: {missing_cols}"
        )

    result_df[output_col] = (
        result_df[hr_col] *
        result_df[rr_col]
    ) / result_df[spo2_col]

    return result_df

# This function calculates the Pressure Rate Product based on heart rate and systolic blood pressure values.
def calculate_pressure_rate_product(
    df: pd.DataFrame,
    hr_col: str = "heart_rate",
    sbp_col: str = "sbp",
    output_col: str = "pressure_rate_product",
) -> pd.DataFrame:
    """
    Calculate Pressure Rate Product.

    Pressure Rate Product = heart_rate * sbp

    Purpose:
    Estimate cardiovascular workload.
    """

    result_df = df.copy()

    required_cols = [hr_col, sbp_col]
    missing_cols = [col for col in required_cols if col not in result_df.columns]

    if missing_cols:
        raise ValueError(
            f"Cannot calculate Pressure Rate Product. Missing required columns: {missing_cols}"
        )

    result_df[output_col] = result_df[hr_col] * result_df[sbp_col]

    return result_df

# This function calculates the Blood Pressure Narrowing Index based on pulse pressure and systolic blood pressure values.
def calculate_bp_narrowing_index(
    df: pd.DataFrame,
    pulse_pressure_col: str = "pulse_pressure",
    sbp_col: str = "sbp",
    output_col: str = "bp_narrowing_index",
) -> pd.DataFrame:
    """
    Calculate BP Narrowing Index.

    BP Narrowing Index = pulse_pressure / sbp

    Purpose:
    Evaluate pulse pressure relative to systolic pressure.
    """

    result_df = df.copy()

    required_cols = [pulse_pressure_col, sbp_col]
    missing_cols = [col for col in required_cols if col not in result_df.columns]

    if missing_cols:
        raise ValueError(
            f"Cannot calculate BP Narrowing Index. Missing required columns: {missing_cols}"
        )

    result_df[output_col] = (
        result_df[pulse_pressure_col] /
        result_df[sbp_col]
    )

    return result_df

# This function calculates the Diastolic Pressure Ratio based on diastolic and systolic blood pressure values.
def calculate_diastolic_pressure_ratio(
    df: pd.DataFrame,
    dbp_col: str = "dbp",
    sbp_col: str = "sbp",
    output_col: str = "diastolic_pressure_ratio",
) -> pd.DataFrame:
    """
    Calculate Diastolic Pressure Ratio.

    Diastolic Pressure Ratio = dbp / sbp

    Purpose:
    Evaluate the proportion of diastolic pressure
    relative to systolic pressure.
    """

    result_df = df.copy()

    required_cols = [dbp_col, sbp_col]
    missing_cols = [col for col in required_cols if col not in result_df.columns]

    if missing_cols:
        raise ValueError(
            f"Cannot calculate Diastolic Pressure Ratio. Missing required columns: {missing_cols}"
        )

    result_df[output_col] = (
        result_df[dbp_col] /
        result_df[sbp_col]
    )

    return result_df

# This function calculates the Hemodynamic Ratio based on heart rate and mean arterial pressure estimate values.
def calculate_hemodynamic_ratio(
    df: pd.DataFrame,
    hr_col: str = "heart_rate",
    map_col: str = "map_estimate",
    output_col: str = "hemodynamic_ratio",
) -> pd.DataFrame:
    """
    Calculate Hemodynamic Ratio.

    Hemodynamic Ratio = heart_rate / map_estimate

    Purpose:
    Evaluate cardiac activity relative
    to perfusion pressure.
    """

    result_df = df.copy()

    required_cols = [hr_col, map_col]
    missing_cols = [col for col in required_cols if col not in result_df.columns]

    if missing_cols:
        raise ValueError(
            f"Cannot calculate Hemodynamic Ratio. Missing required columns: {missing_cols}"
        )

    result_df[output_col] = (
        result_df[hr_col] /
        result_df[map_col]
    )

    return result_df

# This function calculates the Respiratory Load Index based on respiratory rate and oxygen gap values.
def calculate_respiratory_load_index(
    df: pd.DataFrame,
    rr_col: str = "resp_rate",
    oxygen_gap_col: str = "oxygen_gap",
    output_col: str = "respiratory_load_index",
) -> pd.DataFrame:
    """
    Calculate Respiratory Load Index.

    Respiratory Load Index =
    resp_rate * oxygen_gap

    Purpose:
    Evaluate respiratory strain relative
    to oxygen reserve.
    """

    result_df = df.copy()

    required_cols = [rr_col, oxygen_gap_col]
    missing_cols = [col for col in required_cols if col not in result_df.columns]

    if missing_cols:
        raise ValueError(
            f"Cannot calculate Respiratory Load Index. Missing required columns: {missing_cols}"
        )

    result_df[output_col] = (
        result_df[rr_col] *
        result_df[oxygen_gap_col]
    )

    return result_df

# this function calculates oxgencation pressure stress based on oxygen gap and systolic blood pressure values.
# It is designed to evaluate the burden of impaired oxygenation in the context of circulatory pressure, which can help identify
# patients at risk of hypoxic stress despite normal SpO2 readings.
def calculate_oxygenation_pressure_stress(
    df: pd.DataFrame,
    oxygen_gap_col: str = "oxygen_gap",
    sbp_col: str = "sbp",
    output_col: str = "oxygenation_pressure_stress",
) -> pd.DataFrame:
    """
    Calculate Oxygenation Pressure Stress.

    Oxygenation Pressure Stress =
    oxygen_gap / sbp

    Purpose:
    Evaluate oxygenation burden relative
    to circulatory pressure.
    """

    result_df = df.copy()

    required_cols = [oxygen_gap_col, sbp_col]
    missing_cols = [col for col in required_cols if col not in result_df.columns]

    if missing_cols:
        raise ValueError(
            f"Cannot calculate Oxygenation Pressure Stress. Missing required columns: {missing_cols}"
        )

    result_df[output_col] = (
        result_df[oxygen_gap_col] /
        result_df[sbp_col]
    )

    return result_df

# This function calculates the Shock Respiratory Combo based on shock index and respiratory load index values.
def calculate_shock_respiratory_combo(
    df: pd.DataFrame,
    shock_index_col: str = "shock_index",
    respiratory_load_col: str = "respiratory_load_index",
    output_col: str = "shock_respiratory_combo",
) -> pd.DataFrame:
    """
    Calculate Shock Respiratory Combo.

    Shock Respiratory Combo =
    shock_index * respiratory_load_index

    Purpose:
    Evaluate combined circulatory strain and
    respiratory burden.
    """

    result_df = df.copy()

    required_cols = [shock_index_col, respiratory_load_col]
    missing_cols = [col for col in required_cols if col not in result_df.columns]

    if missing_cols:
        raise ValueError(
            f"Cannot calculate Shock Respiratory Combo. Missing required columns: {missing_cols}"
        )

    result_df[output_col] = (
        result_df[shock_index_col] *
        result_df[respiratory_load_col]
    )

    return result_df

# This function calculates the Shock Oxygen Combo based on shock index and oxygen gap values.
def calculate_shock_oxygen_combo(
    df: pd.DataFrame,
    shock_index_col: str = "shock_index",
    oxygen_gap_col: str = "oxygen_gap",
    output_col: str = "shock_oxygen_combo",
) -> pd.DataFrame:
    """
    Calculate Shock Oxygen Combo.

    Shock Oxygen Combo =
    shock_index * oxygen_gap

    Purpose:
    Evaluate combined circulatory strain
    and oxygenation burden.
    """

    result_df = df.copy()

    required_cols = [shock_index_col, oxygen_gap_col]
    missing_cols = [col for col in required_cols if col not in result_df.columns]

    if missing_cols:
        raise ValueError(
            f"Cannot calculate Shock Oxygen Combo. Missing required columns: {missing_cols}"
        )

    result_df[output_col] = (
        result_df[shock_index_col] *
        result_df[oxygen_gap_col]
    )

    return result_df

# This function calculates the Shock Temperature Combo based on shock index and temperature deviation values.
def calculate_shock_temperature_combo(
    df: pd.DataFrame,
    shock_index_col: str = "shock_index",
    temperature_deviation_col: str = "temperature_deviation",
    output_col: str = "shock_temperature_combo",
) -> pd.DataFrame:
    """
    Calculate Shock Temperature Combo.

    Shock Temperature Combo =
    shock_index * temperature_deviation

    Purpose:
    Evaluate combined circulatory strain
    and thermal burden.
    """

    result_df = df.copy()

    required_cols = [
        shock_index_col,
        temperature_deviation_col,
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in result_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Cannot calculate Shock Temperature Combo. Missing required columns: {missing_cols}"
        )

    result_df[output_col] = (
        result_df[shock_index_col]
        * result_df[temperature_deviation_col]
    )

    return result_df

# This function calculates the Cardio Pressure Stress based on pressure rate product and hemodynamic ratio values.
def calculate_cardio_pressure_stress(
    df: pd.DataFrame,
    pressure_rate_product_col: str = "pressure_rate_product",
    hemodynamic_ratio_col: str = "hemodynamic_ratio",
    output_col: str = "cardio_pressure_stress",
) -> pd.DataFrame:
    """
    Calculate Cardio Pressure Stress.

    Cardio Pressure Stress =
    pressure_rate_product * hemodynamic_ratio

    Purpose:
    Evaluate combined cardiovascular workload
    and perfusion-pressure burden.
    """

    result_df = df.copy()

    required_cols = [
        pressure_rate_product_col,
        hemodynamic_ratio_col,
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in result_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Cannot calculate Cardio Pressure Stress. Missing required columns: {missing_cols}"
        )

    result_df[output_col] = (
        result_df[pressure_rate_product_col]
        * result_df[hemodynamic_ratio_col]
    )

    return result_df


# This function calculates the Hemodynamic Oxygen Stress based on hemodynamic ratio and oxygenation pressure stress values.
def calculate_hemodynamic_oxygen_stress(
    df: pd.DataFrame,
    hemodynamic_ratio_col: str = "hemodynamic_ratio",
    oxygenation_pressure_stress_col: str = "oxygenation_pressure_stress",
    output_col: str = "hemodynamic_oxygen_stress",
) -> pd.DataFrame:
    """
    Calculate Hemodynamic Oxygen Stress.

    Hemodynamic Oxygen Stress =
    hemodynamic_ratio *
    oxygenation_pressure_stress

    Purpose:
    Evaluate combined hemodynamic burden
    and oxygen-pressure burden.
    """

    result_df = df.copy()

    required_cols = [
        hemodynamic_ratio_col,
        oxygenation_pressure_stress_col,
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in result_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Cannot calculate Hemodynamic Oxygen Stress. Missing required columns: {missing_cols}"
        )

    result_df[output_col] = (
        result_df[hemodynamic_ratio_col]
        *
        result_df[oxygenation_pressure_stress_col]
    )

    return result_df

# This function calculates the Respiratory Oxygen Temperature Stress based on respiratory load index and temperature deviation values.
def calculate_respiratory_oxygen_temperature_stress(
    df: pd.DataFrame,
    respiratory_load_index_col: str = "respiratory_load_index",
    temperature_deviation_col: str = "temperature_deviation",
    output_col: str = "respiratory_oxygen_temperature_stress",
) -> pd.DataFrame:
    """
    Calculate Respiratory Oxygen Temperature Stress.

    Respiratory Oxygen Temperature Stress =
    respiratory_load_index * temperature_deviation

    Purpose:
    Evaluate the interaction between respiratory-oxygen burden
    and thermal deviation.
    """

    result_df = df.copy()

    required_cols = [
        respiratory_load_index_col,
        temperature_deviation_col,
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in result_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Cannot calculate Respiratory Oxygen Temperature Stress. Missing required columns: {missing_cols}"
        )

    result_df[output_col] = (
        result_df[respiratory_load_index_col]
        *
        result_df[temperature_deviation_col]
    )

    return result_df

# This function calculates the Narrow Pressure Shock based on blood pressure narrowing index and shock index values.
def calculate_narrow_pressure_shock(
    df: pd.DataFrame,
    bp_narrowing_index_col: str = "bp_narrowing_index",
    shock_index_col: str = "shock_index",
    output_col: str = "narrow_pressure_shock",
) -> pd.DataFrame:
    """
    Calculate Narrow Pressure Shock.

    Narrow Pressure Shock =
    bp_narrowing_index * shock_index

    Purpose:
    Evaluate the interaction between
    pressure narrowing burden and
    shock burden.
    """

    result_df = df.copy()

    required_cols = [
        bp_narrowing_index_col,
        shock_index_col,
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in result_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Cannot calculate Narrow Pressure Shock. Missing required columns: {missing_cols}"
        )

    result_df[output_col] = (
        result_df[bp_narrowing_index_col]
        *
        result_df[shock_index_col]
    )

    return result_df

# This function calculates the MAP Oxygen Stress based on mean arterial pressure estimate and oxygen gap values.
def calculate_map_oxygen_stress(
    df: pd.DataFrame,
    map_estimate_col: str = "map_estimate",
    oxygen_gap_col: str = "oxygen_gap",
    output_col: str = "map_oxygen_stress",
) -> pd.DataFrame:
    """
    Calculate MAP Oxygen Stress.

    MAP Oxygen Stress =
    map_estimate * oxygen_gap

    Purpose:
    Evaluate the interaction between
    perfusion pressure burden and
    oxygen burden.
    """

    result_df = df.copy()

    required_cols = [
        map_estimate_col,
        oxygen_gap_col,
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in result_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Cannot calculate MAP Oxygen Stress. Missing required columns: {missing_cols}"
        )

    result_df[output_col] = (
        result_df[map_estimate_col]
        * result_df[oxygen_gap_col]
    )

    return result_df

# This function calculates the Cardiorespiratory Hemodynamic Stress based on cardiorespiratory stress and hemodynamic ratio values.
def calculate_cardiorespiratory_hemodynamic_stress(
    df: pd.DataFrame,
    cardiorespiratory_stress_col: str = "cardiorespiratory_stress",
    hemodynamic_ratio_col: str = "hemodynamic_ratio",
    output_col: str = "cardiorespiratory_hemodynamic_stress",
) -> pd.DataFrame:
    """
    Calculate Cardiorespiratory Hemodynamic Stress.

    Cardiorespiratory Hemodynamic Stress =
    cardiorespiratory_stress * hemodynamic_ratio

    Purpose:
    Evaluate the interaction between
    cardiorespiratory burden and
    hemodynamic burden.
    """

    result_df = df.copy()

    required_cols = [
        cardiorespiratory_stress_col,
        hemodynamic_ratio_col,
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in result_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Cannot calculate Cardiorespiratory Hemodynamic Stress. Missing required columns: {missing_cols}"
        )

    result_df[output_col] = (
        result_df[cardiorespiratory_stress_col]
        * result_df[hemodynamic_ratio_col]
    )

    return result_df

# This function calculates the Circulatory Respiratory Burden based on shock respiratory combo and cardio pressure stress values.
def calculate_circulatory_respiratory_burden(
    df: pd.DataFrame,
    shock_respiratory_combo_col: str = "shock_respiratory_combo",
    cardio_pressure_stress_col: str = "cardio_pressure_stress",
    output_col: str = "circulatory_respiratory_burden",
) -> pd.DataFrame:
    """
    Calculate Circulatory Respiratory Burden.

    Circulatory Respiratory Burden =
    shock_respiratory_combo +
    cardio_pressure_stress

    Purpose:
    Evaluate combined circulatory,
    cardiovascular, and respiratory burden.
    """

    result_df = df.copy()

    required_cols = [
        shock_respiratory_combo_col,
        cardio_pressure_stress_col,
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in result_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Cannot calculate Circulatory Respiratory Burden. Missing required columns: {missing_cols}"
        )

    result_df[output_col] = (
        result_df[shock_respiratory_combo_col]
        +
        result_df[cardio_pressure_stress_col]
    )

    return result_df

# This function calculates the Oxygen Compensation Burden based on shock oxygen combo and hemodynamic oxygen stress values.
def calculate_oxygen_compensation_burden(
    df: pd.DataFrame,
    shock_oxygen_combo_col: str = "shock_oxygen_combo",
    hemodynamic_oxygen_stress_col: str = "hemodynamic_oxygen_stress",
    output_col: str = "oxygen_compensation_burden",
) -> pd.DataFrame:
    """
    Calculate Oxygen Compensation Burden.

    Oxygen Compensation Burden =
    shock_oxygen_combo +
    hemodynamic_oxygen_stress

    Purpose:
    Evaluate combined oxygen-centered
    burden families.
    """

    result_df = df.copy()

    required_cols = [
        shock_oxygen_combo_col,
        hemodynamic_oxygen_stress_col,
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in result_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Cannot calculate Oxygen Compensation Burden. Missing required columns: {missing_cols}"
        )

    result_df[output_col] = (
        result_df[shock_oxygen_combo_col]
        +
        result_df[hemodynamic_oxygen_stress_col]
    )

    return result_df

# This function calculates the Thermal Respiratory Burden based on shock temperature combo and respiratory oxygen temperature stress values.
def calculate_thermal_respiratory_burden(
    df: pd.DataFrame,
    shock_temperature_combo_col: str = "shock_temperature_combo",
    respiratory_oxygen_temperature_stress_col: str = (
        "respiratory_oxygen_temperature_stress"
    ),
    output_col: str = "thermal_respiratory_burden",
) -> pd.DataFrame:
    """
    Calculate Thermal Respiratory Burden.

    Thermal Respiratory Burden =
    shock_temperature_combo +
    respiratory_oxygen_temperature_stress

    Purpose:
    Evaluate combined thermal,
    respiratory, oxygen,
    and circulatory burden.
    """

    result_df = df.copy()

    required_cols = [
        shock_temperature_combo_col,
        respiratory_oxygen_temperature_stress_col,
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in result_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Cannot calculate Thermal Respiratory Burden. Missing required columns: {missing_cols}"
        )

    result_df[output_col] = (
        result_df[shock_temperature_combo_col]
        +
        result_df[respiratory_oxygen_temperature_stress_col]
    )

    return result_df

# This function calculates the Pressure Oxygen Burden based on MAP oxygen stress and hemodynamic oxygen stress values.
def calculate_pressure_oxygen_burden(
    df: pd.DataFrame,
    map_oxygen_stress_col: str = "map_oxygen_stress",
    hemodynamic_oxygen_stress_col: str = "hemodynamic_oxygen_stress",
    output_col: str = "pressure_oxygen_burden",
) -> pd.DataFrame:
    """
    Calculate Pressure Oxygen Burden.

    Pressure Oxygen Burden =
    map_oxygen_stress + hemodynamic_oxygen_stress

    Purpose:
    Evaluate combined perfusion-pressure,
    hemodynamic, and oxygen burden.
    """

    result_df = df.copy()

    required_cols = [
        map_oxygen_stress_col,
        hemodynamic_oxygen_stress_col,
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in result_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Cannot calculate Pressure Oxygen Burden. Missing required columns: {missing_cols}"
        )

    result_df[output_col] = (
        result_df[map_oxygen_stress_col]
        +
        result_df[hemodynamic_oxygen_stress_col]
    )

    return result_df

# This function calculates the Multi-System Burden Index based on circulatory respiratory burden, oxygen compensation burden,
# thermal respiratory burden, and pressure oxygen burden values.
def calculate_multi_system_burden_index(
    df: pd.DataFrame,
    circulatory_respiratory_burden_col: str = "circulatory_respiratory_burden",
    oxygen_compensation_burden_col: str = "oxygen_compensation_burden",
    thermal_respiratory_burden_col: str = "thermal_respiratory_burden",
    pressure_oxygen_burden_col: str = "pressure_oxygen_burden",
    output_col: str = "multi_system_burden_index",
) -> pd.DataFrame:
    """
    Calculate Multi-System Burden Index.

    Multi-System Burden Index =
    circulatory_respiratory_burden
    + oxygen_compensation_burden
    + thermal_respiratory_burden
    + pressure_oxygen_burden

    Purpose:
    Evaluate combined exploratory burden across
    multiple complex hidden relationship families.
    """

    result_df = df.copy()

    required_cols = [
        circulatory_respiratory_burden_col,
        oxygen_compensation_burden_col,
        thermal_respiratory_burden_col,
        pressure_oxygen_burden_col,
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in result_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Cannot calculate Multi-System Burden Index. Missing required columns: {missing_cols}"
        )

    result_df[output_col] = (
        result_df[circulatory_respiratory_burden_col]
        + result_df[oxygen_compensation_burden_col]
        + result_df[thermal_respiratory_burden_col]
        + result_df[pressure_oxygen_burden_col]
    )

    return result_df

# This function calculates the Normal Vitals Hidden Burden Flag based on heart rate, respiratory rate, SpO2, systolic blood pressure, temperature,
#  and hidden instability score values.
def calculate_normal_vitals_hidden_burden_flag(
    df: pd.DataFrame,
    heart_rate_col: str = "heart_rate",
    resp_rate_col: str = "resp_rate",
    spo2_col: str = "spo2",
    sbp_col: str = "sbp",
    temperature_col: str = "temperature",
    hidden_instability_score_col: str = "hidden_instability_score",
    output_col: str = "normal_vitals_hidden_burden_flag",
) -> pd.DataFrame:
    """
    Calculate Normal Vitals Hidden Burden Flag.

    Purpose:
    Identify observations where surface vitals
    appear normal while hidden instability
    remains elevated.
    """

    result_df = df.copy()

    required_cols = [
        heart_rate_col,
        resp_rate_col,
        spo2_col,
        sbp_col,
        temperature_col,
        hidden_instability_score_col,
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in result_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Cannot calculate Normal Vitals Hidden Burden Flag. Missing required columns: {missing_cols}"
        )

    normal_surface_vitals = (
        result_df[heart_rate_col].between(60, 100)
        &
        result_df[resp_rate_col].between(12, 20)
        &
        (result_df[spo2_col] >= 95)
        &
        (result_df[sbp_col] >= 100)
        &
        result_df[temperature_col].between(97.0, 100.0)
    )

    hidden_instability_threshold = (
        result_df[hidden_instability_score_col]
        >
        result_df[hidden_instability_score_col].quantile(0.75)
    )

    result_df[output_col] = (
        normal_surface_vitals
        &
        hidden_instability_threshold
    )

    return result_df

# This function calculates the High Shock With Acceptable Blood Pressure Flag based on shock index and systolic blood pressure values.
def calculate_high_shock_acceptable_bp_flag(
    df: pd.DataFrame,
    shock_index_col: str = "shock_index",
    sbp_col: str = "sbp",
    output_col: str = "high_shock_acceptable_bp_flag",
) -> pd.DataFrame:
    """
    Calculate High Shock With Acceptable Blood Pressure Flag.

    Purpose:
    Identify observations where shock burden
    appears elevated while visible systolic
    blood pressure remains acceptable.
    """

    result_df = df.copy()

    required_cols = [
        shock_index_col,
        sbp_col,
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in result_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Cannot calculate High Shock With Acceptable Blood Pressure Flag. Missing required columns: {missing_cols}"
        )

    elevated_shock = (
        result_df[shock_index_col]
        >
        result_df[shock_index_col].quantile(0.75)
    )

    acceptable_bp = (
        result_df[sbp_col] >= 100
    )

    result_df[output_col] = (
        elevated_shock
        &
        acceptable_bp
    )

    return result_df

# This function calculates the Hidden Respiratory Strain Flag based on SpO2 and respiratory load index values.
def calculate_hidden_respiratory_strain_flag(
    df: pd.DataFrame,
    spo2_col: str = "spo2",
    respiratory_load_index_col: str = "respiratory_load_index",
    output_col: str = "hidden_respiratory_strain_flag",
) -> pd.DataFrame:
    """
    Calculate Hidden Respiratory Strain Flag.

    Purpose:
    Identify observations where oxygen
    saturation appears acceptable while
    hidden respiratory burden remains elevated.
    """

    result_df = df.copy()

    required_cols = [
        spo2_col,
        respiratory_load_index_col,
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in result_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Cannot calculate Hidden Respiratory Strain Flag. Missing required columns: {missing_cols}"
        )

    acceptable_spo2 = (
        result_df[spo2_col] >= 95
    )

    elevated_respiratory_load = (
        result_df[respiratory_load_index_col]
        >
        result_df[respiratory_load_index_col].quantile(0.75)
    )

    result_df[output_col] = (
        acceptable_spo2
        &
        elevated_respiratory_load
    )

    return result_df

# This function calculates the False Recovery Signal based on stabilization durability score and hidden instability score values.
def calculate_false_recovery_signal(
    df: pd.DataFrame,
    false_vital_recovery_signal_col: str = "false_vital_recovery_signal",
    recovery_authenticity_state_col: str = "recovery_authenticity_state",
    output_col: str = "false_recovery_signal",
) -> pd.DataFrame:
    """
    Calculate False Recovery Signal.

    Purpose:
    Identify observations where vital-based
    recovery appears unreliable and recovery
    authenticity is not trustworthy.
    """

    result_df = df.copy()

    required_cols = [
        false_vital_recovery_signal_col,
        recovery_authenticity_state_col,
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in result_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Cannot calculate False Recovery Signal. Missing required columns: {missing_cols}"
        )

    result_df[output_col] = (
        (result_df[false_vital_recovery_signal_col] == 1)
        &
        (
            result_df[recovery_authenticity_state_col]
            == "RECOVERY_NOT_TRUSTWORTHY"
        )
    )

    return result_df

# This function calculates the Normal Range Deception Signal based on normal range deception score values.
def calculate_normal_range_deception_signal(
    df: pd.DataFrame,
    normal_range_deception_score_col: str = "normal_range_deception_score",
    output_col: str = "normal_range_deception_signal",
) -> pd.DataFrame:
    """
    Calculate Normal Range Deception Signal.

    Purpose:
    Identify observations where normal-range
    deception pressure is elevated.
    """

    result_df = df.copy()

    required_cols = [
        normal_range_deception_score_col,
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in result_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Cannot calculate Normal Range Deception Signal. Missing required columns: {missing_cols}"
        )

    elevated_normal_range_deception = (
        result_df[normal_range_deception_score_col]
        >
        result_df[normal_range_deception_score_col].quantile(0.75)
    )

    result_df[output_col] = elevated_normal_range_deception

    return result_df

# This function reviews HVI candidate signals for trajectory-aware measurement eligibility based on availability, missingness, and basic usability.
def review_trajectory_hvi_candidates(df, candidate_columns=None):
    """
    Review HVI candidate signals for trajectory-aware measurement eligibility.

    This function does not calculate velocity, acceleration, drift, or interpretation.
    It only reviews availability, missingness, and basic usability of candidate
    HVI outputs before trajectory measurements are created.

    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe containing HVI outputs.

    candidate_columns : list[str], optional
        Candidate HVI columns to review. If None, a default candidate list is used.

    Returns
    -------
    pandas.DataFrame
        Contributor review table.
    """

    import pandas as pd

    if candidate_columns is None:
        candidate_columns = [
            "hidden_instability_score",
            "hidden_burden_score",
            "compensation_burden_score",
            "multi_system_burden_index",
            "normal_range_deception_score",
            "re_escalation_pressure_score",
            "shock_index",
            "respiratory_load_index",
            "narrow_pressure_shock",
        ]

    total_rows = len(df)

    review_rows = []

    for col in candidate_columns:
        if col not in df.columns:
            review_rows.append({
                "candidate_signal": col,
                "column_available": False,
                "total_rows": total_rows,
                "non_null_count": 0,
                "missing_count": total_rows,
                "missing_rate": 1.0,
                "usable_for_trajectory_review": False,
                "review_note": "Column not available in source dataframe.",
            })
            continue

        non_null_count = df[col].notna().sum()
        missing_count = df[col].isna().sum()
        missing_rate = missing_count / total_rows if total_rows else 0

        usable = non_null_count > 0 and missing_rate < 0.50

        if missing_rate == 0:
            note = "Signal is fully available."
        elif missing_rate < 0.10:
            note = "Signal has minor missingness."
        elif missing_rate < 0.50:
            note = "Signal has moderate missingness and requires continuity review."
        else:
            note = "Signal is highly fragmented or insufficient for trajectory review."

        review_rows.append({
            "candidate_signal": col,
            "column_available": True,
            "total_rows": total_rows,
            "non_null_count": int(non_null_count),
            "missing_count": int(missing_count),
            "missing_rate": round(float(missing_rate), 4),
            "usable_for_trajectory_review": usable,
            "review_note": note,
        })

    return pd.DataFrame(review_rows)

# This function builds the exploratory Recovery Hidden Vitals candidate family review table, which groups candidate signals into observational categories
#  based on guiding questions.
def build_recovery_hidden_vitals_family_review():
    """
    Build the exploratory Recovery Hidden Vitals candidate family review table.

    These groupings are observational, not finalized doctrine.
    """

    import pandas as pd

    recovery_family_review = pd.DataFrame({
        "candidate_family": [
            "Recovery Authenticity",
            "Recovery Resilience",
            "Recovery Stability",
            "Recovery Legitimacy",
            "Recovery Contradiction",
            "Recovery Re-Escalation",
        ],
        "guiding_question": [
            "Does recovery appear authentic?",
            "Does recovery appear resilient?",
            "How stable does recovery appear?",
            "How trustworthy does recovery appear?",
            "Do other domains appear to disagree with recovery?",
            "Does recovery appear vulnerable to re-escalation?",
        ],
        "candidate_signals": [
            [
                "recovery_authenticity_state",
                "vital_recovery_authenticity_warning",
                "false_vital_recovery_signal",
                "false_recovery_risk_flag",
                "recovery_claim_contradicted_flag",
                "contradictory_recovery_signature",
                "dependency_misinterpreted_as_recovery",
            ],
            [
                "recovery_resilience",
                "recovery_without_resilience",
                "context_recovery_resilience_pressure",
            ],
            [
                "recovery_instability_score",
                "recovery_momentum",
                "stabilization_durability_score",
                "trajectory_reversal_after_recovery",
            ],
            [
                "recovery_legitimacy_score",
                "recovery_quality_score",
                "recovery_trust_state",
            ],
            [
                "handoff_false_recovery_pressure_score",
                "imaging_recovery_contradiction_score",
                "lab_recovery_contradiction_score",
            ],
            [
                "re_escalation_pressure_score",
                "re_escalation_risk_flag",
                "quiet_gap_before_escalation_flag",
                "care_escalation_occurred",
            ],
        ],
        "status": [
            "Observation",
            "Observation",
            "Observation",
            "Observation",
            "Observation",
            "Observation",
        ],
    })

    return recovery_family_review