# ============================================================
# BIRE OS — Advanced Trajectory Intelligence (ATI)
# File: src/bire/ati/scoring.py
# Chapter: 64.0
#
# Created: 2026-06-15
# Updated: 2026-06-15
#
# CHANGED:
# - Added ATI output distribution review helpers
# - Added deceptive recovery driver and trigger reviews
# - Added rebound vulnerability driver and trigger reviews
#
# Purpose:
# Backend helper for ATI output scoring, interpretation review,
# trigger inspection, and trajectory output analysis.
# ============================================================
import pandas as pd
from sympy import python

def build_ati_output_distribution_review(df):
    """
    Review ATI interpretation output distributions.
    """


    output_cols = [
        "ati_trajectory_state",
        "ati_recovery_state",
        "ati_instability_pattern",
        "ati_rebound_vulnerability",
    ]

    rows = []

    for col in output_cols:
        if col not in df.columns:
            rows.append({
                "ati_output": col,
                "state": "MISSING_OUTPUT_COLUMN",
                "count": None,
            })
            continue

        counts = df[col].value_counts(dropna=False)

        for state, count in counts.items():
            rows.append({
                "ati_output": col,
                "state": state,
                "count": int(count),
            })

    return pd.DataFrame(rows)


def build_ati_deceptive_recovery_driver_review(df):
    """
    Review existing Playground signals associated with
    ATI deceptive recovery outputs.
    """

    deceptive_df = df[
        df["ati_trajectory_state"].eq("DECEPTIVE_RECOVERY")
    ]

    driver_cols = [
        "false_vital_recovery_signal",
        "recovery_trust_state",
        "recovery_instability_score",
        "recovery_momentum",
        "re_escalation_pressure_score",
        "ati_trajectory_state",
        "ati_recovery_state",
        "ati_trajectory_summary",
    ]

    available_cols = [
        col for col in driver_cols
        if col in deceptive_df.columns
    ]

    return deceptive_df[available_cols].describe(include="all").T


def build_ati_deceptive_recovery_trigger_review(df):
    """
    Review trigger counts contributing to ATI deceptive
    recovery outputs.
    """



    deceptive_df = df[
        df["ati_trajectory_state"].eq("DECEPTIVE_RECOVERY")
    ]

    false_recovery = (
        deceptive_df["false_vital_recovery_signal"].eq(1)
        if "false_vital_recovery_signal" in deceptive_df.columns
        else pd.Series(False, index=deceptive_df.index)
    )

    fragmented_trust = (
        deceptive_df["recovery_trust_state"]
        .astype(str)
        .str.contains("fragmented", case=False, na=False)
        if "recovery_trust_state" in deceptive_df.columns
        else pd.Series(False, index=deceptive_df.index)
    )

    return pd.DataFrame({
        "trigger": [
            "false_vital_recovery_signal",
            "fragmented_recovery_trust_state",
            "both_false_recovery_and_fragmented_trust",
        ],
        "count": [
            int(false_recovery.sum()),
            int(fragmented_trust.sum()),
            int((false_recovery & fragmented_trust).sum()),
        ],
    })


def build_ati_rebound_vulnerability_driver_review(df):
    """
    Review existing Playground signals associated with
    ATI rebound vulnerability outputs.
    """

    rebound_df = df[
        df["ati_trajectory_state"].eq("REBOUND_VULNERABILITY")
    ]

    driver_cols = [
        "re_escalation_pressure_score",
        "recovery_instability_score",
        "recovery_momentum",
        "false_vital_recovery_signal",
        "recovery_trust_state",
        "ati_trajectory_state",
        "ati_recovery_state",
        "ati_rebound_vulnerability",
        "ati_trajectory_summary",
    ]

    available_cols = [
        col for col in driver_cols
        if col in rebound_df.columns
    ]

    return rebound_df[available_cols].describe(include="all").T


def build_ati_rebound_vulnerability_trigger_review(df):
    """
    Review trigger counts contributing to ATI rebound
    vulnerability outputs.
    """



    rebound_df = df[
        df["ati_trajectory_state"].eq("REBOUND_VULNERABILITY")
    ]

    high_pressure = (
        rebound_df["re_escalation_pressure_score"].ge(0.40)
        if "re_escalation_pressure_score" in rebound_df.columns
        else pd.Series(False, index=rebound_df.index)
    )

    high_recovery_instability = (
        rebound_df["recovery_instability_score"].ge(0.40)
        if "recovery_instability_score" in rebound_df.columns
        else pd.Series(False, index=rebound_df.index)
    )

    return pd.DataFrame({
        "trigger": [
            "high_re_escalation_pressure",
            "high_recovery_instability",
            "both_pressure_and_instability",
        ],
        "count": [
            int(high_pressure.sum()),
            int(high_recovery_instability.sum()),
            int((high_pressure & high_recovery_instability).sum()),
        ],
    })

def build_ati_rebound_vulnerability_threshold_review(df):
    """
    Review rebound vulnerability threshold behavior.
    """

    import pandas as pd

    rebound_df = df[
        df["ati_trajectory_state"].eq("REBOUND_VULNERABILITY")
    ]

    return pd.DataFrame({
        "metric": [
            "re_escalation_pressure_min",
            "re_escalation_pressure_mean",
            "re_escalation_pressure_median",
            "recovery_instability_min",
            "recovery_instability_mean",
            "recovery_instability_median",
        ],
        "value": [
            rebound_df["re_escalation_pressure_score"].min(),
            rebound_df["re_escalation_pressure_score"].mean(),
            rebound_df["re_escalation_pressure_score"].median(),
            rebound_df["recovery_instability_score"].min(),
            rebound_df["recovery_instability_score"].mean(),
            rebound_df["recovery_instability_score"].median(),
        ],
    })



# This function investigates minority ATI recovery interpretation outputs.
def build_ati_minority_recovery_investigation(df):
    """
    Investigate minority ATI recovery interpretation outputs.
    """

    minority_states = [
        "TEMPORARY_NORMALIZATION",
        "FRAGILE_STABILIZATION",
        "RESILIENT_RECOVERY",
    ]

    driver_cols = [
        "recovery_instability_score",
        "recovery_momentum",
        "re_escalation_pressure_score",
        "false_vital_recovery_signal",
        "recovery_trust_state",
        "ati_trajectory_state",
        "ati_recovery_state",
        "ati_rebound_vulnerability",
        "ati_trajectory_summary",
    ]

    rows = []

    for state in minority_states:
        state_df = df[
            df["ati_trajectory_state"].eq(state)
            | df["ati_recovery_state"].eq(state)
        ]

        row = {
            "ati_state": state,
            "record_count": len(state_df),
        }

        for col in driver_cols:
            if col not in state_df.columns:
                continue

            if pd.api.types.is_numeric_dtype(state_df[col]):
                row[f"{col}_mean"] = state_df[col].mean()
                row[f"{col}_median"] = state_df[col].median()
                row[f"{col}_min"] = state_df[col].min()
                row[f"{col}_max"] = state_df[col].max()
            else:
                mode = state_df[col].mode(dropna=False)
                row[f"{col}_top"] = mode.iloc[0] if not mode.empty else None
                row[f"{col}_top_count"] = (
                    state_df[col].value_counts(dropna=False).iloc[0]
                    if len(state_df) > 0 else 0
                )

        rows.append(row)

    return pd.DataFrame(rows)


# This function investigates ATI instability pattern outputs.
def build_ati_instability_pattern_investigation(df):
    """
    Investigate ATI instability pattern outputs.
    """

    instability_states = [
        "COMPENSATION_WEAKENING",
        "PROGRESSIVE_WORSENING",
    ]

    driver_cols = [
        "hidden_instability_score",
        "hidden_instability_score_velocity",
        "hidden_instability_score_acceleration",
        "hidden_instability_persistence",
        "vital_burden_score",
        "vital_burden_score_velocity",
        "compensation_reserve_score",
        "compensation_reserve_score_velocity",
        "ati_trajectory_state",
        "ati_instability_pattern",
        "ati_trajectory_summary",
    ]

    rows = []

    for state in instability_states:
        state_df = df[
            df["ati_instability_pattern"].eq(state)
        ]

        row = {
            "ati_instability_pattern": state,
            "record_count": len(state_df),
        }

        for col in driver_cols:
            if col not in state_df.columns:
                continue

            if pd.api.types.is_numeric_dtype(state_df[col]):
                row[f"{col}_mean"] = state_df[col].mean()
                row[f"{col}_median"] = state_df[col].median()
                row[f"{col}_min"] = state_df[col].min()
                row[f"{col}_max"] = state_df[col].max()
            else:
                mode = state_df[col].mode(dropna=False)
                row[f"{col}_top"] = mode.iloc[0] if not mode.empty else None
                row[f"{col}_top_count"] = (
                    state_df[col].value_counts(dropna=False).iloc[0]
                    if len(state_df) > 0 else 0
                )

        rows.append(row)

    return pd.DataFrame(rows)


def build_ati_output_crosswalk(df):
    """
    Build ATI output crosswalk.

    Compares ATI interpretation outputs against one another
    to inspect alignment between trajectory state, recovery state,
    instability pattern, and rebound vulnerability.
    """

    required_cols = [
        "ati_trajectory_state",
        "ati_recovery_state",
        "ati_instability_pattern",
        "ati_rebound_vulnerability",
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        return pd.DataFrame({
            "missing_required_columns": [", ".join(missing_cols)]
        })

    crosswalk = (
        df.groupby(required_cols)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    return crosswalk

# This function builds a distribution table for the ATI trajectory behavior.
def build_ati_trajectory_behavior_distribution(df):
    """
    Build ATI trajectory behavior distribution table.
    """

    import pandas as pd

    if "ati_trajectory_behavior" not in df.columns:
        return pd.DataFrame({
            "missing_required_column": [
                "ati_trajectory_behavior"
            ]
        })

    distribution = (
        df["ati_trajectory_behavior"]
        .value_counts(dropna=False)
        .reset_index()
    )

    distribution.columns = [
        "ati_trajectory_behavior",
        "count",
    ]

    return distribution


# This function identifies gaps in the integrated interpretation of ATI data.
def build_ati_integrated_interpretation_gaps(df):
    """
    Identify ATI output combinations that currently fail
    integrated interpretation.
    """

    required_cols = [
        "ati_trajectory_state",
        "ati_recovery_state",
        "ati_instability_pattern",
        "ati_rebound_vulnerability",
        "ati_trajectory_behavior",
        "ati_interpretation_trust_state",
        "ati_integrated_interpretation",
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        return pd.DataFrame({
            "missing_required_columns": [", ".join(missing_cols)]
        })

    gap_df = df[
        df["ati_integrated_interpretation"].eq(
            "Insufficient ATI context available for integrated interpretation."
        )
    ]

    gaps = (
        gap_df.groupby([
            "ati_trajectory_state",
            "ati_recovery_state",
            "ati_instability_pattern",
            "ati_rebound_vulnerability",
            "ati_trajectory_behavior",
            "ati_interpretation_trust_state",
        ])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    return gaps


# This function identifies gaps in the integrated interpretation of ATI data
# involving compensation weakening.
def build_ati_compensation_weakening_gaps(df):
    """
    Identify remaining ATI integrated interpretation gaps
    involving compensation weakening.
    """

    required_cols = [
        "ati_trajectory_state",
        "ati_recovery_state",
        "ati_instability_pattern",
        "ati_rebound_vulnerability",
        "ati_trajectory_behavior",
        "ati_interpretation_trust_state",
        "ati_integrated_interpretation",
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        return pd.DataFrame({
            "missing_required_columns": [", ".join(missing_cols)]
        })

    gap_text = (
        "Insufficient ATI context available for integrated interpretation."
    )

    gap_df = df[
        df["ati_integrated_interpretation"].eq(gap_text)
        &
        df["ati_instability_pattern"].eq("COMPENSATION_WEAKENING")
    ]

    return (
        gap_df.groupby([
            "ati_trajectory_state",
            "ati_recovery_state",
            "ati_instability_pattern",
            "ati_rebound_vulnerability",
            "ati_trajectory_behavior",
            "ati_interpretation_trust_state",
        ])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


# This function investigates limited trajectory context outputs in the ATI data.
def build_ati_limited_context_investigation(df):
    """
    Investigate ATI limited trajectory context outputs.
    """

    import pandas as pd

    limited_df = df[
        df["ati_context_sufficiency"]
        .eq("LIMITED_TRAJECTORY_CONTEXT")
    ]

    return (
        limited_df.groupby([
            "ati_trajectory_state",
            "ati_recovery_state",
            "ati_instability_pattern",
            "ati_rebound_vulnerability",
            "ati_trajectory_behavior",
            "ati_interpretation_trust_state",
        ])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
# This function classifies recurring ATI context-deficit patterns.
def build_ati_context_deficit_classification(df):
    """
    Classify recurring ATI context-deficit patterns.
    """

    limited_df = df[
        df["ati_context_sufficiency"]
        .eq("LIMITED_TRAJECTORY_CONTEXT")
    ].copy()

    limited_df["context_deficit_type"] = "OTHER_CONTEXT_DEFICIT"

    limited_df.loc[
        limited_df["ati_instability_pattern"]
        .eq("INSUFFICIENT_TRAJECTORY_CONTEXT"),
        "context_deficit_type"
    ] = "MISSING_INSTABILITY_CONTEXT"

    limited_df.loc[
        limited_df["ati_interpretation_trust_state"]
        .eq("LOW_TRAJECTORY_TRUST"),
        "context_deficit_type"
    ] = "LOW_INTERPRETATION_TRUST"

    limited_df.loc[
        limited_df["ati_trajectory_behavior"]
        .eq("INSUFFICIENT_TRAJECTORY_CONTEXT"),
        "context_deficit_type"
    ] = "MISSING_BEHAVIOR_CONTEXT"

    return (
        limited_df["context_deficit_type"]
        .value_counts(dropna=False)
        .reset_index()
    )

# This function analyzes drivers associated with ATI low interpretation trust.
def build_ati_low_trust_driver_analysis(df):
    """
    Analyze drivers associated with ATI low interpretation trust.
    """

    low_trust_df = df[
        df["ati_interpretation_trust_state"].eq("LOW_TRAJECTORY_TRUST")
    ].copy()

    driver_cols = [
        "recovery_trust_state",
        "false_vital_recovery_signal",
        "ati_trajectory_state",
        "ati_recovery_state",
        "ati_instability_pattern",
        "ati_rebound_vulnerability",
        "ati_trajectory_behavior",
        "ati_context_sufficiency",
    ]

    rows = []

    for col in driver_cols:
        if col not in low_trust_df.columns:
            continue

        counts = (
            low_trust_df[col]
            .value_counts(dropna=False)
            .head(10)
        )

        for value, count in counts.items():
            rows.append({
                "driver_column": col,
                "value": value,
                "count": int(count),
            })

    return pd.DataFrame(rows)


# This function checks whether reconstructable fragmented context produces
# usable ATI integrated interpretations and trust levels.
def build_ati_reconstructable_context_interpretation_check(df):
    """
    Check whether reconstructable fragmented context produces
    usable ATI integrated interpretations and trust levels.
    """

    required_cols = [
        "ati_fragmented_context_state",
        "ati_integrated_interpretation",
        "ati_interpretation_trust_state",
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        return pd.DataFrame({
            "missing_required_columns": [", ".join(missing_cols)]
        })

    recon_df = df[
        df["ati_fragmented_context_state"].eq(
            "RECONSTRUCTABLE_CONTEXT"
        )
    ].copy()

    return (
        recon_df.groupby([
            "ati_interpretation_trust_state",
            "ati_integrated_interpretation",
        ])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )