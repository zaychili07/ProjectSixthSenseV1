"""
BIRE OS Ecosystem Schema Navigation Utilities

Chapter 54 purpose:
Build a usable navigation layer for the Chapter 53 synthetic healthcare
ecosystem so BIRE OS can understand, categorize, and explore its own
large-scale intelligence schema.

This module helps answer:
"Where do I look when I want to investigate a specific kind of failure?"

We Detect What Others Miss.
"""

from __future__ import annotations

import pandas as pd


NAVIGATION_RULES = {
    "Patient / Encounter Identity": [
        "patient_id", "encounter_id", "encounter_start", "encounter_end",
        "encounter_year", "encounter_sequence", "patient_age",
    ],
    "Care Mode / Acuity": [
        "care_mode", "initial_care_mode", "highest_acuity_mode", "final_care_mode",
        "acuity", "esi", "icu", "transfer",
    ],
    "Lifecycle / State Logic": [
        "lifecycle", "state", "phase", "post_event", "transition",
        "monitor", "escalation", "reintegration", "de_escalation",
    ],
    "Hidden Instability / HVI": [
        "hidden_instability", "small_signal", "deceptive_stability",
        "masked_instability", "oscillatory", "compensation", "silent",
    ],
    "Deterioration / Collapse": [
        "deterioration", "collapse", "rebound", "crash", "terminal",
        "decline", "worsening",
    ],
    "False Reassurance / Deception": [
        "false_reassurance", "deceived", "deception", "misleading",
        "masked", "false_stability", "false_recovery",
    ],
    "Vitals / Physiology": [
        "heart_rate", "resp_rate", "spo2", "temperature", "sbp", "dbp",
        "map", "shock_index", "pulse_pressure", "physiology", "vital",
    ],
    "Labs / Chemistry": [
        "lab", "lactate", "wbc", "creatinine", "bicarbonate", "ph",
        "pco2", "troponin", "hemoglobin", "glucose", "chemistry",
    ],
    "Imaging / Evidence": [
        "imaging", "scan", "radiology", "ct", "mri", "xray", "finding",
        "artifact", "read",
    ],
    "Medication / Intervention": [
        "medication", "therapeutic", "intervention", "treatment",
        "stabilization", "vasopressor", "oxygen", "sedation", "opioid",
    ],
    "Handoffs / Continuity": [
        "handoff", "continuity", "memory_decay", "fragmentation",
        "provider_change", "cross_cover", "reassessment_plan",
    ],
    "Operations / Hospital Pressure": [
        "operational", "hospital", "resource", "capacity", "bed",
        "boarding", "queue", "staffing", "nurse", "telemetry",
        "throughput", "overload",
    ],
    "Diagnosis / Interpretation": [
        "diagnosis", "diagnostic", "missed", "revision",
        "premature_closure", "underestimated",
    ],
    "Outcomes / Consequences": [
        "outcome", "final_outcome", "mortality", "readmission",
        "failure_mode", "recovery_quality", "discharge", "icu_transfer",
    ],
    "Uncertainty / Confidence": [
        "uncertainty", "confidence", "trust", "self_doubt",
        "contradiction", "conflict",
    ],
    "Learning / Upgrade Signals": [
        "learning", "upgrade", "teaching", "regret", "preventability",
        "avoidable",
    ],
}


def classify_column(column_name: str, navigation_rules: dict | None = None) -> list[str]:
    """
    Classify one ecosystem column into one or more navigation domains.
    """

    if navigation_rules is None:
        navigation_rules = NAVIGATION_RULES

    lower_col = str(column_name).lower()
    matched_domains = []

    for domain, keywords in navigation_rules.items():
        if any(keyword in lower_col for keyword in keywords):
            matched_domains.append(domain)

    if not matched_domains:
        matched_domains.append("Unmapped / General")

    return matched_domains


def build_ecosystem_navigation_map(
    df: pd.DataFrame,
    navigation_rules: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a schema navigation map and domain summary for a BIRE OS ecosystem dataframe.

    Returns
    -------
    ecosystem_navigation_map_df:
        One row per column with domain classification, dtype, missingness, and uniqueness.

    domain_summary_df:
        Aggregated summary by primary domain.
    """

    if navigation_rules is None:
        navigation_rules = NAVIGATION_RULES

    navigation_rows = []

    for column in df.columns:
        matched_domains = classify_column(
            column_name=column,
            navigation_rules=navigation_rules,
        )

        series = df[column]

        navigation_rows.append(
            {
                "column_name": column,
                "matched_domains": " | ".join(matched_domains),
                "primary_domain": matched_domains[0],
                "domain_count": len(matched_domains),
                "dtype": str(series.dtype),
                "non_null_count": int(series.notna().sum()),
                "null_count": int(series.isna().sum()),
                "null_rate": round(float(series.isna().mean()), 4),
                "unique_values": int(series.nunique(dropna=True)),
            }
        )

    ecosystem_navigation_map_df = pd.DataFrame(navigation_rows)

    domain_summary_df = (
        ecosystem_navigation_map_df
        .groupby("primary_domain")
        .agg(
            column_count=("column_name", "count"),
            avg_null_rate=("null_rate", "mean"),
            avg_unique_values=("unique_values", "mean"),
        )
        .reset_index()
        .sort_values("column_count", ascending=False)
    )

    return ecosystem_navigation_map_df, domain_summary_df


def get_columns_by_domain(
    navigation_map_df: pd.DataFrame,
    domain: str,
    include_secondary_matches: bool = True,
) -> list[str]:
    """
    Return all ecosystem columns associated with a selected navigation domain.
    """

    if include_secondary_matches:
        mask = navigation_map_df["matched_domains"].str.contains(
            domain,
            regex=False,
            na=False,
        )
    else:
        mask = navigation_map_df["primary_domain"].eq(domain)

    return navigation_map_df.loc[mask, "column_name"].tolist()


def search_ecosystem_columns(
    navigation_map_df: pd.DataFrame,
    keyword: str,
) -> pd.DataFrame:
    """
    Search ecosystem schema columns by keyword.
    """

    keyword = str(keyword).lower()

    return navigation_map_df[
        navigation_map_df["column_name"].str.lower().str.contains(
            keyword,
            na=False,
        )
    ].copy()