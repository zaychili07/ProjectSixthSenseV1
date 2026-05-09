import pandas as pd

def build_latest_patient_view(df):
    """
    Get the latest available row for each patient.
    """

    latest_patient_view = (
        df.sort_values("timestamp")
        .groupby("patient_id")
        .tail(1)
        .copy()
    )

    return latest_patient_view


def build_cohort_intelligence_view(df):
    """
    Build the first cohort-level PSR intelligence view.
    """

    latest_patient_view = build_latest_patient_view(df)

    latest_patient_view = calculate_psr_scores(latest_patient_view)

    latest_patient_view["psr_attention_band"] = (
        latest_patient_view["psr_attention_score"]
        .apply(assign_psr_attention_band)
    )

    cohort_cols = [
        "patient_id",
        "timestamp",
        "pred_proba",
        "risk_band",
        "bire_final_tier",
        "bire_state",
        "lead_time_min",
        "abnormal_count",
        "risk_delta",
        "risk_std",
        "monitor_state",
        "re_escalate_flag",
        "critical_flag",
        "psr_attention_score",
        "psr_attention_band",
    ]

    available_cols = [
        col for col in cohort_cols
        if col in latest_patient_view.columns
    ]

    cohort_view = (
        latest_patient_view[available_cols]
        .sort_values("psr_attention_score", ascending=False)
        .reset_index(drop=True)
    )

    return cohort_view

def build_psr_score_breakdown(df):
    """
    Build an interpretable PSR score component breakdown.
    """

    psr_df = calculate_psr_scores(df)

    breakdown_cols = [
        "patient_id",
        "timestamp",
        "bire_final_tier",
        "bire_state",
        "pred_proba",
        "abnormal_count",
        "critical_flag",
        "re_escalate_flag",
        "risk_score_component",
        "abnormal_burden_component",
        "critical_component",
        "re_escalation_component",
        "psr_attention_score",
    ]

    available_cols = [
        col for col in breakdown_cols
        if col in psr_df.columns
    ]

    breakdown_view = (
        psr_df[available_cols]
        .sort_values("psr_attention_score", ascending=False)
        .reset_index(drop=True)
    )

    return breakdown_view


# 39.13 — PSR Attention Bands
def assign_psr_attention_band(score):

    if score >= 200:
        return "CRITICAL ATTENTION"

    elif score >= 120:
        return "HIGH ATTENTION"

    elif score >= 60:
        return "MODERATE ATTENTION"

    else:
        return "LOW ATTENTION"


# Apply Attention Bands
psr_operational_queue["psr_attention_band"] = (
    psr_operational_queue["psr_attention_score"]
    .apply(assign_psr_attention_band)
)


# Display Updated Queue
band_cols = [
    "patient_id",
    "timestamp",
    "psr_attention_score",
    "psr_attention_band",
    "pred_proba",
    "bire_final_tier",
    "bire_state",
    "monitor_state",
    "re_escalate_flag",
    "critical_flag",
]

available_band_cols = [
    col for col in band_cols
    if col in psr_operational_queue.columns
]



def get_critical_attention_cohort(df):
    return df[
        df["psr_attention_band"] == "CRITICAL ATTENTION"
    ]


def get_re_escalation_cohort(df):
    return df[
        df["re_escalate_flag"] == True
    ]


def get_post_event_monitoring_cohort(df):
    return df[
        df["bire_state"] == "Post-event monitoring"
    ]

def build_context_tags(row):

    tags = []

    if row.get("re_escalate_flag") == True:
        tags.append("RE_ESCALATION")

    if row.get("critical_flag") == True:
        tags.append("CRITICAL")

    if row.get("bire_state") == "Post-event monitoring":
        tags.append("POST_EVENT")

    if row.get("monitor_state") == "DECLINING_MONITOR":
        tags.append("DECLINING_MONITOR")

    return ", ".join(tags)



# Build Context Tags
psr_operational_queue["context_tags"] = (
    psr_operational_queue
    .apply(build_context_tags, axis=1)
)


# Example Dynamic Current Status
def determine_current_status(row):

    if row.get("critical_flag") == True:
        return "WORSENING"

    if row.get("re_escalate_flag") == True:
        return "UNSTABLE"

    return "STABLE"


psr_operational_queue["current_status"] = (
    psr_operational_queue
    .apply(determine_current_status, axis=1)
)


# Add Operational Rank
psr_operational_queue = (
    psr_operational_queue
    .sort_values("psr_attention_score", ascending=False)
    .reset_index(drop=True)
)

psr_operational_queue["operational_rank"] = (
    psr_operational_queue.index + 1
)

def build_monitoring_summary(row):
    
    if (
        row.get("critical_flag") == True
        and row.get("re_escalate_flag") == True
        and row.get("bire_state") == "Post-event monitoring"
    ):
        return "Post-event decline with re-escalation concern"
    
    if row.get("critical_flag") == True:
        return "Critical instability requiring close review"
    
    if row.get("re_escalate_flag") == True:
        return "Re-escalation concern detected"
    
    if row.get("bire_state") == "Post-event monitoring":
        return "Post-event monitoring active"
    
    if row.get("psr_attention_band") == "HIGH ATTENTION":
        return "High-priority monitoring case"
    
    if row.get("psr_attention_band") == "MODERATE ATTENTION":
        return "Moderate concern; continue monitoring"
    
    return "Low current operational concern"


def simplify_attention_band(band):
    
    band_map = {
        "CRITICAL ATTENTION": "Critical",
        "HIGH ATTENTION": "High",
        "MODERATE ATTENTION": "Moderate",
        "LOW ATTENTION": "Low",
    }
    
    return band_map.get(band, band)


def simplify_status(status):
    
    status_map = {
        "WORSENING": "Worsening",
        "UNSTABLE": "Unstable",
        "STABLE": "Stable",
        "IMPROVING": "Improving",
    }
    
    return status_map.get(status, status)


# ------------------------------------------
# Build Human-Readable Queue
# ------------------------------------------

human_queue = psr_operational_queue.copy()

human_queue["Attention"] = (
    human_queue["psr_attention_band"]
    .apply(simplify_attention_band)
)

human_queue["Status"] = (
    human_queue["current_status"]
    .apply(simplify_status)
)

human_queue["Monitoring Summary"] = (
    human_queue
    .apply(build_monitoring_summary, axis=1)
)

human_queue["PSR Score"] = (
    human_queue["psr_attention_score"]
    .round(1)
)

human_queue_view = human_queue[[
    "operational_rank",
    "patient_id",
    "Attention",
    "Status",
    "Monitoring Summary",
    "PSR Score",
]].rename(columns={
    "operational_rank": "Rank",
    "patient_id": "Patient",
})

display(human_queue_view)


# Final Primary Queue
primary_queue_cols = [
    "operational_rank",
    "patient_id",
    "psr_attention_band",
    "current_status",
    "context_tags",
    "psr_attention_score",
]

display(
    psr_operational_queue[primary_queue_cols]
)

import pandas as pd


def classify_psr_drift(delta):
    """
    Classify PSR score movement over time.
    """

    if pd.isna(delta):
        return "INITIAL"

    if delta >= 25:
        return "RAPIDLY WORSENING"

    if delta >= 5:
        return "WORSENING"

    if delta <= -25:
        return "RAPIDLY IMPROVING"

    if delta <= -5:
        return "IMPROVING"

    return "STABLE"


def add_psr_drift_tracking(df):
    """
    Add PSR temporal drift and persistence tracking.
    """

    out = (
        df.sort_values(["patient_id", "timestamp"])
        .copy()
    )

    out["psr_delta"] = (
        out.groupby("patient_id")["psr_attention_score"]
        .diff()
    )

    out["psr_rolling_mean"] = (
        out.groupby("patient_id")["psr_attention_score"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    out["psr_drift_status"] = (
        out["psr_delta"]
        .apply(classify_psr_drift)
    )

    return out

def determine_queue_state(row):
    """
    Determine operational queue state from PSR status flags.
    """

    if (
        row.get("critical_flag") == True
        and row.get("re_escalate_flag") == True
    ):
        return "RE_ESCALATED"

    if row.get("critical_flag") == True:
        return "ESCALATED"

    if row.get("current_status") == "WORSENING":
        return "ACTIVE_REVIEW"

    if row.get("current_status") == "UNSTABLE":
        return "MONITORING"

    if row.get("current_status") == "STABLE":
        return "STABLE_OBSERVATION"

    return "UNREVIEWED"


def add_queue_states(df):
    """
    Add operational queue states to the PSR queue.
    """

    out = df.copy()

    out["queue_state"] = (
        out.apply(determine_queue_state, axis=1)
    )

    return out

def add_patient_summary_fields(
    df,
    default_summary="No verified clinical summary available.",
    default_verified_by="Pending clinician verification",
):
    """
    Add patient summary and verification fields to the PSR queue.
    """

    out = df.copy()

    out["patient_summary"] = default_summary
    out["verified_by"] = default_verified_by

    return out


def update_patient_summary(
    df,
    patient_id,
    patient_summary,
    verified_by,
):
    """
    Update verified patient summary context for a specific patient.
    """

    out = df.copy()

    patient_mask = out["patient_id"] == patient_id

    out.loc[patient_mask, "patient_summary"] = patient_summary
    out.loc[patient_mask, "verified_by"] = verified_by

    return out
