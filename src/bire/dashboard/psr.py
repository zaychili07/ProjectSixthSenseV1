import pandas as pd


PSR_WEIGHTS = {
    "risk_score_weight": 100,
    "abnormal_burden_weight": 10,
    "critical_weight": 50,
    "re_escalation_weight": 25,
}


def build_latest_patient_view(df):
    """
    Get the latest available row for each patient.
    """
    return (
        df.sort_values("timestamp")
        .groupby("patient_id")
        .tail(1)
        .copy()
    )


def calculate_psr_scores(df, weights=None):
    """
    Calculate PSR attention score and component breakdown.
    """
    weights = weights or PSR_WEIGHTS

    out = df.copy()

    out["risk_score_component"] = (
        out["pred_proba"].fillna(0)
        * weights["risk_score_weight"]
    )

    out["abnormal_burden_component"] = (
        out["abnormal_count"].fillna(0)
        * weights["abnormal_burden_weight"]
    )

    out["critical_component"] = (
        out["critical_flag"].fillna(False).astype(int)
        * weights["critical_weight"]
    )

    out["re_escalation_component"] = (
        out["re_escalate_flag"].fillna(False).astype(int)
        * weights["re_escalation_weight"]
    )

    out["psr_attention_score"] = (
        out["risk_score_component"]
        + out["abnormal_burden_component"]
        + out["critical_component"]
        + out["re_escalation_component"]
    )

    return out


def assign_psr_attention_band(score):
    """
    Assign operational attention band from PSR score.
    """
    if score >= 200:
        return "CRITICAL ATTENTION"
    if score >= 120:
        return "HIGH ATTENTION"
    if score >= 60:
        return "MODERATE ATTENTION"
    return "LOW ATTENTION"


def determine_current_status(row):
    """
    Determine current patient trajectory status.
    """
    if row.get("critical_flag") is True:
        return "WORSENING"

    if row.get("re_escalate_flag") is True:
        return "UNSTABLE"

    return "STABLE"


def determine_queue_state(row):
    """
    Determine operational queue state from PSR status flags.
    """
    if (
        row.get("critical_flag") is True
        and row.get("re_escalate_flag") is True
    ):
        return "RE_ESCALATED"

    if row.get("critical_flag") is True:
        return "ESCALATED"

    if row.get("current_status") == "WORSENING":
        return "ACTIVE_REVIEW"

    if row.get("current_status") == "UNSTABLE":
        return "MONITORING"

    if row.get("current_status") == "STABLE":
        return "STABLE_OBSERVATION"

    return "UNREVIEWED"


def build_context_tags(row):
    """
    Build compact backend context tags for a PSR row.
    """
    tags = []

    if row.get("re_escalate_flag") is True:
        tags.append("RE_ESCALATION")

    if row.get("critical_flag") is True:
        tags.append("CRITICAL")

    if row.get("bire_state") == "Post-event monitoring":
        tags.append("POST_EVENT")

    if row.get("monitor_state") == "DECLINING_MONITOR":
        tags.append("DECLINING_MONITOR")

    return ", ".join(tags)


def build_psr_operational_queue(df):
    """
    Build ranked PSR operational queue.
    """
    out = calculate_psr_scores(df)

    out["psr_attention_band"] = (
        out["psr_attention_score"]
        .apply(assign_psr_attention_band)
    )

    out["current_status"] = (
        out.apply(determine_current_status, axis=1)
    )

    out["queue_state"] = (
        out.apply(determine_queue_state, axis=1)
    )

    out["context_tags"] = (
        out.apply(build_context_tags, axis=1)
    )

    out = (
        out.sort_values("psr_attention_score", ascending=False)
        .reset_index(drop=True)
    )

    out["operational_rank"] = out.index + 1

    return out


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

    return (
        latest_patient_view[available_cols]
        .sort_values("psr_attention_score", ascending=False)
        .reset_index(drop=True)
    )


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

    return (
        psr_df[available_cols]
        .sort_values("psr_attention_score", ascending=False)
        .reset_index(drop=True)
    )


def get_critical_attention_cohort(df):
    """
    Filter CRITICAL ATTENTION patients.
    """
    return df[df["psr_attention_band"] == "CRITICAL ATTENTION"]


def get_re_escalation_cohort(df):
    """
    Filter patients with active re-escalation behavior.
    """
    return df[df["re_escalate_flag"] == True]


def get_post_event_monitoring_cohort(df):
    """
    Filter patients in post-event monitoring.
    """
    return df[df["bire_state"] == "Post-event monitoring"]


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


def add_queue_states(df):
    """
    Add operational queue states to the PSR queue.
    """
    out = df.copy()

    out["queue_state"] = (
        out.apply(determine_queue_state, axis=1)
    )

    return out


def simplify_attention_band(band):
    """
    Convert PSR attention band to human-readable display text.
    """
    band_map = {
        "CRITICAL ATTENTION": "Critical",
        "HIGH ATTENTION": "High",
        "MODERATE ATTENTION": "Moderate",
        "LOW ATTENTION": "Low",
    }

    return band_map.get(band, band)


def simplify_status(status):
    """
    Convert backend status to human-readable display text.
    """
    status_map = {
        "WORSENING": "Worsening",
        "UNSTABLE": "Unstable",
        "STABLE": "Stable",
        "IMPROVING": "Improving",
    }

    return status_map.get(status, status)


def build_human_readable_psr_queue(df):
    """
    Build clean human-readable PSR queue view.
    """
    out = df.copy()

    out["Attention"] = (
        out["psr_attention_band"]
        .apply(simplify_attention_band)
    )

    out["Status"] = (
        out["current_status"]
        .apply(simplify_status)
    )

    out["PSR Score"] = (
        out["psr_attention_score"]
        .round(1)
    )

    view = out[[
        "operational_rank",
        "patient_id",
        "Attention",
        "Status",
        "queue_state",
        "PSR Score",
    ]].rename(columns={
        "operational_rank": "Rank",
        "patient_id": "Patient",
        "queue_state": "Queue State",
    })

    return view


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


def build_summary_queue_view(df):
    """
    Build final PSR summary queue display view.
    """
    summary_cols = [
    "operational_rank",
    "patient_id",
    "psr_attention_band",
    "current_status",
    "queue_state",
    "psr_attention_score",
    "patient_summary",
    "verified_by",
]

    available_cols = [
        col for col in summary_cols
        if col in df.columns
    ]

    view = df[available_cols].rename(columns={
        "operational_rank": "Rank",
        "patient_id": "Patient",
        "psr_attention_band": "Attention",
        "current_status": "Status",
        "queue_state": "Queue State",
        "psr_attention_score": "PSR Score",
        "patient_summary": "Patient Summary",
        "verified_by": "Verified By",
    })

    if "PSR Score" in view.columns:
        view["PSR Score"] = view["PSR Score"].round(1)

    return view
