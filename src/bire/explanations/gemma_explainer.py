import pandas as pd


def _safe_get(row, key, default=None):
    try:
        value = row.get(key, default)
        if pd.isna(value):
            return default
        return value
    except Exception:
        return default


def build_patient_chart_context(patient_df):
    """
    Build a compact patient chart summary from the patient's timeline.

    This does NOT make clinical decisions.
    It only summarizes BIRE outputs and signal trends for Gemma.
    """
    p = patient_df.copy()

    if p.empty:
        raise ValueError("patient_df is empty.")

    if "timestamp" in p.columns:
        p["timestamp"] = pd.to_datetime(p["timestamp"])
        p = p.sort_values("timestamp")

    latest = p.iloc[-1]

    patient_id = _safe_get(latest, "patient_id", "UNKNOWN")

    first_time = _safe_get(p.iloc[0], "timestamp", None)
    last_time = _safe_get(latest, "timestamp", None)

    final_tier = _safe_get(latest, "bire_final_tier", "UNKNOWN")
    risk = _safe_get(latest, "pred_proba", None)
    timing = _safe_get(latest, "bire_timing", _safe_get(latest, "timing_category", "unknown"))

    monitor_state = _safe_get(latest, "monitor_state", None)
    re_reason = _safe_get(latest, "re_escalate_reason", None)
    critical_reason = _safe_get(latest, "critical_reason", None)
    decision_reason = _safe_get(latest, "bire_decision_reason", None)

    risk_start = _safe_get(p.iloc[0], "pred_proba", None)
    risk_max = p["pred_proba"].max() if "pred_proba" in p.columns else None
    risk_min = p["pred_proba"].min() if "pred_proba" in p.columns else None

    if "risk_trend" in p.columns:
        latest_risk_trend = _safe_get(latest, "risk_trend", None)
    else:
        latest_risk_trend = None

    if "abnormal_count" in p.columns:
        abnormal_count = _safe_get(latest, "abnormal_count", None)
    else:
        abnormal_count = _safe_get(latest, "ibpip_n_abnormal_signals", None)

    event_count = int(p["event_now"].sum()) if "event_now" in p.columns else 0

    tier_path = (
        p["bire_final_tier"]
        .dropna()
        .astype(str)
        .tolist()
        if "bire_final_tier" in p.columns
        else []
    )

    compressed_path = []
    for tier in tier_path:
        if not compressed_path or compressed_path[-1] != tier:
            compressed_path.append(tier)

    context = f"""
Patient Chart Context:
- Patient ID: {patient_id}
- Timeline start: {first_time}
- Timeline end: {last_time}
- Events observed: {event_count}

Current BIRE State:
- Final tier: {final_tier}
- Risk score: {risk}
- Timing category: {timing}
- Monitor state: {monitor_state}
- Re-escalation reason: {re_reason}
- Critical reason: {critical_reason}
- BIRE decision reason: {decision_reason}

Risk Trajectory:
- Starting risk: {risk_start}
- Minimum risk: {risk_min}
- Maximum risk: {risk_max}
- Latest risk trend: {latest_risk_trend}
- Latest abnormal signal count: {abnormal_count}

Lifecycle Path:
- {' → '.join(compressed_path)}
"""
    return context.strip()


def build_bire_patient_explanation_prompt(patient_df):
    """
    Build a safe Gemma prompt for explaining what is happening
    across a patient's BIRE chart/timeline.

    Gemma explains.
    BIRE decides.
    """
    chart_context = build_patient_chart_context(patient_df)

    prompt = f"""
You are explaining the output of BIRE, a research prototype clinical intelligence system.

Important rules:
- Do not diagnose.
- Do not recommend treatment.
- Do not claim the system is clinically validated.
- Do not say the patient definitely has a condition.
- Explain the system state cautiously and clearly.
- Focus only on risk score, trajectory, timing, vitals instability, and BIRE decision logic.
- Use concise clinical-style language.
- Make it understandable to a clinical reviewer or project evaluator.

Core principle:
BIRE decides. You explain BIRE's decision.

{chart_context}

Write the explanation with this exact structure:

1. Summary:
Explain what BIRE is currently showing for this patient.

2. Timeline interpretation:
Explain how the patient moved through the BIRE lifecycle.

3. Why BIRE flagged this:
Explain the signals supporting the current tier.

4. Post-event interpretation:
If the patient is in MONITOR, RE-ESCALATE, or CRITICAL, explain what BIRE is observing after deterioration onset.

5. Safety note:
State that this is a research prototype explanation and not a diagnosis or treatment recommendation.
"""
    return prompt.strip()


def build_bire_row_explanation_prompt(row):
    """
    Build a safe explanation prompt for one row/timestamp.
    Useful for explaining a single CRITICAL, URGENT, or RE-ESCALATE decision.
    """
    tier = _safe_get(row, "bire_final_tier", "UNKNOWN")
    risk = _safe_get(row, "pred_proba", None)
    timing = _safe_get(row, "bire_timing", _safe_get(row, "timing_category", "unknown"))

    initial_reason = _safe_get(row, "bire_decision_reason", None)
    monitor_state = _safe_get(row, "monitor_state", None)
    re_reason = _safe_get(row, "re_escalate_reason", None)
    critical_reason = _safe_get(row, "critical_reason", None)

    abnormal_count = _safe_get(row, "abnormal_count", _safe_get(row, "ibpip_n_abnormal_signals", None))
    risk_trend = _safe_get(row, "risk_trend", None)

    prompt = f"""
You are explaining a single BIRE decision from a research prototype clinical intelligence system.

Important rules:
- Do not diagnose.
- Do not recommend treatment.
- Do not claim the system is clinically validated.
- Explain the risk state clearly and cautiously.
- Focus on signals, trajectory, timing, and system reasoning.

BIRE Decision Context:
- Final BIRE tier: {tier}
- Risk score: {risk}
- Timing category: {timing}
- Initial system reasoning: {initial_reason}
- Monitor state: {monitor_state}
- Re-escalation reason: {re_reason}
- Critical reason: {critical_reason}
- Abnormal signal count: {abnormal_count}
- Risk trend: {risk_trend}

Write a concise explanation with this structure:

1. Summary:
2. Why BIRE flagged this:
3. Timing interpretation:
4. Safety note:
"""
    return prompt.strip()
