import pandas as pd


def _safe_get(row, key, default=None):
    """
    Safely get a value from a pandas row/Series.
    """
    try:
        value = row.get(key, default)
        if pd.isna(value):
            return default
        return value
    except Exception:
        return default


def _compress_path(values):
    """
    Compress repeated lifecycle states.

    Example:
    WATCH, WATCH, URGENT, URGENT, CRITICAL
    becomes:
    WATCH → URGENT → CRITICAL
    """
    compressed = []

    for value in values:
        if value is None:
            continue

        value = str(value)

        if not compressed or compressed[-1] != value:
            compressed.append(value)

    return compressed


def _get_latest_vitals(latest):
    """
    Extract latest vital signs from the most recent patient row.
    """
    return {
        "heart_rate": _safe_get(latest, "heart_rate", None),
        "resp_rate": _safe_get(latest, "resp_rate", None),
        "spo2": _safe_get(latest, "spo2", None),
        "temperature": _safe_get(latest, "temperature", None),
        "sbp": _safe_get(latest, "sbp", None),
        "dbp": _safe_get(latest, "dbp", None),
    }


def _detect_abnormal_findings(vitals):
    """
    Detect abnormal findings using BIRE's research thresholds.

    These are not diagnostic thresholds.
    They are signal thresholds used by the research prototype.
    """
    findings = []

    hr = vitals.get("heart_rate")
    rr = vitals.get("resp_rate")
    spo2 = vitals.get("spo2")
    temp = vitals.get("temperature")
    sbp = vitals.get("sbp")

    if hr is not None and hr > 120:
        findings.append(f"elevated heart rate ({hr})")

    if rr is not None and rr > 24:
        findings.append(f"elevated respiratory rate ({rr})")

    if spo2 is not None and spo2 < 92:
        findings.append(f"low oxygen saturation / SpO2 ({spo2})")

    if sbp is not None and sbp < 100:
        findings.append(f"low systolic blood pressure / SBP ({sbp})")

    if temp is not None and (temp > 38.5 or temp < 36):
        findings.append(f"abnormal temperature ({temp})")

    return findings


def build_patient_chart_context(patient_df):
    """
    Build a clinician-readable patient chart context for Gemma.

    This summarizes:
    - Current BIRE tier
    - Risk trajectory
    - Latest vitals
    - Abnormal findings
    - Timing category
    - Post-event status
    - Lifecycle path

    This function does NOT make clinical decisions.
    It only summarizes BIRE outputs and available patient signals.
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

    timing = _safe_get(
        latest,
        "bire_timing",
        _safe_get(latest, "timing_category", "unknown"),
    )

    monitor_state = _safe_get(latest, "monitor_state", None)
    re_reason = _safe_get(latest, "re_escalate_reason", None)
    critical_reason = _safe_get(latest, "critical_reason", None)
    decision_reason = _safe_get(latest, "bire_decision_reason", None)

    # Risk trajectory
    risk_start = _safe_get(p.iloc[0], "pred_proba", None)
    risk_latest = risk
    risk_max = p["pred_proba"].max() if "pred_proba" in p.columns else None
    risk_min = p["pred_proba"].min() if "pred_proba" in p.columns else None
    latest_risk_trend = _safe_get(latest, "risk_trend", None)

    # Event information
    event_count = int(p["event_now"].sum()) if "event_now" in p.columns else 0

    event_time = None
    if "event_now" in p.columns and "timestamp" in p.columns and p["event_now"].eq(1).any():
        event_time = p.loc[p["event_now"] == 1, "timestamp"].min()

    # Latest vitals + abnormal findings
    vitals = _get_latest_vitals(latest)
    abnormal_findings = _detect_abnormal_findings(vitals)

    abnormal_count = _safe_get(
        latest,
        "abnormal_count",
        _safe_get(latest, "ibpip_n_abnormal_signals", len(abnormal_findings)),
    )

    # Lifecycle path
    if "bire_final_tier" in p.columns:
        tier_values = p["bire_final_tier"].dropna().astype(str).tolist()
    else:
        tier_values = []

    compressed_path = _compress_path(tier_values)

    # Meaningful pre-event signal lead time
    meaningful_lead_time = None
    first_meaningful_signal_time = None

    if event_time is not None and "pred_proba" in p.columns and "timestamp" in p.columns:
        meaningful_pre_event = p[
            (p["timestamp"] < event_time)
            & (p["pred_proba"] >= 0.40)
        ]

        if not meaningful_pre_event.empty:
            first_meaningful_signal_time = meaningful_pre_event["timestamp"].min()
            meaningful_lead_time = int(
                (event_time - first_meaningful_signal_time).total_seconds() / 60
            )

    context = f"""
Patient Chart Context:
- Patient ID: {patient_id}
- Timeline start: {first_time}
- Timeline end: {last_time}
- Events observed: {event_count}
- Event time: {event_time}
- First meaningful pre-event signal time: {first_meaningful_signal_time}
- Meaningful lead time minutes: {meaningful_lead_time}

Current BIRE State:
- Final tier: {final_tier}
- Risk score: {risk_latest}
- Timing category: {timing}
- Monitor state: {monitor_state}
- Re-escalation reason: {re_reason}
- Critical reason: {critical_reason}
- BIRE decision reason: {decision_reason}

Latest Vitals:
- Heart rate: {vitals["heart_rate"]}
- Respiratory rate: {vitals["resp_rate"]}
- SpO2: {vitals["spo2"]}
- Temperature: {vitals["temperature"]}
- SBP: {vitals["sbp"]}
- DBP: {vitals["dbp"]}

Abnormal Findings:
- Abnormal signal count: {abnormal_count}
- Abnormal findings: {abnormal_findings}

Risk Trajectory:
- Starting risk: {risk_start}
- Minimum risk: {risk_min}
- Maximum risk: {risk_max}
- Latest risk trend: {latest_risk_trend}

Lifecycle Path:
- {' → '.join(compressed_path)}
"""
    return context.strip()


def build_bire_patient_explanation_prompt(patient_df):
    """
    Build a safe clinician-facing Gemma prompt for explaining
    what is happening across a patient's BIRE timeline.

    Gemma explains.
    BIRE decides.
    """
    chart_context = build_patient_chart_context(patient_df)

    prompt = f"""
You are generating a clinician-facing explanation for BIRE, a research prototype clinical intelligence system.

Core principle:
BIRE decides. You explain BIRE's decision.

Important safety rules:
- Do not diagnose.
- Do not recommend treatment.
- Do not claim the system is clinically validated.
- Do not say the patient definitely has a specific disease or condition.
- Do not use alarming language beyond what the BIRE signal supports.
- Explain what BIRE is observing from risk score, vitals, trajectory, timing, and system logic.
- Use clear clinical-style language that a clinician or evaluator can quickly understand.
- Be concise but useful.

{chart_context}

Write the explanation with this exact structure:

1. Clinical summary:
Briefly explain what BIRE is currently showing for this patient and why the clinician is looking at this case.

2. Current concern:
Explain the current tier using the latest risk score, risk trend, abnormal findings, and escalation reason.

3. Timeline interpretation:
Explain how the patient moved through the BIRE lifecycle over time.

4. Post-event interpretation:
If the patient is in MONITOR, RE-ESCALATE, or CRITICAL, explain what BIRE is observing after deterioration onset.

5. What to review:
List the specific signals a clinician may want to review in the chart, without recommending treatment.

6. Safety note:
State that this is a research prototype explanation and not a diagnosis, clinical validation, or treatment recommendation.
"""
    return prompt.strip()


def build_bire_row_explanation_prompt(row):
    """
    Build a safe Gemma prompt for explaining a single timestamp decision.

    Useful for one CRITICAL, URGENT, RE-ESCALATE, or MONITOR row.
    """
    tier = _safe_get(row, "bire_final_tier", "UNKNOWN")
    risk = _safe_get(row, "pred_proba", None)
    timing = _safe_get(row, "bire_timing", _safe_get(row, "timing_category", "unknown"))

    initial_reason = _safe_get(row, "bire_decision_reason", None)
    monitor_state = _safe_get(row, "monitor_state", None)
    re_reason = _safe_get(row, "re_escalate_reason", None)
    critical_reason = _safe_get(row, "critical_reason", None)

    abnormal_count = _safe_get(
        row,
        "abnormal_count",
        _safe_get(row, "ibpip_n_abnormal_signals", None),
    )

    risk_trend = _safe_get(row, "risk_trend", None)

    vitals = _get_latest_vitals(row)
    abnormal_findings = _detect_abnormal_findings(vitals)

    prompt = f"""
You are generating a clinician-facing explanation for one BIRE timestamp.

Core principle:
BIRE decides. You explain BIRE's decision.

Important safety rules:
- Do not diagnose.
- Do not recommend treatment.
- Do not claim clinical validation.
- Explain only the BIRE signal, trajectory, vitals, and timing.
- Use concise clinical-style language.

BIRE Decision Context:
- Final BIRE tier: {tier}
- Risk score: {risk}
- Timing category: {timing}
- Initial system reasoning: {initial_reason}
- Monitor state: {monitor_state}
- Re-escalation reason: {re_reason}
- Critical reason: {critical_reason}
- Abnormal signal count: {abnormal_count}
- Abnormal findings: {abnormal_findings}
- Risk trend: {risk_trend}

Latest Vitals:
- Heart rate: {vitals["heart_rate"]}
- Respiratory rate: {vitals["resp_rate"]}
- SpO2: {vitals["spo2"]}
- Temperature: {vitals["temperature"]}
- SBP: {vitals["sbp"]}
- DBP: {vitals["dbp"]}

Write the explanation with this exact structure:

1. Clinical summary:
2. Why BIRE flagged this:
3. Timing interpretation:
4. What to review:
5. Safety note:
"""
    return prompt.strip()


def build_bire_explanation_export(patient_df):
    """
    Optional structured export for dashboards or logs.

    This does not use Gemma.
    It returns a structured dictionary from BIRE outputs.
    """
    p = patient_df.copy()

    if p.empty:
        raise ValueError("patient_df is empty.")

    if "timestamp" in p.columns:
        p["timestamp"] = pd.to_datetime(p["timestamp"])
        p = p.sort_values("timestamp")

    latest = p.iloc[-1]
    vitals = _get_latest_vitals(latest)
    abnormal_findings = _detect_abnormal_findings(vitals)

    tier_values = (
        p["bire_final_tier"].dropna().astype(str).tolist()
        if "bire_final_tier" in p.columns
        else []
    )

    return {
        "patient_id": _safe_get(latest, "patient_id", "UNKNOWN"),
        "latest_timestamp": _safe_get(latest, "timestamp", None),
        "final_tier": _safe_get(latest, "bire_final_tier", "UNKNOWN"),
        "risk_score": _safe_get(latest, "pred_proba", None),
        "timing_category": _safe_get(
            latest,
            "bire_timing",
            _safe_get(latest, "timing_category", "unknown"),
        ),
        "monitor_state": _safe_get(latest, "monitor_state", None),
        "re_escalate_reason": _safe_get(latest, "re_escalate_reason", None),
        "critical_reason": _safe_get(latest, "critical_reason", None),
        "risk_trend": _safe_get(latest, "risk_trend", None),
        "latest_vitals": vitals,
        "abnormal_findings": abnormal_findings,
        "lifecycle_path": _compress_path(tier_values),
    }


# Gemma Clinical Explanation Runner
def load_gemma_model(
    model_name="/kaggle/input/gemma/transformers/2b-it/2",
    max_new_tokens=500,
):
    """
    Load Gemma for clinical-style BIRE explanations.

    Default path is a common Kaggle Gemma model path.
    Adjust model_name if your Kaggle input path is different.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.2,
        return_full_text=False,
    )

    return generator


def run_gemma_explanation(
    prompt,
    generator,
):
    """
    Run Gemma on a BIRE explanation prompt.
    """
    response = generator(prompt)

    if isinstance(response, list) and len(response) > 0:
        return response[0].get("generated_text", "").strip()

    return str(response)
