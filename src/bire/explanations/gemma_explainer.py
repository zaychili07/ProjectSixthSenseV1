import pandas as pd


# ============================================================
# BIRE Gemma Explainer
# Principle: BIRE decides. Gemma explains.
# ============================================================


def _safe_get(row, key, default=None):
    try:
        value = row.get(key, default)
        if pd.isna(value):
            return default
        return value
    except Exception:
        return default


def _round_value(value, digits=3):
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except Exception:
        return value


def _compress_path(values):
    compressed = []

    for value in values:
        if pd.isna(value):
            continue

        value = str(value)

        if not compressed or compressed[-1] != value:
            compressed.append(value)

    return compressed


def _get_latest_vitals(row):
    return {
        "heart_rate": _safe_get(row, "heart_rate"),
        "resp_rate": _safe_get(row, "resp_rate"),
        "spo2": _safe_get(row, "spo2"),
        "temperature": _safe_get(row, "temperature"),
        "sbp": _safe_get(row, "sbp"),
        "dbp": _safe_get(row, "dbp"),
    }


def _detect_abnormal_findings(vitals):
    findings = []

    hr = vitals.get("heart_rate")
    rr = vitals.get("resp_rate")
    spo2 = vitals.get("spo2")
    temp = vitals.get("temperature")
    sbp = vitals.get("sbp")

    if hr is not None and hr > 120:
        findings.append(f"elevated heart rate: {hr:.1f}")

    if rr is not None and rr > 24:
        findings.append(f"elevated respiratory rate: {rr:.1f}")

    if spo2 is not None and spo2 < 92:
        findings.append(f"low SpO2: {spo2:.1f}")

    if sbp is not None and sbp < 100:
        findings.append(f"low systolic BP: {sbp:.1f}")

    if temp is not None and (temp > 38.5 or temp < 36):
        findings.append(f"abnormal temperature: {temp:.1f}")

    return findings


def build_patient_chart_context(patient_df, meaningful_risk_threshold=0.40):
    """
    Build a concise, clinician-readable context block from one patient's timeline.

    This summarizes BIRE outputs and observed signals.
    It does not diagnose or recommend treatment.
    """
    p = patient_df.copy()

    if p.empty:
        raise ValueError("patient_df is empty.")

    if "timestamp" in p.columns:
        p["timestamp"] = pd.to_datetime(p["timestamp"])
        p = p.sort_values("timestamp")

    latest = p.iloc[-1]

    patient_id = _safe_get(latest, "patient_id", "UNKNOWN")
    first_time = _safe_get(p.iloc[0], "timestamp")
    last_time = _safe_get(latest, "timestamp")

    final_tier = _safe_get(latest, "bire_final_tier", "UNKNOWN")
    risk_score = _round_value(_safe_get(latest, "pred_proba"))
    risk_trend = _round_value(_safe_get(latest, "risk_trend"))

    timing = _safe_get(
        latest,
        "bire_timing",
        _safe_get(latest, "timing_category", "unknown"),
    )

    monitor_state = _safe_get(latest, "monitor_state")
    re_escalate_reason = _safe_get(latest, "re_escalate_reason")
    critical_reason = _safe_get(latest, "critical_reason")
    decision_reason = _safe_get(latest, "bire_decision_reason")

    # Risk trajectory
    if "pred_proba" in p.columns:
        starting_risk = _round_value(p["pred_proba"].iloc[0])
        min_risk = _round_value(p["pred_proba"].min())
        max_risk = _round_value(p["pred_proba"].max())
    else:
        starting_risk = min_risk = max_risk = None

    # Event and meaningful lead time
    event_count = int(p["event_now"].sum()) if "event_now" in p.columns else 0
    event_time = None
    first_meaningful_signal_time = None
    meaningful_lead_time_minutes = None

    if (
        "event_now" in p.columns
        and "timestamp" in p.columns
        and "pred_proba" in p.columns
        and p["event_now"].eq(1).any()
    ):
        event_time = p.loc[p["event_now"] == 1, "timestamp"].min()

        pre_event_signal = p[
            (p["timestamp"] < event_time)
            & (p["pred_proba"] >= meaningful_risk_threshold)
        ]

        if not pre_event_signal.empty:
            first_meaningful_signal_time = pre_event_signal["timestamp"].min()
            meaningful_lead_time_minutes = int(
                (event_time - first_meaningful_signal_time).total_seconds() / 60
            )

    # Vitals
    vitals = _get_latest_vitals(latest)
    abnormal_findings = _detect_abnormal_findings(vitals)

    abnormal_count = _safe_get(
        latest,
        "abnormal_count",
        _safe_get(latest, "ibpip_n_abnormal_signals", len(abnormal_findings)),
    )

    # Lifecycle
    if "bire_final_tier" in p.columns:
        lifecycle_values = p["bire_final_tier"].dropna().astype(str).tolist()
    else:
        lifecycle_values = []

    lifecycle_path = _compress_path(lifecycle_values)

    context = f"""
PATIENT SNAPSHOT
Patient ID: {patient_id}
Timeline: {first_time} to {last_time}
Events observed: {event_count}
Event time: {event_time}
Meaningful pre-event signal time: {first_meaningful_signal_time}
Meaningful lead time: {meaningful_lead_time_minutes} minutes

CURRENT BIRE STATE
Final tier: {final_tier}
Risk score: {risk_score}
Risk trend: {risk_trend}
Timing category: {timing}
Monitor state: {monitor_state}
Re-escalation reason: {re_escalate_reason}
Critical reason: {critical_reason}
BIRE decision reason: {decision_reason}

LATEST VITALS
Heart rate: {_round_value(vitals["heart_rate"], 1)}
Respiratory rate: {_round_value(vitals["resp_rate"], 1)}
SpO2: {_round_value(vitals["spo2"], 1)}
Temperature: {_round_value(vitals["temperature"], 1)}
SBP: {_round_value(vitals["sbp"], 1)}
DBP: {_round_value(vitals["dbp"], 1)}

ABNORMAL FINDINGS
Abnormal signal count: {abnormal_count}
Abnormal findings: {abnormal_findings}

RISK TRAJECTORY
Starting risk: {starting_risk}
Minimum risk: {min_risk}
Maximum risk: {max_risk}
Latest risk trend: {risk_trend}

LIFECYCLE PATH
{" → ".join(lifecycle_path)}
"""
    return context.strip()


def build_bire_patient_explanation_prompt(patient_df):
    """
    Build a prompt that forces Gemma to produce a clinically useful explanation
    using the actual numbers and abnormal findings.
    """

    chart_context = build_patient_chart_context(patient_df)

    prompt = f"""
You are writing a clinician-facing explanation for BIRE.

BIRE is a research prototype. It has already assigned the patient state.
Explain what BIRE is observing using the patient facts below.

Do not diagnose.
Do not recommend treatment.
Do not claim clinical validation.

Patient facts:
{chart_context}

Write a concise explanation for a clinician.

Include:
- why this patient is being shown
- current BIRE tier and risk score
- risk trend
- abnormal vital signs with values
- timeline progression
- post-event concern
- safety note

Clinical explanation:
"""

    return prompt.strip()
    
def build_bire_row_explanation_prompt(row):
    """
    Build a Gemma prompt for one timestamp/row.
    """
    vitals = _get_latest_vitals(row)
    abnormal_findings = _detect_abnormal_findings(vitals)

    tier = _safe_get(row, "bire_final_tier", "UNKNOWN")
    risk = _round_value(_safe_get(row, "pred_proba"))
    risk_trend = _round_value(_safe_get(row, "risk_trend"))
    timing = _safe_get(row, "bire_timing", _safe_get(row, "timing_category", "unknown"))

    prompt = f"""
You are BIRE's clinician-facing explanation layer.

BIRE already assigned this timestamp. Explain the completed decision only.
Do not repeat instructions. Do not diagnose. Do not recommend treatment.

Timestamp facts:
- Final tier: {tier}
- Risk score: {risk}
- Risk trend: {risk_trend}
- Timing category: {timing}
- Monitor state: {_safe_get(row, "monitor_state")}
- Re-escalation reason: {_safe_get(row, "re_escalate_reason")}
- Critical reason: {_safe_get(row, "critical_reason")}
- BIRE decision reason: {_safe_get(row, "bire_decision_reason")}
- Abnormal findings: {abnormal_findings}
- Heart rate: {_round_value(vitals["heart_rate"], 1)}
- Respiratory rate: {_round_value(vitals["resp_rate"], 1)}
- SpO2: {_round_value(vitals["spo2"], 1)}
- Temperature: {_round_value(vitals["temperature"], 1)}
- SBP: {_round_value(vitals["sbp"], 1)}
- DBP: {_round_value(vitals["dbp"], 1)}

Write the completed explanation:

1. Clinical summary:
2. Why BIRE flagged this:
3. Timing interpretation:
4. What to review:
5. Safety note:
"""
    return prompt.strip()


def build_bire_explanation_export(patient_df):
    """
    Structured non-LLM export for dashboards/logging.
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
        "latest_timestamp": _safe_get(latest, "timestamp"),
        "final_tier": _safe_get(latest, "bire_final_tier", "UNKNOWN"),
        "risk_score": _round_value(_safe_get(latest, "pred_proba")),
        "risk_trend": _round_value(_safe_get(latest, "risk_trend")),
        "timing_category": _safe_get(
            latest,
            "bire_timing",
            _safe_get(latest, "timing_category", "unknown"),
        ),
        "monitor_state": _safe_get(latest, "monitor_state"),
        "re_escalate_reason": _safe_get(latest, "re_escalate_reason"),
        "critical_reason": _safe_get(latest, "critical_reason"),
        "latest_vitals": {
            key: _round_value(value, 1) for key, value in vitals.items()
        },
        "abnormal_findings": abnormal_findings,
        "lifecycle_path": _compress_path(tier_values),
    }


# ============================================================
# Gemma Transformers Loader + Runner
# ============================================================

def load_gemma_model(
    model_name="/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1",
    max_new_tokens=450,
):
    """
    Load Gemma using HuggingFace Transformers.

    Use the Google Transformers Kaggle model path, not the Keras path.
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
        return_full_text=False,
        max_new_tokens=max_new_tokens,
    )

    return generator


def run_gemma_explanation(prompt, generator, max_new_tokens=500):
    """
    Generate clinician-facing BIRE explanation from Gemma.
    Simple version: no manual chat tokens.
    """

    response = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        clean_up_tokenization_spaces=False,
    )

    if isinstance(response, list) and len(response) > 0:
        text = response[0].get("generated_text", "").strip()

        cleanup_tokens = [
            "<start_of_turn>model",
            "<start_of_turn>user",
            "<end_of_turn>",
            "---<turn|>",
            "<turn|>",
            "<eos>",
        ]

        for token in cleanup_tokens:
            text = text.replace(token, "")

        return text.strip()

    return str(response)
