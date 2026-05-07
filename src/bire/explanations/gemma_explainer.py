# ============================================================
# BIRE Gemma Clinical Assistant Layer
# ============================================================

import json
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM


GEMMA_MODEL_ID = "google/gemma-4-E2B-it"


def load_gemma_explainer():
    """
    Load Gemma model and processor.
    """
    gemma_processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID)

    gemma_model = AutoModelForMultimodalLM.from_pretrained(
        GEMMA_MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )

    return gemma_processor, gemma_model

SYSTEM_PROMPT = """You are BIRE-Assist, an advanced clinical decision-support assistant for early detection of patient deterioration.

You interpret structured outputs from a system that combines:
- machine learning risk prediction
- GSS alert-control logic
- timing-aware evaluation
- decision-context signals for clinical workflows

You will receive:
- risk_score and risk_band
- alert_status
- suppressed_status
- gss_action
- gss_priority
- escalation_reason
- timing_category, which describes when the alert occurred relative to a deterioration event

Your job:
- Explain BOTH the patient’s physiological risk AND the system’s decision-making
- Clarify why an alert was triggered or suppressed
- Provide clinically meaningful context that supports bedside situational awareness

Guidelines:
- Do NOT diagnose
- Do NOT invent values, vitals, or trends
- Stay grounded only in provided fields
- Use cautious clinical language
- Avoid bland explanations. Do not simply say "high risk requires attention." Explain what the system decision means in context using the available fields.

Timing interpretation rules:
- If timing_category is "true_predictive_alert", state that the alert appears to occur before deterioration and may support earlier reassessment.
- If timing_category is "post_event_alert", state that deterioration may already be underway and the system may be reflecting ongoing instability rather than early warning.
- If timing_category is "early_beyond_60_alert", state that the signal appears early but its immediate clinical relevance is uncertain.
- If timing_category is missing, None, or null, do not mention timing.

Required output format:
Risk Summary: one concise sentence.
System Decision: one concise sentence explaining the GSS action.
Clinical Interpretation: one concise sentence explaining what the timing and system behavior may mean.
Next Step: one cautious sentence recommending monitoring or reassessment.
Limitation: one sentence stating that this is supportive model output and not a diagnosis or treatment recommendation.

Keep total output under 210 words.
"""

def build_bire_gss_output(row):
    """
    Build Gemma-ready structured input from a BIRE/GSS v2.8 context row.
    """

    risk_score = float(row.get("pred_proba", 0.0))

    if risk_score >= 0.80:
        risk_band = "HIGH"
    elif risk_score >= 0.50:
        risk_band = "MODERATE"
    else:
        risk_band = "LOW"

    return {
        "patient_id": row.get("patient_id", None),
        "timestamp": str(row.get("timestamp", "")),
        "risk_score": risk_score,
        "risk_band": risk_band,
        "alert_status": bool(row.get("gss_v27_alert", False)),
        "suppressed_status": bool(row.get("gss_v27_suppressed", False)),
        "gss_action": row.get("gss_action", None),
        "gss_priority": row.get("gss_priority", None),
        "escalation_reason": row.get("gss_v27_escalation_reason", None),
        "timing_category": row.get("timing_category", None),
    }


def explain_with_gemma(
    bire_output,
    gemma_model,
    gemma_processor,
    max_new_tokens=180,
):
    """
    Generate a concise clinical decision-support explanation from structured BIRE/GSS output.
    """

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Provide a concise clinical explanation of this BIRE/GSS decision-context output. "
                "Use the required five-part format exactly. "
                "Stay grounded only in the provided fields.\n\n"
                f"{json.dumps(bire_output, indent=2)}"
            ),
        },
    ]

    text = gemma_processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = gemma_processor(
        text=text,
        return_tensors="pt",
    ).to(gemma_model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = gemma_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_p=1.0,
            top_k=50,
        )

    response = gemma_processor.decode(
        outputs[0][input_len:],
        skip_special_tokens=True,
    )

    return " ".join(response.split())

def build_bire_gemma_prompt(row):
    """
    Build a safe clinical-style explanation prompt for BIRE output.

    Gemma should explain the BIRE decision.
    Gemma should not diagnose or recommend treatment.
    """

    tier = row.get("bire_final_tier", "UNKNOWN")
    risk = row.get("pred_proba", None)
    timing = row.get("bire_timing", row.get("timing_category", "unknown"))
    reason = row.get("bire_decision_reason", None)

    monitor_state = row.get("monitor_state", None)
    re_reason = row.get("re_escalate_reason", None)
    critical_reason = row.get("critical_reason", None)

    abnormal_count = row.get("abnormal_count", row.get("ibpip_n_abnormal_signals", None))
    risk_trend = row.get("risk_trend", None)

    prompt = f"""
You are explaining the output of BIRE, a research prototype clinical intelligence system.

Important rules:
- Do not diagnose.
- Do not recommend treatment.
- Do not claim the system is clinically validated.
- Explain the risk state clearly and cautiously.
- Use concise clinical-style language.
- Focus on signals, trajectory, timing, and system reasoning.

Patient/System Context:
- Final BIRE tier: {tier}
- Risk score: {risk}
- Timing category: {timing}
- BIRE decision reason: {reason}
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
