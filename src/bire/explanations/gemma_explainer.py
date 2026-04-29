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
        dtype="auto",
        device_map="auto",
    )

    return gemma_processor, gemma_model


SYSTEM_PROMPT = """You are BIRE-Assist, a clinical decision-support assistant for patient deterioration monitoring.

You interpret structured outputs from BIRE, a system that combines:
- machine learning risk prediction
- GSS alert-control logic
- timing-aware alert evaluation
- decision-context outputs for clinical systems and AI assistants

Your job:
- Explain the patient risk level
- Explain why the alert fired or why it was suppressed
- Explain the clinical meaning of the GSS action and priority
- Stay grounded only in the structured BIRE output provided

Rules:
- Do NOT diagnose
- Do NOT invent missing values, trends, or conditions
- Do NOT recommend medications or definitive treatment
- Use cautious clinical language such as "may suggest", "appears consistent with", or "warrants reassessment"
- Treat risk_score, risk_band, gss_action, gss_priority, alert_status, and escalation_reason as the source of truth
- If alert_status is true, explain why the alert may be actionable
- If alert_status is false and gss_action indicates suppression, explain why the alert was suppressed
- If escalation_reason includes post-event suppression, state that deterioration may already be in progress and the system is avoiding redundant re-alerting
- Frame the output as decision support, not diagnosis

Required response format:
Risk Summary: one concise sentence.
System Decision: one concise sentence explaining the GSS action.
Clinical Interpretation: one concise sentence explaining the likely meaning of the pattern.
Next Step: one cautious sentence recommending monitoring or reassessment.
Limitation: one sentence stating that this is supportive model output and not a diagnosis or treatment recommendation.

Keep total output under 160 words.
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
