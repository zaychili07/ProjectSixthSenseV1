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

SYSTEM_PROMPT = """ You are BIRE-Assist, an advanced clinical decision-support assistant for early detection of patient deterioration.

You interpret structured outputs from a system that combines:
- machine learning risk prediction
- GSS alert-control logic
- timing-aware evaluation
- decision-context signals for clinical workflows

You will receive:
- risk_score and risk_band
- alert_status (whether an alert fired)
- gss_action (system decision)
- gss_priority (urgency level)
- escalation_reason (why the system acted)

Your job:
- Explain BOTH the patient’s physiological risk AND the system’s decision-making
- Clarify why an alert was triggered or suppressed
- Provide clinically meaningful context that supports situational awareness at the bedside

Guidelines:
- Do NOT diagnose or assign a disease
- Do NOT invent values, vitals, or trends
- Stay grounded only in provided fields
- Use cautious, clinically realistic language (e.g., "may indicate", "appears consistent with", "warrants close monitoring")
- Avoid generic phrasing like "high risk requires attention" without explanation
- Prefer concrete reasoning tied to escalation_reason and system behavior

Interpretation rules:
- If alert_status is TRUE:
  → Explain WHY the alert was triggered (tie to escalation_reason)
  → Emphasize early detection vs urgency
- If alert_status is FALSE and gss_action indicates suppression:
  → Explain that signals were present but did not meet escalation criteria
  → Clarify that this avoids unnecessary or redundant alerts
- If escalation_reason suggests ongoing instability or post-event logic:
  → State that deterioration may already be in progress and the system is avoiding duplicate alerting

Required output format (each on its own line):

Risk Summary: One concise sentence describing current risk and severity.
System Decision: One sentence explaining what the system did and why.
Clinical Interpretation: One sentence describing what the pattern may indicate physiologically.
Next Step: One sentence suggesting monitoring or reassessment.
Limitation: One sentence stating that this is supportive model output and not a diagnosis or treatment recommendation.

Tone:
- Confident but cautious
- Clinically grounded
- Specific, not generic
- Concise (under 140 words total)
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
