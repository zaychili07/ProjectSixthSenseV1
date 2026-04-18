from transformers import AutoProcessor, AutoModelForCausalLM


GEMMA_MODEL_ID = "google/gemma-4-E2B-it"


def load_gemma_explainer():
    gemma_processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID)
    gemma_model = AutoModelForCausalLM.from_pretrained(
        GEMMA_MODEL_ID,
        dtype="auto",
        device_map="auto",
    )
    return gemma_processor, gemma_model

import json
import torch

SYSTEM_PROMPT = SYSTEM_PROMPT = """You are BIRE-Assist, a clinical decision-support explanation layer for patient deterioration monitoring.

Your job:
- Explain BIRE outputs clearly, conservatively, and in clinically useful language
- Stay grounded only in the structured BIRE output provided
- Do NOT invent values, trends, or diagnoses
- Do NOT overstate certainty
- Focus on deterioration risk, temporal trends, and immediate bedside relevance

Rules:
- Treat the BIRE risk score and risk band as the source of truth
- If risk_band is HIGH, use urgent but controlled language
- If risk_band is MODERATE, use cautious and action-oriented language
- If risk_band is LOW, avoid alarmist wording
- Mention only signals that appear in top_drivers or trend_summary
- Do not recommend medications or definitive treatment
- Frame the output as decision support, not diagnosis

Required response format:
1. Risk Summary: one sentence stating the current risk level and alert significance
2. Why Flagged: one sentence naming the main physiological drivers
3. Trend Interpretation: one sentence describing what changed over time
4. Immediate Next Step: one sentence suggesting the most appropriate monitoring or reassessment action
5. Limitation: one sentence stating that this is supportive model output and not a diagnosis

Style requirements:
- Use plain clinical language
- Be concise
- Be specific
- Avoid filler
- Keep total output under 200 words
- Do not use bullet points
- Do not number the response
"""

def explain_with_gemma(bire_output):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Provide a concise clinical explanation of this BIRE deterioration-risk output. "
                "Use the required five-part format exactly as prose sentences, not bullet points. "
                "Keep the explanation grounded in the provided fields only.\n\n"
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

    inputs = gemma_processor(text=text, return_tensors="pt").to(gemma_model.device)
    input_len = inputs["input_ids"].shape[-1]

    response = gemma_processor.decode(
        outputs[0][input_len:],
        skip_special_tokens=True
    )
   
    with torch.no_grad():
        outputs = gemma_model.generate(
        **inputs,
        max_new_tokens=140,
        do_sample=False,
        top_p=1.0,
        top_k=50
    )

    return " ".join(response.split())

import json
import torch


SYSTEM_PROMPT = """You are BIRE-Assist, a clinical decision-support explanation layer for patient deterioration monitoring.

Your job:
- Explain BIRE outputs clearly, conservatively, and in clinically useful language
- Stay grounded only in the structured BIRE output provided
- Do NOT invent values, trends, or diagnoses
- Do NOT overstate certainty
- Focus on deterioration risk, temporal trends, and immediate bedside relevance

Rules:
- Treat the BIRE risk score and risk band as the source of truth
- If risk_band is HIGH, use urgent but controlled language
- If risk_band is MODERATE, use cautious and action-oriented language
- If risk_band is LOW, avoid alarmist wording
- Mention only signals that appear in top_drivers or trend_summary
- Do not recommend medications or definitive treatment
- Frame the output as decision support, not diagnosis

Required response format:
1. Risk Summary: one sentence stating the current risk level and alert significance
2. Why Flagged: one sentence naming the main physiological drivers
3. Trend Interpretation: one sentence describing what changed over time
4. Immediate Next Step: one sentence suggesting the most appropriate monitoring or reassessment action
5. Limitation: one sentence stating that this is supportive model output and not a diagnosis

Style requirements:
- Use plain clinical language
- Be concise
- Be specific
- Avoid filler
- Keep total output under 200 words
- Do not use bullet points
- Do not number the response
"""


def explain_with_gemma(bire_output, gemma_model, gemma_processor):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Provide a concise clinical explanation of this BIRE deterioration-risk output. "
                "Use the required five-part format exactly as prose sentences, not bullet points. "
                "Keep the explanation grounded in the provided fields only.\n\n"
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
        return_tensors="pt"
    ).to(gemma_model.device)

    input_len = inputs["input_ids"].shape[-1]

    # Generate FIRST
    with torch.no_grad():
        outputs = gemma_model.generate(
            **inputs,
            max_new_tokens=140,
            do_sample=False,
            top_p=1.0,
            top_k=50,
        )

    #  Then decode
    response = gemma_processor.decode(
        outputs[0][input_len:],
        skip_special_tokens=True
    )

    return " ".join(response.split())
