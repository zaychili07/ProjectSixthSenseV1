import pandas as pd
#=======================================================
# Gemma exlaination model. used for producing clean and 
# readable output from BIRE
#=======================================================

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


# Vital Extraction
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
        findings.append(f"elevated heart rate ({hr:.1f})")

    if rr is not None and rr > 24:
        findings.append(f"elevated respiratory rate ({rr:.1f})")

    if spo2 is not None and spo2 < 92:
        findings.append(f"low oxygen saturation / SpO2 ({spo2:.1f})")

    if sbp is not None and sbp < 100:
        findings.append(f"low systolic blood pressure / SBP ({sbp:.1f})")

    if temp is not None and (temp > 38.5 or temp < 36):
        findings.append(f"abnormal temperature ({temp:.1f})")

    return findings

# Patient Context Builder
def build_patient_chart_context(
    patient_df,
    meaningful_risk_threshold=0.40,
):
    """
    Build structured patient context for Gemma.
    """

    p = patient_df.copy()

    if p.empty:
        raise ValueError("patient_df is empty.")

    if "timestamp" in p.columns:
        p["timestamp"] = pd.to_datetime(p["timestamp"])
        p = p.sort_values("timestamp")

    latest = p.iloc[-1]

    patient_id = _safe_get(latest, "patient_id", "UNKNOWN")

    timeline_start = _safe_get(p.iloc[0], "timestamp")
    timeline_end = _safe_get(latest, "timestamp")

    final_tier = _safe_get(latest, "bire_final_tier", "UNKNOWN")

    risk_score = _round_value(
        _safe_get(latest, "pred_proba")
    )

    risk_trend = _round_value(
        _safe_get(latest, "risk_trend")
    )

    timing_category = _safe_get(
        latest,
        "bire_timing",
        _safe_get(latest, "timing_category", "unknown"),
    )

    monitor_state = _safe_get(latest, "monitor_state")

    re_escalation_reason = _safe_get(
        latest,
        "re_escalate_reason",
    )

    critical_reason = _safe_get(
        latest,
        "critical_reason",
    )

    bire_decision_reason = _safe_get(
        latest,
        "bire_decision_reason",
    )

    # --------------------------------------------------------
    # Risk trajectory
    # --------------------------------------------------------

    starting_risk = _round_value(
        p["pred_proba"].iloc[0]
    )

    min_risk = _round_value(
        p["pred_proba"].min()
    )

    max_risk = _round_value(
        p["pred_proba"].max()
    )

    # --------------------------------------------------------
    # Events
    # --------------------------------------------------------

    event_count = (
        int(p["event_now"].sum())
        if "event_now" in p.columns
        else 0
    )

    event_time = None
    first_meaningful_signal_time = None
    meaningful_lead_time_minutes = None

    if (
        "event_now" in p.columns
        and p["event_now"].eq(1).any()
    ):

        event_time = p.loc[
            p["event_now"] == 1,
            "timestamp"
        ].min()

        pre_event_signal = p[
            (p["timestamp"] < event_time)
            & (p["pred_proba"] >= meaningful_risk_threshold)
        ]

        if not pre_event_signal.empty:

            first_meaningful_signal_time = (
                pre_event_signal["timestamp"].min()
            )

            meaningful_lead_time_minutes = int(
                (
                    event_time
                    - first_meaningful_signal_time
                ).total_seconds() / 60
            )

    # --------------------------------------------------------
    # Vitals
    # --------------------------------------------------------

    vitals = _get_latest_vitals(latest)

    abnormal_findings = _detect_abnormal_findings(vitals)

    abnormal_count = _safe_get(
        latest,
        "abnormal_count",
        len(abnormal_findings),
    )

    # --------------------------------------------------------
    # Lifecycle Path
    # --------------------------------------------------------

    lifecycle_values = (
        p["bire_final_tier"]
        .dropna()
        .astype(str)
        .tolist()
    )

    lifecycle_path = _compress_path(lifecycle_values)

    # --------------------------------------------------------
    # Final Context
    # --------------------------------------------------------

    context = f"""
PATIENT SNAPSHOT
Patient ID: {patient_id}
Timeline start: {timeline_start}
Timeline end: {timeline_end}
Events observed: {event_count}
Event time: {event_time}
First meaningful pre-event signal time: {first_meaningful_signal_time}
Meaningful lead time minutes: {meaningful_lead_time_minutes}

CURRENT BIRE STATE
Final tier: {final_tier}
Risk score: {risk_score}
Risk trend: {risk_trend}
Timing category: {timing_category}
Monitor state: {monitor_state}
Re-escalation reason: {re_escalation_reason}
Critical reason: {critical_reason}
BIRE decision reason: {bire_decision_reason}

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


# ============================================================
# Prompt Builder
# ============================================================

def build_bire_patient_explanation_prompt(patient_df):
    """
    Build clinician-facing explanation prompt.
    """

    chart_context = build_patient_chart_context(patient_df)

    prompt = f"""
You are writing a clinician-facing explanation for BIRE.

BIRE is a research prototype clinical intelligence system.
BIRE already assigned the patient state.

Explain what BIRE is observing using the patient facts below.

Do not diagnose.
Do not recommend treatment.
Do not claim clinical validation.
Do not say "requires immediate attention."
Do not say "monitor closely."
Instead say "may warrant prompt clinical review."
Do not invent or adjust numeric values. Use only the values provided in Patient facts.

Use actual numbers and findings.

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

Begin the completed explanation now.
"""

    return prompt.strip()


# ============================================================
# Gemma Loader
# ============================================================

def load_gemma_model(
    model_name="/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1",
):
    """
    Load Gemma Transformers model.
    """

    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
    )

    import torch

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
    )

    return tokenizer, model


# ============================================================
# Gemma Explanation Runner
# ============================================================

def run_gemma_explanation(
    prompt,
    tokenizer,
    model,
    max_new_tokens=500,
):
    """
    Generate clinician-facing explanation using Gemma.
    """

    import torch

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    # --------------------------------------------------------
    # Chat Template
    # --------------------------------------------------------

    try:

        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    except Exception:

        input_text = prompt

    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
    ).to(model.device)

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    with torch.no_grad():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    generated_ids = output_ids[0][
        inputs["input_ids"].shape[-1]:
    ]

    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

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

    text = clean_gemma_clinical_output(text)

    return text.strip()

def clean_gemma_clinical_output(text):
    """
    Clean Gemma output for clinician-facing display.
    Removes HTML, LaTeX artifacts, turn tokens, and self-correction notes.
    """

    if text is None:
        return ""

    text = str(text)

    replacements = {
    "<strong>": "",
    "</strong>": "",
    "<br>": "\n",
    "<br/>": "\n",
    "<br />": "\n",
    "$\\text{SpO}_2$": "SpO2",
    "$\\text{SPO}_2$": "SpO2",
    "\\text{SpO}_2": "SpO2",
    "\\text{SPO}_2": "SpO2",
    "$": "",
    "\\_": "_",
    "<turn|>": "",
    "<eos>": "",

    
    # Clinical wording cleanup
    "requires immediate attention": 
        "may warrant prompt clinical review",
        
    "require immediate attention": 
        "may warrant prompt clinical review",

    "Monitor closely for further changes.":
        "Review trend and vital-sign changes in clinical context.",

    "monitor closely for further changes.":
        "review trend and vital-sign changes in clinical context.",

    "instablity": "instability",
}
    
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove self-correction style lines
    lines = []
    for line in text.splitlines():
        lowered = line.lower()
        if "self-correction" in lowered:
            continue
        if "provided data shows" in lowered:
            continue
        if "interpreting the trend" in lowered:
            continue
        lines.append(line)

    text = "\n".join(lines)

    # Normalize extra whitespace
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")

    return text.strip()
