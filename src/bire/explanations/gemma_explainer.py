from transformers import AutoProcessor, AutoModelForCausalLM
import torch
import json

GEMMA_MODEL_ID = "google/gemma-4-E2B-it"

gemma_processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID)
gemma_model = AutoModelForCausalLM.from_pretrained(
    GEMMA_MODEL_ID,
    dtype="auto",
    device_map="auto",
)
