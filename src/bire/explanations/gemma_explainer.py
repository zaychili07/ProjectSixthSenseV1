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
