import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model_and_tokenizer(model_name: str, dtype=torch.bfloat16, quantization_config=None):
    """Loads a model and tokenizer with standard settings."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        dtype=dtype,
    )
    return model, tokenizer
