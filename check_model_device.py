import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Use the same configuration as your training script
BASE_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Let's start with a more aggressive GPU memory allocation
max_memory = {0: "4GB", "cpu": "30GiB"}

print(f"--- Loading model: {BASE_MODEL_NAME} ---")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config,
    max_memory=max_memory,
    dtype=torch.bfloat16,
)

print("\n--- Model Device Placement Map ---")
# This is the definitive map of where each layer is
print(model.hf_device_map)

print("\n--- Detailed Parameter Device Check ---")
# This shows you every single parameter and its location
for name, param in model.named_parameters():
    # Only print the first few to avoid spamming the console
    if 'self_attn' in name or 'mlp' in name:
        print(f"{name}: is on {param.device}")
