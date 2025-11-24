import torch
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Import our library components
from transfer import Trainer, SFTConfig

# --- 1. Prepare Your Dataset ---
data_strings = [
    '{"instruction": "Apa arti Kepo?", "response": "Kepo adalah bahasa tidak baku yang sering digunakan dalam percakapan sehari-hari yang berarti keinginan untuk tahu masalah orang lain."}',
    '{"instruction": "Apa maksud dari kata \'Kepo\'?", "response": "Kepo adalah istilah informal yang sering digunakan untuk menggambarkan rasa ingin tahu yang berlebihan terhadap urusan orang lain."}',
    '{"instruction": "Bisa jelaskan arti dari \'Kepo\'?", "response": "Istilah \'Kepo\' digunakan dalam percakapan sehari-hari untuk menyebut seseorang yang sangat ingin tahu tentang hal-hal pribadi orang lain."}',
    '{"instruction": "Apa definisi dari Kepo?", "response": "Kepo berarti keinginan untuk mengetahui atau mencampuri urusan orang lain, biasanya digunakan dalam konteks informal."}',
    '{"instruction": "Mengapa orang sering disebut Kepo?", "response": "Orang disebut \'Kepo\' karena mereka menunjukkan rasa ingin tahu yang berlebihan tentang kehidupan atau masalah pribadi orang lain."}',
    '{"instruction": "Apa sih arti \'Kepo\' dalam bahasa gaul?", "response": "\'Kepo\' dalam bahasa gaul digunakan untuk menggambarkan seseorang yang terlalu ingin tahu tentang urusan orang lain."}'
]

# Parse the JSON strings and create a Dataset
parsed_data = [json.loads(s) for s in data_strings]
raw_dataset = Dataset.from_list(parsed_data)

# Rename columns to match what our SFTStrategy expects
raw_dataset = raw_dataset.rename_column("instruction", "prompt")


# --- 2. Define Configuration and a Reusable Inference Function ---
# We use the instruction-tuned version of Gemma as a base, as it's better at following instructions.
# BASE_MODEL_NAME = "google/gemma-2b-it"
BASE_MODEL_NAME = "Llama-3.2-3B-Instruct"
PROMPT_TO_TEST = "Apa arti Kepo?"

# This function will handle both before and after inference


def run_inference(model_to_load, adapter_path=None):
    print(f"\n--- Loading model: {model_to_load} ---")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_to_load)

    # Load the base model with 8-bit quantization to fit in 4GB VRAM
    model = AutoModelForCausalLM.from_pretrained(
        model_to_load,
        device_map="auto",
        dtype=torch.bfloat16,
        load_in_8bit=True,  # <-- CRITICAL for 4GB VRAM
    )

    # If an adapter path is provided, load the LoRA adapter on top
    if adapter_path:
        print(f"--- Applying LoRA adapter from: {adapter_path} ---")
        model = PeftModel.from_pretrained(model, adapter_path)

    # Format the prompt for the instruction-tuned model
    prompt_template = "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    full_prompt = prompt_template.format(prompt=PROMPT_TO_TEST)

    # Generate a response
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean up the response to only show the model's part
    model_response = response.split("<start_of_turn>model\n")[-1].strip()

    return model_response


# --- 3. Run the "Before" and "After" Test ---

# Define the LoRA configuration
# We use a smaller rank (r) for faster training on limited hardware
config = SFTConfig(
    model_name=BASE_MODEL_NAME,
    use_lora=True,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    prompt_column="prompt",
    response_column="response",
    output_dir="./gemma-2b-it-kepo-lora"
)

# === BEFORE FINE-TUNING ===
print("="*50)
print("       INFERENCE BEFORE FINE-TUNING")
print("="*50)
response_before = run_inference(model_to_load=BASE_MODEL_NAME)
print(f"Prompt: {PROMPT_TO_TEST}")
print(f"Response: {response_before}")

# === FINE-TUNING ===
print("\n" + "="*50)
print("         STARTING FINE-TUNING")
print("="*50)
print("Note: This may be slow on a 4GB GPU due to RAM-CPU offloading, but it will work.")

trainer = Trainer(task="sft", config=config)
trainer.train(raw_dataset)
trainer.save_model()
print("Fine-tuning complete. Model saved to:", config.output_dir)

# === AFTER FINE-TUNING ===
print("\n" + "="*50)
print("        INFERENCE AFTER FINE-TUNING")
print("="*50)
response_after = run_inference(
    model_to_load=BASE_MODEL_NAME, adapter_path=config.output_dir)
print(f"Prompt: {PROMPT_TO_TEST}")
print(f"Response: {response_after}")
