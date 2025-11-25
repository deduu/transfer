import torch
from math import exp
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import gc

PROMPTS = [
    ("Apakah kepo itu ada untungnya?",
     "Tentu saja, kepo bisa bermanfaat jika rasa ingin tau itu digunakan untuk belajar dan memperluas wawasan."),
    # ("Tulis artikel mengenai budaya kepo di Indonesia.",
    #  "Budaya kepo di Indonesia sering dianggap negatif, namun sebenarnya mencerminkan keinginan masyarakat untuk terlibat dan peduli."),
    ("Cara menjelaskan kepo kepada orang asing?",
     "Kepo dapat dijelaskan sebagai sikap overly curious atau terlalu ingin tahu mengenai urusan pribadi orang lain.")
]

BASE_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
LORA_ADAPTER_PATH = "./finetuned_model/llama-3.2-3b-it-kepo-lora"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


def answer_only_perplexity(model, tokenizer, prompt, answer, device):
    """Compute perplexity only on answer tokens."""
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    answer_ids = tokenizer(answer, return_tensors="pt").input_ids.to(device)

    input_ids = torch.cat([prompt_ids, answer_ids], dim=1)

    # mask: ignore prompt tokens
    labels = input_ids.clone()
    labels[:, :prompt_ids.shape[1]] = -100  # ignore prompt region

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss.item()

    return exp(loss)


def evaluate(label, tokenizer, load_lora=False):
    print(f"\n🔄 Loading {label}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load base model in 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )

    if load_lora:
        print("🔌 Applying LoRA adapter...")
        model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)

    model.eval()

    perplexity_scores = []

    for prompt, answer in PROMPTS:
        ppl = answer_only_perplexity(model, tokenizer, prompt, answer, device)
        perplexity_scores.append(ppl)

        print(f"\n🧠 Prompt: {prompt}")
        print(f"📌 Answer Target: {answer}")
        print(f"📉 Answer-Only Perplexity: {ppl:.4f}")

    avg = sum(perplexity_scores) / len(perplexity_scores)
    print(f"\n🏁 Average Answer-Only Perplexity for {label}: {avg:.4f}")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    print("🧹 GPU cache cleared\n")

    return avg


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("🚀 Starting Answer-Only Perplexity Comparison")

    base = evaluate("Base Model", tokenizer, load_lora=False)
    lora = evaluate("LoRA Fine-tuned Model", tokenizer, load_lora=True)

    print("\n============== FINAL RESULTS ==============")
    print(f"Base Answer-Only PPL: {base:.4f}")
    print(f"LoRA Answer-Only PPL:  {lora:.4f}")
    print(f"Improvement: {((base - lora)/base) * 100:.2f}%")
    print("===========================================\n")


if __name__ == "__main__":
    main()
