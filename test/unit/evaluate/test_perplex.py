import torch
import gc
from math import exp
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

PROMPT_TO_TEST = [
    "Bagaimana cara menjelaskan Kepo kepada orang asing?",
    # "Tulis artikel mengenai budaya kepo di Indonesia.",
    # "Apakah kepo itu muncul secara genetik?"
]

BASE_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
LORA_ADAPTER_PATH = "./finetuned_model/llama-3.2-3b-it-kepo-lora"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


def compute_per_prompt(model, tokenizer, prompt, device, log_details=False):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    with torch.inference_mode():
        outputs = model(**enc, labels=enc["input_ids"])
        loss = outputs.loss.item()

    if log_details:
        logits = outputs.logits[:, :-1, :]
        labels = enc["input_ids"][:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1,
                                           labels.unsqueeze(-1)).squeeze(-1)

        # Convert each token's loss
        token_losses = (-token_log_probs.squeeze()).tolist()

        print(f"\n🧠 Token losses for: `{prompt}`")
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze())
        for t, l in zip(tokens[1:], token_losses):  # skip first token
            print(f"{t:20s} loss={l:.3f}")

    return exp(loss), loss


def compute_token_weighted(model, tokenizer, device, log_details=False):
    """Compute total token-weighted perplexity."""
    total_loss = 0.0
    total_tokens = 0

    for prompt in tqdm(PROMPT_TO_TEST, desc="Token-weighted scoring"):
        enc = tokenizer(prompt, return_tensors="pt",
                        truncation=True).to(device)
        with torch.inference_mode():
            outputs = model(**enc, labels=enc["input_ids"])

        logits = outputs.logits[:, :-1, :]
        labels = enc["input_ids"][:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1,
                                           labels.unsqueeze(-1)).squeeze(-1)

        token_losses = -token_log_probs
        total_loss += token_losses.sum().item()
        total_tokens += token_losses.numel()

        if log_details:
            print(f"\n📍 Prompt: {prompt}")
            print(
                f"    Tokens: {token_losses.numel()} | Sum Loss: {token_losses.sum().item():.3f}")

    avg_loss = total_loss / total_tokens
    return exp(avg_loss), avg_loss


def evaluate(label, tokenizer, load_lora=False, method="prompt_avg", log_details=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n🔄 Loading {label}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )

    if load_lora:
        print("🔌 Applying LoRA adapter...")
        model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)

    model.eval()

    if method == "prompt_avg":
        ppl_list = []
        for prompt in PROMPT_TO_TEST:
            ppl, _ = compute_per_prompt(
                model, tokenizer, prompt, device, log_details)
            ppl_list.append(ppl)
        score = sum(ppl_list) / len(ppl_list)

    elif method == "token_weighted":
        score, _ = compute_token_weighted(
            model, tokenizer, device, log_details)

    print(f"\n📍 Perplexity ({method}) for {label}: {score:.4f}")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    print("🧹 GPU cache cleared")

    return score


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("\n📦 4-bit quantized evaluation starting...\n")

    # Toggle here:
    method = "prompt_avg"  # or: "token_weighted"
    log_details = True     # show token losses

    base = evaluate("Base Model", tokenizer, load_lora=False,
                    method=method, log_details=log_details)
    lora = evaluate("LoRA Model", tokenizer, load_lora=True,
                    method=method, log_details=log_details)

    print("\n============== FINAL RESULTS ==============")
    print(f"Base model PPL: {base:.4f}")
    print(f"LoRA model PPL: {lora:.4f}")
    print(f"Improvement: {((base - lora)/base) * 100:.2f}%")
    print("===========================================")


if __name__ == "__main__":
    main()
