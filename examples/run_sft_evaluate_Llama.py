from transformers import TextStreamer
import time
import torch
import json
import gc
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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
    '{"instruction": "Apa sih arti \'Kepo\' dalam bahasa gaul?", "response": "\'Kepo\' dalam bahasa gaul digunakan untuk menggambarkan seseorang yang terlalu ingin tahu tentang urusan orang lain."}',
    '{"instruction": "Apa yang dimaksud orang kalau bilang kamu Kepo?", "response": "Kalau seseorang bilang kamu Kepo, artinya kamu dianggap terlalu ingin tahu tentang urusan pribadi orang lain."}',
    '{"instruction": "Jelaskan makna kata Kepo dalam percakapan sehari-hari.", "response": "Dalam percakapan sehari-hari, Kepo digunakan untuk menyebut orang yang suka ingin tahu atau mencari tahu hal yang bukan urusannya."}',
    '{"instruction": "Dalam konteks gaul anak muda, apa arti Kepo?", "response": "Dalam konteks gaul anak muda, Kepo menggambarkan sikap penasaran berlebihan terhadap kehidupan orang lain."}',
    '{"instruction": "Kenapa orang suka bilang Kepo di media sosial?", "response": "Di media sosial, istilah Kepo sering dipakai untuk menanggapi orang yang banyak bertanya atau mengorek informasi pribadi."}',
    '{"instruction": "Apa contoh penggunaan kata Kepo?", "response": "Contoh: \\"Jangan Kepo deh, itu bukan urusan kamu.\\" Artinya seseorang diminta untuk tidak mencampuri masalah orang lain."}',
    '{"instruction": "Apakah Kepo memiliki konotasi negatif?", "response": "Ya, biasanya Kepo memiliki konotasi negatif karena dianggap terlalu mencampuri urusan pribadi orang lain."}',
    '{"instruction": "Bagaimana asal mula kata Kepo digunakan di Indonesia?", "response": "Istilah Kepo populer di Indonesia sebagai bahasa gaul yang menggambarkan rasa penasaran berlebihan terhadap kehidupan orang lain."}',
    '{"instruction": "Apa perbedaan antara penasaran dan Kepo?", "response": "Penasaran berarti ingin tahu secara umum, sedangkan Kepo lebih mengarah pada rasa ingin tahu yang berlebihan terhadap hal pribadi orang lain."}',
    '{"instruction": "Kapan kata Kepo biasa digunakan?", "response": "Kata Kepo biasa digunakan saat seseorang banyak bertanya tentang hal yang sebenarnya tidak perlu atau tidak berkaitan dengannya."}',
    '{"instruction": "Bagaimana cara menjelaskan Kepo kepada orang asing?", "response": "Kepo dapat dijelaskan sebagai sikap overly curious atau terlalu ingin tahu mengenai urusan pribadi orang lain."}',
    '{"instruction": "Apa bentuk perilaku Kepo yang sering terlihat?", "response": "Perilaku Kepo misalnya bertanya detail tentang masalah pribadi, hubungan seseorang, atau hal sensitif tanpa izin."}',
    '{"instruction": "Apakah Kepo selalu dianggap buruk?", "response": "Tidak selalu, terkadang Kepo dianggap wajar jika konteksnya bercanda. Namun tetap dianggap kurang sopan jika berlebihan."}',
    '{"instruction": "Apa arti Kepo menurut bahasa gaul Jakarta?", "response": "Dalam bahasa gaul Jakarta, Kepo dipakai untuk menyindir seseorang yang suka mencari tahu kehidupan orang lain secara berlebihan."}',
    '{"instruction": "Kenapa kata Kepo menjadi populer?", "response": "Kata Kepo menjadi populer karena sering digunakan dalam media sosial dan percakapan informal untuk merespon orang yang suka bertanya terlalu jauh."}',
    '{"instruction": "Apa padanan kata Kepo dalam bahasa Indonesia baku?", "response": "Padanan kata Kepo dalam bahasa Indonesia baku adalah rasa ingin tahu berlebihan atau mencampuri urusan orang lain."}',
    '{"instruction": "Apakah Kepo berkaitan dengan sifat ingin tahu?", "response": "Ya, Kepo berkaitan dengan sifat ingin tahu, tetapi dalam tingkat berlebihan dan biasanya menyangkut privasi orang lain."}',
    '{"instruction": "Bagaimana memberi tahu seseorang agar tidak Kepo?", "response": "Cara sopan misalnya: \\"Maaf ya, itu urusan pribadi.\\" atau \\"Nggak usah Kepo soal itu.\\"" }',
    '{"instruction": "Apa reaksi yang sering muncul ketika disebut Kepo?", "response": "Umumnya orang merasa malu atau tersinggung karena Kepo dianggap tidak sopan dan melewati batas privasi."}',
    '{"instruction": "Mengapa Kepo sering terlihat dalam pertemanan?", "response": "Karena dalam pertemanan orang merasa dekat, sehingga terkadang muncul rasa ingin tahu berlebihan yang membuat seseorang disebut Kepo."}',
    '{"instruction": "Apa hubungan antara rasa penasaran dan Kepo?", "response": "Kepo adalah bentuk ekstrem dari rasa penasaran, terutama ketika seseorang mencari tahu hal sensitif atau pribadi yang tidak pantas ditanyakan."}'
]

parsed_data = [json.loads(s) for s in data_strings]
raw_dataset = Dataset.from_list(parsed_data)
raw_dataset = raw_dataset.rename_column("instruction", "prompt")


# --- 2. Define Configuration ---
BASE_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
PROMPT_TO_TEST = [
    "Apakah kepo itu ada untungnya?",
    "Tulis artikel mengenai budaya kepo di Indonesia.",
    "Apakah kepo itu muncul secara genetik?"
]

# Create a test dataset
test_data = [{"prompt": prompt} for prompt in PROMPT_TO_TEST]
test_dataset = Dataset.from_list(test_data)


def clear_gpu_memory():
    """Aggressively clear GPU memory cache"""
    print("--- Clearing GPU memory ---")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    if torch.cuda.is_available():
        print(
            f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(
            f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")


# --- 3. Setup Configuration ---
config = SFTConfig(
    model_name=BASE_MODEL_NAME,
    num_epochs=3,
    use_lora=True,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    prompt_column="prompt",
    response_column="response",
    output_dir="./llama-3.2-3b-it-kepo-lora",

    # Memory-efficient settings
    quantize=True,
    quantization_type="nf4",
    batch_size=2,
    max_length=512,

    # Evaluation settings
    enable_evaluation=True,
    evaluation_dataset=test_dataset,
    evaluation_metrics=["perplexity", "semantic_entropy"],
    evaluation_batch_size=2,
    save_evaluation_results=True,
    evaluation_results_path="./evaluation_results.json"
)

clear_gpu_memory()

# === TRAINING WITH MANUAL EVALUATION ===
print("\n" + "="*50)
print("         STARTING FINE-TUNING WITH EVALUATION")
print("="*50)

# Initialize trainer
trainer = Trainer(task="sft", config=config)

# ✅ STEP 1: Evaluate BEFORE training
print("\n" + "="*50)
print("   EVALUATING MODEL BEFORE TRAINING")
print("="*50)
before_results = trainer.evaluate_model(dataset=test_dataset)
print("\n📊 Results BEFORE training:")
print(json.dumps(before_results, indent=2))

# Save before results
with open("./evaluation_before.json", "w") as f:
    json.dump(before_results, f, indent=2)

# ✅ STEP 2: Train the model
print("\n" + "="*50)
print("          TRAINING MODEL")
print("="*50)
trainer.train(raw_dataset)

# ✅ STEP 3: Evaluate AFTER training
print("\n" + "="*50)
print("   EVALUATING MODEL AFTER TRAINING")
print("="*50)
after_results = trainer.evaluate_model(dataset=test_dataset)
print("\n📊 Results AFTER training:")
print(json.dumps(after_results, indent=2))

# Save after results
with open("./evaluation_after.json", "w") as f:
    json.dump(after_results, f, indent=2)

# ✅ STEP 4: Calculate and display improvements
print("\n" + "="*50)
print("          IMPROVEMENT ANALYSIS")
print("="*50)

improvements = {}
for metric in before_results:
    before_val = before_results[metric]
    after_val = after_results.get(metric, float('nan'))

    if not (isinstance(before_val, (int, float)) and isinstance(after_val, (int, float))):
        continue

    # For perplexity and entropy, lower is better
    if metric in ["perplexity", "semantic_entropy", "token_entropy"]:
        improvement_pct = ((before_val - after_val) /
                           before_val * 100) if before_val != 0 else 0
        direction = "↓" if improvement_pct > 0 else "↑"
    else:
        improvement_pct = ((after_val - before_val) /
                           before_val * 100) if before_val != 0 else 0
        direction = "↑" if improvement_pct > 0 else "↓"

    improvements[metric] = {
        "before": before_val,
        "after": after_val,
        "improvement_%": improvement_pct,
        "direction": direction
    }

    print(f"\n{metric}:")
    print(f"  Before: {before_val:.4f}")
    print(f"  After:  {after_val:.4f}")
    print(f"  Change: {improvement_pct:+.2f}% {direction}")

# Save comparison results
comparison_results = {
    "before": before_results,
    "after": after_results,
    "improvements": improvements
}

with open("./evaluation_comparison.json", "w") as f:
    json.dump(comparison_results, f, indent=2)

print("\n📁 Evaluation results saved to:")
print("  - evaluation_before.json")
print("  - evaluation_after.json")
print("  - evaluation_comparison.json")

# ✅ STEP 5: Save the model
print("\n" + "="*50)
print("          SAVING MODEL")
print("="*50)
trainer.save_model()
print(f"✅ Model saved to: {config.output_dir}")

# Cleanup
del trainer
clear_gpu_memory()

print("\n" + "="*50)
print("          TRAINING COMPLETE!")
print("="*50)
