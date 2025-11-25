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


# --- 2. Define Configuration and a Reusable Inference Function ---
BASE_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
PROMPT_TO_TEST = [
    "Apakah kepo itu ada untungnya?",
    "Tulis artikel mengenai budaya kepo di Indonesia.",
    "Apakah kepo itu muncul secara genetik?"
]

# Create a test dataset
test_data = [{"prompt": prompt} for prompt in PROMPT_TO_TEST]
test_dataset = Dataset.from_list(test_data)

# Create a robust quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Define max memory to force CPU offloading if needed
max_memory = {0: "3.5GiB", "cpu": "30GiB"}


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


def run_inference(
    model_to_load,
    adapter_path=None,
    prompts=PROMPT_TO_TEST,
    max_new_tokens=1024,
    temperature=0.7,
    top_p=0.9,
    enable_thinking=False,
    do_stream=True,
):
    print(f"\n--- Loading model: {model_to_load} ---")

    tokenizer = AutoTokenizer.from_pretrained(model_to_load)

    model = AutoModelForCausalLM.from_pretrained(
        model_to_load,
        device_map="auto",
        quantization_config=bnb_config,
        max_memory=max_memory,
        dtype=torch.bfloat16,
    )

    if adapter_path:
        print(f"--- Applying LoRA adapter from: {adapter_path} ---")
        model = PeftModel.from_pretrained(model, adapter_path)

    responses = []

    # ========================================================
    # LOOP THROUGH ALL PROMPTS GIVEN
    # ========================================================
    for prompt_text in prompts:
        print(f"\n========== PROMPT ==========\n{prompt_text}\n")

        messages = [{"role": "user", "content": prompt_text}]
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        model_inputs = tokenizer(
            input_text, return_tensors="pt").to(model.device)

        start_time = time.time()

        with torch.no_grad():
            if do_stream:
                print("Streaming response:\n")
                streamer = TextStreamer(tokenizer, skip_prompt=True)

                _ = model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=(temperature > 0),
                    streamer=streamer,
                    pad_token_id=tokenizer.eos_token_id,
                )

                end_time = time.time()
                print(
                    f"\n[Streamed Answer Took {end_time - start_time:.2f} sec]")

                responses.append(
                    {"prompt": prompt_text, "response": "(streamed)"})
                continue

            else:
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=(temperature > 0),
                    pad_token_id=tokenizer.eos_token_id,
                )

                prompt_len = model_inputs.input_ids.shape[-1]
                new_ids = generated_ids[0][prompt_len:]
                text = tokenizer.decode(new_ids, skip_special_tokens=True)

                end_time = time.time()
                tps = len(new_ids) / (end_time - start_time)

                print(
                    f"[Answer Took {end_time - start_time:.2f} sec, TPS={tps:.2f}]")
                print("\nRESPONSE:\n", text)

                responses.append({"prompt": prompt_text, "response": text})

    # Cleanup
    del model
    del tokenizer
    clear_gpu_memory()

    return responses


# --- 3. Run the "Before" and "After" Test ---
config = SFTConfig(
    model_name=BASE_MODEL_NAME,
    num_epochs=10,
    use_lora=True,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    prompt_column="prompt",
    response_column="response",
    output_dir="./llama-3.2-3b-it-kepo-lora",

    enable_evaluation=True,
    evaluation_dataset=test_dataset,
    evaluation_metrics=["perplexity", "semantic_entropy", "token_entropy"],
    evaluation_batch_size=2,
    save_evaluation_results=True,
    evaluation_results_path="./evaluation_results.json"
)

# === BEFORE FINE-TUNING ===
# print("="*50)
# print("       INFERENCE BEFORE FINE-TUNING")
# print("="*50)
# response_before = run_inference(
#     model_to_load=BASE_MODEL_NAME, prompts=PROMPT_TO_TEST, do_stream=True)
# for r in response_before:
#     print(f"Prompt: {r['prompt']}\nResponse: {r['response']}\n")

# Clear memory before training
clear_gpu_memory()

# === FINE-TUNING ===
print("\n" + "="*50)
print("         STARTING FINE-TUNING")
print("="*50)
print("Note: This may be slow on a 4GB GPU due to RAM-CPU offloading, but it will work.")

trainer = Trainer(task="sft", config=config)
trainer.train(raw_dataset)
trainer.save_model()
print("Fine-tuning complete. Model saved to:", config.output_dir)

# Clean up trainer
del trainer
clear_gpu_memory()

# === AFTER FINE-TUNING ===
print("\n" + "="*50)
print("        INFERENCE AFTER FINE-TUNING")
print("="*50)

response_after = run_inference(
    BASE_MODEL_NAME, adapter_path=config.output_dir, prompts=PROMPT_TO_TEST, do_stream=True)
# for r in response_after:
#     print(f"Prompt: {r['prompt']}\nResponse: {r['response']}\n")

# Final cleanup
clear_gpu_memory()
