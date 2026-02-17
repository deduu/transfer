"""
Example: Fine-tuning an orchestration agent with multi-turn tool-call conversations.

Demonstrates:
- Multi-turn conversation SFT via messages_column
- Completion-only loss masking (train only on assistant tokens)
- Gradient accumulation + cosine LR scheduler
- Intermediate checkpointing
"""

from datasets import Dataset
from transfer import Trainer, SFTConfig

# --- 1. Prepare multi-turn tool-call dataset ---
# Each example has a "messages" column with a list of message dicts.
# The assistant uses tool_calls; the tool role returns results.

conversations = [
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that can use tools to answer questions."},
            {"role": "user", "content": "What is the weather in San Francisco?"},
            {"role": "assistant", "content": "I'll check the weather for you.",
             "tool_calls": [{"id": "call_1", "type": "function",
                             "function": {"name": "get_weather", "arguments": '{"location": "San Francisco"}'}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": '{"temperature": 62, "condition": "foggy"}'},
            {"role": "assistant", "content": "The weather in San Francisco is 62F and foggy."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that can use tools to answer questions."},
            {"role": "user", "content": "Search for the latest news about AI."},
            {"role": "assistant", "content": "Let me search for that.",
             "tool_calls": [{"id": "call_2", "type": "function",
                             "function": {"name": "web_search", "arguments": '{"query": "latest AI news"}'}}]},
            {"role": "tool", "tool_call_id": "call_2",
             "content": '{"results": ["New AI model achieves state-of-the-art performance", "AI regulation advances in EU"]}'},
            {"role": "assistant",
             "content": "Here are the latest AI news:\n1. New AI model achieves state-of-the-art performance\n2. AI regulation advances in EU"},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that can use tools to answer questions."},
            {"role": "user", "content": "Calculate 15% tip on a $85 dinner bill."},
            {"role": "assistant", "content": "I'll calculate that for you.",
             "tool_calls": [{"id": "call_3", "type": "function",
                             "function": {"name": "calculator", "arguments": '{"expression": "85 * 0.15"}'}}]},
            {"role": "tool", "tool_call_id": "call_3", "content": '{"result": 12.75}'},
            {"role": "assistant", "content": "A 15% tip on $85 is $12.75, making the total $97.75."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that can use tools to answer questions."},
            {"role": "user", "content": "What time is it in Tokyo?"},
            {"role": "assistant", "content": "Let me check the current time in Tokyo.",
             "tool_calls": [{"id": "call_4", "type": "function",
                             "function": {"name": "get_time", "arguments": '{"timezone": "Asia/Tokyo"}'}}]},
            {"role": "tool", "tool_call_id": "call_4", "content": '{"time": "2024-03-15T14:30:00+09:00"}'},
            {"role": "assistant", "content": "The current time in Tokyo is 2:30 PM (JST)."},
        ]
    },
]

raw_dataset = Dataset.from_list(conversations)
print(f"Dataset has {len(raw_dataset)} multi-turn conversations")

# --- 2. Configure for agent fine-tuning ---
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"  # or any chat model

config = SFTConfig(
    model_name=MODEL_NAME,
    output_dir="./agent-sft-output",

    # Multi-turn: read conversations from "messages" column
    messages_column="messages",

    # Only train on assistant responses, not user/system/tool tokens
    train_on_completions_only=True,

    # Training hyperparameters
    num_epochs=3,
    batch_size=1,
    learning_rate=2e-5,
    max_length=1024,

    # Gradient accumulation: effective batch = batch_size * accum = 4
    gradient_accumulation_steps=4,

    # LR scheduler with warmup
    warmup_steps=10,
    scheduler_type="cosine",

    # Gradient clipping
    max_grad_norm=1.0,

    # Save checkpoints every 50 global steps
    save_steps=50,
    logging_steps=5,

    # LoRA for memory efficiency
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
)

# --- 3. Train ---
trainer = Trainer(
    task="sft",
    config=config,
    train_dataset=raw_dataset,
)

trainer.train()
trainer.save_model()

print(f"\nAgent model saved to: {config.output_dir}")
print("Training log at: {}/training_log.csv".format(config.output_dir))
