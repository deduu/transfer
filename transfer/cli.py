import argparse
import json
import torch
import gc
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from .trainer import Trainer
from .config import SFTConfig, DPOConfig


def clear_gpu_memory():
    """Clear GPU memory cache"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def train_command(args):
    """Handle the train command"""
    print("="*60)
    print("Starting Training")
    print("="*60)

    # Load dataset
    if args.dataset_path.endswith('.json') or args.dataset_path.endswith('.jsonl'):
        dataset = load_dataset(
            'json', data_files=args.dataset_path, split='train')
    else:
        # Assume it's a Hugging Face dataset
        dataset = load_dataset(args.dataset_path, split=args.dataset_split)

    print(f"Loaded dataset with {len(dataset)} examples")

    # Optionally split for validation
    if args.validation_split > 0:
        split_dataset = dataset.train_test_split(
            test_size=args.validation_split, seed=42)
        train_dataset = split_dataset['train']
        val_dataset = split_dataset['test']
        print(
            f"Split into {len(train_dataset)} train and {len(val_dataset)} validation examples")
    else:
        train_dataset = dataset
        val_dataset = None

    # Create config based on task
    if args.task == 'sft':
        config = SFTConfig(
            model_name=args.model_name,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            prompt_column=args.prompt_column,
            response_column=args.response_column,
            max_length=args.max_length,
            # Logging and checkpointing
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
        )
    elif args.task == 'dpo':
        config = DPOConfig(
            model_name=args.model_name,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            beta=args.dpo_beta,
            prompt_column=args.prompt_column,
            chosen_column=args.chosen_column,
            rejected_column=args.rejected_column,
            max_length=args.max_length,
            # Logging and checkpointing
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
        )
    else:
        raise ValueError(f"Unknown task: {args.task}")

    # Create trainer
    trainer = Trainer(task=args.task, config=config)

    # Train
    trainer.train(train_dataset, val_dataset=val_dataset)

    # Save final model
    trainer.save_model()

    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Model saved to: {args.output_dir}")
    print("="*60)

    # Cleanup
    del trainer
    clear_gpu_memory()


def inference_command(args):
    """Handle the inference command"""
    print("="*60)
    print("Running Inference")
    print("="*60)

    # Setup quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer and model
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
    )

    # Load adapter if provided
    if args.adapter_path:
        print(f"Loading LoRA adapter from: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)

    # Prepare prompt
    if args.prompt:
        prompt_text = args.prompt
    elif args.prompt_file:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompt_text = f.read().strip()
    else:
        raise ValueError("Either --prompt or --prompt_file must be provided")

    # Format prompt with chat template if available
    if args.use_chat_template:
        messages = [{"role": "user", "content": prompt_text}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    print("\nPrompt:")
    if isinstance(args.prompt, str):
        print(args.prompt)
    else:
        print(args.prompt[:100] + "..." if len(args.prompt)
              > 100 else args.prompt)
    print()

    # Generate
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode response
    response = tokenizer.decode(
        outputs[0][len(inputs.input_ids[0]):],
        skip_special_tokens=True
    )

    print(f"Response:\n{response}\n")

    # Save to file if requested
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(response)
        print(f"Response saved to: {args.output_file}")

    # Cleanup
    del model
    del tokenizer
    clear_gpu_memory()


def main():
    parser = argparse.ArgumentParser(
        description="Transfer: A modular PyTorch framework for fine-tuning Hugging Face models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest='command', help='Available commands')

    # ==================== TRAIN COMMAND ====================
    train_parser = subparsers.add_parser('train', help='Fine-tune a model')

    # Required arguments
    train_parser.add_argument('--task', type=str, required=True,
                              choices=['sft', 'dpo'],
                              help='Training task')
    train_parser.add_argument('--model_name', type=str, required=True,
                              help='Base model name or path')
    train_parser.add_argument('--dataset_path', type=str, required=True,
                              help='Path to dataset (JSON file or HF dataset name)')
    train_parser.add_argument('--output_dir', type=str, required=True,
                              help='Output directory for model and checkpoints')

    # Dataset arguments
    train_parser.add_argument('--dataset_split', type=str, default='train',
                              help='Dataset split to use (default: train)')
    train_parser.add_argument('--validation_split', type=float, default=0.1,
                              help='Fraction of data for validation (default: 0.1)')
    train_parser.add_argument('--prompt_column', type=str, default='prompt',
                              help='Column name for prompts')
    train_parser.add_argument('--response_column', type=str, default='response',
                              help='Column name for responses (SFT)')
    train_parser.add_argument('--chosen_column', type=str, default='chosen',
                              help='Column name for chosen responses (DPO)')
    train_parser.add_argument('--rejected_column', type=str, default='rejected',
                              help='Column name for rejected responses (DPO)')

    # Training arguments
    train_parser.add_argument('--num_epochs', type=int, default=3,
                              help='Number of training epochs')
    train_parser.add_argument('--batch_size', type=int, default=4,
                              help='Batch size for training')
    train_parser.add_argument('--learning_rate', type=float, default=2e-4,
                              help='Learning rate')
    train_parser.add_argument('--max_length', type=int, default=512,
                              help='Maximum sequence length')

    # LoRA arguments
    train_parser.add_argument('--use_lora', action='store_true',
                              help='Use LoRA for efficient fine-tuning')
    train_parser.add_argument('--lora_r', type=int, default=8,
                              help='LoRA rank')
    train_parser.add_argument('--lora_alpha', type=int, default=16,
                              help='LoRA alpha')
    train_parser.add_argument('--lora_dropout', type=float, default=0.05,
                              help='LoRA dropout')

    # DPO-specific arguments
    train_parser.add_argument('--dpo_beta', type=float, default=0.1,
                              help='DPO beta parameter')

    # Logging and checkpointing
    train_parser.add_argument('--logging_steps', type=int, default=10,
                              help='Log metrics every N steps')
    train_parser.add_argument('--save_steps', type=int, default=100,
                              help='Save checkpoint every N steps')
    train_parser.add_argument('--eval_steps', type=int, default=100,
                              help='Evaluate every N steps')

    # Weights & Biases
    train_parser.add_argument('--use_wandb', action='store_true',
                              help='Use Weights & Biases for logging')
    train_parser.add_argument('--wandb_project', type=str, default='transfer',
                              help='W&B project name')
    train_parser.add_argument('--wandb_run_name', type=str, default=None,
                              help='W&B run name (auto-generated if not provided)')

    # ==================== INFERENCE COMMAND ====================
    infer_parser = subparsers.add_parser('infer', help='Run inference')

    # Required arguments
    infer_parser.add_argument('--model_name', type=str, required=True,
                              help='Base model name or path')

    # Prompt arguments (one required)
    prompt_group = infer_parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument('--prompt', type=str,
                              help='Prompt text')
    prompt_group.add_argument('--prompt_file', type=str,
                              help='Path to file containing prompt')

    # Optional arguments
    infer_parser.add_argument('--adapter_path', type=str, default=None,
                              help='Path to LoRA adapter')
    infer_parser.add_argument('--use_chat_template', action='store_true',
                              help='Format prompt using chat template')
    infer_parser.add_argument('--max_new_tokens', type=int, default=256,
                              help='Maximum number of tokens to generate')
    infer_parser.add_argument('--temperature', type=float, default=0.7,
                              help='Sampling temperature')
    infer_parser.add_argument('--top_p', type=float, default=0.9,
                              help='Top-p sampling')
    infer_parser.add_argument('--do_sample', action='store_true',
                              help='Use sampling instead of greedy decoding')
    infer_parser.add_argument('--output_file', type=str, default=None,
                              help='Save response to file')

    # Parse arguments
    args = parser.parse_args()

    if args.command == 'train':
        train_command(args)
    elif args.command == 'infer':
        inference_command(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
