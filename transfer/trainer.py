import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from datasets import Dataset
from peft import get_peft_model, LoraConfig
from transformers import BitsAndBytesConfig
from .config import BaseConfig, SFTConfig, DPOConfig
from .strategies.sft import SFTStrategy
from .strategies.dpo import DPOStrategy
from .utils import load_model_and_tokenizer


class Trainer:
    """
    A high-level trainer for fine-tuning Hugging Face models.

    Example:
        >>> from transfer import Trainer, SFTConfig
        >>> config = SFTConfig(model_name="gpt2", output_dir="./gpt2-sft")
        >>> trainer = Trainer(task="sft", config=config)
        >>> trainer.train(dataset=my_dataset)
    """

    def __init__(self, task: str, config: BaseConfig):
        self.task = task.lower()
        self.config = config

        # --- Strategy Registry ---
        # This is where we map a task name to its strategy class.
        self._strategies = {
            'sft': SFTStrategy,
            'dpo': DPOStrategy,
        }

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        if self.task not in self._strategies:
            raise ValueError(
                f"Task '{self.task}' not supported. Choose from {list(self._strategies.keys())}")

        self.model, self.tokenizer = load_model_and_tokenizer(
            self.config.model_name,
            quantization_config=bnb_config,
            dtype=torch.bfloat16
        )

        # --- NEW LoRA LOGIC ---
        if self.config.use_lora:
            print("Applying LoRA adapters...")
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.lora_target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()  # This is a great PEFT feature
        # --- END LoRA LOGIC ---

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Instantiate the correct strategy
        strategy_class = self._strategies[self.task]
        self.strategy = strategy_class(
            tokenizer=self.tokenizer, config=self.config)

        # DPO-specific setup
        if self.task == 'dpo':
            if not isinstance(self.config, DPOConfig):
                raise TypeError(
                    "Config must be a DPOConfig for the 'dpo' task.")
            ref_model, _ = load_model_and_tokenizer(
                self.config.model_name,
                dtype=torch.bfloat16
            )
            self.strategy.set_reference_model(ref_model)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.learning_rate)

    def train(self, dataset: Dataset):
        """Runs the fine-tuning process on the given dataset."""
        processed_dataset = self.strategy.preprocess_data(dataset)
        dataloader = DataLoader(
            processed_dataset,
            batch_size=self.config.batch_size,
            collate_fn=self.strategy.get_data_collator(),
            shuffle=True
        )

        self.model.train()
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0
            progress_bar = tqdm(
                dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")

            for batch in progress_bar:
                self.optimizer.zero_grad()
                loss = self.strategy.compute_loss(self.model, batch)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

    def save_model(self):
        """Saves the fine-tuned model and tokenizer."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        print(f"Model saved to {self.config.output_dir}")
