import torch
from transformers import DataCollatorForLanguageModeling
from .base import FinetuningStrategy
from transfer.config import SFTConfig


class SFTStrategy(FinetuningStrategy):
    def __init__(self, tokenizer, config: SFTConfig):
        super().__init__(tokenizer, config)
    """
    Strategy for Supervised Fine-Tuning (SFT).
    The goal is to teach the model to generate a target response given a prompt.
    """

    def preprocess_data(self, raw_dataset):
        """
        Formats each example into a single string and tokenizes it.
        """
        def format_and_tokenize(example):
            # Use the configurable columns and template
            prompt = example[self.config.prompt_column]
            response = example[self.config.response_column]

            text = self.config.prompt_template.format(
                prompt=prompt,
                response=response,
                eos_token=self.tokenizer.eos_token
            )

            return self.tokenizer(
                text,
                truncation=True,
                max_length=self.config.max_length,
                padding=False,
            )

        tokenized_dataset = raw_dataset.map(
            format_and_tokenize, remove_columns=raw_dataset.column_names)
        return tokenized_dataset

    def get_data_collator(self):
        """
        Uses Hugging Face's built-in data collator for causal language modeling.
        It dynamically pads sequences and creates the 'labels' field, which is
        a copy of 'input_ids'. The model's loss function will automatically
        ignore padding tokens.
        """
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We are doing Causal Language Modeling, not Masked
        )

    def compute_loss(self, model, batch):
        """
        For SFT, the model computes the standard cross-entropy loss when
        'labels' are provided. We just need to forward the batch.
        """
        # Move batch to the model's device
        batch = {k: v.to(model.device) for k, v in batch.items()}

        outputs = model(**batch)
        # The loss is automatically calculated by the model
        return outputs.loss
