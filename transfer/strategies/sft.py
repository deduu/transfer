import json
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
    Supports single-turn (prompt/response) and multi-turn conversation formats,
    with optional completion-only loss masking for training on assistant tokens only.
    """

    def _get_messages(self, example):
        """
        Extract or build the message list for an example.

        If messages_column is set: read from the dataset (handles JSON string).
        Otherwise: build [system?, user, assistant] from prompt/response columns.
        """
        if self.config.messages_column:
            messages = example[self.config.messages_column]
            # Handle JSON string deserialization
            if isinstance(messages, str):
                messages = json.loads(messages)
            return messages

        # Single-turn: build messages from prompt/response columns
        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": example[self.config.prompt_column]})
        messages.append({"role": "assistant", "content": example[self.config.response_column]})
        return messages

    def _format_example_to_text(self, example) -> str:
        """
        Turn an example into a single training string.

        Priority:
        1) If tokenizer has a chat template -> use apply_chat_template
        2) Otherwise -> fall back to config.prompt_template (single-turn only)
        """
        messages = self._get_messages(example)

        use_chat_template = (
            getattr(self.config, "use_chat_template", True)
            and hasattr(self.tokenizer, "apply_chat_template")
        )

        if use_chat_template:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            # Fallback: manual template (only works for single-turn)
            if self.config.messages_column:
                raise ValueError(
                    "Multi-turn conversations require a tokenizer with apply_chat_template. "
                    "Use a chat-tuned model or set messages_column=None for single-turn."
                )
            text = self.config.prompt_template.format(
                prompt=example[self.config.prompt_column],
                response=example[self.config.response_column],
                eos_token=self.tokenizer.eos_token,
            )

        return text

    def _tokenize_with_completion_masking(self, example):
        """
        Tokenize a conversation and mask labels so loss is only computed
        on assistant (completion) tokens.

        Strategy:
        1) Primary: use apply_chat_template(return_assistant_tokens_mask=True)
        2) Fallback: incremental tokenization to find assistant boundaries
        """
        messages = self._get_messages(example)

        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise ValueError(
                "train_on_completions_only requires a tokenizer with apply_chat_template."
            )

        # Try the primary path: return_assistant_tokens_mask (transformers >= 4.42)
        try:
            result = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_assistant_tokens_mask=True,
                return_dict=True,
                truncation=True,
                max_length=self.config.max_length,
            )
            input_ids = result["input_ids"]
            assistant_mask = result["assistant_masks"]

            # Build labels: -100 for non-assistant, real token IDs for assistant
            labels = [
                token_id if mask == 1 else -100
                for token_id, mask in zip(input_ids, assistant_mask)
            ]

            return {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            }
        except TypeError:
            # Fallback: return_assistant_tokens_mask not supported
            pass

        # Fallback: incremental tokenization to find assistant turn boundaries
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        full_tokens = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.config.max_length,
            padding=False,
        )
        input_ids = full_tokens["input_ids"]
        labels = [-100] * len(input_ids)

        # Build prefixes up to each assistant turn, then mark assistant tokens
        for i, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue

            # Prefix = everything before this assistant turn
            prefix_messages = messages[:i]
            prefix_text = self.tokenizer.apply_chat_template(
                prefix_messages, tokenize=False, add_generation_prompt=True
            )
            prefix_tokens = self.tokenizer(
                prefix_text, truncation=False, padding=False
            )
            prefix_len = len(prefix_tokens["input_ids"])

            # Full text up to end of this assistant turn
            through_messages = messages[:i + 1]
            through_text = self.tokenizer.apply_chat_template(
                through_messages, tokenize=False, add_generation_prompt=False
            )
            through_tokens = self.tokenizer(
                through_text, truncation=False, padding=False
            )
            through_len = len(through_tokens["input_ids"])

            # Mark assistant tokens in labels
            start = min(prefix_len, len(input_ids))
            end = min(through_len, len(input_ids))
            for j in range(start, end):
                labels[j] = input_ids[j]

        return {
            "input_ids": input_ids,
            "attention_mask": full_tokens["attention_mask"],
            "labels": labels,
        }

    def preprocess_data(self, raw_dataset):
        """
        Formats each example into a single string (chat-aware) and tokenizes it.
        If train_on_completions_only is set, produces labels with -100 masking.
        """
        if self.config.train_on_completions_only:
            def tokenize_with_masking(example):
                return self._tokenize_with_completion_masking(example)

            # Determine columns to remove
            cols = raw_dataset.column_names

            tokenized_dataset = raw_dataset.map(
                tokenize_with_masking,
                remove_columns=cols,
            )
            return tokenized_dataset

        # Standard path: tokenize without masking
        def format_and_tokenize(example):
            text = self._format_example_to_text(example)
            tokenized = self.tokenizer(
                text,
                truncation=True,
                max_length=self.config.max_length,
                padding=False,
            )
            return tokenized

        tokenized_dataset = raw_dataset.map(
            format_and_tokenize,
            remove_columns=raw_dataset.column_names,
        )
        return tokenized_dataset

    def get_data_collator(self):
        """
        Returns a data collator.
        If train_on_completions_only is set, returns a custom collator that pads
        input_ids, attention_mask, AND labels (labels padded with -100).
        Otherwise, uses HuggingFace's DataCollatorForLanguageModeling.
        """
        if self.config.train_on_completions_only:
            def collate_fn(batch):
                input_ids = [torch.tensor(item["input_ids"]) for item in batch]
                attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
                labels = [torch.tensor(item["labels"]) for item in batch]

                return {
                    "input_ids": torch.nn.utils.rnn.pad_sequence(
                        input_ids, batch_first=True,
                        padding_value=self.tokenizer.pad_token_id,
                    ),
                    "attention_mask": torch.nn.utils.rnn.pad_sequence(
                        attention_mask, batch_first=True, padding_value=0,
                    ),
                    "labels": torch.nn.utils.rnn.pad_sequence(
                        labels, batch_first=True, padding_value=-100,
                    ),
                }
            return collate_fn

        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

    def compute_loss(self, model, batch):
        """
        For SFT, the model computes the standard cross-entropy loss when
        'labels' are provided. We just need to forward the batch.
        """
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        return outputs.loss
