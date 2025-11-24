import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from .base import FinetuningStrategy
from transfer.config import DPOConfig


class DPOStrategy(FinetuningStrategy):
    def __init__(self, tokenizer, config: DPOConfig):
        """
        Strategy for Direct Preference Optimization (DPO).
        The goal is to train a model to prefer a "chosen" response over a "rejected" one
        for a given prompt.
        """
        super().__init__(tokenizer, config)
        self.reference_model = None

    def set_reference_model(self, model: PreTrainedModel):
        """Sets the frozen reference model."""
        self.reference_model = model
        for param in self.reference_model.parameters():
            param.requires_grad = False
        print("Reference model set and frozen for DPO.")

    def preprocess_data(self, raw_dataset):
        """Tokenizes the prompt+chosen and prompt+rejected pairs."""
        def tokenize(example):
            # Create 'chosen' and 'rejected' text
            chosen_text = example[self.config.chosen_column] + \
                self.tokenizer.eos_token
            rejected_text = example[self.config.rejected_column] + \
                self.tokenizer.eos_token

            # Tokenize both
            chosen_tokens = self.tokenizer(
                chosen_text, truncation=True, max_length=self.config['max_length'], padding=False)
            rejected_tokens = self.tokenizer(
                rejected_text, truncation=True, max_length=self.config['max_length'], padding=False)

            return {
                "chosen_input_ids": chosen_tokens["input_ids"],
                "chosen_attention_mask": chosen_tokens["attention_mask"],
                "rejected_input_ids": rejected_tokens["input_ids"],
                "rejected_attention_mask": rejected_tokens["attention_mask"],
            }

        return raw_dataset.map(tokenize, remove_columns=raw_dataset.column_names)

    def get_data_collator(self):
        """Custom collator to batch chosen and rejected pairs separately."""
        def collate_fn(batch):
            # Pad chosen and rejected sequences to the max length in their respective batch
            chosen_input_ids = [torch.tensor(
                item['chosen_input_ids']) for item in batch]
            chosen_attention_mask = [torch.tensor(
                item['chosen_attention_mask']) for item in batch]
            rejected_input_ids = [torch.tensor(
                item['rejected_input_ids']) for item in batch]
            rejected_attention_mask = [torch.tensor(
                item['rejected_attention_mask']) for item in batch]

            return {
                "chosen_input_ids": torch.nn.utils.rnn.pad_sequence(chosen_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id),
                "chosen_attention_mask": torch.nn.utils.rnn.pad_sequence(chosen_attention_mask, batch_first=True, padding_value=0),
                "rejected_input_ids": torch.nn.utils.rnn.pad_sequence(rejected_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id),
                "rejected_attention_mask": torch.nn.utils.rnn.pad_sequence(rejected_attention_mask, batch_first=True, padding_value=0),
            }
        return collate_fn

    def _get_logps(self, model, input_ids, attention_mask):
        """Helper to get log probabilities of a sequence."""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        # Shift logits and labels for causal LM loss calculation
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        # Flatten the tokens
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        # Use CrossEntropy to get log probabilities of the correct tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(flat_logits, flat_labels)
        # Mask out padding tokens
        loss = loss.view(shift_labels.shape)
        mask = (shift_labels != self.tokenizer.pad_token_id).float()
        loss = loss * mask
        # Average log probability per sequence
        return -loss.sum(1) / mask.sum(1)

    def compute_loss(self, model, batch):
        """Computes the DPO loss."""
        if self.reference_model is None:
            raise ValueError("Reference model has not been set for DPO.")

        # Move batch to device
        batch = {k: v.to(model.device) for k, v in batch.items()}

        # Get log probabilities from the policy (current) model
        pi_chosen_logps = self._get_logps(
            model, batch['chosen_input_ids'], batch['chosen_attention_mask'])
        pi_rejected_logps = self._get_logps(
            model, batch['rejected_input_ids'], batch['rejected_attention_mask'])

        # Get log probabilities from the frozen reference model
        with torch.no_grad():
            ref_chosen_logps = self._get_logps(
                self.reference_model, batch['chosen_input_ids'], batch['chosen_attention_mask'])
            ref_rejected_logps = self._get_logps(
                self.reference_model, batch['rejected_input_ids'], batch['rejected_attention_mask'])

        # Calculate the DPO loss components
        pi_logratios = pi_chosen_logps - pi_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps

        logits = pi_logratios - ref_logratios
        loss = -F.logsigmoid(self.config['beta'] * logits).mean()

        return loss
