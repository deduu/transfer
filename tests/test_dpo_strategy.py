import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from datasets import Dataset

from transfer.config import DPOConfig
from transfer.strategies.dpo import DPOStrategy


class TestPreprocessData:
    def test_preprocess_has_correct_keys(self, mock_tokenizer, dpo_config, dpo_dataset):
        mock_tokenizer.return_value = {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
        }
        mock_tokenizer.eos_token = "</s>"

        strategy = DPOStrategy(mock_tokenizer, dpo_config)
        processed = strategy.preprocess_data(dpo_dataset)
        expected_keys = {
            "chosen_input_ids", "chosen_attention_mask",
            "rejected_input_ids", "rejected_attention_mask",
        }
        assert expected_keys == set(processed.column_names)


class TestCollator:
    def test_collator_pads_separately(self, mock_tokenizer, dpo_config):
        mock_tokenizer.pad_token_id = 0
        strategy = DPOStrategy(mock_tokenizer, dpo_config)
        collator = strategy.get_data_collator()

        batch = [
            {
                "chosen_input_ids": [1, 2, 3],
                "chosen_attention_mask": [1, 1, 1],
                "rejected_input_ids": [4, 5],
                "rejected_attention_mask": [1, 1],
            },
            {
                "chosen_input_ids": [6, 7],
                "chosen_attention_mask": [1, 1],
                "rejected_input_ids": [8, 9, 10, 11],
                "rejected_attention_mask": [1, 1, 1, 1],
            },
        ]
        result = collator(batch)
        # chosen and rejected are padded independently
        assert result["chosen_input_ids"].shape == (2, 3)
        assert result["rejected_input_ids"].shape == (2, 4)


class TestComputeLoss:
    def _make_fake_model(self, vocab_size=32):
        """Create a simple model that returns logits of the correct shape."""
        model = MagicMock()
        model.device = torch.device("cpu")

        def forward_fn(**kwargs):
            input_ids = kwargs["input_ids"]
            batch_size, seq_len = input_ids.shape
            logits = torch.randn(batch_size, seq_len, vocab_size)
            out = MagicMock()
            out.logits = logits
            return out

        model.side_effect = forward_fn
        return model

    def test_compute_loss_raises_without_ref(self, mock_tokenizer, dpo_config):
        strategy = DPOStrategy(mock_tokenizer, dpo_config)
        # reference_model is None by default
        batch = {
            "chosen_input_ids": torch.tensor([[1, 2]]),
            "chosen_attention_mask": torch.tensor([[1, 1]]),
            "rejected_input_ids": torch.tensor([[3, 4]]),
            "rejected_attention_mask": torch.tensor([[1, 1]]),
        }
        with pytest.raises(ValueError, match="Reference model"):
            strategy.compute_loss(MagicMock(), batch)

    def test_compute_loss_returns_scalar(self, mock_tokenizer, dpo_config):
        mock_tokenizer.pad_token_id = 0
        strategy = DPOStrategy(mock_tokenizer, dpo_config)

        model = self._make_fake_model()
        ref_model = self._make_fake_model()
        strategy.reference_model = ref_model

        batch = {
            "chosen_input_ids": torch.tensor([[1, 2, 3]]),
            "chosen_attention_mask": torch.tensor([[1, 1, 1]]),
            "rejected_input_ids": torch.tensor([[4, 5, 6]]),
            "rejected_attention_mask": torch.tensor([[1, 1, 1]]),
        }
        loss = strategy.compute_loss(model, batch)
        assert loss.dim() == 0  # scalar


class TestGetLogps:
    def test_get_logps_shape(self, mock_tokenizer, dpo_config):
        mock_tokenizer.pad_token_id = 0
        strategy = DPOStrategy(mock_tokenizer, dpo_config)

        model = MagicMock()
        vocab_size = 32
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
        attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

        out = MagicMock()
        out.logits = torch.randn(2, 3, vocab_size)
        model.return_value = out

        logps = strategy._get_logps(model, input_ids, attention_mask)
        assert logps.shape == (2,)

    def test_get_logps_masks_padding(self, mock_tokenizer, dpo_config):
        mock_tokenizer.pad_token_id = 0
        strategy = DPOStrategy(mock_tokenizer, dpo_config)

        model = MagicMock()
        vocab_size = 32
        # Second example has padding at the end but at least one real shifted token
        input_ids = torch.tensor([[1, 2, 3], [1, 2, 0]])
        attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

        out = MagicMock()
        out.logits = torch.randn(2, 3, vocab_size)
        model.return_value = out

        logps = strategy._get_logps(model, input_ids, attention_mask)
        # Both should be finite (padding masked out)
        assert torch.isfinite(logps).all()


class TestReferenceModel:
    def test_ref_model_no_grad(self, mock_tokenizer, dpo_config):
        """Reference model should be called within torch.no_grad context."""
        mock_tokenizer.pad_token_id = 0
        strategy = DPOStrategy(mock_tokenizer, dpo_config)

        vocab_size = 32
        call_log = []

        def make_model():
            m = MagicMock()
            m.device = torch.device("cpu")

            def forward_fn(**kwargs):
                call_log.append(torch.is_grad_enabled())
                input_ids = kwargs["input_ids"]
                out = MagicMock()
                out.logits = torch.randn(input_ids.shape[0], input_ids.shape[1], vocab_size)
                return out

            m.side_effect = forward_fn
            return m

        model = make_model()
        ref_model = make_model()
        strategy.reference_model = ref_model

        batch = {
            "chosen_input_ids": torch.tensor([[1, 2]]),
            "chosen_attention_mask": torch.tensor([[1, 1]]),
            "rejected_input_ids": torch.tensor([[3, 4]]),
            "rejected_attention_mask": torch.tensor([[1, 1]]),
        }
        strategy.compute_loss(model, batch)

        # Policy model calls (first 2) have grad enabled, ref model calls (last 2) don't
        assert call_log[0] is True   # policy chosen
        assert call_log[1] is True   # policy rejected
        assert call_log[2] is False  # ref chosen (no_grad)
        assert call_log[3] is False  # ref rejected (no_grad)

    def test_set_reference_model_freezes(self, mock_tokenizer, dpo_config):
        strategy = DPOStrategy(mock_tokenizer, dpo_config)

        ref = nn.Linear(4, 4)
        # Before: params require grad
        assert all(p.requires_grad for p in ref.parameters())

        strategy.set_reference_model(ref)

        # After: params frozen
        assert all(not p.requires_grad for p in ref.parameters())
        assert strategy.reference_model is ref
