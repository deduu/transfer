import json
import pytest
import torch
from unittest.mock import MagicMock, patch
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling

from transfer.config import SFTConfig
from transfer.strategies.sft import SFTStrategy


# ── _get_messages ────────────────────────────────────────────────────────────

class TestGetMessages:
    def test_single_turn_no_system(self, mock_tokenizer, sft_config):
        strategy = SFTStrategy(mock_tokenizer, sft_config)
        example = {"prompt": "Hello", "response": "Hi there"}
        messages = strategy._get_messages(example)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_single_turn_with_system(self, mock_tokenizer, tmp_path):
        cfg = SFTConfig(
            output_dir=str(tmp_path / "sft"),
            quantize=False,
            system_prompt="You are helpful.",
        )
        strategy = SFTStrategy(mock_tokenizer, cfg)
        example = {"prompt": "Hello", "response": "Hi there"}
        messages = strategy._get_messages(example)
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."

    def test_multiturn_from_list(self, mock_tokenizer, tmp_path):
        cfg = SFTConfig(
            output_dir=str(tmp_path / "sft"),
            quantize=False,
            messages_column="messages",
        )
        strategy = SFTStrategy(mock_tokenizer, cfg)
        msg_list = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        example = {"messages": msg_list}
        result = strategy._get_messages(example)
        assert result == msg_list

    def test_multiturn_from_json_string(self, mock_tokenizer, tmp_path):
        cfg = SFTConfig(
            output_dir=str(tmp_path / "sft"),
            quantize=False,
            messages_column="messages",
        )
        strategy = SFTStrategy(mock_tokenizer, cfg)
        msg_list = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        example = {"messages": json.dumps(msg_list)}
        result = strategy._get_messages(example)
        assert result == msg_list


# ── _tokenize_with_completion_masking ────────────────────────────────────────

class TestTokenizeWithCompletionMasking:
    def test_primary_path_labels(self, mock_tokenizer, sft_config):
        """When apply_chat_template returns assistant_masks, labels should be -100
        for non-assistant tokens and real IDs for assistant tokens."""
        sft_config.train_on_completions_only = True

        input_ids = [1, 10, 20, 30, 40, 2]
        assistant_masks = [0, 0, 0, 1, 1, 0]

        mock_tokenizer.apply_chat_template.return_value = {
            "input_ids": input_ids,
            "assistant_masks": assistant_masks,
        }

        strategy = SFTStrategy(mock_tokenizer, sft_config)
        result = strategy._tokenize_with_completion_masking(
            {"prompt": "Hi", "response": "Hello"}
        )

        assert result["input_ids"] == input_ids
        assert result["labels"] == [-100, -100, -100, 30, 40, -100]
        assert result["attention_mask"] == [1] * 6

    def test_fallback_path_labels(self, mock_tokenizer, sft_config):
        """When apply_chat_template raises TypeError for return_assistant_tokens_mask,
        the fallback should produce valid labels."""
        sft_config.train_on_completions_only = True

        # First call (with return_assistant_tokens_mask) raises TypeError
        # Subsequent calls (tokenize=False) return strings
        def side_effect(*args, **kwargs):
            if kwargs.get("return_assistant_tokens_mask"):
                raise TypeError("unexpected keyword argument")
            if kwargs.get("tokenize") is False:
                return "User: Hi\nAssistant: Hello"
            return {"input_ids": [1, 2, 3], "assistant_masks": [0, 0, 1]}

        mock_tokenizer.apply_chat_template.side_effect = side_effect

        # Mock the __call__ for tokenization
        mock_tokenizer.return_value = {
            "input_ids": [1, 10, 20, 30, 40],
            "attention_mask": [1, 1, 1, 1, 1],
        }

        strategy = SFTStrategy(mock_tokenizer, sft_config)
        result = strategy._tokenize_with_completion_masking(
            {"prompt": "Hi", "response": "Hello"}
        )

        assert "input_ids" in result
        assert "labels" in result
        assert "attention_mask" in result

    def test_labels_have_minus_100_and_real_ids(self, tiny_tokenizer, tmp_path):
        """With a real tokenizer via the fallback path, labels should have a mix
        of -100 and real token IDs."""
        cfg = SFTConfig(
            output_dir=str(tmp_path / "sft"),
            quantize=False,
            max_length=64,
            train_on_completions_only=True,
        )
        strategy = SFTStrategy(tiny_tokenizer, cfg)
        example = {"prompt": "What is Python?", "response": "A programming language."}

        # The tiny tokenizer's template lacks {% generation %} markers, so the
        # primary path returns all-zero assistant_masks.  Force the fallback by
        # making the first apply_chat_template call raise TypeError.
        original_act = tiny_tokenizer.apply_chat_template

        def patched_act(*args, **kwargs):
            if kwargs.get("return_assistant_tokens_mask"):
                raise TypeError("not supported")
            return original_act(*args, **kwargs)

        with patch.object(tiny_tokenizer, "apply_chat_template", side_effect=patched_act):
            result = strategy._tokenize_with_completion_masking(example)

        labels = result["labels"]
        has_masked = any(l == -100 for l in labels)
        has_real = any(l != -100 for l in labels)
        assert has_masked, "Expected some labels to be -100 (non-assistant)"
        assert has_real, "Expected some labels to be real token IDs (assistant)"

    def test_truncation_respected(self, tiny_tokenizer, tmp_path):
        cfg = SFTConfig(
            output_dir=str(tmp_path / "sft"),
            quantize=False,
            max_length=32,
            train_on_completions_only=True,
        )
        strategy = SFTStrategy(tiny_tokenizer, cfg)
        # Long example to trigger truncation
        example = {
            "prompt": "Tell me a very long story " * 20,
            "response": "Once upon a time " * 20,
        }
        result = strategy._tokenize_with_completion_masking(example)
        assert len(result["input_ids"]) <= 32


# ── preprocess_data ──────────────────────────────────────────────────────────

class TestPreprocessData:
    def test_standard_path(self, tiny_tokenizer, sft_config, sft_dataset):
        sft_config.train_on_completions_only = False
        strategy = SFTStrategy(tiny_tokenizer, sft_config)
        processed = strategy.preprocess_data(sft_dataset)
        assert "input_ids" in processed.column_names
        # Standard path does NOT produce an explicit labels column
        # (DataCollatorForLanguageModeling creates labels at collation time)

    def test_masking_path(self, tiny_tokenizer, tmp_path, sft_dataset):
        cfg = SFTConfig(
            output_dir=str(tmp_path / "sft"),
            quantize=False,
            max_length=64,
            train_on_completions_only=True,
        )
        strategy = SFTStrategy(tiny_tokenizer, cfg)
        processed = strategy.preprocess_data(sft_dataset)
        assert "input_ids" in processed.column_names
        assert "labels" in processed.column_names

    def test_removes_original_columns(self, tiny_tokenizer, sft_config, sft_dataset):
        strategy = SFTStrategy(tiny_tokenizer, sft_config)
        processed = strategy.preprocess_data(sft_dataset)
        assert "prompt" not in processed.column_names
        assert "response" not in processed.column_names


# ── get_data_collator ────────────────────────────────────────────────────────

class TestGetDataCollator:
    def test_standard_collator(self, mock_tokenizer, sft_config):
        sft_config.train_on_completions_only = False
        strategy = SFTStrategy(mock_tokenizer, sft_config)
        collator = strategy.get_data_collator()
        assert isinstance(collator, DataCollatorForLanguageModeling)

    def test_masking_collator_pads_labels(self, mock_tokenizer, sft_config):
        sft_config.train_on_completions_only = True
        mock_tokenizer.pad_token_id = 0
        strategy = SFTStrategy(mock_tokenizer, sft_config)
        collator = strategy.get_data_collator()

        batch = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, -100, 3]},
            {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [-100, 5]},
        ]
        result = collator(batch)
        # Shorter sequence should be padded with -100 for labels
        assert result["labels"].shape[0] == 2
        assert result["labels"][1, -1].item() == -100  # padding position

    def test_masking_collator_output_shapes(self, mock_tokenizer, sft_config):
        sft_config.train_on_completions_only = True
        mock_tokenizer.pad_token_id = 0
        strategy = SFTStrategy(mock_tokenizer, sft_config)
        collator = strategy.get_data_collator()

        batch = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]},
            {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [4, 5]},
        ]
        result = collator(batch)

        assert set(result.keys()) == {"input_ids", "attention_mask", "labels"}
        seq_len = result["input_ids"].shape[1]
        assert result["attention_mask"].shape[1] == seq_len
        assert result["labels"].shape[1] == seq_len


# ── compute_loss ─────────────────────────────────────────────────────────────

class TestComputeLoss:
    def test_forwards_batch(self, mock_tokenizer, sft_config):
        strategy = SFTStrategy(mock_tokenizer, sft_config)
        model = MagicMock()
        model.device = torch.device("cpu")
        fake_loss = torch.tensor(1.5)

        class FakeOutput:
            loss = fake_loss
        model.return_value = FakeOutput()

        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([[1, 2, 3]]),
        }
        loss = strategy.compute_loss(model, batch)
        assert loss == fake_loss
        model.assert_called_once()

    def test_moves_to_device(self, mock_tokenizer, sft_config):
        strategy = SFTStrategy(mock_tokenizer, sft_config)
        model = MagicMock()
        model.device = torch.device("cpu")

        class FakeOutput:
            loss = torch.tensor(0.5)
        model.return_value = FakeOutput()

        batch = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        strategy.compute_loss(model, batch)
        # model was called; the batch tensors were moved to model.device
        call_kwargs = model.call_args[1]
        for v in call_kwargs.values():
            assert v.device == torch.device("cpu")
