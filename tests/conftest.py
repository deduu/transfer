import json
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from datasets import Dataset
from transfer.config import SFTConfig, DPOConfig


# ---------------------------------------------------------------------------
# Tiny real tokenizer (session-scoped, ~1 MB download)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def tiny_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM"
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ---------------------------------------------------------------------------
# Mock tokenizer (function-scoped so tests can mutate freely)
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.pad_token = "<pad>"
    tok.eos_token = "</s>"
    tok.eos_token_id = 2
    tok.model_max_length = 512
    return tok


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
@pytest.fixture
def sft_dataset():
    return Dataset.from_dict({
        "prompt": [
            "What is Python?",
            "Explain recursion.",
            "What is 2+2?",
        ],
        "response": [
            "Python is a programming language.",
            "Recursion is when a function calls itself.",
            "2+2 is 4.",
        ],
    })


@pytest.fixture
def multiturn_dataset():
    conversations = [
        json.dumps([
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm fine, thanks!"},
        ]),
        json.dumps([
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is artificial intelligence."},
        ]),
    ]
    return Dataset.from_dict({"messages": conversations})


@pytest.fixture
def dpo_dataset():
    return Dataset.from_dict({
        "chosen": [
            "Python is a popular programming language.",
            "Recursion is a function calling itself.",
        ],
        "rejected": [
            "Python is a snake.",
            "Recursion is looping.",
        ],
    })


# ---------------------------------------------------------------------------
# Configs (use tmp_path so __post_init__ can create directories)
# ---------------------------------------------------------------------------
@pytest.fixture
def sft_config(tmp_path):
    return SFTConfig(
        output_dir=str(tmp_path / "sft_out"),
        quantize=False,
        max_length=64,
        num_epochs=1,
        batch_size=2,
    )


@pytest.fixture
def dpo_config(tmp_path):
    return DPOConfig(
        output_dir=str(tmp_path / "dpo_out"),
        quantize=False,
        max_length=64,
        num_epochs=1,
        batch_size=2,
    )


# ---------------------------------------------------------------------------
# FakeModel — lightweight nn.Module with real parameters for AdamW
# ---------------------------------------------------------------------------
class FakeModel(nn.Module):
    """Minimal model that satisfies the Trainer's expectations."""

    def __init__(self, vocab_size=32, hidden_size=8):
        super().__init__()
        self.linear = nn.Linear(hidden_size, vocab_size)
        self._device = torch.device("cpu")

    @property
    def device(self):
        return self._device

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        # Produce dummy logits
        logits = torch.randn(batch_size, seq_len, self.linear.out_features)

        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        # Return a namespace-like object
        return _Outputs(loss=loss, logits=logits)

    def save_pretrained(self, path):
        pass  # no-op for tests

    def train(self, mode=True):
        return super().train(mode)


class _Outputs:
    """Simple namespace to mimic HuggingFace model outputs."""
    def __init__(self, loss=None, logits=None):
        self.loss = loss
        self.logits = logits


@pytest.fixture
def fake_model():
    return FakeModel()
