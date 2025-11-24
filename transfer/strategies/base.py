from abc import ABC, abstractmethod
from typing import Any, Dict, List
from transformers import PreTrainedTokenizerBase


class FinetuningStrategy(ABC):
    """Abstract base class for a fine-tuning strategy."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, config: Any):
        self.tokenizer = tokenizer
        self.config = config

    @abstractmethod
    def preprocess_data(self, raw_dataset):
        """Processes the raw dataset into the required format."""
        pass

    @abstractmethod
    def get_data_collator(self):
        """Returns a data collator function."""
        pass

    @abstractmethod
    def compute_loss(self, model, batch):
        """Computes the loss for a given batch."""
        pass
