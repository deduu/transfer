import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BaseConfig:
    """Base configuration for all fine-tuning tasks."""
    model_name: str = "google/gemma-2b"
    learning_rate: float = 5e-5
    batch_size: int = 1
    num_epochs: int = 30
    max_length: int = 256
    output_dir: str = "./finetuned_model"

    # Add LoRA parameters
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    # Common target modules for Gemma, Llama, Mistral models
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])


@dataclass
class SFTConfig(BaseConfig):
    """Configuration for Supervised Fine-Tuning."""
    # Add any SFT-specific parameters here
    output_dir: str = "./sft_finetuned_model"
    # Add configurable column names with sensible defaults
    prompt_column: str = "prompt"
    response_column: str = "response"

    # You can also add a prompt template for more flexibility
    prompt_template: str = "### Prompt:\n{prompt}\n\n### Response:\n{response}{eos_token}"

    def __post_init__(self):
        # Automatically nest output directory inside BaseConfig.output_dir
        self.output_dir = os.path.join(super().output_dir, self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class DPOConfig(BaseConfig):
    """Configuration for Direct Preference Optimization."""
    # DPO often uses a smaller learning rate
    learning_rate: float = 1e-6
    beta: float = 0.1  # DPO-specific temperature parameter
    # DPO datasets might need longer sequences
    max_length: int = 512
    output_dir: str = "./dpo_finetuned_model"

    # Add configurable column names
    chosen_column: str = "chosen"
    rejected_column: str = "rejected"

    def __post_init__(self):
        # Automatically nest output directory inside BaseConfig.output_dir
        self.output_dir = os.path.join(super().output_dir, self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
