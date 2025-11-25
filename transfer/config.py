import os
from dataclasses import dataclass, field
from typing import Optional, Any, List


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

    # Quantization settings
    quantize: bool = True  # Enable/disable 4-bit quantization
    quantization_type: str = "nf4"  # "nf4" or "fp4"
    # "bfloat16", "float16", or "float32"
    quantization_compute_dtype: str = "bfloat16"
    use_double_quant: bool = True  # Enable nested quantization

    # Evaluation parameters
    enable_evaluation: bool = False
    evaluation_dataset: Optional[Any] = None
    evaluation_metrics: List[str] = field(
        default_factory=lambda: ["perplexity"])
    evaluation_batch_size: int = 8
    save_evaluation_results: bool = True
    evaluation_results_path: str = "evaluation_results.json"  # relative path now
    evaluate_during_training: bool = False  # Evaluate after each epoch
    # Semantic entropy specific settings
    semantic_entropy_samples: int = 5  # Number of responses to generate
    semantic_entropy_temperature: float = 1.0  # Sampling temperature
    semantic_entropy_max_tokens: int = 100  # Max tokens per response
    semantic_entropy_eps: float = 0.3  # DBSCAN clustering epsilon
    semantic_entropy_min_samples: int = 2  # DBSCAN min samples

    # Memory and device settings
    max_memory: Optional[dict] = None  # e.g., {0: "10GB", "cpu": "30GB"}
    device_map: str = "auto"  # "auto", "cuda:0", etc.

    def update_paths(self):
        # after output_dir finalized, update evaluation location
        self.evaluation_results_path = os.path.join(
            self.output_dir, self.evaluation_results_path)


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

        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.quantization_type not in ["nf4", "fp4"]:
            raise ValueError("quantization_type must be 'nf4' or 'fp4'")
        if self.quantization_compute_dtype not in ["bfloat16", "float16", "float32"]:
            raise ValueError(
                "quantization_compute_dtype must be 'bfloat16', 'float16', or 'float32'")

        self.output_dir = os.path.join(super().output_dir, self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        # Update evaluation result path relative to output_dir
        self.update_paths()


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

        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.quantization_type not in ["nf4", "fp4"]:
            raise ValueError("quantization_type must be 'nf4' or 'fp4'")

        self.output_dir = os.path.join(super().output_dir, self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        # Update evaluation result path relative to output_dir
        self.update_paths()
