# Transfer

A modular and extensible PyTorch framework for fine-tuning Hugging Face models.

Transfer simplifies the process of applying various fine-tuning techniques like Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) to any Hugging Face causal language model.

## Installation

You can install `transfer` directly from pip (once published) or from a local source.

### From Source

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/transfer.git
    cd transfer
    ```
2.  Install the package:
    ```bash
    pip install .
    ```

## Quick Start

Using `transfer` is designed to be simple and intuitive.

### 1. Supervised Fine-Tuning (SFT)

```python
from datasets import load_dataset
from transfer import Trainer, SFTConfig

# 1. Load your data
# For this example, we create a dummy dataset
data = {
    "prompt": ["What is the capital of France?", "Explain the theory of relativity."],
    "response": ["The capital of France is Paris.", "The theory of relativity is..."]
}
dataset = Dataset.from_dict(data)

# 2. Define your configuration
config = SFTConfig(
    model_name="google/gemma-2b",
    num_epochs=3,
    batch_size=2,
    learning_rate=5e-5,
    output_dir="./gemma-sft"
)

# 3. Create and run the trainer
trainer = Trainer(task="sft", config=config)
trainer.train(dataset)
trainer.save_model()
```
