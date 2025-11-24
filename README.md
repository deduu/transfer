# Transfer

A modular and extensible PyTorch framework for fine-tuning Hugging Face models.

Transfer simplifies the process of applying various fine-tuning techniques like Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) to any Hugging Face causal language model.

## Installation

You can install `transfer` directly from pip (once published) or from a local source.

### From Source

1.  Clone the repository:
    ```bash
    git clone https://github.com/deduu/transfer.git
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

### 2. Direct Preference Optimization (DPO)

```python
from datasets import load_dataset
from transfer import Trainer, DPOConfig

# 1. Load a preference dataset
dataset = load_dataset("Anthropic/hh-rlhf", split="train[:1%]")

# 2. Define your DPO configuration
config = DPOConfig(
    model_name="google/gemma-2b",
    num_epochs=1,
    batch_size=2,
    beta=0.1,
    output_dir="./gemma-dpo"
)

# 3. Create and run the trainer
trainer = Trainer(task="dpo", config=config)
trainer.train(dataset)
trainer.save_model()
```

## Features

Modular Design: Easily add new fine-tuning strategies by creating a new class that inherits from FinetuningStrategy.
Simple API: A clean Trainer class abstracts away the complex training loops.
Type-Safe Configuration: Uses dataclasses for robust and readable configuration management.
Hugging Face Integration: Seamlessly works with any model or dataset from the Hugging Face Hub.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

### 4. How to Build and Install Your Library Locally

1.  Navigate to the root of your `transfer/` project directory in your terminal.
2.  Make sure you have a recent version of `pip` and `setuptools`.
3.  Run the install command in "editable" mode. This is great for development, as any changes you make to the code are immediately reflected without needing to reinstall.

    ```bash
    pip install -e .
    ```

You have now successfully created and installed your `transfer` library! You can import it in any Python script on your system:
