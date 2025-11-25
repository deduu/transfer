import os
import torch
import json
import copy
from tqdm import tqdm
from typing import Dict, Any, Optional, Union
from transformers import PreTrainedModel, PreTrainedTokenizerBase, AutoModelForCausalLM
from torch.utils.data import DataLoader
from datasets import Dataset
from peft import get_peft_model, LoraConfig, PeftModel
from transformers import BitsAndBytesConfig
from .config import BaseConfig, SFTConfig, DPOConfig
from .strategies.sft import SFTStrategy
from .strategies.dpo import DPOStrategy
from .utils import load_model_and_tokenizer
from .evaluation import HallucinationDetector, PerplexityMetric, SemanticEntropyMetric


class Trainer:
    """
    A high-level trainer for fine-tuning Hugging Face models.

    Example:
        >>> from transfer import Trainer, SFTConfig
        >>> config = SFTConfig(model_name="gpt2", output_dir="./gpt2-sft")
        >>> trainer = Trainer(task="sft", config=config)
        >>> trainer.train(dataset=my_dataset)
    """

    def __init__(self, task: str, config: BaseConfig):
        self.task = task.lower()
        self.config = config

        # --- Strategy Registry ---
        self._strategies = {
            'sft': SFTStrategy,
            'dpo': DPOStrategy,
        }

        # ✅ FIXED: Make quantization configurable
        self.is_quantized = getattr(config, 'quantize', True)

        # Setup quantization config if enabled
        if self.is_quantized:
            # Map string dtype to torch dtype
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32
            }
            compute_dtype = dtype_map.get(
                getattr(config, 'quantization_compute_dtype', 'bfloat16'),
                torch.bfloat16
            )

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=getattr(
                    config, 'quantization_type', 'nf4'),
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=getattr(
                    config, 'use_double_quant', True),
            )
            model_dtype = compute_dtype
            print(
                f"✅ Quantization enabled: {getattr(config, 'quantization_type', 'nf4')} with {getattr(config, 'quantization_compute_dtype', 'bfloat16')}")
        else:
            bnb_config = None
            model_dtype = torch.bfloat16
            print("✅ Quantization disabled - loading full precision model")

        if self.task not in self._strategies:
            raise ValueError(
                f"Task '{self.task}' not supported. Choose from {list(self._strategies.keys())}")

        self.model, self.tokenizer = load_model_and_tokenizer(
            self.config.model_name,
            quantization_config=bnb_config,
            dtype=model_dtype
        )

        # --- LoRA LOGIC ---
        if self.config.use_lora:
            print("Applying LoRA adapters...")
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.lora_target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Store base model info for comparison
        self.original_model_state = None
        self.supports_comparison = False  # Will be set in _setup_evaluation
        self.base_model_class = type(
            self.model.base_model if self.config.use_lora else self.model)

        # Instantiate the correct strategy
        strategy_class = self._strategies[self.task]
        self.strategy = strategy_class(
            tokenizer=self.tokenizer, config=self.config)

        # DPO-specific setup
        if self.task == 'dpo':
            if not isinstance(self.config, DPOConfig):
                raise TypeError(
                    "Config must be a DPOConfig for the 'dpo' task.")
            ref_model, _ = load_model_and_tokenizer(
                self.config.model_name,
                dtype=torch.bfloat16
            )
            self.strategy.set_reference_model(ref_model)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.learning_rate)

        # Setup evaluation if enabled
        self.evaluation_detector = None
        if self.config.enable_evaluation:
            self._setup_evaluation()

    def _setup_evaluation(self):
        """Setup the evaluation detector with specified metrics."""
        metrics = {}

        if "perplexity" in self.config.evaluation_metrics:
            metrics["perplexity"] = PerplexityMetric()

        if "semantic_entropy" in self.config.evaluation_metrics:
            metrics["semantic_entropy"] = SemanticEntropyMetric(
                num_samples=getattr(
                    self.config, 'semantic_entropy_samples', 5),
                temperature=getattr(
                    self.config, 'semantic_entropy_temperature', 1.0)
            )

        self.evaluation_detector = HallucinationDetector(metrics)

        # Disable comparison for quantized models
        # Quantized models have complex BitsAndBytes metadata that can't be easily restored
        self.supports_comparison = not self.is_quantized

        if self.config.save_evaluation_results and self.supports_comparison:
            print("Saving original model state for comparison...")
            self.original_model_state = {
                k: v.clone().cpu() for k, v in self.model.state_dict().items()
            }
        else:
            self.original_model_state = None
            if self.is_quantized:
                print("\n" + "="*60)
                print("⚠️  Model comparison disabled for quantized models.")
                print("💡 Use trainer.evaluate_model() before and after training")
                print("   to manually compare results.")
                print("="*60 + "\n")

    def _prepare_evaluation_dataset(self, dataset: Dataset) -> Dataset:
        """
        Prepare the evaluation dataset by tokenizing it.

        Args:
            dataset: The dataset to prepare

        Returns:
            The tokenized dataset
        """
        def tokenize_function(examples):
            # Tokenize the prompts
            tokenized = self.tokenizer(
                examples["prompt"],
                truncation=True,
                max_length=self.config.max_length,
                padding="max_length",
            )

            # Don't return tensors here, let DataLoader handle batching
            return tokenized

        # Remove columns that aren't needed for evaluation
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        # Set format for PyTorch
        tokenized_dataset.set_format(type='torch', columns=[
                                     'input_ids', 'attention_mask'])

        return tokenized_dataset

    def evaluate_model(self, model: Optional[PreTrainedModel] = None,
                       dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Evaluate the model using the configured metrics.

        Args:
            model: The model to evaluate. If None, uses self.model.
            dataset: The dataset to evaluate on. If None, uses self.config.evaluation_dataset.

        Returns:
            Dictionary of metric name to score (empty dict if evaluation fails)
        """
        if not self.config.enable_evaluation or self.evaluation_detector is None:
            print("⚠️  Evaluation is not enabled in config.")
            return {}

        if model is None:
            model = self.model

        was_training = model.training
        model.eval()

        if dataset is None:
            if self.config.evaluation_dataset is None:
                print("⚠️  Evaluation dataset is not specified in config.")
                return {}
            dataset = self.config.evaluation_dataset

        try:
            # Prepare the evaluation dataset
            tokenized_dataset = self._prepare_evaluation_dataset(dataset)

            # Create data loader
            eval_dataloader = DataLoader(
                tokenized_dataset,
                batch_size=self.config.evaluation_batch_size,
                shuffle=False
            )

            # Evaluate
            results = self.evaluation_detector.evaluate(
                model, self.tokenizer, eval_dataloader, self.config
            )

            return results

        except torch.cuda.OutOfMemoryError as e:
            print("\n" + "="*60)
            print("❌ OUT OF MEMORY ERROR")
            print("="*60)
            print(f"Error: {str(e)}")
            print("\n💡 Solutions:")
            print("   1. Reduce evaluation_batch_size (currently: {})".format(
                self.config.evaluation_batch_size))
            print("   2. Reduce semantic_entropy_samples (currently: {})".format(
                getattr(self.config, 'semantic_entropy_samples', 5)))
            print("   3. Use a smaller evaluation dataset")
            print("   4. Close other GPU processes")
            print("="*60 + "\n")

            # Clear GPU memory
            torch.cuda.empty_cache()
            return {}

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("\n" + "="*60)
                print("❌ OUT OF MEMORY ERROR")
                print("="*60)
                print(f"Error: {str(e)}")
                print("\n💡 Solutions:")
                print("   1. Reduce evaluation_batch_size")
                print("   2. Reduce semantic_entropy_samples")
                print("   3. Use a smaller evaluation dataset")
                print("="*60 + "\n")
                torch.cuda.empty_cache()
            else:
                print(f"\n❌ Runtime error during evaluation: {str(e)}")
                import traceback
                traceback.print_exc()
            return {}

        except ValueError as e:
            print(f"\n❌ Value error during evaluation: {str(e)}")
            print("💡 Check your evaluation dataset format and config settings")
            return {}

        except Exception as e:
            print("\n" + "="*60)
            print("❌ UNEXPECTED ERROR DURING EVALUATION")
            print("="*60)
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("\n💡 This might be a bug. Full traceback:")
            import traceback
            traceback.print_exc()
            print("="*60 + "\n")
            return {}
        finally:
            # Restore mode
            if was_training:
                model.train()

    def compare_models(self) -> Dict[str, Dict[str, float]]:
        """
        Compare the original model with the fine-tuned model.

        Note: This method does not support quantized models due to BitsAndBytes
        state dict incompatibility. For quantized models, use evaluate_model()
        separately before and after training.

        Returns:
            Dictionary with "before" and "after" keys containing metric scores
        """
        if not self.config.enable_evaluation or self.evaluation_detector is None:
            print("⚠️  Evaluation is not enabled in config.")
            return {}

        if not self.supports_comparison:
            print("\n" + "="*60)
            print("⚠️  Model comparison not supported for quantized models.")
            print("💡 Tip: Use trainer.evaluate_model() before and after")
            print("   training separately to compare results.")
            print("="*60 + "\n")
            return {}

        if self.original_model_state is None:
            print("⚠️  Original model state is not available for comparison.")
            return {}

        if self.config.evaluation_dataset is None:
            raise ValueError("Evaluation dataset is not specified in config.")

        # This code path should never execute for quantized models
        # But keeping it here for non-quantized models in the future
        print("Loading original model for comparison...")

        # Load the base model again
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        original_model, _ = load_model_and_tokenizer(
            self.config.model_name,
            quantization_config=bnb_config,
            dtype=torch.bfloat16
        )

        # Apply LoRA if it was used
        if self.config.use_lora:
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.lora_target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            original_model = get_peft_model(original_model, lora_config)

        # Load the original state
        device = next(self.model.parameters()).device
        original_state = {k: v.to(device)
                          for k, v in self.original_model_state.items()}
        original_model.load_state_dict(original_state)
        original_model.to(device)

        # Prepare the evaluation dataset
        tokenized_dataset = self._prepare_evaluation_dataset(
            self.config.evaluation_dataset)

        # Create data loader
        eval_dataloader = DataLoader(
            tokenized_dataset,
            batch_size=self.config.evaluation_batch_size,
            shuffle=False
        )

        # Compare models
        print("Comparing models...")
        results = self.evaluation_detector.compare_models(
            original_model, self.model, self.tokenizer, eval_dataloader, self.config
        )

        # Save results if configured
        if self.config.save_evaluation_results:
            os.makedirs(os.path.dirname(
                self.config.evaluation_results_path) or ".", exist_ok=True)
            with open(self.config.evaluation_results_path, "w") as f:
                json.dump(results, f, indent=2)
            print(
                f"Evaluation results saved to {self.config.evaluation_results_path}")

        # Clean up
        del original_model
        torch.cuda.empty_cache()

        return results

    def train(self, dataset: Dataset):
        """Runs the fine-tuning process on the given dataset."""

        try:
            # Preprocess data
            processed_dataset = self.strategy.preprocess_data(dataset)

            # Create data loader
            dataloader = DataLoader(
                processed_dataset,
                batch_size=self.config.batch_size,
                collate_fn=self.strategy.get_data_collator(),
                shuffle=True
            )

            self.model.train()
            for epoch in range(self.config.num_epochs):
                epoch_loss = 0
                progress_bar = tqdm(
                    dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")

                for batch in progress_bar:
                    try:
                        self.optimizer.zero_grad()
                        loss = self.strategy.compute_loss(self.model, batch)
                        loss.backward()
                        self.optimizer.step()

                        epoch_loss += loss.item()
                        progress_bar.set_postfix(loss=loss.item())

                    except torch.cuda.OutOfMemoryError as e:
                        print("\n" + "="*60)
                        print("❌ OUT OF MEMORY ERROR DURING TRAINING")
                        print("="*60)
                        print(f"Error at epoch {epoch+1}")
                        print("\n💡 Solutions:")
                        print("   1. Reduce batch_size (currently: {})".format(
                            self.config.batch_size))
                        print("   2. Reduce max_length (currently: {})".format(
                            self.config.max_length))
                        print("   3. Enable gradient accumulation")
                        print("   4. Enable quantization if disabled")
                        print("="*60 + "\n")
                        torch.cuda.empty_cache()
                        raise  # Re-raise to stop training

                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print(
                                "\n❌ OUT OF MEMORY - Reduce batch_size or max_length")
                            torch.cuda.empty_cache()
                            raise
                        else:
                            print(
                                f"\n❌ Runtime error in training step: {str(e)}")
                            raise

                avg_loss = epoch_loss / len(dataloader)
                print(
                    f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

                # Optional: Run evaluation after each epoch
                if self.config.enable_evaluation and hasattr(self.config, 'evaluate_during_training') and self.config.evaluate_during_training:
                    print(f"\nEvaluating after epoch {epoch+1}...")
                    eval_results = self.evaluate_model()
                    if eval_results:
                        print(f"Epoch {epoch+1} Evaluation: {eval_results}")
                    else:
                        print(
                            f"⚠️  Evaluation skipped or failed for epoch {epoch+1}")

        except KeyboardInterrupt:
            print("\n" + "="*60)
            print("⚠️  Training interrupted by user")
            print("💡 You can still save the model with trainer.save_model()")
            print("="*60 + "\n")
            raise

        except Exception as e:
            print("\n" + "="*60)
            print("❌ TRAINING FAILED")
            print("="*60)
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            import traceback
            traceback.print_exc()
            print("="*60 + "\n")
            raise

    def save_model(self):
        """Saves the fine-tuned model and tokenizer."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        if self.config.use_lora:
            # Save only LoRA adapters
            self.model.save_pretrained(self.config.output_dir)
        else:
            # Save full model
            self.model.save_pretrained(self.config.output_dir)

        self.tokenizer.save_pretrained(self.config.output_dir)

        # Check if we should attempt comparison
        if self.config.enable_evaluation:
            # Only try comparison if model supports it AND we have original state
            if self.supports_comparison and self.original_model_state is not None:
                print("\nRunning model comparison...")
                comparison_results = self.compare_models()
                print("\n=== Evaluation Results ===")
                print(json.dumps(comparison_results, indent=2))
            else:
                # Provide helpful message
                if self.is_quantized:
                    print("\n" + "="*60)
                    print("⚠️  Skipping automatic model comparison")
                    print("    (not supported for quantized models)")
                    print()
                    print("💡 To compare before/after performance:")
                    print("   1. Run trainer.evaluate_model() BEFORE training")
                    print("   2. Train the model")
                    print("   3. Run trainer.evaluate_model() AFTER training")
                    print("   4. Compare the results manually")
                    print("="*60)
                else:
                    print("\n⚠️  Skipping model comparison (original state not saved)")

        print(f"\n✅ Model saved to {self.config.output_dir}")
