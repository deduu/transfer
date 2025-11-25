import torch
import numpy as np
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from scipy.special import softmax


class EvaluationMetric(ABC):
    """Abstract base class for evaluation metrics."""

    @abstractmethod
    def compute(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                dataset: DataLoader, config: Any) -> Dict[str, float]:
        """Computes the metric on the given dataset."""
        pass


class PerplexityMetric(EvaluationMetric):
    """Computes perplexity of a model on a given dataset."""

    def compute(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                dataset: DataLoader, config: Any) -> Dict[str, float]:
        """
        Compute perplexity on the dataset.

        Args:
            model: The model to evaluate
            tokenizer: The tokenizer
            dataset: The dataset to evaluate on
            config: Configuration object

        Returns:
            Dictionary containing the perplexity score
        """
        model.eval()
        total_loss = 0.0
        total_batches = 0

        device = next(model.parameters()).device

        with torch.no_grad():
            for batch in tqdm(dataset, desc="Computing perplexity"):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}

                # Get model outputs - loss is already averaged
                outputs = model(**batch, labels=batch["input_ids"])

                # ✅ FIXED: Use pre-averaged loss correctly
                total_loss += outputs.loss.item()
                total_batches += 1

        # Calculate perplexity
        avg_loss = total_loss / total_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return {"perplexity": perplexity}


class SemanticEntropyMetric(EvaluationMetric):
    """
    Computes semantic entropy to detect hallucination via response uncertainty.

    This metric works by:
    1. Generating multiple responses for the same prompt
    2. Clustering semantically similar responses
    3. Computing entropy over the cluster distribution

    High entropy indicates the model is uncertain and more likely to hallucinate.
    """

    def __init__(self, num_samples: int = 5, temperature: float = 1.0,
                 max_new_tokens: int = 100, eps: float = 0.3, min_samples: int = 2):
        """
        Initialize semantic entropy metric.

        Args:
            num_samples: Number of responses to generate per prompt
            temperature: Sampling temperature for generation
            max_new_tokens: Maximum tokens to generate
            eps: DBSCAN epsilon parameter for clustering
            min_samples: DBSCAN minimum samples parameter
        """
        self.num_samples = num_samples
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.eps = eps
        self.min_samples = min_samples

        # Load sentence transformer for semantic similarity
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def _generate_multiple_responses(self, model: PreTrainedModel,
                                     tokenizer: PreTrainedTokenizerBase,
                                     input_ids: torch.Tensor,
                                     attention_mask: torch.Tensor) -> List[str]:
        """Generate multiple responses for a given input."""
        device = input_ids.device

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=self.num_samples,
                do_sample=True,
                temperature=self.temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_length = input_ids.shape[1]
        responses = []
        for output in outputs:
            response = tokenizer.decode(
                output[prompt_length:],
                skip_special_tokens=True
            )
            responses.append(response)

        return responses

    def _cluster_responses(self, responses: List[str]) -> np.ndarray:
        """
        Cluster semantically similar responses using DBSCAN.

        Returns:
            Cluster labels (-1 for noise points)
        """
        if len(responses) < 2:
            return np.array([0])

        # Get embeddings
        embeddings = self.embedder.encode(responses)

        # Cluster using DBSCAN
        clustering = DBSCAN(
            eps=self.eps, min_samples=self.min_samples, metric='cosine')
        labels = clustering.fit_predict(embeddings)

        # Handle noise points by assigning each to its own cluster
        noise_mask = labels == -1
        if noise_mask.any():
            max_label = labels.max() if labels.max() >= 0 else -1
            noise_indices = np.where(noise_mask)[0]
            for i, idx in enumerate(noise_indices):
                labels[idx] = max_label + 1 + i

        return labels

    def _compute_entropy(self, labels: np.ndarray) -> float:
        """Compute entropy over cluster distribution."""
        unique_labels, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        return entropy

    def compute(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                dataset: DataLoader, config: Any) -> Dict[str, float]:
        """
        Compute semantic entropy on the dataset.

        Args:
            model: The model to evaluate
            tokenizer: The tokenizer
            dataset: The dataset to evaluate on
            config: Configuration object

        Returns:
            Dictionary containing the semantic entropy score
        """
        model.eval()
        entropies = []
        num_clusters_list = []

        device = next(model.parameters()).device

        with torch.no_grad():
            for batch in tqdm(dataset, desc="Computing semantic entropy"):
                # Process each item in the batch
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                batch_size = input_ids.shape[0]

                for i in range(batch_size):
                    # Get single input
                    single_input_ids = input_ids[i:i+1]
                    single_attention_mask = attention_mask[i:i+1]

                    # Generate multiple responses
                    responses = self._generate_multiple_responses(
                        model, tokenizer, single_input_ids, single_attention_mask
                    )

                    # Skip if all responses are empty
                    responses = [r for r in responses if r.strip()]
                    if len(responses) < 2:
                        continue

                    # Cluster responses
                    labels = self._cluster_responses(responses)

                    # Compute entropy
                    entropy = self._compute_entropy(labels)
                    entropies.append(entropy)

                    # Track number of unique clusters
                    num_clusters = len(np.unique(labels))
                    num_clusters_list.append(num_clusters)

        if not entropies:
            return {
                "semantic_entropy": float('nan'),
                "avg_num_clusters": float('nan')
            }

        return {
            "semantic_entropy": float(np.mean(entropies)),
            "semantic_entropy_std": float(np.std(entropies)),
            "avg_num_clusters": float(np.mean(num_clusters_list))
        }


class TokenEntropyMetric(EvaluationMetric):
    """
    Computes average token-level entropy as a measure of model uncertainty.
    High entropy indicates the model is uncertain about its predictions.
    """

    def compute(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                dataset: DataLoader, config: Any) -> Dict[str, float]:
        """
        Compute token-level entropy on the dataset.

        Args:
            model: The model to evaluate
            tokenizer: The tokenizer
            dataset: The dataset to evaluate on
            config: Configuration object

        Returns:
            Dictionary containing the token entropy score
        """
        model.eval()
        total_entropy = 0.0
        total_tokens = 0

        device = next(model.parameters()).device

        with torch.no_grad():
            for batch in tqdm(dataset, desc="Computing token entropy"):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}

                # Get model outputs
                outputs = model(**batch, labels=batch["input_ids"])
                logits = outputs.logits

                # Compute probabilities
                probs = torch.softmax(logits, dim=-1)

                # Compute entropy: -sum(p * log(p))
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

                # Mask padding tokens
                if "attention_mask" in batch:
                    mask = batch["attention_mask"]
                    entropy = entropy * mask
                    tokens = mask.sum().item()
                else:
                    tokens = entropy.numel()

                total_entropy += entropy.sum().item()
                total_tokens += tokens

        avg_entropy = total_entropy / total_tokens if total_tokens > 0 else 0.0

        return {"token_entropy": avg_entropy}


class HallucinationDetector:
    """Class to detect hallucinations using various metrics."""

    def __init__(self, metrics: Dict[str, EvaluationMetric]):
        """
        Initialize the detector with specified metrics.

        Args:
            metrics: Dictionary of metric name to metric instance
        """
        self.metrics = metrics

    def evaluate(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                 dataset: DataLoader, config: Any) -> Dict[str, float]:
        """
        Evaluate the model using all registered metrics.

        Args:
            model: The model to evaluate
            tokenizer: The tokenizer
            dataset: The dataset to evaluate on
            config: Configuration object

        Returns:
            Dictionary of metric name to score
        """
        results = {}
        torch.manual_seed(42)
        np.random.seed(42)

        for name, metric in self.metrics.items():
            print(f"\nComputing {name}...")
            metric_results = metric.compute(model, tokenizer, dataset, config)
            results.update(metric_results)

        return results

    def compare_models(self, model_before: PreTrainedModel, model_after: PreTrainedModel,
                       tokenizer: PreTrainedTokenizerBase, dataset: DataLoader,
                       config: Any) -> Dict[str, Any]:
        """
        Compare two models using all registered metrics.

        Args:
            model_before: The model before fine-tuning
            model_after: The model after fine-tuning
            tokenizer: The tokenizer
            dataset: The dataset to evaluate on
            config: Configuration object

        Returns:
            Dictionary with "before" and "after" keys containing metric scores
        """
        print("\nEvaluating model BEFORE fine-tuning...")
        results = {
            "before": self.evaluate(model_before, tokenizer, dataset, config),
        }

        print("\nEvaluating model AFTER fine-tuning...")
        results["after"] = self.evaluate(
            model_after, tokenizer, dataset, config)

        # Calculate improvement for each metric
        print("\nCalculating improvements...")
        for metric_name in results["before"]:
            before_val = results["before"][metric_name]
            after_val = results["after"][metric_name]

            # Skip if values are NaN
            if np.isnan(before_val) or np.isnan(after_val):
                results[f"{metric_name}_improvement"] = float('nan')
                continue

            # For perplexity, entropy, lower is better
            if metric_name in ["perplexity", "semantic_entropy", "token_entropy"]:
                improvement = ((before_val - after_val) /
                               before_val * 100) if before_val != 0 else 0
            else:
                # For other metrics, assume higher is better
                improvement = ((after_val - before_val) /
                               before_val * 100) if before_val != 0 else 0

            results[f"{metric_name}_improvement_%"] = improvement

        return results
