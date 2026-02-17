from .trainer import Trainer
from .config import BaseConfig, SFTConfig, DPOConfig
from .evaluation import (
    EvaluationMetric,
    HallucinationDetector,
    PerplexityMetric,
    SemanticEntropyMetric,
    TokenEntropyMetric,
)

__version__ = "0.1.0"
__all__ = [
    "Trainer",
    "BaseConfig",
    "SFTConfig",
    "DPOConfig",
    "EvaluationMetric",
    "HallucinationDetector",
    "PerplexityMetric",
    "SemanticEntropyMetric",
    "TokenEntropyMetric",
]
