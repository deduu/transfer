# transfer/__init__.py (updated)
from .trainer import Trainer
from .config import BaseConfig, SFTConfig, DPOConfig
from .evaluation import HallucinationDetector, PerplexityMetric, EvaluationMetric

__version__ = "0.1.0"
__all__ = ["Trainer", "BaseConfig", "SFTConfig", "DPOConfig",
           "HallucinationDetector", "PerplexityMetric", "EvaluationMetric"]
