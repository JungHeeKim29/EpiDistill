"""EpiDistill -- Geometric Distillation from Rectified Stereo."""

from .config import EpiDistillConfig
from .inference import EpiDistillPredictor, normalize_image
from .model import EpiDistill
from .utils.checkpoint import load_checkpoint

__version__ = "0.1.0"

__all__ = [
    "EpiDistill",
    "EpiDistillConfig",
    "EpiDistillPredictor",
    "normalize_image",
    "load_checkpoint",
    "__version__",
]
