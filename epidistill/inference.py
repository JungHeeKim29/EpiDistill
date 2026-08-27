"""High-level inference wrapper.

Mirrors the evaluation protocol used for the numbers in the paper: bf16 autocast
plus horizontal-flip test-time augmentation, with the depth and the intrinsics
averaged over the two passes.
"""

from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from .config import EpiDistillConfig
from .model import EpiDistill
from .utils.checkpoint import load_checkpoint

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def normalize_image(image: Union[np.ndarray, torch.Tensor], device="cuda") -> torch.Tensor:
    """uint8 HWC / float CHW / batched tensor -> ImageNet-normalised (B, 3, H, W) float32."""
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    if image.ndim == 3 and image.shape[-1] in (1, 3):     # HWC -> CHW
        image = image.permute(2, 0, 1)
    if image.ndim == 3:
        image = image.unsqueeze(0)
    image = image.to(device).float()
    if image.max() > 1.5:                                  # uint8 range
        image = image / 255.0

    mean = torch.tensor(IMAGENET_MEAN, device=image.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image.device).view(1, 3, 1, 1)
    return (image - mean) / std


class EpiDistillPredictor:
    """Load once, call on images.

    >>> predictor = EpiDistillPredictor.from_pretrained("epidistill_unidepthv2.pt")
    >>> depth, K = predictor(rgb_uint8_hwc)
    """

    def __init__(self, model: EpiDistill, device="cuda"):
        self.model = model.eval().to(device)
        self.device = device
        self.config = model.config

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        config: EpiDistillConfig = None,
        device: str = "cuda",
        strict: bool = True,
    ) -> "EpiDistillPredictor":
        config = config or EpiDistillConfig()
        model = EpiDistill(config, device=torch.device(device))
        load_checkpoint(model, checkpoint_path, strict=strict)
        return cls(model, device=device)

    @torch.no_grad()
    def infer_tensor(self, image: torch.Tensor):
        """image: (B, 3, H, W) ImageNet-normalised.  Returns (depth, K) in float32."""
        cfg = self.config
        dtype = _DTYPES[cfg.autocast_dtype]
        autocast = torch.amp.autocast(
            "cuda", dtype=dtype, enabled=(self.device.startswith("cuda") and dtype != torch.float32)
        )

        with autocast:
            depth, K = self.model(image)
            if cfg.use_hflip_tta:
                depth_f, K_f = self.model(torch.flip(image, [-1]))
                depth = 0.5 * (depth + torch.flip(depth_f, [-1]))
                K = 0.5 * (K + K_f)

        return depth.float(), K.float()

    @torch.no_grad()
    def __call__(self, image, resize_to_input: bool = True):
        """Convenience path for a single numpy/torch image.

        Returns ``(depth, K)`` with depth as (H, W) float32 metres.
        """
        was_numpy = isinstance(image, np.ndarray)
        H0, W0 = (image.shape[:2] if was_numpy and image.ndim == 3 else (None, None))

        tensor = normalize_image(image, device=self.device)
        depth, K = self.infer_tensor(tensor)

        if resize_to_input and H0 is not None and depth.shape[-2:] != (H0, W0):
            depth = F.interpolate(depth, size=(H0, W0), mode="bilinear", align_corners=False)

        depth = depth[0, 0]
        return (depth.cpu().numpy(), K[0].cpu().numpy()) if was_numpy else (depth, K[0])
