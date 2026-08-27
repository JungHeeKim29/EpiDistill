"""Configuration for EpiDistill.

These are the released single-view settings.  Ablation switches from development
are not exposed: the code implements one architecture -- depth-guided epipolar
attention with rectified stereo tokens, a scale head driven by the scale token,
and a structural residual on the frozen backbone's depth.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class EpiDistillConfig:
    # ── backbone ──────────────────────────────────────────────────────────────
    backbone: str = "unidepthv2"          # 'unidepthv2' | 'depthpro'
    backbone_checkpoint: str = None       # None -> the backbone's own default weights
    depthpro_levels: Tuple[str, ...] = ("latent0", "latent1", "highres", "global")

    # ── token geometry ────────────────────────────────────────────────────────
    token_dim: int = 1024
    patch_size: int = 14
    num_levels: int = 4                   # frame / epipolar layers (Sec. 5.1)

    # ── attention ─────────────────────────────────────────────────────────────
    frame_num_heads: int = 32
    epipolar_num_heads: int = 32
    epipolar_num_samples: int = 16        # M sampled points along the epipolar line
    epipolar_layer_scale: float = 0.1
    rope_frequency: float = 100.0

    # ── rectified stereo tokens (Sec. 4.2) ────────────────────────────────────
    # A learnable grid per level; 518 / 14 = 37 -> 37x37 spatial tokens + 1 CLS slot.
    rst_grid_pixels: int = 518
    rst_num_heads: int = 8

    # ── scale token read-out ──────────────────────────────────────────────────
    scale_num_heads: int = 8

    # ── scale head ────────────────────────────────────────────────────────────
    log_scale_min: float = -3.0
    log_scale_max: float = 6.0

    # ── depth composition (Eq. 1) ─────────────────────────────────────────────
    # Multiplicative:  D_struct = D_base * exp(tau * tanh(Delta d / tau))
    # Additive:        D_struct = D_base + Delta d          (the form printed in Eq. 1)
    # The released weights are trained with the multiplicative form -- see
    # EpiDistill._compose_struct for why. tau <= 0 disables the squash.
    struct_multiplicative: bool = True
    struct_tau: float = 3.0
    backbone_valid_eps: float = 1e-3
    min_depth: float = 0.01

    # ── input conditioning ────────────────────────────────────────────────────
    # UniDepthV2's own shape constraints; DepthPro skips the aspect padding
    # (it is a fixed-resolution stretch-resize network, so any ratio is in-distribution).
    ratio_bounds: Tuple[float, float] = (0.5, 2.5)
    pixels_min: int = 200_000
    pixels_max: int = 600_000

    # ── inference ─────────────────────────────────────────────────────────────
    use_hflip_tta: bool = True
    autocast_dtype: str = "bfloat16"      # 'bfloat16' | 'float32'

    @property
    def rst_grid_size(self) -> int:
        return self.rst_grid_pixels // self.patch_size          # 37

    @property
    def rst_num_tokens(self) -> int:
        return self.rst_grid_size ** 2 + 1                      # 1370
