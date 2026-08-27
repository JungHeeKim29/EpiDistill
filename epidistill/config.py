"""Configuration for EpiDistill.

Defaults reproduce the released single-view recipe: depth-guided epipolar
attention with rectified stereo tokens, a scale head driven by the scale token
alone, and a multiplicative structural residual on the frozen backbone's depth.
"""

from dataclasses import dataclass, field
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
    # Two role-bound slots: slot 0 reads the reference view, slot 1 reads the source
    # views (multi-view training) or T_rst-ref (single-view inference).
    scale_ref_src_tokens: bool = True
    # Slot 0's key/value set is every view, not the reference alone.
    scale_ref_reads_all_views: bool = True
    # Each level gets an additive learnable seed on top of the carried-over state.
    scale_level_seed: bool = True

    # ── scale head ────────────────────────────────────────────────────────────
    scale_from_scratch: bool = True       # predict log s_hat directly (no backbone anchor)
    scale_use_depth_stats: bool = False
    scale_use_focal_norm: bool = False
    scale_use_cls: bool = False
    log_scale_min: float = -3.0
    log_scale_max: float = 6.0

    # ── depth composition (Eq. 1) ─────────────────────────────────────────────
    # D_struct = D_base * exp(tau * tanh(Delta d / tau)); tau <= 0 disables the squash.
    struct_multiplicative: bool = True
    struct_tau: float = 3.0
    # Normalise D_struct over the pixels the backbone itself declared valid.
    struct_norm_masked: bool = True
    backbone_valid_eps: float = 1e-3

    # ── input conditioning ────────────────────────────────────────────────────
    # UniDepthV2's own shape constraints; DepthPro skips the aspect padding
    # (it is a fixed-resolution stretch-resize network, so any ratio is in-distribution).
    ratio_bounds: Tuple[float, float] = (0.5, 2.5)
    pixels_min: int = 200_000
    pixels_max: int = 600_000
    min_depth: float = 0.01

    # ── inference ─────────────────────────────────────────────────────────────
    use_hflip_tta: bool = True
    autocast_dtype: str = "bfloat16"      # 'bfloat16' | 'float32'

    @property
    def rst_grid_size(self) -> int:
        return self.rst_grid_pixels // self.patch_size          # 37

    @property
    def rst_num_tokens(self) -> int:
        return self.rst_grid_size ** 2 + 1                      # 1370
