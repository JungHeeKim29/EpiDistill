"""Smoke test: build EpiDistill on a stub backbone and run a single-view forward.

Runs on CPU in a few seconds and needs no backbone weights, so it exercises the
whole EpiDistill path -- token plumbing, rectified epipolar attention, the three
heads and the depth composition -- independently of UniDepthV2 / DepthPro.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import epidistill.model as model_mod  # noqa: E402
from epidistill.backbones.base import DepthBackbone  # noqa: E402
from epidistill.config import EpiDistillConfig  # noqa: E402


class StubBackbone(DepthBackbone):
    """Deterministic stand-in with the same contract as the real backbones."""

    patch_size = 14
    token_dim = 1024
    num_levels = 4

    def __init__(self, device=torch.device("cpu"), **kwargs):
        self.device = device
        self.model = torch.nn.Identity()

    def encode(self, image, intrinsics=None):
        B, _, H, W = image.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size
        g = torch.Generator().manual_seed(0)
        return {
            "image": image,
            "features": [
                torch.randn(B, Hp, Wp, self.token_dim, generator=g) for _ in range(self.num_levels)
            ],
            "tokens": [
                torch.randn(B, 1, self.token_dim, generator=g) for _ in range(self.num_levels)
            ],
            # DepthPro's fov encoder: 24x24 patches + CLS at 1024-d
            "fov_tokens": torch.randn(B, 1 + 24 * 24, self.token_dim, generator=g),
            "shape": (H, W),
        }

    def decode(self, inputs):
        B, _, H, W = inputs["image"].shape
        depth = torch.linspace(1.0, 20.0, H * W).view(1, 1, H, W).expand(B, -1, -1, -1).clone()
        K = torch.eye(3).unsqueeze(0).repeat(B, 1, 1)
        K[:, 0, 0] = K[:, 1, 1] = 500.0
        K[:, 0, 2], K[:, 1, 2] = W / 2, H / 2
        return depth, K, None, None

    def to(self, device):
        self.device = device
        return self


def test_single_view_forward(backbone="unidepthv2"):
    model_mod.build_backbone = lambda name, **kwargs: StubBackbone(**kwargs)

    torch.manual_seed(0)
    config = EpiDistillConfig(backbone=backbone)
    model = model_mod.EpiDistill(config, device=torch.device("cpu")).eval()

    image = torch.randn(1, 3, 476, 630)
    with torch.no_grad():
        depth, K = model(image)

    assert depth.shape == (1, 1, 476, 630), depth.shape
    assert K.shape == (1, 3, 3), K.shape
    assert torch.isfinite(depth).all(), "non-finite depth"
    assert (depth > 0).all(), "non-positive depth"
    assert torch.isfinite(K).all(), "non-finite intrinsics"
    return depth, K


def test_zero_init_preserves_backbone_structure(backbone="unidepthv2"):
    """Every head is zero-init, so an untrained model must reproduce the backbone's
    STRUCTURE exactly -- the prediction differs from it by one constant per image
    (the median normalisation times s_hat).  On the DepthPro variant s_hat starts
    at the backbone's own scale, so the prediction must equal the backbone depth."""
    model_mod.build_backbone = lambda name, **kwargs: StubBackbone(**kwargs)

    torch.manual_seed(0)
    model = model_mod.EpiDistill(EpiDistillConfig(backbone=backbone), device=torch.device("cpu")).eval()

    image = torch.randn(1, 3, 476, 630)
    backbone_depth = model.backbone.decode(model.backbone.encode(image))[0]

    with torch.no_grad():
        depth, _ = model(image)

    # D_adj = (D_base / _sm) * s_hat with Delta d == 0, so the ratio to the backbone
    # depth must be a single constant per image (s_hat / median(D_orig)).
    ratio = (depth / backbone_depth).flatten()
    spread = (ratio.max() - ratio.min()) / ratio.mean()
    assert spread < 1e-3, f"structure is not the backbone's at init (spread {spread:.2e})"
    if backbone == "depthpro":
        assert abs(ratio.mean().item() - 1.0) < 1e-3, "DepthPro variant must start at the backbone's scale"
    return ratio.mean().item()


if __name__ == "__main__":
    for backbone in ("unidepthv2", "depthpro"):
        depth, K = test_single_view_forward(backbone)
        print(f"[{backbone}] forward ok: depth {tuple(depth.shape)} "
              f"[{depth.min():.3f}, {depth.max():.3f}] m, fx={K[0, 0, 0]:.1f}")
        scale = test_zero_init_preserves_backbone_structure(backbone)
        print(f"[{backbone}] zero-init structure ok: ratio to backbone is the constant {scale:.4f}")
