"""DepthPro backbone.

DepthPro is a fixed 1536x1536 network reached by a STRETCH resize (exactly what
native ``DepthPro.infer`` does), and it reports

    f_px = 0.5 * W / tan(fov / 2)        inverse_depth = canonical * (W / f_px)

so ``W`` cancels: the metric depth is invariant to any aspect-preserving resize of
the input and depends on the aspect ratio only through the stretch.  That is why
the feature maps can be resampled onto EpiDistill's (H//14, W//14) token grid with
no crop and no padding -- the 1536x1536 buffer covers the input uniformly, so no
pixel in the backbone's field of view is fabricated.

The four exposed levels, ordered EARLY -> LATE to match UniDepthV2's block order:

    latent0 : ViT block 5,  5x5 sliding-window high-res stream  -> 96x96
    latent1 : ViT block 11, high-res stream                     -> 96x96
    highres : ViT block 23, high-res stream                     -> 96x96
    global  : image encoder (whole image at 384), block 23      -> 24x24

The first three carry spatial detail with a local receptive field (each 384 crop
sees 1/25 of the image); ``global`` is the only level with a genuinely global
receptive field, and it is the one whose class token the camera and scale heads
read.
"""

import dataclasses
import os
import sys

import torch
import torch.nn.functional as F

from .base import DepthBackbone

_THIRD_PARTY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "third_party")
)

# ImageNet statistics the EpiDistill input pipeline normalises with; undone here
# before applying DepthPro's own [-1, 1] normalisation.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_LEVELS = ("latent0", "latent1", "highres", "global")


def _import_depth_pro():
    if _THIRD_PARTY not in sys.path:
        sys.path.insert(0, _THIRD_PARTY)
    from DepthPro.depth_pro import (  # noqa: E402
        DEFAULT_MONODEPTH_CONFIG_DICT,
        create_model_and_transforms,
    )

    return create_model_and_transforms, DEFAULT_MONODEPTH_CONFIG_DICT


class DepthProBackbone(DepthBackbone):
    patch_size = 14
    token_dim = 1024
    num_levels = 4

    def __init__(
        self,
        device=torch.device("cuda"),
        checkpoint_path: str = None,
        target_size: int = 1536,
        patch_size: int = 14,
        levels=DEFAULT_LEVELS,
    ):
        create_model_and_transforms, default_config = _import_depth_pro()

        self.device = device
        self.target_size = target_size
        self.patch_size = patch_size
        self.levels = tuple(levels)
        assert len(self.levels) == self.num_levels, (
            f"DepthProBackbone needs exactly {self.num_levels} levels "
            f"(the attention stack, the RST grid and the DPT head are built for it); "
            f"got {self.levels}"
        )

        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                _THIRD_PARTY, "DepthPro", "checkpoints", "depth_pro.pt"
            )
        # dataclasses.replace, not in-place mutation: the default config is a shared
        # module-level instance.
        config = dataclasses.replace(default_config, checkpoint_uri=checkpoint_path)

        self.model, self.transform = create_model_and_transforms(
            config=config, device=device, precision=torch.float32
        )
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    # ── input conditioning ────────────────────────────────────────────────────
    def _to_network_input(self, image):
        mean = torch.tensor(_IMAGENET_MEAN, device=image.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=image.device).view(1, 3, 1, 1)
        image_01 = torch.clamp(image * std + mean, 0.0, 1.0)
        image_norm = (image_01 - 0.5) / 0.5
        return F.interpolate(
            image_norm,
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False,
        )

    def _levels_from_aux(self, aux, patch_h, patch_w):
        features, tokens = [], []
        for name in self.levels:
            feat = aux["feat_" + name]                                   # (B, 1024, h, w)
            feat = F.interpolate(
                feat, size=(patch_h, patch_w), mode="bilinear", align_corners=False
            )
            features.append(feat.permute(0, 2, 3, 1).contiguous())       # (B, Hp, Wp, 1024)
            tokens.append(aux["cls_" + name].unsqueeze(1).contiguous())  # (B, 1, 1024)
        return features, tokens

    def _metric_depth(self, canonical_inverse_depth, fov_deg, H, W):
        """Canonical inverse depth -> metric depth, in native ``infer``'s operation order
        (interpolate the INVERSE depth, then reciprocate)."""
        f_px = 0.5 * W / torch.tan(0.5 * torch.deg2rad(fov_deg.to(torch.float)))
        inverse_depth = canonical_inverse_depth * (W / f_px)
        inverse_depth = F.interpolate(
            inverse_depth, size=(H, W), mode="bilinear", align_corners=False
        )
        depth = 1.0 / torch.clamp(inverse_depth, min=1e-4, max=1e4)

        B = depth.shape[0]
        f_px_flat = f_px.reshape(B)
        intrinsics = torch.eye(3, device=depth.device, dtype=depth.dtype)
        intrinsics = intrinsics.unsqueeze(0).repeat(B, 1, 1)
        intrinsics[:, 0, 0] = f_px_flat
        intrinsics[:, 1, 1] = f_px_flat
        intrinsics[:, 0, 2] = W / 2.0
        intrinsics[:, 1, 2] = H / 2.0
        return depth, intrinsics

    # ── contract ──────────────────────────────────────────────────────────────
    def encode(self, image, intrinsics=None, decode_depth: bool = True):
        """image: (B, 3, H, W), ImageNet-normalised.

        ``intrinsics`` is accepted for signature parity and stored, not consumed:
        DepthPro has no camera-conditioning input.

        ``decode_depth=False`` skips the conv pyramid and the FOV pass, which is
        what multi-view training uses for source views (features only).
        """
        B, _, H, W = image.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        # fp32 throughout: DepthPro was trained in fp32 and its depth conversion
        # (1/x, tan) is precision sensitive, so the base depth must not move
        # between an autocast context and a plain one.
        with torch.amp.autocast("cuda", enabled=False), torch.no_grad():
            image_net = self._to_network_input(image.float())

            if decode_depth:
                canonical_inverse_depth, fov_deg, aux = self.model.forward_aux(image_net)
                depth, pred_intrinsics = self._metric_depth(
                    canonical_inverse_depth, fov_deg, H, W
                )
                del canonical_inverse_depth
            else:
                _encodings, aux = self.model.encoder(image_net)
                del _encodings
                depth, pred_intrinsics = None, None

            features, tokens = self._levels_from_aux(aux, patch_h, patch_w)
            del aux

        return {
            "image": image,
            "original_shape": (H, W),
            "features": features,
            "tokens": tokens,
            "depth": depth,
            "intrinsics": pred_intrinsics,
            # DepthPro emits neither camera rays nor a confidence map; the slots
            # exist for parity with the UniDepthV2 backbone.
            "rays": None,
            "confidence": None,
            "gt_intrinsics": intrinsics,
        }

    def decode(self, inputs):
        """Returns the depth already computed by ``encode``.

        The decode work happens inside ``encode`` on purpose: it needs the 5-level
        conv pyramid, by far the largest tensor the backbone produces, and consuming
        it there lets it be freed before the caller's own activations are allocated.
        """
        if inputs.get("depth", None) is None:
            raise RuntimeError(
                "decode() called on an encode(decode_depth=False) result -- that path "
                "returns features only."
            )
        return (
            inputs["depth"],
            inputs["intrinsics"],
            inputs["rays"],
            inputs["confidence"],
        )
