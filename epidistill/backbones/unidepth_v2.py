import os
import sys

import torch

from .base import DepthBackbone

_THIRD_PARTY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "UniDepth")
)


def _import_unidepth_v2():
    if _THIRD_PARTY not in sys.path:
        sys.path.insert(0, _THIRD_PARTY)
        sys.path.insert(0, os.path.join(_THIRD_PARTY, "unidepth", "models"))
    from unidepthv2 import UniDepthV2  # noqa: E402

    return UniDepthV2


class UniDepthV2Backbone(DepthBackbone):
    patch_size = 14
    token_dim = 1024
    num_levels = 4

    def __init__(
        self,
        device=torch.device("cuda"),
        pretrained: str = "lpiccinelli/unidepth-v2-vitl14",
        checkpoint_path: str = None,
    ):
        UniDepthV2 = _import_unidepth_v2()
        if checkpoint_path is not None:
            self.model = UniDepthV2.from_pretrained(checkpoint_path)
        else:
            self.model = UniDepthV2.from_pretrained(pretrained)

        self.device = device
        self.model.to(device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def encode(self, image, intrinsics=None):
        """image: (B, 3, H, W), ImageNet-normalised.

        ``intrinsics`` conditions UniDepthV2's camera pathway.  EpiDistill leaves it
        ``None`` at inference -- the backbone estimates its own K, and the camera
        head then predicts the residual of Eq. 11.
        """
        inputs = {"image": image}
        if intrinsics is not None:
            inputs["camera"] = intrinsics
        return self.model.encode(inputs, None)

    def decode(self, inputs):
        outputs = self.model.decode(inputs)
        return (
            outputs["depth"],
            outputs["intrinsics"],
            outputs["rays"],
            outputs["confidence"],
        )
