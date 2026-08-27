"""The contract EpiDistill expects from a frozen monocular depth backbone.

EpiDistill never fine-tunes the backbone; it only reads its tokens and corrects
its depth (Eq. 1).  A backbone therefore has to expose exactly two calls:

    encode(image)  -> dict with
                        'features' : list of L tensors, (B, H//p, W//p, C)
                        'tokens'   : list of L tensors, (B, 1, C)   -- one CLS per level
    decode(inputs) -> (depth (B,1,H,W), intrinsics (B,3,3), rays, confidence)

``L`` must equal ``num_levels`` (4) and ``C`` must equal ``token_dim`` (1024):
the frame/epipolar stack, the rectified stereo token grid and the DPT head are
all built for those numbers.

Both released backbones return their levels ordered EARLY -> LATE.  The ordering
is load-bearing: the DPT head upsamples level 0 the most and level 3 the least,
and the camera/scale heads read the CLS of the LAST level.
"""

from abc import ABC, abstractmethod

import torch


class DepthBackbone(ABC):
    """Frozen backbone interface.

    Deliberately a plain object rather than an ``nn.Module``: the backbone must not
    appear in ``EpiDistill.state_dict()`` (released checkpoints carry only the
    EpiDistill modules, a few hundred MB instead of several GB) nor in a
    distributed parameter walk.
    """

    patch_size: int = 14
    token_dim: int = 1024
    num_levels: int = 4

    @abstractmethod
    def encode(self, image: torch.Tensor, intrinsics=None) -> dict:
        ...

    @abstractmethod
    def decode(self, inputs: dict):
        ...

    # ── frozen-module shims ───────────────────────────────────────────────────
    def eval(self):
        self.model.eval()
        return self

    def train(self, mode: bool = True):
        """No-op by design -- the backbone is frozen and stays in eval mode."""
        self.model.eval()
        return self

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

    def parameters(self):
        return list(self.model.parameters())
