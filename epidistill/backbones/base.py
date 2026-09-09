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
