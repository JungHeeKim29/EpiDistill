"""Scale head: the scale token -> the global metric factor s_hat (Eq. 10).

The first linear layer is 1285-wide because the trained checkpoints carry three
auxiliary input channels that the released configuration holds at zero; they are
kept as zeros rather than sliced away so the weights load unchanged.
"""

import torch
import torch.nn as nn

# Widths of the retained-but-inactive input channels: CLS projection, focal_norm, depth stats.
_INACTIVE_CHANNELS = 256 + 2 + 3


class ScaleHead(nn.Module):
    def __init__(
        self,
        dim_in: int = 1024,
        hidden_dim: int = 1024,
        output_dim: int = 1,
        log_scale_min: float = -3.0,
        log_scale_max: float = 6.0,
    ):
        super().__init__()
        self.log_scale_min = log_scale_min
        self.log_scale_max = log_scale_max

        self.net = nn.Sequential(
            nn.Linear(dim_in + _INACTIVE_CHANNELS, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        # Zero-init the output layer: log s_hat starts at 0, i.e. the scale is
        # learned from the token rather than inherited from an initialisation.
        nn.init.constant_(self.net[-1].weight, 0.0)
        with torch.no_grad():
            self.net[-1].bias[0] = 0.0

    def forward(self, scale_token):
        """scale_token: (B, C) -- F_scale after frame + epipolar attention.
        Returns (B, 1), the strictly positive metric scale factor s_hat."""
        pad = torch.zeros(
            scale_token.shape[0], _INACTIVE_CHANNELS,
            device=scale_token.device, dtype=scale_token.dtype,
        )
        log_scale = self.net(torch.cat([scale_token, pad], dim=-1))
        log_scale = torch.clamp(log_scale, min=self.log_scale_min, max=self.log_scale_max)
        return torch.exp(log_scale)
