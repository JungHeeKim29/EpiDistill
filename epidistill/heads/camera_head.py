"""Camera head -- predicts the normalised focal-length residuals (delta_x, delta_y).

Sec. 4 / Eq. 11:  f_adj = f_base + (delta_x * W, delta_y * H).

Zero-initialised output layer, so an untrained head leaves the backbone's own
intrinsics untouched.
"""

import torch.nn as nn


class CameraHead(nn.Module):
    def __init__(self, dim_in: int = 1024, hidden_dim: int = 1024, output_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.constant_(self.net[-1].weight, 0.0)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, x):
        """x: (B, C) or (B, 1, C).  Returns (B, 2) = [delta_x, delta_y]."""
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.net(x)
