"""Scale head -- predicts the global metric scale factor s_hat (Eq. 1).

The scale token produced by the frame/epipolar stack is mapped to log s_hat by a
small MLP.  Three auxiliary input channels exist for ablations and are kept (as
zeros when disabled) so that the parameter shapes -- and therefore released
checkpoints -- stay fixed:

    [ scale token (C) | CLS projection (256) | log focal_norm (2) | depth stats (3) ]

The released configuration uses the scale token alone (``use_cls=False``,
``use_focal_norm=False``, ``use_depth_stats=False``, ``from_scratch=True``):
supervision comes from ``L_scale`` (Eq. 10) and the head predicts absolute scale
rather than a residual around the frozen backbone's own median depth.
"""

import torch
import torch.nn as nn


class ScaleHead(nn.Module):
    def __init__(
        self,
        dim_in: int = 1024,
        hidden_dim: int = 1024,
        output_dim: int = 1,
        cls_token_dim: int = 1024,
        cls_proj_dim: int = 256,
        from_scratch: bool = True,
        use_depth_stats: bool = False,
        use_focal_norm: bool = False,
        log_scale_min: float = -3.0,
        log_scale_max: float = 6.0,
        residual_clamp: float = 0.4,
    ):
        super().__init__()
        self.cls_proj_dim = cls_proj_dim
        self.from_scratch = from_scratch
        self.use_depth_stats = use_depth_stats
        self.use_focal_norm = use_focal_norm
        self.log_scale_min = log_scale_min
        self.log_scale_max = log_scale_max
        # Only used when ``from_scratch=False`` (anchored parameterisation).
        self.residual_clamp = residual_clamp

        self.cls_proj = nn.Sequential(
            nn.Linear(cls_token_dim, cls_proj_dim),
            nn.GELU(),
        )
        nn.init.normal_(self.cls_proj[0].weight, std=0.01)
        nn.init.zeros_(self.cls_proj[0].bias)

        self.net = nn.Sequential(
            nn.Linear(dim_in + cls_proj_dim + 2 + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        # Zero-init the output layer: log s_hat starts at 0 (from-scratch) or at the
        # backbone's own median depth (anchored), i.e. no correction at step 0.
        nn.init.constant_(self.net[-1].weight, 0.0)
        with torch.no_grad():
            self.net[-1].bias[0] = 0.0

    def forward(self, scale_token, focal_norm=None, depth_stats=None, cls_token=None):
        """
        Args:
            scale_token: (B, C)   -- F_scale after frame + epipolar attention.
            focal_norm:  (B, 2)   -- [fx / W, fy / H] from the adjusted intrinsics.
            depth_stats: (B, 3)   -- log [median, std, range] of the frozen backbone depth.
            cls_token:   (B, C)   -- backbone class token (scene-level context).
        Returns:
            (B, 1) strictly positive metric scale factor s_hat.
        """
        B = scale_token.shape[0]
        kw = dict(device=scale_token.device, dtype=scale_token.dtype)

        if focal_norm is not None and self.use_focal_norm:
            focal_feat = torch.log(torch.clamp(focal_norm, min=0.01))
        else:
            focal_feat = torch.zeros(B, 2, **kw)

        if depth_stats is not None and self.use_depth_stats:
            depth_feat = depth_stats
        else:
            depth_feat = torch.zeros(B, 3, **kw)

        if cls_token is not None:
            cls_feat = self.cls_proj(cls_token.detach())
        else:
            cls_feat = torch.zeros(B, self.cls_proj_dim, **kw)

        raw = self.net(torch.cat([scale_token, cls_feat, focal_feat, depth_feat], dim=-1))

        if self.from_scratch:
            log_scale = raw
        else:
            delta = torch.clamp(raw, min=-self.residual_clamp, max=self.residual_clamp)
            if depth_stats is not None and self.use_depth_stats:
                log_scale = depth_stats[:, 0:1] + delta      # anchored on the backbone median
            else:
                log_scale = delta

        log_scale = torch.clamp(log_scale, min=self.log_scale_min, max=self.log_scale_max)
        return torch.exp(log_scale)
