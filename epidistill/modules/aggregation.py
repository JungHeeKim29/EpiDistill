"""Token mixing around the scale token and the rectified stereo tokens (Fig. 3)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


class CrossViewScaleAggregation(nn.Module):
    """Scale token (query) attends to spatial tokens (key/value).

    At single-view inference the key/value set is either the reference view's own
    post-epipolar tokens or the reference-guided rectified stereo tokens
    T_rst-ref, depending on which scale slot is querying.  The update is a gated
    residual, so the token is preserved at initialisation.
    """

    def __init__(self, dim: int = 1024, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        # Gated residual, opened slowly: sigmoid(-2) ~= 0.12.
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        nn.init.constant_(self.gate[0].bias, -2.0)

    def forward(self, scale_token, views_spatial):
        """
        Args:
            scale_token:   (B, Q, C) -- the querying scale slot(s).
            views_spatial: (B, V, N, C) -- key/value tokens.
        Returns:
            (B, Q, C) updated scale token.
        """
        B, V, N, C = views_spatial.shape
        kv_tokens = views_spatial.reshape(B, V * N, C)

        q = self.q_proj(self.norm_q(scale_token))
        k = self.k_proj(self.norm_kv(kv_tokens))
        v = self.v_proj(self.norm_kv(kv_tokens))

        Q = q.shape[1]
        q = q.view(B, Q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, V * N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, V * N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, Q, C)
        out = self.o_proj(out)
        return scale_token + self.gate(out) * out


class ScaleLevelSelfAttention(nn.Module):
    """Self-attention over the per-level scale states.

    The scale token is refreshed once per encoder level; this lets the final
    state attend over its own history across levels before the scale and camera
    heads read it.
    """

    def __init__(self, dim: int = 1024, num_heads: int = 8, num_levels: int = 4):
        super().__init__()
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

        self.level_embed = nn.Parameter(torch.randn(num_levels, dim) * 0.02)

        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        nn.init.constant_(self.gate[0].bias, -2.0)

    def forward(self, scale_states):
        """scale_states: list of (B, 1, C), one per level.  Returns (B, 1, C)."""
        B = scale_states[0].shape[0]
        C = scale_states[0].shape[-1]

        stacked = torch.stack([s.squeeze(1) for s in scale_states], dim=1)   # (B, L, C)
        stacked = stacked + self.level_embed.unsqueeze(0)

        normed = self.norm(stacked)
        q = self.q_proj(normed)
        k = self.k_proj(normed)
        v = self.v_proj(normed)

        L = self.num_levels
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        out = self.o_proj((attn @ v).transpose(1, 2).reshape(B, L, C))

        last_out = out[:, -1:, :]
        return scale_states[-1] + self.gate(last_out) * last_out


class RSTImageConditioning(nn.Module):
    """Rectified Stereo Tokens (T_rst) cross-attend to the reference features.

    Sec. 4.2 / Fig. 3: ``T_rst`` is a learnable spatial token grid; it queries the
    encoded reference features ``F_ref`` to pick up scene context, producing the
    reference-guided tokens ``T_rst-ref`` that stand in for the missing source
    views inside the (rectified) epipolar attention.
    """

    def __init__(self, dim: int = 1024, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        nn.init.constant_(self.gate[0].bias, -2.0)

    def forward(self, rst_feat, ref_feat):
        """rst_feat / ref_feat: (B, C, H, W).  Returns T_rst-ref as (B, C, H, W)."""
        if self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, rst_feat, ref_feat, use_reentrant=False
            )
        return self._forward_impl(rst_feat, ref_feat)

    def _forward_impl(self, rst_feat, ref_feat):
        B, C, H, W = rst_feat.shape
        N = H * W

        q_seq = rst_feat.flatten(2).transpose(1, 2)      # (B, N, C)
        kv_seq = ref_feat.flatten(2).transpose(1, 2)     # (B, N, C)

        q = self.q_proj(self.norm_q(q_seq))
        k = self.k_proj(self.norm_kv(kv_seq))
        v = self.v_proj(self.norm_kv(kv_seq))

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)    # memory-efficient, no NxN matrix
        out = self.o_proj(out.transpose(1, 2).reshape(B, N, C))

        updated = q_seq + self.gate(out) * out
        return updated.transpose(1, 2).view(B, C, H, W)
