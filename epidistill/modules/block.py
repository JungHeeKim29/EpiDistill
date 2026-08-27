# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#
# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#
# EpiDistill: trimmed to the two block flavours the released model uses --
#   attention_type='default'  -> frame attention (Sec. 3, "Frame and Global Attention")
#   attention_type='epipolar' -> depth-guided epipolar attention (Sec. 4.1)

from typing import Callable

from torch import Tensor, nn

from .attention import Attention
from .drop_path import DropPath
from .epipolar_attention import DepthGuidedEpipolarAttention
from .layer_scale import LayerScale
from .mlp import Mlp


class Block(nn.Module):
    """Pre-norm transformer block.

    With ``attention_type='epipolar'`` the attention operator is the depth-guided
    epipolar cross-attention of Sec. 4.1 and the block consumes/produces spatial
    maps ``(B, C, H, W)``; otherwise it is standard self-attention over token
    sequences ``(B, N, C)``.
    """

    def __init__(
        self,
        dim: int = 1024,
        num_heads: int = 32,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = True,
        fused_attn: bool = True,
        rope=None,
        attention_type: str = "default",
        num_samples: int = 32,
        grad_checkpointing: bool = True,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)

        if attention_type == "epipolar":
            self.attn = DepthGuidedEpipolarAttention(
                dim=dim,
                num_heads=num_heads,
                num_samples=num_samples,
                grad_checkpointing=grad_checkpointing,
                rope=rope,
            )
        else:
            self.attn = attn_class(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                qk_norm=qk_norm,
                fused_attn=fused_attn,
                rope=rope,
            )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    # ── helpers ───────────────────────────────────────────────────────────────
    @property
    def _is_epipolar(self) -> bool:
        return isinstance(self.attn, DepthGuidedEpipolarAttention)

    @staticmethod
    def _nchw_norm(norm, x: Tensor) -> Tensor:
        """LayerNorm over the channel axis of an (B, C, H, W) map."""
        return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def forward(
        self,
        x: Tensor,
        pos=None,
        key_feat=None,
        value_feat=None,
        K_ref=None,
        K_src=None,
        T_ref2src=None,
        depth=None,
        rectified: bool = False,
        apply_depth_bias: bool = False,
        return_attn_loss: bool = False,
    ):
        attn_loss = None

        def attn_residual(inp: Tensor) -> Tensor:
            nonlocal attn_loss
            if not self._is_epipolar:
                return self.attn(self.norm1(inp), pos=pos)

            q = self._nchw_norm(self.norm1, inp)
            k = self._nchw_norm(self.norm1, key_feat) if key_feat is not None else q
            v = self._nchw_norm(self.norm1, value_feat) if value_feat is not None else q
            out = self.attn(
                query_feat=q,
                key_feat=k,
                value_feat=v,
                K_ref=K_ref,
                K_src=K_src,
                T_ref2src=T_ref2src,
                depth=depth,
                rectified=rectified,
                pos=pos,
                apply_depth_bias=apply_depth_bias,
                return_matching_loss=return_attn_loss,
            )
            if return_attn_loss:
                out, attn_loss = out
            return out

        def ffn_residual(inp: Tensor) -> Tensor:
            if self._is_epipolar:
                y = self.mlp(self.norm2(inp.permute(0, 2, 3, 1)))
                return y.permute(0, 3, 1, 2)
            return self.mlp(self.norm2(inp))

        def scaled(ls, t: Tensor) -> Tensor:
            if self._is_epipolar:
                return ls(t.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            return ls(t)

        x = x + self.drop_path1(scaled(self.ls1, attn_residual(x)))
        x = x + self.drop_path2(scaled(self.ls2, ffn_residual(x)))

        if return_attn_loss:
            return x, attn_loss
        return x
