"""EpiDistill: a frozen depth backbone corrected into a single-view metric model.

Paper: "Geometric Distillation from Rectified Stereo" (Sec. 4, Fig. 3).
"""

from contextlib import contextmanager
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import build_backbone
from .config import EpiDistillConfig
from .heads.camera_head import CameraHead
from .heads.dpt_head import DPTHead
from .heads.scale_head import ScaleHead
from .modules.aggregation import (
    CrossViewScaleAggregation,
    RSTImageConditioning,
    ScaleLevelSelfAttention,
)
from .modules.block import Block
from .modules.rope import PositionGetter, RotaryPositionEmbedding2D


@contextmanager
def freeze_module(module: nn.Module):
    """Freeze a module's weights for one pass; gradients still reach its inputs.

    The student pass reuses the teacher's epipolar block without training it.
    """
    flags = [(q, q.requires_grad) for q in module.parameters()]
    for q, _ in flags:
        q.requires_grad_(False)
    # Activation checkpointing must be off for this pass: the recomputation at
    # backward time would see the weights with gradient re-enabled and no longer
    # match the forward that was recorded.
    ckpt = [(m, m.grad_checkpointing) for m in module.modules() if hasattr(m, "grad_checkpointing")]
    for m, _ in ckpt:
        m.grad_checkpointing = False
    try:
        yield
    finally:
        for q, flag in flags:
            q.requires_grad_(flag)
        for m, flag in ckpt:
            m.grad_checkpointing = flag


class EpiDistill(nn.Module):
    def __init__(self, config: EpiDistillConfig = None, device=torch.device("cuda")):
        super().__init__()
        self.config = config or EpiDistillConfig()
        cfg = self.config
        self.device = device

        # frozen backbone (never in state_dict -- see backbones/base.py)
        backbone_kwargs = {"device": device}
        if cfg.backbone_checkpoint is not None:
            backbone_kwargs["checkpoint_path"] = cfg.backbone_checkpoint
        if cfg.backbone == "depthpro":
            backbone_kwargs["levels"] = cfg.depthpro_levels
            backbone_kwargs["patch_size"] = cfg.patch_size
        object.__setattr__(self, "_backbone", build_backbone(cfg.backbone, **backbone_kwargs))

        C, L = cfg.token_dim, cfg.num_levels
        self.patch_size = cfg.patch_size

        # positional encoding
        self.rope = RotaryPositionEmbedding2D(frequency=cfg.rope_frequency)
        self.position_getter = PositionGetter()

        # attention stack (Sec. 3 / Sec. 4.1)
        self.frame_blocks = nn.ModuleList(
            [Block(dim=C, num_heads=cfg.frame_num_heads, rope=self.rope) for _ in range(L)]
        )
        self.epipolar_blocks = nn.ModuleList(
            [
                Block(
                    dim=C,
                    rope=self.rope,
                    attention_type="epipolar",
                    num_heads=cfg.epipolar_num_heads,
                    num_samples=cfg.epipolar_num_samples,
                    init_values=cfg.epipolar_layer_scale,
                )
                for _ in range(L)
            ]
        )

        # rectified stereo tokens (Sec. 4.2)
        # (L, C, N+1): a spatial token grid per level, plus one CLS slot.
        self.rectified_stereo_tokens = nn.Parameter(torch.zeros(L, C, cfg.rst_num_tokens))
        nn.init.normal_(self.rectified_stereo_tokens, std=0.02)
        self.reference_guided_attn = nn.ModuleList(
            [RSTImageConditioning(dim=C, num_heads=cfg.rst_num_heads) for _ in range(L)]
        )

        # scale token (F_scale)
        # Two role-bound slots: slot 0 reads the reference view, slot 1 reads the
        # source content -- the source views for the multi-view teacher, T_rst-ref
        # for the single-view model.  Seeded as copies so both start somewhere sane;
        # they diverge immediately because their key/value sets differ.
        self.scale_token = nn.Parameter(torch.randn(1, C) * 0.02)
        self.scale_src_token = nn.Parameter(self.scale_token.detach().clone())
        # Per-level additive seed, zero-init: an untrained seed is an exact no-op.
        self.scale_level_seed = nn.Parameter(torch.zeros(L, 1, C))
        self.scale_level_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(C), nn.Linear(C, C), nn.GELU(), nn.Linear(C, C)
                )
                for _ in range(L)
            ]
        )
        for proj in self.scale_level_proj:
            nn.init.zeros_(proj[-1].weight)
            nn.init.zeros_(proj[-1].bias)

        self.scale_cross_view_attn = nn.ModuleList(
            [CrossViewScaleAggregation(dim=C, num_heads=cfg.scale_num_heads) for _ in range(L)]
        )
        self.scale_level_attn = ScaleLevelSelfAttention(
            dim=C, num_heads=cfg.scale_num_heads, num_levels=L
        )
        if cfg.reads_fov_tokens:
            # DepthPro: a second read-out per level over the fov encoder's tokens,
            # added to every scale slot's read-out (see _add_fov_readout).
            self.scale_fov_proj = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, C))
            self.scale_fov_attn = nn.ModuleList(
                [CrossViewScaleAggregation(dim=C, num_heads=cfg.scale_num_heads) for _ in range(L)]
            )

        # norms
        self.token_norm = nn.LayerNorm(C, eps=1e-6)
        self.output_norm = nn.LayerNorm(C)

        # heads
        self.structure_head = DPTHead(
            C,
            features=64,
            output_dim=2,                       # 0 = structural shift Delta d, 1 = confidence
            out_channels=[64, 128, 256, 256],
            use_bn=True,
        )
        if cfg.scale_via_focal:
            # DepthPro: the focal residual is read from the scale token and the
            # scale follows from it (config.scale_via_focal).
            self.scale_focal_head = CameraHead(dim_in=C, output_dim=2)
        else:
            self.focal_head = CameraHead(dim_in=C, output_dim=2)
            self.scale_head = ScaleHead(
                dim_in=C,
                output_dim=1,
                log_scale_min=cfg.log_scale_min,
                log_scale_max=cfg.log_scale_max,
            )

        self.to(device)

    # backbone accessor (kept out of the module tree so it stays out of state_dict)
    @property
    def backbone(self):
        return self._backbone

    def to(self, *args, **kwargs):
        out = super().to(*args, **kwargs)
        device = args[0] if args else kwargs.get("device", None)
        if device is not None and hasattr(self, "_backbone"):
            self._backbone.to(device)
            self.device = device
        return out

    # input conditioning
    def _preprocess(self, image):
        """Pad to the backbone's aspect bounds, then resize into its pixel band.

        Returns the conditioned image and the bookkeeping needed to map the depth
        and intrinsics back to the caller's resolution.
        """
        cfg = self.config
        _, _, H0, W0 = image.shape
        ratio = W0 / H0

        if cfg.backbone == "depthpro":
            # No padding: DepthPro is a fixed 1536x1536 stretch-resize network, so
            # every aspect ratio is in-distribution, and padded constant pixels would
            # enter its field of view with nothing marking them invalid.
            target_ratio = ratio
        else:
            target_ratio = min(cfg.ratio_bounds[1], max(cfg.ratio_bounds[0], ratio))

        if ratio > target_ratio:                       # too wide -> pad height
            padded_W, padded_H = W0, int(W0 / target_ratio)
            pad_top = (padded_H - H0) // 2
            pad = (0, 0, pad_top, padded_H - H0 - pad_top)
        elif ratio < target_ratio:                     # too tall -> pad width
            padded_H, padded_W = H0, int(H0 * target_ratio)
            pad_left = (padded_W - W0) // 2
            pad = (pad_left, padded_W - W0 - pad_left, 0, 0)
        else:
            padded_H, padded_W = H0, W0
            pad = (0, 0, 0, 0)

        if any(pad):
            image = F.pad(image, pad, value=0.0)

        n_pixels = padded_H * padded_W
        target_pixels = min(cfg.pixels_max, max(cfg.pixels_min, n_pixels))
        factor = (target_pixels / n_pixels) ** 0.5
        p = self.patch_size
        new_H = int(ceil(padded_H * factor / p) * p)
        new_W = int(ceil(padded_W * factor / p) * p)

        if (new_H, new_W) != (padded_H, padded_W):
            image = F.interpolate(
                image, size=(new_H, new_W), mode="bilinear", align_corners=False
            )

        meta = {
            "pad": pad,                                # (left, right, top, bottom)
            "padded": (padded_H, padded_W),
            "resized": image.shape[-2:],
        }
        return image, meta

    def _postprocess(self, depth, K, meta):
        """Undo ``_preprocess`` for the depth map and the intrinsics."""
        pad_left, pad_right, pad_top, pad_bottom = meta["pad"]
        padded_H, padded_W = meta["padded"]
        H, W = meta["resized"]

        if (H, W) != (padded_H, padded_W) or any(meta["pad"]):
            depth = F.interpolate(
                depth, size=(padded_H, padded_W), mode="bilinear", align_corners=False
            )
            K = K.clone()
            K[:, 0, :] *= padded_W / W
            K[:, 1, :] *= padded_H / H

        if any(meta["pad"]):
            depth = depth[..., pad_top: padded_H - pad_bottom, pad_left: padded_W - pad_right]
            K[:, 0, 2] -= pad_left
            K[:, 1, 2] -= pad_top

        return depth, K

    # scale-token plumbing
    def _scale_seed(self, batch_size, n_src: int = 1):
        """Level-0 scale query: (B, 1 + n_src, C), slot 0 = reference, slots 1.. = sources."""
        seed = torch.cat([self.scale_token, self.scale_src_token.expand(n_src, -1)], dim=0)
        return seed.unsqueeze(0).expand(batch_size, -1, -1)

    def _level_scale_query(self, level, state):
        """Per-level query = proj_L(carried-over state) + learnable seed_L."""
        return self.scale_level_proj[level](state) + self.scale_level_seed[level].unsqueeze(0)

    def _scale_readout(self, level, level_scale, ref_spatial, src_spatial, fov_kv=None):
        """Read the scale token out: slot 0 over the reference view, slot 1 over T_rst-ref."""
        ref_out = self.scale_cross_view_attn[level](level_scale[:, 0:1], ref_spatial)
        src_out = self.scale_cross_view_attn[level](level_scale[:, 1:2], src_spatial)
        return self._add_fov_readout(level, level_scale, torch.cat([ref_out, src_out], dim=1), fov_kv)

    def _fov_kv(self, fov_tokens):
        """Project the fov-encoder tokens of every view: list of (B, 1+N, C) -> (B, V, N, C).

        The tokens come from the frozen backbone (detached); the projection is
        ours and trains.  Returns None when the backbone has no fov encoder.
        """
        if not self.config.reads_fov_tokens or any(t is None for t in fov_tokens):
            return None
        x = torch.stack([t[:, 1:, :] for t in fov_tokens], dim=1)          # drop CLS
        x = x.detach().to(self.scale_fov_proj[1].weight.dtype)
        B, V, N, C = x.shape
        return self.scale_fov_proj(x.reshape(B * V * N, C)).view(B, V, N, -1)

    def _add_fov_readout(self, level, level_scale, out, fov_kv):
        """DepthPro: add a read-out over the fov-encoder tokens to every scale slot."""
        if fov_kv is None:
            return out
        V = fov_kv.shape[1]
        adds = []
        for s in range(out.shape[1]):
            v = min(s, V - 1)
            adds.append(self.scale_fov_attn[level](level_scale[:, s:s + 1], fov_kv[:, v:v + 1]))
        return out + torch.cat(adds, dim=1)

    def _teacher_scale_readout(self, level, level_scale, views_spatial, fov_kv=None):
        """Teacher read-out with one slot per view: slot 0 reads every view's patches,
        slot i >= 1 reads source view i alone.  views_spatial: (B, V, N, C)."""
        outs = [self.scale_cross_view_attn[level](level_scale[:, 0:1], views_spatial)]
        for i in range(1, level_scale.shape[1]):
            outs.append(self.scale_cross_view_attn[level](level_scale[:, i:i + 1], views_spatial[:, i:i + 1]))
        return self._add_fov_readout(level, level_scale, torch.cat(outs, dim=1), fov_kv)

    def _pool_scale_history(self, history):
        """Self-attention across levels; the slot axis is folded into the batch,
        so slots mix across levels but never with each other."""
        B, V, C = history[0].shape
        folded = [s.reshape(B * V, 1, C) for s in history]
        return self.scale_level_attn(folded).squeeze(1).view(B, V, C)

    # rectified stereo tokens
    def _rst_grid(self, H, W):
        """Interpolate T_rst from its native 37x37 grid onto the current token grid."""
        g = self.config.rst_grid_size
        th, tw = H // self.patch_size, W // self.patch_size
        if (th, tw) == (g, g):
            return self.rectified_stereo_tokens

        L, C, _ = self.rectified_stereo_tokens.shape
        cls = self.rectified_stereo_tokens[..., :1]
        spatial = self.rectified_stereo_tokens[..., 1:].view(L, C, g, g)
        spatial = F.interpolate(spatial, size=(th, tw), mode="bicubic", align_corners=False)
        return torch.cat([cls, spatial.flatten(2)], dim=2)

    # attention passes
    def _frame_attention(self, tokens, level, pos):
        """Intra-view self-attention.  tokens: (B, V, P, C)."""
        B, V, P, C = tokens.shape
        x = torch.clamp(tokens.reshape(B * V, P, C), min=-1e4, max=1e4)
        x = self.frame_blocks[level](x, pos=pos.reshape(B * V, P, 2))
        return x.view(B, V, P, C)

    def _rectified_epipolar(self, level, view_tokens, rst_map, K, grid):
        """Rectified epipolar cross-attention: F_ref (query) x T_rst-ref (key/value).

        R = I and t = [b, 0, 0] (Sec. 4.2), so the epipolar line is the query's own
        horizontal scanline and the geometry arguments are placeholders.
        The CLS slot is carried through untouched.
        """
        B, V, P, C = view_tokens.shape
        Hp, Wp = grid
        cls_tokens = view_tokens[:, :, 0:1, :]
        ref_feat = view_tokens[:, 0, 1:, :].permute(0, 2, 1).reshape(B, C, Hp, Wp)

        eye = torch.eye(4, device=view_tokens.device, dtype=view_tokens.dtype)
        eye = eye.unsqueeze(0).expand(B, -1, -1)

        out = self.epipolar_blocks[level](
            ref_feat,
            key_feat=rst_map,
            value_feat=rst_map,
            K_ref=K,
            K_src=K,
            T_ref2src=eye,
            rectified=True,
        )
        out = out.permute(0, 2, 3, 1).reshape(B, 1, Hp * Wp, C)
        return torch.cat([cls_tokens, out], dim=2)

    # depth composition (Eq. 1)
    def _compose_struct(self, base_depth_norm, delta_d):
        """D_struct = D_base + Delta d / (1 + |Delta d| / tau)  (Eq. 1)."""
        tau = float(self.config.struct_tau)
        return base_depth_norm + delta_d / (1.0 + delta_d.abs() / tau)

    @torch.no_grad()
    def _struct_norm(self, struct, valid_mask):
        """Per-image normaliser ``_sm``, so ``D_adj = (D_struct / _sm) * s_hat``."""
        flat = struct.reshape(struct.shape[0], -1)
        w = valid_mask.reshape(flat.shape[0], -1).to(flat.dtype)
        # Same fallback as the D_base normalisation: too few valid pixels -> use all.
        w = torch.where(w.sum(dim=1, keepdim=True) > 100, w, torch.ones_like(w))

        # Masked median without a loop: push invalid pixels above every real value,
        # sort, and take each image's own (n_valid / 2)-th order statistic.
        big = flat.max(dim=1, keepdim=True).values + 1.0
        ordered = torch.where(w > 0, flat, big).sort(dim=1).values
        idx = (w.sum(dim=1) * 0.5).long().clamp(min=0, max=flat.shape[1] - 1)
        v = ordered.gather(1, idx.view(-1, 1)).squeeze(1)

        return v.view(-1, 1, 1, 1).clamp(min=1e-3).detach()

    def _base_statistics(self, depth):
        """Normalised base depth ``D_base = D_orig / median_valid(D_orig)``.

        The scale-invariant structural map of Sec. 4.  The median (not the mean)
        is used because it is robust to the sky/far tail, and the mask excludes
        pixels the backbone itself declared invalid.
        """
        valid = depth > self.config.backbone_valid_eps
        depth = torch.clamp(depth, min=self.config.min_depth)

        base, centers = [], []
        for b in range(depth.shape[0]):
            d = depth[b]
            v = d[valid[b]]
            center = v.median() if v.numel() > 100 else d.median()
            center = torch.clamp(center, min=self.config.min_depth)
            base.append(d / center)
            centers.append(center)
        return torch.stack(base, 0), valid, torch.stack(centers, 0)

    # shared per-level plumbing (single-view inference and multi-view training)
    def _view_positions(self, num_seqs, grid, device):
        """RoPE positions for ``num_seqs`` sequences of [special, patches]: (num_seqs, 1+N, 2)."""
        Hp, Wp = grid
        pos = self.position_getter(num_seqs, Hp, Wp, device=device) + 1
        return torch.cat([torch.zeros(num_seqs, 1, 2, device=device, dtype=pos.dtype), pos], 1)

    def _level_inputs(self, level, tokens, feats, rst_all):
        """token_norm over [CLS, patches] of every view and of T_rst, for one level.

        tokens/feats: lists over views of (B, 1, C) / (B, Hp, Wp, C).
        Returns views (B, V, 1+N, C) and rst (B, 1, 1+N, C).
        """
        views = []
        for tok, feat in zip(tokens, feats):
            feat = feat.reshape(feat.shape[0], -1, self.config.token_dim)
            views.append(self.token_norm(torch.cat([tok, feat], dim=1)))
        views = torch.stack(views, dim=1)
        rst = self.token_norm(rst_all[:, level].permute(0, 2, 1)).unsqueeze(1)
        return views, rst

    def _frame_pass(self, level, views, rst, pos):
        """Frame attention over the V real views and T_rst together.

        Returns the attended views (B, V, 1+N, C) and T_rst (B, 1+N, C), both
        through ``output_norm``.
        """
        V = views.shape[1]
        attended = self._frame_attention(torch.cat([views, rst], dim=1), level, pos)
        return self.output_norm(attended[:, :V]), self.output_norm(attended[:, V])

    def _student_level(self, level, views, rst_view, level_scale, K, grid, fov_kv=None):
        """One level of the single-view (student) path.

        ``views`` is (B, V, 1+N, C) with view 0 the reference; V == 1 at inference,
        while during training the source views ride along through the post-fusion
        block so the DPT head can decode every view from one token set.
        Returns the fused tokens, the read-out scale token, and T_rst-ref.
        """
        B, V, S, C = views.shape
        Hp, Wp = grid

        # T_rst -> T_rst-ref: pick up this image's structure (Fig. 3, "Cross Attn.")
        rst_map = rst_view[:, 1:].transpose(1, 2).reshape(B, C, Hp, Wp)
        ref_map = views[:, 0, 1:, :].permute(0, 2, 1).reshape(B, C, Hp, Wp)
        rst_ref = self.reference_guided_attn[level](rst_map, ref_map)

        # Rectified epipolar attention over T_rst-ref.  The epipolar weights belong to
        # the multi-view teacher; the student pass leaves them where they are.
        if self.training:
            with freeze_module(self.epipolar_blocks[level]):
                ref_out = self._rectified_epipolar(level, views[:, 0:1], rst_ref, K, grid)
        else:
            ref_out = self._rectified_epipolar(level, views[:, 0:1], rst_ref, K, grid)

        seq = torch.cat([ref_out, views[:, 1:]], dim=1) if V > 1 else ref_out
        pos_post = self._view_positions(B * V, grid, views.device)
        fused = self.output_norm(
            self.frame_blocks[level](seq.reshape(B * V, S, C), pos=pos_post)
        ).view(B, V, S, C)

        # Scale read-out: slot 0 reads the reference patches, slot 1 reads T_rst-ref.
        src_kv = rst_ref.flatten(2).permute(0, 2, 1).unsqueeze(1)          # (B, 1, N, C)
        scale_state = self._scale_readout(level, level_scale, fused[:, :1, 1:, :], src_kv, fov_kv)
        return fused, scale_state, rst_ref

    def _adjust_intrinsics(self, K_base, delta_f, H, W):
        """Eq. 11: f_adj = f_base + (delta_x * W, delta_y * H); principal point passes through."""
        K = torch.zeros_like(K_base)
        K[:, 0, 0] = torch.clamp(K_base[:, 0, 0] + delta_f[:, 0] * W, min=10.0, max=10000.0)
        K[:, 1, 1] = torch.clamp(K_base[:, 1, 1] + delta_f[:, 1] * H, min=10.0, max=10000.0)
        K[:, 0, 2] = K_base[:, 0, 2]
        K[:, 1, 2] = K_base[:, 1, 2]
        K[:, 2, 2] = 1.0
        return K

    def _scale_and_intrinsics(self, scale_token_n, cls_token, K_base, base_center, H, W,
                              detach_camera=False):
        """s_hat (B, 1) and the adjusted intrinsics (B, 3, 3) for one view (Eq. 10, Eq. 11).

        On DepthPro the focal residual is read from the scale token and
        ``s_hat = median_valid(D_orig) * f_adj / f_base`` follows from it.
        """
        if self.config.scale_via_focal:
            K = self._adjust_intrinsics(K_base, self.scale_focal_head(scale_token_n), H, W)
            ratio = K[:, 0, 0] / torch.clamp(K_base[:, 0, 0], min=1e-3)
            return (base_center * ratio).view(-1, 1), K
        # The camera head is supervised on the reference view only.
        delta_f = self.focal_head(cls_token.detach() if detach_camera else cls_token)
        if detach_camera:
            delta_f = delta_f.detach()
        return self.scale_head(scale_token_n), self._adjust_intrinsics(K_base, delta_f, H, W)

    @staticmethod
    def _clamp_intrinsics(K):
        K = K.clone()
        K[:, 0, 0] = torch.clamp(K[:, 0, 0], min=10.0, max=10000.0)
        K[:, 1, 1] = torch.clamp(K[:, 1, 1], min=10.0, max=10000.0)
        K[:, 0, 2] = torch.clamp(K[:, 0, 2], min=-10000.0, max=10000.0)
        K[:, 1, 2] = torch.clamp(K[:, 1, 2], min=-10000.0, max=10000.0)
        return K

    def _compose(self, base_depth_norm, backbone_valid, delta_d, s_hat):
        """Eq. 1: D_adj = s_hat * D_struct / median_valid(D_struct)."""
        struct = self._compose_struct(base_depth_norm, delta_d)
        struct_norm = self._struct_norm(struct, backbone_valid)
        depth = (struct / struct_norm) * s_hat.view(-1, 1, 1, 1)
        return torch.clamp(depth, min=self.config.min_depth)

    # single-view forward
    def forward(self, image):
        """Single-view inference.

        Args:
            image: (B, 3, H, W), ImageNet-normalised.
        Returns:
            depth: (B, 1, H, W) metric depth, at the input resolution.
            K:     (B, 3, 3) adjusted intrinsics (Eq. 11).
        """
        cfg = self.config
        image, meta = self._preprocess(image)
        B, _, H, W = image.shape
        grid = (H // self.patch_size, W // self.patch_size)

        # 1. frozen backbone
        inputs = self.backbone.encode(image)
        depth, K_base, _, _ = self.backbone.decode(inputs)
        K_base = self._clamp_intrinsics(K_base)
        tokens, feats = inputs["tokens"], inputs["features"]
        fov_kv = self._fov_kv([inputs.get("fov_tokens")])

        # 2. positions for the (reference, T_rst) frame pass
        pos = self._view_positions(B * 2, grid, image.device)
        rst_all = self._rst_grid(H, W).unsqueeze(0).expand(B, -1, -1, -1)

        # 3. per-level frame + rectified epipolar attention
        scale_state = self._scale_seed(B)
        scale_history, level_outputs = [], []

        for level in range(cfg.num_levels):
            views, rst = self._level_inputs(level, [tokens[level]], [feats[level]], rst_all)
            views, rst_view = self._frame_pass(level, views, rst, pos)

            level_scale = self._level_scale_query(level, scale_state)
            fused, scale_state, _ = self._student_level(
                level, views, rst_view, level_scale, K_base, grid, fov_kv
            )
            scale_history.append(scale_state)
            level_outputs.append(fused[:, 0])                              # (B, 1+N, C)

        # 4. token read-outs
        pooled = self._pool_scale_history(scale_history)
        scale_token = pooled[:, 0, :]                                      # reference slot
        cls_token = level_outputs[-1][:, 0, :]

        # 5. heads
        base_depth_norm, backbone_valid, base_center = self._base_statistics(depth)
        scale_token_n = F.layer_norm(scale_token, [scale_token.shape[-1]])
        s_hat, K = self._scale_and_intrinsics(scale_token_n, cls_token, K_base, base_center, H, W)

        dpt_out = self.structure_head(
            level_outputs,
            patch_h=grid[0],
            patch_w=grid[1],
            base_depth=base_depth_norm,
            scale_token=scale_token_n,
        )
        delta_d = dpt_out[:, 0:1, :, :]

        # 6. fuse (Eq. 1)
        depth_adj = self._compose(base_depth_norm, backbone_valid, delta_d, s_hat)
        return self._postprocess(depth_adj, K, meta)

    # multi-view training forward (teacher + student)
    def _encode_views(self, images):
        """Frozen-backbone encoding and decoding of every view.

        images: list of (B, 3, H, W).  Returns per-view lists of tokens, features,
        backbone depth (each view's own D_orig) and backbone intrinsics.
        """
        view_tokens, view_feats, depths, Ks, fov = [], [], [], [], []
        with torch.no_grad():
            for image in images:
                inputs = self.backbone.encode(image)
                depth, K_base, _, _ = self.backbone.decode(inputs)
                view_tokens.append(inputs["tokens"])
                view_feats.append(inputs["features"])
                depths.append(torch.clamp(depth.detach(), min=self.config.min_depth))
                Ks.append(self._clamp_intrinsics(K_base.detach()))
                fov.append(inputs.get("fov_tokens"))
        return view_tokens, view_feats, depths, Ks, self._fov_kv(fov)

    def _epipolar_teacher(self, level, views, Ks, T_ref2src, ref_depth_tok, grid):
        """Depth-guided epipolar attention across the real views (Sec. 4.1).

        Every view attends to each other view along the epipolar lines induced by
        their intrinsics and relative pose, and the per-source outputs are averaged.
        The reference view also gets the ground-truth-depth bias and yields the
        attention losses (Eq. 5); the CLS row passes through untouched.
        """
        B, V, S, C = views.shape
        Hp, Wp = grid
        eye = torch.eye(4, device=views.device, dtype=views.dtype).unsqueeze(0).expand(B, -1, -1)
        T = [eye] + [T_ref2src[:, j - 1].to(views.dtype) for j in range(1, V)]
        T_inv = [torch.inverse(t) for t in T]

        def rel(i, j):                                  # points of view i -> view j
            if i == 0:
                return T[j]
            if j == 0:
                return T_inv[i]
            return T[j] @ T_inv[i]

        cls_tokens = views[:, :, 0:1, :]
        maps = views[:, :, 1:, :].permute(0, 1, 3, 2).reshape(B, V, C, Hp, Wp)

        outputs, attn_losses = [], []
        for i in range(V):
            agg = torch.zeros_like(maps[:, i])
            for j in range(V):
                if i == j:
                    continue
                out = self.epipolar_blocks[level](
                    maps[:, i],
                    key_feat=maps[:, j],
                    value_feat=maps[:, j],
                    K_ref=Ks[i],
                    K_src=Ks[j],
                    T_ref2src=rel(i, j),
                    depth=ref_depth_tok if i == 0 else None,
                    rectified=False,
                    apply_depth_bias=True,
                    return_attn_loss=True,
                )
                out, attn_loss = out
                if attn_loss is not None:
                    attn_losses.append(attn_loss)
                agg = agg + out
            outputs.append(agg / (V - 1))

        out = torch.stack(outputs, dim=1).permute(0, 1, 3, 4, 2).reshape(B, V, Hp * Wp, C)
        return torch.cat([cls_tokens, out], dim=2), attn_losses

    def forward_multiview(self, ref_image, src_images, K_ref, K_src, T_ref2src, ref_depth_gt):
        """Training forward: the multi-view teacher and the single-view student.

        Args:
            ref_image (B,3,H,W), src_images (B,S,3,H,W)  ImageNet-normalised, sides
                multiples of 14; K_ref (B,3,3), K_src (B,S,3,3) ground-truth intrinsics;
                T_ref2src (B,S,4,4) maps reference-camera points into each source camera;
                ref_depth_gt (B,1,H,W) metres, 0 = invalid.

        Returns a dict of per-view predictions (depth, intrinsics, confidence, s_hat),
        the teacher and student tokens and pooled scale tokens for the distillation
        terms, and ``attn_loss`` (Eq. 5).  Every view is decoded from the STUDENT
        tokens on its own backbone depth, with ``s_hat`` read from the teacher's
        slot for that view.
        """
        cfg = self.config
        B, S_views, _, H, W = src_images.shape
        V = 1 + S_views
        grid = (H // self.patch_size, W // self.patch_size)
        device = ref_image.device

        # 1. frozen backbone
        images = [ref_image] + [src_images[:, i] for i in range(S_views)]
        view_tokens, view_feats, base_depths, K_bases, fov_kv = self._encode_views(images)

        # 2. geometry for the teacher
        scale = 1.0 / self.patch_size
        Ks_tok = []
        for K in [K_ref] + [K_src[:, i] for i in range(S_views)]:
            K = K.clone()
            K[:, :2, :] *= scale
            Ks_tok.append(K)
        ref_depth_tok = F.interpolate(ref_depth_gt, size=grid, mode="nearest")

        pos = self._view_positions(B * (V + 1), grid, device)
        pos_teacher = self._view_positions(B * V, grid, device)
        rst_all = self._rst_grid(H, W).unsqueeze(0).expand(B, -1, -1, -1)

        # 3. per-level teacher + student
        t_state, s_state = self._scale_seed(B, n_src=S_views), self._scale_seed(B)
        t_hist, s_hist, t_tokens, s_tokens, attn_losses = [], [], [], [], []

        for level in range(cfg.num_levels):
            views, rst = self._level_inputs(
                level, [t[level] for t in view_tokens], [f[level] for f in view_feats], rst_all
            )
            views, rst_view = self._frame_pass(level, views, rst, pos)

            # teacher: depth-guided epipolar attention over the real views
            t_out, losses = self._epipolar_teacher(level, views, Ks_tok, T_ref2src, ref_depth_tok, grid)
            attn_losses.extend(losses)
            t_fused = self.output_norm(self.frame_blocks[level](
                t_out.reshape(B * V, -1, cfg.token_dim), pos=pos_teacher
            )).view(B, V, -1, cfg.token_dim)
            t_state = self._teacher_scale_readout(
                level, self._level_scale_query(level, t_state), t_fused[:, :, 1:, :], fov_kv
            )

            # student: rectified epipolar attention over T_rst-ref, its own query
            s_fused, s_state, _ = self._student_level(
                level, views, rst_view, self._level_scale_query(level, s_state), K_bases[0], grid,
                fov_kv[:, :1] if fov_kv is not None else None,
            )

            t_hist.append(t_state)
            s_hist.append(s_state)
            t_tokens.append(t_fused)
            s_tokens.append(s_fused)

        teacher_scale = self._pool_scale_history(t_hist)
        student_scale = self._pool_scale_history(s_hist)

        # 4. decode every view from the student tokens
        depths, intrinsics, confidences, s_hats = [], [], [], []
        for k in range(V):
            level_outputs = [tok[:, k] for tok in s_tokens]               # (B, 1+N, C) x L
            cls_token = level_outputs[-1][:, 0, :]
            base_depth_norm, backbone_valid, base_center = self._base_statistics(base_depths[k])

            scale_token = teacher_scale[:, k, :]                           # this view's slot
            scale_token_n = F.layer_norm(scale_token, [scale_token.shape[-1]])
            s_hat, K = self._scale_and_intrinsics(
                scale_token_n, cls_token, K_bases[k], base_center, H, W, detach_camera=(k > 0)
            )
            s_hats.append(s_hat)
            intrinsics.append(K)
            if k == 0:
                s_tok = F.layer_norm(student_scale[:, 0, :], [cfg.token_dim])
                s_hat_student, _ = self._scale_and_intrinsics(
                    s_tok, cls_token, K_bases[0], base_center, H, W
                )

            dpt_out = self.structure_head(
                level_outputs, patch_h=grid[0], patch_w=grid[1],
                base_depth=base_depth_norm, scale_token=scale_token_n.detach(),
            )
            depths.append(self._compose(base_depth_norm, backbone_valid, dpt_out[:, 0:1], s_hat))
            confidences.append(dpt_out[:, 1:2])

        attn_loss = (torch.stack(attn_losses).mean() if attn_losses
                     else torch.zeros((), device=device))
        return {
            "depth": depths,
            "intrinsics": intrinsics,
            "confidence": confidences,
            "s_hat": s_hats,
            "s_hat_student": s_hat_student,
            "teacher_tokens": t_tokens,
            "student_tokens": s_tokens,
            "teacher_scale_history": t_hist,
            "student_scale_history": s_hist,
            "teacher_scale": teacher_scale,
            "student_scale": student_scale,
            "attn_loss": attn_loss,
        }

    # Alias kept for readability at call sites.
    @torch.no_grad()
    def infer(self, image):
        return self.forward(image)
