"""EpiDistill -- single-view metric depth with distilled multi-view geometry.

Paper: "Geometric Distillation from Rectified Stereo: Leveraging Epipolar Cues
for Monocular Depth" (Sec. 4, Fig. 3).

At inference only the single-view pipeline runs:

    1. a frozen ViT depth foundation model (UniDepthV2 or DepthPro) encodes the
       image into L=4 levels of spatial tokens F_ref and class tokens, and decodes
       its own metric depth D_orig and intrinsics f_base;
    2. per level, the learnable Rectified Stereo Tokens T_rst cross-attend to
       F_ref to become the reference-guided tokens T_rst-ref (Sec. 4.2), which
       substitute for the absent source views;
    3. the shared frame attention and the rectified epipolar attention run over
       (F_ref, T_rst-ref), and a scale token F_scale reads the result out;
    4. three lightweight heads produce the structural shift Delta d (DPT), the
       global scale s_hat (MLP) and the focal residual (delta_x, delta_y) (MLP);
    5. the outputs are fused as

           D_base = D_orig / mean_valid(D_orig)          (scale-invariant structure)
           D_adj  = s_hat * (D_base + Delta d)           (Eq. 1)
           f_adj  = f_base + (delta_x * W, delta_y * H)  (Eq. 11)

       The released configuration composes the structural residual multiplicatively,
       D_struct = D_base * exp(tau * tanh(Delta d / tau)); see ``_compose_struct``.

Training (the multi-view teacher, L_attn / L_distill / L_rel / L_scale / L_ray)
is not part of this release yet -- see README, "Roadmap".
"""

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


class EpiDistill(nn.Module):
    def __init__(self, config: EpiDistillConfig = None, device=torch.device("cuda")):
        super().__init__()
        self.config = config or EpiDistillConfig()
        cfg = self.config
        self.device = device

        # ── frozen backbone (never in state_dict -- see backbones/base.py) ────
        backbone_kwargs = {"device": device}
        if cfg.backbone_checkpoint is not None:
            backbone_kwargs["checkpoint_path"] = cfg.backbone_checkpoint
        if cfg.backbone == "depthpro":
            backbone_kwargs["levels"] = cfg.depthpro_levels
            backbone_kwargs["patch_size"] = cfg.patch_size
        object.__setattr__(self, "_backbone", build_backbone(cfg.backbone, **backbone_kwargs))

        C, L = cfg.token_dim, cfg.num_levels
        self.patch_size = cfg.patch_size

        # ── positional encoding ───────────────────────────────────────────────
        self.rope = RotaryPositionEmbedding2D(frequency=cfg.rope_frequency)
        self.position_getter = PositionGetter()

        # ── attention stack (Sec. 3 / Sec. 4.1) ───────────────────────────────
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

        # ── rectified stereo tokens (Sec. 4.2) ────────────────────────────────
        # (L, C, N+1): a spatial token grid per level, plus one CLS slot.
        self.rst_tokens = nn.Parameter(torch.zeros(L, C, cfg.rst_num_tokens))
        nn.init.normal_(self.rst_tokens, std=0.02)
        self.rst_conditioning = nn.ModuleList(
            [RSTImageConditioning(dim=C, num_heads=cfg.rst_num_heads) for _ in range(L)]
        )

        # ── scale token (F_scale) ─────────────────────────────────────────────
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

        # ── norms ─────────────────────────────────────────────────────────────
        self.token_norm = nn.LayerNorm(C, eps=1e-6)
        self.output_norm = nn.LayerNorm(C)

        # ── heads ─────────────────────────────────────────────────────────────
        self.dpt_head = DPTHead(
            C,
            features=64,
            output_dim=2,                       # 0 = structural shift Delta d, 1 = confidence
            out_channels=[64, 128, 256, 256],
            use_bn=True,
        )
        self.camera_head = CameraHead(dim_in=C, output_dim=2)
        self.scale_head = ScaleHead(
            dim_in=C,
            output_dim=1,
            log_scale_min=cfg.log_scale_min,
            log_scale_max=cfg.log_scale_max,
        )

        self.to(device)

    # ══════════════════════════════════════════════════════════════════════════
    # backbone accessor (kept out of the module tree so it stays out of state_dict)
    # ══════════════════════════════════════════════════════════════════════════
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

    # ══════════════════════════════════════════════════════════════════════════
    # input conditioning
    # ══════════════════════════════════════════════════════════════════════════
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

    # ══════════════════════════════════════════════════════════════════════════
    # scale-token plumbing
    # ══════════════════════════════════════════════════════════════════════════
    def _scale_seed(self, batch_size):
        """Level-0 scale query: (B, 2, C), slot 0 = reference, slot 1 = source."""
        seed = torch.cat([self.scale_token, self.scale_src_token], dim=0)
        return seed.unsqueeze(0).expand(batch_size, -1, -1)

    def _level_scale_query(self, level, state):
        """Per-level query = proj_L(carried-over state) + learnable seed_L.

        The projection makes each level a transform of one shared state; the
        additive seed lets a level ask its own question regardless of the answer
        the previous level produced.
        """
        return self.scale_level_proj[level](state) + self.scale_level_seed[level].unsqueeze(0)

    def _scale_readout(self, level, level_scale, ref_spatial, src_spatial):
        """Read the scale token out of the post-epipolar tokens.

        Slot 0 reads the reference view's spatial tokens, slot 1 reads the
        reference-guided rectified stereo tokens.  Both go through the same
        module, so the read-out weights are shared and only the token differs;
        the separation *is* the architecture, there is no auxiliary loss holding
        the two apart.
        """
        ref_out = self.scale_cross_view_attn[level](level_scale[:, 0:1], ref_spatial)
        src_out = self.scale_cross_view_attn[level](level_scale[:, 1:2], src_spatial)
        return torch.cat([ref_out, src_out], dim=1)

    def _pool_scale_history(self, history):
        """Self-attention across levels; the slot axis is folded into the batch,
        so slots mix across levels but never with each other."""
        B, V, C = history[0].shape
        folded = [s.reshape(B * V, 1, C) for s in history]
        return self.scale_level_attn(folded).squeeze(1).view(B, V, C)

    # ══════════════════════════════════════════════════════════════════════════
    # rectified stereo tokens
    # ══════════════════════════════════════════════════════════════════════════
    def _rst_grid(self, H, W):
        """Interpolate T_rst from its native 37x37 grid onto the current token grid."""
        g = self.config.rst_grid_size
        th, tw = H // self.patch_size, W // self.patch_size
        if (th, tw) == (g, g):
            return self.rst_tokens

        L, C, _ = self.rst_tokens.shape
        cls = self.rst_tokens[..., :1]
        spatial = self.rst_tokens[..., 1:].view(L, C, g, g)
        spatial = F.interpolate(spatial, size=(th, tw), mode="bicubic", align_corners=False)
        return torch.cat([cls, spatial.flatten(2)], dim=2)

    # ══════════════════════════════════════════════════════════════════════════
    # attention passes
    # ══════════════════════════════════════════════════════════════════════════
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

    # ══════════════════════════════════════════════════════════════════════════
    # depth composition (Eq. 1)
    # ══════════════════════════════════════════════════════════════════════════
    def _compose_struct(self, base_depth_norm, delta_d):
        """How the frozen backbone and the DPT head combine into the structure.

        Additive (as written in Eq. 1)::

            D_struct = D_base + Delta d

        Multiplicative (the released default)::

            D_struct = D_base * exp(tau * tanh(Delta d / tau))

        Why the multiplicative form is the default: ``D_base`` is normalised, so
        ``median(D_base) == 1``, while ``Delta d`` is free to be any size -- and
        ``D_adj = (D_struct / _sm) * s_hat`` is exactly invariant under
        ``(Delta d, s_hat) -> (c * Delta d + (c - 1) * D_base, s_hat / c)``.  That
        overall magnitude is a gauge direction no loss penalises, and it drifts:
        the backbone's contribution to the structure decays until the shape is the
        DPT head alone.  As a multiplicative factor the same gauge is exactly inert
        (``exp(Delta d + c)`` cancels in ``D_struct / _sm``), so the backbone's
        structure can never be diluted.

        ``tau`` is a numerical guard, not a treatment: ``tau * tanh(x / tau)`` is the
        identity to third order at 0 -- so at initialisation (``Delta d == 0``, the
        DPT output layer is zero-init) ``D_struct == D_base`` exactly, i.e. the
        structure is the frozen backbone's -- and it saturates at ``e^{+-tau}``, far
        outside any correction actually observed.

        The *scale* is a separate story: the scale head is zero-init too, so an
        untrained model emits ``s_hat = exp(0) = 1`` m and the metric scale has to
        be learned from the distilled geometry -- only the structure is inherited.
        """
        if not self.config.struct_multiplicative:
            return base_depth_norm + delta_d
        tau = float(self.config.struct_tau or 0.0)
        s = (tau * torch.tanh(delta_d / tau)) if tau > 0 else delta_d
        return base_depth_norm * torch.exp(s)

    @torch.no_grad()
    def _struct_norm(self, struct, valid_mask):
        """Per-image normaliser ``_sm``, so ``D_adj = (D_struct / _sm) * s_hat``.

        ``torch.no_grad`` is part of the design, not an optimisation: ``_sm`` must be
        a constant per image, otherwise the division opens a gradient path from the
        scale loss into the structural offset and the scale/structure decoupling the
        composition rests on is lost.

        The median is taken over the pixels the backbone itself declared valid --
        the same support set the ``D_base`` normalisation uses, so sky and
        backbone-invalid pixels cannot fold into one anchor but not the other.
        """
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

        base = []
        for b in range(depth.shape[0]):
            d = depth[b]
            v = d[valid[b]]
            center = v.median() if v.numel() > 100 else d.median()
            base.append(d / torch.clamp(center, min=self.config.min_depth))
        return torch.stack(base, 0), valid

    # ══════════════════════════════════════════════════════════════════════════
    # single-view forward
    # ══════════════════════════════════════════════════════════════════════════
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
        Hp, Wp = H // self.patch_size, W // self.patch_size

        # ── 1. frozen backbone ────────────────────────────────────────────────
        inputs = self.backbone.encode(image)
        depth, K_base, _, _ = self.backbone.decode(inputs)

        K_base = K_base.clone()
        K_base[:, 0, 0] = torch.clamp(K_base[:, 0, 0], min=10.0, max=10000.0)
        K_base[:, 1, 1] = torch.clamp(K_base[:, 1, 1], min=10.0, max=10000.0)
        K_base[:, 0, 2] = torch.clamp(K_base[:, 0, 2], min=-10000.0, max=10000.0)
        K_base[:, 1, 2] = torch.clamp(K_base[:, 1, 2], min=-10000.0, max=10000.0)

        tokens, feats = inputs["tokens"], inputs["features"]

        # ── 2. positional encodings (reference view + T_rst view) ─────────────
        pos = self.position_getter(B * 2, Hp, Wp, device=image.device) + 1
        pos = torch.cat([torch.zeros(B * 2, 1, 2, device=image.device, dtype=pos.dtype), pos], 1)
        # The post-epipolar sequence carries two special slots (scale token + CLS).
        pos_post = self.position_getter(B, Hp, Wp, device=image.device) + 1
        pos_post = torch.cat(
            [torch.zeros(B, 2, 2, device=image.device, dtype=pos_post.dtype), pos_post], 1
        )

        rst_all = self._rst_grid(H, W).unsqueeze(0).expand(B, -1, -1, -1)

        # ── 3. per-level frame + rectified epipolar attention ─────────────────
        scale_state = self._scale_seed(B)
        scale_history, level_outputs = [], []

        for level in range(cfg.num_levels):
            feat = feats[level].reshape(B, -1, cfg.token_dim)
            view = self.token_norm(torch.cat([tokens[level], feat], dim=1)).unsqueeze(1)

            rst = self.token_norm(rst_all[:, level].permute(0, 2, 1)).unsqueeze(1)
            attended = self._frame_attention(torch.cat([view, rst], dim=1), level, pos)

            ref_view = self.output_norm(attended[:, 0:1])                 # (B, 1, 1+N, C)
            rst_view = self.output_norm(attended[:, 1])                   # (B, 1+N, C)
            rst_map = rst_view[:, 1:].transpose(1, 2).reshape(B, cfg.token_dim, Hp, Wp)

            # T_rst -> T_rst-ref: pick up this image's structure (Fig. 3, "Cross Attn.")
            ref_map = ref_view[:, 0, 1:, :].permute(0, 2, 1).reshape(B, cfg.token_dim, Hp, Wp)
            rst_ref = self.rst_conditioning[level](rst_map, ref_map)

            attn_out = self._rectified_epipolar(level, ref_view, rst_ref, K_base, (Hp, Wp))

            level_scale = self._level_scale_query(level, scale_state)
            # Single view: only the reference slot sits in the token sequence.
            fused = torch.cat([level_scale[:, 0:1].unsqueeze(1), attn_out], dim=2)
            fused = self.output_norm(self.frame_blocks[level](
                fused.reshape(B, -1, cfg.token_dim), pos=pos_post
            )).unsqueeze(1)

            spatial = fused[:, :, 1:, :]                                   # CLS + patches
            src_kv = rst_ref.flatten(2).permute(0, 2, 1).unsqueeze(1)      # (B, 1, N, C)
            scale_state = self._scale_readout(
                level, level_scale, spatial[:, :, 1:, :], src_kv
            )
            scale_history.append(scale_state)
            level_outputs.append(spatial[:, 0])                            # (B, 1+N, C)

        # ── 4. token read-outs ────────────────────────────────────────────────
        pooled = self._pool_scale_history(scale_history)
        scale_token = pooled[:, 0, :]                                      # reference slot
        cls_token = level_outputs[-1][:, 0, :]

        # ── 5. heads ──────────────────────────────────────────────────────────
        delta_f = self.camera_head(cls_token)                              # (B, 2)
        K = torch.zeros_like(K_base)
        K[:, 0, 0] = torch.clamp(K_base[:, 0, 0] + delta_f[:, 0] * W, min=10.0, max=10000.0)
        K[:, 1, 1] = torch.clamp(K_base[:, 1, 1] + delta_f[:, 1] * H, min=10.0, max=10000.0)
        K[:, 0, 2] = K_base[:, 0, 2]
        K[:, 1, 2] = K_base[:, 1, 2]
        K[:, 2, 2] = 1.0

        base_depth_norm, backbone_valid = self._base_statistics(depth)
        scale_token_n = F.layer_norm(scale_token, [scale_token.shape[-1]])
        s_hat = self.scale_head(scale_token_n)

        dpt_out = self.dpt_head(
            level_outputs,
            patch_h=Hp,
            patch_w=Wp,
            base_depth=base_depth_norm,
            scale_token=scale_token_n,
        )
        delta_d = dpt_out[:, 0:1, :, :]

        # ── 6. fuse (Eq. 1) ───────────────────────────────────────────────────
        struct = self._compose_struct(base_depth_norm, delta_d)
        struct_norm = self._struct_norm(struct, backbone_valid)
        depth_adj = (struct / struct_norm) * s_hat.view(B, 1, 1, 1)
        depth_adj = torch.clamp(depth_adj, min=cfg.min_depth)

        return self._postprocess(depth_adj, K, meta)

    # Alias kept for readability at call sites.
    @torch.no_grad()
    def infer(self, image):
        return self.forward(image)
