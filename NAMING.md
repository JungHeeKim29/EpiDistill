# Naming: paper ↔ code ↔ research code base

EpiDistill grew out of an internal research code base whose names predate the
paper. This release uses the paper's terminology throughout; the table below is
the authoritative mapping, and `epidistill/utils/checkpoint.py` applies exactly
these renames when loading a research checkpoint.

## Concepts

| Paper (Sec. / Eq.) | Release code | Research code base |
|---|---|---|
| Scale token `F_scale` | `scale_token`, `scale_src_token` | `focal_token`, `focal_src_token` |
| Rectified Stereo Tokens `T_rst` | `rst_tokens` | `virtual_tokens` |
| Reference-guided tokens `T_rst-ref` | `rst_ref` (local), `rst_conditioning` | `updated_virtual_token`, `virtual_img_conditioning` |
| Reference image tokens `F_ref` | `ref_view`, `ref_map` | `fmv_token`, `ref_spatial` |
| Depth-Guided Epipolar Attention | `DepthGuidedEpipolarAttention` | `EpipolarAttention` |
| Frame attention | `frame_blocks`, `_frame_attention` | `viewwise_blocks`, `framewise_attention` |
| Global scale `ŝ` (Eq. 1) | `s_hat`, `ScaleHead` | `metric_factor` / `mf`, `MetricFactorHead` |
| Structural shift `Δd` (Eq. 1) | `delta_d`, `DPTHead` | `shift_adjust`, `light_dpt` |
| Normalised base depth `D_base` | `base_depth_norm` | `base_depth_norm` |
| Adjusted depth `D_adj` | `depth_adj` | `final_depth_map` |
| Focal residuals `(δx, δy)` (Eq. 11) | `delta_f`, `CameraHead` | `delta_intrinsics`, `SimpleCameraHead` |
| Attention loss `L_attn` (Eq. 5) | `return_attn_loss` | `return_matching_loss` |

`focal token` → **scale token** is the single most important rename: the token was
originally introduced to carry focal-length information, but what it actually
learns — and what the paper describes — is the scene's absolute scale, which the
scale head reads and the camera head reads only as a residual.

`virtual token` → **rectified stereo token** likewise: the name in the paper says
what the token *is* (a stand-in source view under a rectified stereo geometry),
not merely that it is synthetic.

## State-dict prefixes

| Release | Research |
|---|---|
| `frame_blocks.*` | `viewwise_blocks.*` |
| `epipolar_blocks.*` | `epipolar_blocks.*` |
| `rst_tokens` | `virtual_tokens` |
| `rst_conditioning.*` | `virtual_img_conditioning.*` |
| `scale_token`, `scale_src_token` | `focal_token`, `focal_src_token` |
| `scale_level_seed` | `focal_level_seed` |
| `scale_level_proj.*` | `focal_level_proj.*` |
| `scale_cross_view_attn.*` | `focal_cross_view_attn.*` |
| `scale_level_attn.*` | `focal_level_self_attn.*` |
| `dpt_head.*` | `light_dpt.*` |
| `scale_head.*`, `camera_head.*`, `token_norm.*`, `output_norm.*` | unchanged |

Research-only modules (ablations and the frozen backbone itself) are dropped on
load; the list is `DROP_PREFIXES` in `epidistill/utils/checkpoint.py`.
