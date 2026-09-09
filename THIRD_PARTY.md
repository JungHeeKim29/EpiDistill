# Third-party code and licences

## Vendored under `third_party/`

### UniDepth (`third_party/UniDepth`)
Piccinelli et al., *UniDepthV2: Universal Monocular Metric Depth Estimation Made
Simpler*. Source: https://github.com/lpiccinelli-eth/UniDepth
Licence: **Creative Commons Attribution-NonCommercial 4.0 International
(CC BY-NC 4.0)** — see `third_party/UniDepth/LICENSE`.

> Because UniDepth is non-commercial, any use of EpiDistill **with the UniDepthV2
> backbone** inherits that restriction.

### DepthPro (`third_party/DepthPro`)
Bochkovskii et al., *Depth Pro: Sharp Monocular Metric Depth in Less Than a Second*.
Source: https://github.com/apple/ml-depth-pro
Licence: Apple's licence, as published in that repository.

Licence: see `third_party/DepthPro/LICENSE`. Model weights (`depth_pro.pt`) are
**not** redistributed here; download them from the upstream repository.

### Local modification to the vendored DepthPro

`third_party/DepthPro/network/fov.py` — `FOVNetwork.forward` splits its
`self.encoder` Sequential into its two steps so the raw tokens of the fov
encoder can be read out (EpiDistill's scale token attends to them on the
DepthPro variant). Same operations, same numbers; the depth DepthPro produces is
unchanged.

## Derived source files

| File | Origin | Licence |
|---|---|---|
| `epidistill/modules/{attention,block,layer_scale,drop_path,mlp}.py` | DINOv2 (Meta) | Apache-2.0 |
| `epidistill/modules/rope.py` | RoPE-2D, as used by VGGT / DINOv2 derivatives | Apache-2.0 |
| `epidistill/heads/dpt_head.py` | DPT (Intel ISL) | MIT |

## EpiDistill's own code

Released under **CC BY-NC 4.0** (see `LICENSE`), matching UniDepth so that the
combined distribution is self-consistent.
