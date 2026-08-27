# EpiDistill

**Geometric Distillation from Rectified Stereo: Leveraging Epipolar Cues for Monocular Depth**

Jung-Hee Kim, Xiaoming Liu · Michigan State University / UNC Chapel Hill

[Paper](#) · [Project page](#)

EpiDistill transfers the scale-aware geometric priors of multi-view foundation
models into a **single-view** metric depth model. A multi-view teacher learns
correspondence through *Depth-Guided Epipolar Attention*; at inference the source
views are replaced by learnable **Rectified Stereo Tokens**, so the cross-view
attention pathway never collapses into plain intra-frame self-attention. The
method is model-agnostic and is released on top of two frozen ViT depth
backbones, **UniDepthV2 (ViT-L)** and **DepthPro**.

---

## Status

This repository currently ships the **model definition and the single-view
inference / evaluation path**. See [Roadmap](#roadmap) for what is coming.

| Component | State |
|---|---|
| Single-view model + inference (UniDepthV2, DepthPro) | ✅ in this release |
| Packed zero-shot evaluation sets + evaluation script | ✅ code; 📦 data to be published |
| Pretrained EpiDistill weights | 📦 to be published |
| Multi-view teacher + training code (`L_attn`, `L_distill`, ...) | 🚧 planned |

---

## Installation

```bash
git clone <repo-url> && cd EpiDistill
conda create -n epidistill python=3.10 -y && conda activate epidistill
# install the torch build that matches your CUDA first, then:
pip install -r requirements.txt
pip install -e .
```

Backbone weights:

* **UniDepthV2** downloads automatically from the Hugging Face Hub
  (`lpiccinelli/unidepth-v2-vitl14`) on first use.
* **DepthPro** needs `depth_pro.pt` from
  [apple/ml-depth-pro](https://github.com/apple/ml-depth-pro); place it at
  `third_party/DepthPro/checkpoints/depth_pro.pt`, or pass
  `--backbone-checkpoint /path/to/depth_pro.pt`.

Both backbones are vendored under `third_party/` (see
[THIRD_PARTY.md](THIRD_PARTY.md) for licences — note UniDepth is **CC BY-NC 4.0**).

---

## Quick start

```python
import numpy as np
from PIL import Image
from epidistill import EpiDistillConfig, EpiDistillPredictor

predictor = EpiDistillPredictor.from_pretrained(
    "weights/epidistill_unidepthv2.pt",
    config=EpiDistillConfig(backbone="unidepthv2"),
)

rgb = np.array(Image.open("assets/example.jpg").convert("RGB"))
depth, K = predictor(rgb)     # depth: (H, W) float32 metres; K: (3, 3) adjusted intrinsics
```

Command line:

```bash
python scripts/demo.py --checkpoint weights/epidistill_unidepthv2.pt \
                       --input assets/ --output out/ --save-npy
```

---

## Evaluation

Zero-shot benchmarks are distributed as **one self-contained pickle per dataset**
(`epidistill-eval-v1`): the exact evaluated frames, their ground-truth depth and
their intrinsics, packed together. This removes the usual per-dataset differences
in directory layout, depth encoding and split files, so a number produced here is
reproducible without any dataset SDK.

```bash
python scripts/evaluate.py --checkpoint weights/epidistill_unidepthv2.pt \
                           --eval-sets data/eval/{kitti,nyu,bonn,diode,eth3d}.pkl.gz
```

To pack your own split, arrange it as `rgb/ depth/ intrinsics/` and run:

```bash
python scripts/export_eval_set.py --root data/raw/nyu --dataset NYU \
                                  --output data/eval/nyu.pkl.gz --max-side 640
```

Metrics follow the UniDepthV2 protocol: a pixel counts when its ground truth is
inside the dataset's depth range, predictions are clipped to the same range, and
no crop is applied. Per-dataset ranges are in
[`epidistill/eval/protocols.py`](epidistill/eval/protocols.py). Inference uses
bf16 autocast with horizontal-flip test-time augmentation, matching the paper.

---

## Method

```
                    frozen ViT depth backbone  (UniDepthV2 / DepthPro)
                                 │
        ┌────────────────────────┼─────────────────────────┐
        │                        │                         │
   F_ref (L levels)         CLS tokens              D_orig , f_base
        │                        │                         │
        │   T_rst ──cross-attn──►│                         │
        │        (Sec. 4.2)      │                         │
        ▼                        │                         │
   frame attention               │                         │
        │                        │                         │
        ▼                        │                         │
   rectified epipolar attention  │                         │
        │                        │                         │
        ├──► F_scale ──► scale head ──► ŝ                  │
        │           └──► camera head ──► (δx, δy) ─────────┼──► f_adj = f_base + (δx W, δy H)
        ▼                                                  │
     DPT head ──► Δd ──────────────────────────────────────┴──► D_adj = ŝ · (D_base + Δd)
```

Two components carry the method:

**Depth-Guided Epipolar Attention** (`epidistill/modules/epipolar_attention.py`).
A reference query attends only to points sampled along its epipolar line in the
source view. During multi-view training the ground-truth depth gives the exact
correspondence, a Gaussian spatial bias builds a target attention map around it,
and a cross-entropy loss distils that target into the model's own attention.

**Rectified Stereo Tokens** (`epidistill/modules/aggregation.py`,
`EpiDistill._rectified_epipolar`). In a rectified stereo setup `R = I` and
`t = [b, 0, 0]`, so the baseline cancels out of the epipolar line equation and the
search space collapses to the query's own horizontal scanline — a *baseline-agnostic*
geometry that a learnable token grid can stand in for. `T_rst` cross-attends to the
reference features to become `T_rst-ref`, then plays the source view's role inside
the epipolar attention, keeping the cross-view pathway alive with one image.

The prediction is decoupled into structure and scale (Eq. 1). `D_base` is the
backbone's depth normalised by its own masked median; the DPT head predicts a
dense residual `Δd`, and the scale token drives a single global factor `ŝ`. The
released configuration composes the residual multiplicatively —
`D_struct = D_base · exp(τ·tanh(Δd/τ))` — which keeps the backbone's structure a
*factor* rather than a *summand*; see `EpiDistill._compose_struct` for why that
matters. At initialisation every head is zero-init, so the model reproduces the
frozen backbone exactly.

---

## Repository layout

```
epidistill/
  model.py               EpiDistill: preprocessing, attention stack, heads, composition
  config.py              EpiDistillConfig -- the released defaults
  inference.py           EpiDistillPredictor (bf16 + hflip TTA)
  modules/
    epipolar_attention.py  depth-guided / rectified epipolar attention (Sec. 4.1-4.2)
    aggregation.py         scale-token read-out, level fusion, T_rst conditioning
    block.py               frame / epipolar transformer block
    attention.py mlp.py rope.py layer_scale.py drop_path.py
  heads/
    scale_head.py          global metric scale  ŝ        (Eq. 10)
    camera_head.py         focal residual (δx, δy)       (Eq. 11)
    dpt_head.py            dense structural shift Δd     (Eq. 1)
  backbones/
    unidepth_v2.py depth_pro.py base.py
  eval/
    pickle_dataset.py protocols.py metrics.py evaluate.py
  utils/checkpoint.py    checkpoint loading + research-name conversion
scripts/
  demo.py evaluate.py export_eval_set.py
third_party/
  UniDepth/ DepthPro/    vendored backbones
```

Variable names follow the paper; see [NAMING.md](NAMING.md) for the mapping from
the research code base (`focal token` → **scale token**, `virtual token` →
**rectified stereo token**).

---

## Roadmap

- [ ] Publish pretrained EpiDistill weights for both backbones
- [ ] Publish packed evaluation sets (KITTI, NYU, Bonn, DIODE, ETH3D)
- [ ] Multi-view teacher forward + the full training objective
- [ ] Training data pipeline (multi-view tuple sampling with the 20% overlap rule)
- [ ] Point-cloud reconstruction evaluation (Fig. 1)

---

## Citation

```bibtex
@inproceedings{kim2026epidistill,
  title     = {Geometric Distillation from Rectified Stereo:
               Leveraging Epipolar Cues for Monocular Depth},
  author    = {Kim, Jung-Hee and Liu, Xiaoming},
  booktitle = {ECCV},
  year      = {2026}
}
```

## Acknowledgements

Built on [UniDepth](https://github.com/lpiccinelli-eth/UniDepth),
[DepthPro](https://github.com/apple/ml-depth-pro), [DPT](https://github.com/isl-org/DPT)
and [DINOv2](https://github.com/facebookresearch/dinov2).
