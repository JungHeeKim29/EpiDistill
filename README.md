# EpiDistill

**Geometric Distillation from Rectified Stereo: Leveraging Epipolar Cues for Monocular Depth**

Jung-Hee Kim, Xiaoming Liu · Michigan State University / UNC Chapel Hill

[Paper](https://arxiv.org/abs/2607.15600) · [Project page](https://jungheekim29.github.io/EpiDistill/)

EpiDistill turns a frozen depth foundation model into a **single-view** metric
depth model. A multi-view teacher learns correspondence through depth-guided
epipolar attention; at inference the source views are replaced by learnable
**Rectified Stereo Tokens**, so the cross-view pathway survives with one image.
It is released on two frozen ViT backbones, **UniDepthV2 (ViT-L)** and
**DepthPro**.

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

EpiDistill's own weights are on the Hugging Face Hub
([kimjun84/EpiDistill](https://huggingface.co/kimjun84/EpiDistill)):

```python
from huggingface_hub import hf_hub_download

for name in ("epidistill_unidepthv2.pt", "epidistill_depthpro.pt"):
    hf_hub_download("kimjun84/EpiDistill", name, local_dir="weights")
```

**For training only**, the edge-guided loss needs UniDepth's CUDA
patch-extraction op:

```bash
cd third_party/UniDepth/unidepth/ops/extract_patches && bash compile.sh
```

---

## Inference

```python
import numpy as np
from PIL import Image
from epidistill import EpiDistillConfig, EpiDistillPredictor

predictor = EpiDistillPredictor.from_pretrained(
    "weights/epidistill_unidepthv2.pt",
    config=EpiDistillConfig(backbone="unidepthv2"),
)

rgb = np.array(Image.open("your_image.jpg").convert("RGB"))
depth, K = predictor(rgb)     # depth: (H, W) float32 metres; K: (3, 3) adjusted intrinsics
```

Command line:

```bash
python scripts/demo.py --checkpoint weights/epidistill_unidepthv2.pt \
                       --input images/ --output out/ --save-npy
```

Add `--backbone depthpro --checkpoint weights/epidistill_depthpro.pt` for the
other model.

---

## Training

`EpiDistill.forward_multiview` runs the multi-view teacher and the single-view
student together: the teacher attends across the real source views along their
epipolar lines, the student replaces them with the rectified stereo tokens, and
the objective supervises the teacher's geometry while distilling it into the
student. `epidistill/train/` turns their outputs into a loss and a step:

```
epidistill/train/losses.py    the objective
epidistill/train/optim.py     optimiser parameter groups, learning-rate schedule, weight EMA
```

A step takes a reference view with `S` source views: their images, ground-truth
depths, intrinsics, and the transforms mapping reference-camera points into each
source camera. To run one:

```bash
python tests/test_training_step.py
```

That is a complete step on a stub backbone on CPU — the executable description of
the batch contract and of every loss term.

---

## License

EpiDistill is released under **CC BY-NC 4.0** (see [LICENSE](LICENSE)) —
non-commercial, because the repository vendors UniDepth, which is itself
CC BY-NC 4.0. Third-party components keep their own licences; see
[THIRD_PARTY.md](THIRD_PARTY.md).

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
