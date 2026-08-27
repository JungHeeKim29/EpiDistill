#!/usr/bin/env python3
"""Run EpiDistill on an image or a folder of images.

    python scripts/demo.py --checkpoint weights/epidistill_unidepthv2.pt \
                           --input assets/example.jpg --output out/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from epidistill import EpiDistillConfig, EpiDistillPredictor  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def colorize(depth: np.ndarray) -> np.ndarray:
    """Inverse-depth turbo-ish colouring, for a quick look."""
    import matplotlib

    inv = 1.0 / np.clip(depth, 1e-3, None)
    lo, hi = np.percentile(inv, 2), np.percentile(inv, 98)
    norm = np.clip((inv - lo) / max(hi - lo, 1e-6), 0, 1)
    cmap = matplotlib.colormaps["turbo"]
    return (cmap(norm)[..., :3] * 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="EpiDistill weights (.pt)")
    p.add_argument("--input", required=True, help="image file or directory")
    p.add_argument("--output", default="out", help="output directory")
    p.add_argument("--backbone", default="unidepthv2", choices=["unidepthv2", "depthpro"])
    p.add_argument("--backbone-checkpoint", default=None,
                   help="override the frozen backbone weights")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-tta", action="store_true", help="disable horizontal-flip TTA")
    p.add_argument("--save-npy", action="store_true", help="also write raw metric depth")
    args = p.parse_args()

    config = EpiDistillConfig(
        backbone=args.backbone,
        backbone_checkpoint=args.backbone_checkpoint,
        use_hflip_tta=not args.no_tta,
    )
    predictor = EpiDistillPredictor.from_pretrained(
        args.checkpoint, config=config, device=args.device
    )

    src = Path(args.input)
    files = (
        sorted(f for f in src.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
        if src.is_dir()
        else [src]
    )
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        rgb = np.array(Image.open(f).convert("RGB"))
        depth, K = predictor(rgb)

        Image.fromarray(colorize(depth)).save(out_dir / f"{f.stem}_depth.png")
        if args.save_npy:
            np.save(out_dir / f"{f.stem}_depth.npy", depth)
        np.savetxt(out_dir / f"{f.stem}_intrinsics.txt", K, fmt="%.6f")

        print(
            f"{f.name}: depth {depth.min():.2f}-{depth.max():.2f} m "
            f"(median {np.median(depth):.2f} m), fx {K[0, 0]:.1f}, fy {K[1, 1]:.1f}"
        )


if __name__ == "__main__":
    main()
