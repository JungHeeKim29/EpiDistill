#!/usr/bin/env python3
"""Pack an evaluation split into a single ``epidistill-eval-v1`` pickle.

The released benchmark pickles are produced with this script.  It reads a flat
directory layout so that it stays independent of any particular dataset SDK::

    root/
      rgb/000000.png          uint8 RGB
      depth/000000.npy        float32 (H, W) metres, 0 or NaN = invalid
      intrinsics/000000.txt   3x3 pixel intrinsics for the RGB above

File stems are matched across the three directories; anything present in ``rgb/``
but missing elsewhere is skipped with a warning.

    python scripts/export_eval_set.py --root data/raw/nyu --dataset NYU \
                                      --output data/eval/nyu.pkl.gz

Depth PNGs are supported via ``--depth-suffix .png --depth-scale 1000`` (the usual
millimetre encoding).  ``--max-side`` downsamples both modalities and rescales the
intrinsics accordingly, which is how the released sets are kept small.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from epidistill.eval import EvalSetWriter, depth_range  # noqa: E402


def load_depth(path: Path, scale: float) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    if path.suffix == ".npz":
        data = np.load(path)
        return data[list(data.keys())[0]].astype(np.float32)
    return np.array(Image.open(path)).astype(np.float32) / scale


def resize_sample(rgb, depth, K, max_side):
    H, W = rgb.shape[:2]
    if max_side is None or max(H, W) <= max_side:
        return rgb, depth, K

    factor = max_side / max(H, W)
    new_W, new_H = int(round(W * factor)), int(round(H * factor))

    rgb = np.array(Image.fromarray(rgb).resize((new_W, new_H), Image.BILINEAR))
    # NEAREST for depth: bilinear would blend valid depth with the 0 sentinel and
    # invent geometry at every occlusion boundary.
    depth = np.array(Image.fromarray(depth).resize((new_W, new_H), Image.NEAREST))

    K = K.copy()
    K[0, :] *= new_W / W
    K[1, :] *= new_H / H
    return rgb, depth, K


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="directory with rgb/ depth/ intrinsics/")
    p.add_argument("--dataset", required=True, help="dataset name, e.g. NYU (sets the range)")
    p.add_argument("--output", required=True, help="destination .pkl or .pkl.gz")
    p.add_argument("--depth-suffix", default=".npy")
    p.add_argument("--depth-scale", type=float, default=1000.0,
                   help="divisor applied to PNG depth (ignored for .npy)")
    p.add_argument("--max-side", type=int, default=None,
                   help="downsample so the longer image side is at most this many pixels")
    p.add_argument("--min-depth", type=float, default=None)
    p.add_argument("--max-depth", type=float, default=None)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    root = Path(args.root)
    rgb_dir, depth_dir, K_dir = root / "rgb", root / "depth", root / "intrinsics"
    for d in (rgb_dir, depth_dir, K_dir):
        if not d.is_dir():
            raise SystemExit(f"missing directory: {d}")

    lo, hi = depth_range(args.dataset)
    lo = args.min_depth if args.min_depth is not None else lo
    hi = args.max_depth if args.max_depth is not None else hi

    writer = EvalSetWriter(args.dataset, min_depth=lo, max_depth=hi)

    files = sorted(f for f in rgb_dir.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if args.limit:
        files = files[: args.limit]

    skipped = 0
    for f in tqdm(files, desc=args.dataset, ncols=88):
        depth_path = depth_dir / (f.stem + args.depth_suffix)
        K_path = K_dir / (f.stem + ".txt")
        if not depth_path.exists() or not K_path.exists():
            skipped += 1
            continue

        rgb = np.array(Image.open(f).convert("RGB"))
        depth = load_depth(depth_path, args.depth_scale)
        K = np.loadtxt(K_path).astype(np.float32).reshape(3, 3)

        rgb, depth, K = resize_sample(rgb, depth, K, args.max_side)
        writer.add(f.stem, rgb, depth, K)

    path = writer.save(args.output)
    size_mb = Path(path).stat().st_size / 1e6
    print(f"wrote {len(writer)} samples to {path} ({size_mb:.1f} MB), "
          f"range [{lo}, {hi}] m" + (f", skipped {skipped}" if skipped else ""))


if __name__ == "__main__":
    main()
