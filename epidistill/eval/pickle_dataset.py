"""Self-contained evaluation sets, packed as a single pickle per dataset.

Motivation: the zero-shot benchmarks (KITTI, NYU, Bonn, DIODE, ETH3D, ...) each
ship a different directory layout, depth encoding and split file, which makes
"same protocol" hard to guarantee across papers.  Packing the exact evaluated
frames into one file removes every one of those degrees of freedom -- a reader
only has to load the file.

Format (``epidistill-eval-v1``)::

    {
      "format":     "epidistill-eval-v1",
      "dataset":    "NYU",
      "min_depth":  0.01,
      "max_depth":  10.0,
      "samples": [
          {"name":       "000000",
           "rgb":        <PNG-encoded bytes, uint8 HxWx3>,
           "depth":      <np.float16 (H, W), 0 = invalid>,
           "intrinsics": <np.float32 (3, 3)>},
          ...
      ],
    }

RGB is stored PNG-encoded and depth as float16, which keeps a few-hundred-frame
benchmark in the tens of megabytes.  Files may be gzip-compressed (``.pkl.gz``).
"""

import gzip
import io
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

FORMAT_VERSION = "epidistill-eval-v1"


# ── encoding helpers ──────────────────────────────────────────────────────────
def encode_rgb(rgb: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb.astype(np.uint8))).save(buf, format="PNG")
    return buf.getvalue()


def decode_rgb(blob: bytes) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(io.BytesIO(blob)).convert("RGB"))


def _open(path, mode):
    path = str(path)
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)


# ── writing ───────────────────────────────────────────────────────────────────
class EvalSetWriter:
    """Collect samples, then :meth:`save` them as one evaluation pickle."""

    def __init__(self, dataset: str, min_depth: float, max_depth: float):
        self.dataset = dataset
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.samples = []

    def add(self, name: str, rgb: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray):
        """rgb: uint8 (H, W, 3).  depth: float (H, W) in metres, 0 = invalid.
        intrinsics: (3, 3) pixel intrinsics for the *stored* RGB resolution."""
        assert rgb.ndim == 3 and rgb.shape[2] == 3, f"rgb must be HxWx3, got {rgb.shape}"
        assert depth.shape[:2] == rgb.shape[:2], (
            f"depth {depth.shape[:2]} must match rgb {rgb.shape[:2]}; "
            "resize before adding so the stored intrinsics stay valid"
        )
        self.samples.append(
            {
                "name": str(name),
                "rgb": encode_rgb(rgb),
                "depth": np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16),
                "intrinsics": np.asarray(intrinsics, dtype=np.float32).reshape(3, 3),
            }
        )

    def save(self, path):
        payload = {
            "format": FORMAT_VERSION,
            "dataset": self.dataset,
            "min_depth": self.min_depth,
            "max_depth": self.max_depth,
            "samples": self.samples,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with _open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    def __len__(self):
        return len(self.samples)


# ── reading ───────────────────────────────────────────────────────────────────
class EvalPickleDataset(Dataset):
    """Reads an ``epidistill-eval-v1`` pickle.

    Each item is ``{'name', 'rgb' (uint8 HWC), 'depth' (1, H, W), 'intrinsics' (3, 3)}``.
    RGB is returned raw; normalise it with :func:`epidistill.inference.normalize_image`.
    """

    def __init__(self, path):
        with _open(path, "rb") as f:
            payload = pickle.load(f)

        fmt = payload.get("format")
        if fmt != FORMAT_VERSION:
            raise ValueError(f"{path}: expected format {FORMAT_VERSION!r}, got {fmt!r}")

        self.path = str(path)
        self.dataset = payload["dataset"]
        self.min_depth = payload["min_depth"]
        self.max_depth = payload["max_depth"]
        self.samples = payload["samples"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]
        depth = torch.from_numpy(s["depth"].astype(np.float32)).unsqueeze(0)
        return {
            "name": s["name"],
            "rgb": decode_rgb(s["rgb"]),
            "depth": depth,
            "intrinsics": torch.from_numpy(s["intrinsics"]),
        }
