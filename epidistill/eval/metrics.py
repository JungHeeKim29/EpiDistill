"""Zero-shot metric depth metrics.

Protocol follows UniDepthV2: a pixel counts when its ground truth lies inside the
dataset's depth range; predictions are clipped to the same range; no crop is
applied.  Per-dataset ranges live in :mod:`epidistill.eval.protocols`.
"""

from collections import defaultdict

import torch


def depth_errors(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    """Both tensors are 1-D vectors of valid, already-clipped depths."""
    abs_rel = torch.mean(torch.abs(pred - gt) / gt)
    sq_rel = torch.mean((pred - gt) ** 2 / gt)
    rms = torch.sqrt(torch.mean((pred - gt) ** 2))
    log_rms = torch.sqrt(torch.mean((torch.log(gt) - torch.log(pred)) ** 2))

    thresh = torch.max(gt / pred, pred / gt)
    d1 = (thresh < 1.25).float().mean()
    d2 = (thresh < 1.25 ** 2).float().mean()
    d3 = (thresh < 1.25 ** 3).float().mean()

    # Scale-invariant log error: the spread of the per-pixel log ratio once that
    # image's own scale offset is removed.  Exactly invariant to the multiplicative
    # scale the scale head predicts, so it separates "the shape is wrong" from "one
    # scalar per image is wrong".
    e = torch.log(pred) - torch.log(gt)
    silog = torch.sqrt(torch.clamp((e ** 2).mean() - e.mean() ** 2, min=0.0))

    return {
        "abs_rel": abs_rel.item(),
        "sq_rel": sq_rel.item(),
        "rms": rms.item(),
        "log_rms": log_rms.item(),
        "d1": d1.item(),
        "d2": d2.item(),
        "d3": d3.item(),
        "silog": silog.item(),
    }


def median_scale(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Median-alignment factor -- for reporting scale-aligned (shape-only) numbers."""
    return (gt.median() / pred.median().clamp(min=1e-6)).item()


class MetricAccumulator:
    """Averages per-image metrics over a dataset."""

    def __init__(self):
        self.sums = defaultdict(float)
        self.count = 0

    def update(self, metrics: dict):
        for k, v in metrics.items():
            self.sums[k] += v
        self.count += 1

    def result(self) -> dict:
        if self.count == 0:
            return {}
        return {k: v / self.count for k, v in self.sums.items()}


@torch.no_grad()
def evaluate_sample(pred, gt, min_depth: float, max_depth: float, align_scale: bool = False):
    """pred / gt: (H, W) or (1, H, W) tensors in metres.  Returns a metric dict or None."""
    pred = pred.reshape(-1).float()
    gt = gt.reshape(-1).float()

    mask = (gt > min_depth) & (gt < max_depth)
    if mask.sum() == 0:
        return None

    pred, gt = pred[mask], gt[mask]
    if align_scale:
        pred = pred * median_scale(pred, gt)
    pred = torch.clamp(pred, min=min_depth, max=max_depth)
    return depth_errors(pred, gt)
