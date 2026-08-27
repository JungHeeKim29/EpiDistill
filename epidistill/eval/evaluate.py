"""Evaluation loop over a packed evaluation set."""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..inference import EpiDistillPredictor, normalize_image
from .metrics import MetricAccumulator, evaluate_sample
from .pickle_dataset import EvalPickleDataset


@torch.no_grad()
def evaluate_dataset(
    predictor: EpiDistillPredictor,
    dataset: EvalPickleDataset,
    align_scale: bool = False,
    limit: int = None,
    progress: bool = True,
) -> dict:
    """Run the predictor over ``dataset`` and average the per-image metrics.

    The prediction is resized to the ground truth resolution before scoring, so
    the metrics never depend on the network's internal working resolution.
    """
    metric = MetricAccumulator()
    n = len(dataset) if limit is None else min(limit, len(dataset))
    it = range(n)
    if progress:
        it = tqdm(it, desc=dataset.dataset, ncols=88)

    for i in it:
        sample = dataset[i]
        image = normalize_image(sample["rgb"], device=predictor.device)
        pred, _ = predictor.infer_tensor(image)

        gt = sample["depth"].to(pred.device)
        if pred.shape[-2:] != gt.shape[-2:]:
            pred = F.interpolate(
                pred, size=gt.shape[-2:], mode="bilinear", align_corners=False
            )

        scores = evaluate_sample(
            pred[0], gt, dataset.min_depth, dataset.max_depth, align_scale=align_scale
        )
        if scores is not None:
            metric.update(scores)

    out = metric.result()
    out["num_samples"] = metric.count
    return out


def format_results(name: str, results: dict) -> str:
    keys = ["abs_rel", "sq_rel", "rms", "log_rms", "d1", "d2", "d3", "silog"]
    body = " | ".join(f"{k}: {results[k]:.4f}" for k in keys if k in results)
    return f"{name:>14s}  ({results.get('num_samples', 0):4d} imgs)  {body}"
