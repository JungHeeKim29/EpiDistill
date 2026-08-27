#!/usr/bin/env python3
"""Zero-shot evaluation on packed evaluation sets.

    python scripts/evaluate.py --checkpoint weights/epidistill_unidepthv2.pt \
                               --eval-sets data/eval/*.pkl.gz
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from epidistill import EpiDistillConfig, EpiDistillPredictor  # noqa: E402
from epidistill.eval import EvalPickleDataset, evaluate_dataset, format_results  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval-sets", nargs="+", required=True,
                   help="one or more epidistill-eval-v1 pickles")
    p.add_argument("--backbone", default="unidepthv2", choices=["unidepthv2", "depthpro"])
    p.add_argument("--backbone-checkpoint", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--align-scale", action="store_true",
                   help="median-align each prediction (shape-only numbers)")
    p.add_argument("--limit", type=int, default=None, help="evaluate the first N images only")
    args = p.parse_args()

    config = EpiDistillConfig(
        backbone=args.backbone,
        backbone_checkpoint=args.backbone_checkpoint,
        use_hflip_tta=not args.no_tta,
    )
    predictor = EpiDistillPredictor.from_pretrained(
        args.checkpoint, config=config, device=args.device
    )

    rows = []
    for path in args.eval_sets:
        dataset = EvalPickleDataset(path)
        results = evaluate_dataset(
            predictor, dataset, align_scale=args.align_scale, limit=args.limit
        )
        rows.append((dataset.dataset, results))
        print(format_results(dataset.dataset, results))

    if len(rows) > 1:
        mean_abs_rel = sum(r["abs_rel"] for _, r in rows) / len(rows)
        mean_d1 = sum(r["d1"] for _, r in rows) / len(rows)
        print("-" * 88)
        print(f"{'mean':>14s}  over {len(rows)} sets   "
              f"abs_rel: {mean_abs_rel:.4f} | d1: {mean_d1:.4f}")


if __name__ == "__main__":
    main()
