"""Smoke test: one complete training step on a stub backbone, on CPU.

Exercises the multi-view teacher (depth-guided epipolar attention with the GT
bias and L_attn), the student, every loss term and one optimiser/EMA update.

The edge-guided SSI term is UniDepth's and relies on its CUDA patch-extraction
op, which has no CPU kernel; on CPU it is replaced by a zero so the rest of the
objective can still be checked.  Run with a GPU to exercise it for real.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import epidistill.model as model_mod  # noqa: E402
from epidistill.config import EpiDistillConfig  # noqa: E402
from epidistill.train import (  # noqa: E402
    SupervisedLosses,
    WeightEMA,
    attention_loss,
    build_optimizer,
    build_scheduler,
    clip_gradients,
    distillation_losses,
    student_scale_loss,
    token_distillation_loss,
    token_distillation_weight,
)
from test_forward_shapes import StubBackbone  # noqa: E402


def make_batch(B=1, S=2, H=224, W=308):
    g = torch.Generator().manual_seed(0)
    K = torch.tensor([[300.0, 0.0, W / 2], [0.0, 300.0, H / 2], [0.0, 0.0, 1.0]])
    T = torch.eye(4).repeat(B, S, 1, 1)
    T[:, :, 0, 3] = torch.tensor([-0.2, 0.2])[:S]                 # sideways baselines
    depth = 2.0 + 3.0 * torch.rand(B, 1, H, W, generator=g)
    depth[..., :10, :] = 0.0                                       # some invalid GT
    return {
        "ref_image": torch.randn(B, 3, H, W, generator=g),
        "src_images": torch.randn(B, S, 3, H, W, generator=g),
        "ref_depth": depth,
        "src_depths": 2.0 + 3.0 * torch.rand(B, S, 1, H, W, generator=g),
        "K_ref": K.repeat(B, 1, 1),
        "K_src": K.repeat(B, S, 1, 1),
        "T_ref2src": T,
    }


def test_training_step(backbone="unidepthv2"):
    model_mod.build_backbone = lambda name, **kwargs: StubBackbone(**kwargs)
    torch.manual_seed(0)
    model = model_mod.EpiDistill(EpiDistillConfig(backbone=backbone), device=torch.device("cpu")).train()
    supervised = SupervisedLosses()
    if not torch.cuda.is_available():
        del supervised.edge
        supervised.edge = lambda pred, gt, mask, image, vmask: pred.new_zeros(())
    optimizer = build_optimizer(model, 3e-5)
    scheduler = build_scheduler(optimizer, 5e-5, warmup_steps=300, total_steps=30000)
    ema = WeightEMA(model)
    batch = make_batch()

    out = model.forward_multiview(
        batch["ref_image"], batch["src_images"], batch["K_ref"], batch["K_src"],
        batch["T_ref2src"], batch["ref_depth"],
    )
    V = 1 + batch["src_images"].shape[1]
    assert len(out["depth"]) == V and out["depth"][0].shape == (1, 1, 224, 308)
    assert len(out["teacher_tokens"]) == 4 and out["teacher_tokens"][0].shape[1] == V
    assert out["teacher_scale"].shape == (1, 1 + batch["src_images"].shape[1], 1024)
    assert out["student_scale"].shape == (1, 2, 1024)
    assert len(out["s_hat"]) == V
    assert torch.isfinite(out["attn_loss"]) and out["attn_loss"] > 0, "L_attn should be live"

    sup, target, n_scale = supervised(out, batch)
    tok = token_distillation_loss(out["teacher_tokens"], out["student_tokens"])
    distill, _, _ = distillation_losses(out)
    loss = (sup.total + token_distillation_weight(4000) * tok + distill
            + attention_loss(out) + student_scale_loss(out, target, n_scale))
    assert torch.isfinite(loss), loss
    for name, value in sup.terms.items():
        assert torch.isfinite(value), name

    optimizer.zero_grad()
    loss.backward()
    grads = {n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    # (scale_token / scale_src_token are excluded: the level-0 query projection is
    # zero-init, so the seeds receive an exactly-zero gradient at initialisation.)
    heads = ("scale_focal_head",) if backbone == "depthpro" else ("scale_head", "focal_head")
    for must in ("epipolar_blocks.0", "rectified_stereo_tokens", "reference_guided_attn.0", "structure_head",
                 "scale_cross_view_attn.0", "scale_level_attn", "frame_blocks.0") + heads:
        assert any(n.startswith(must) for n in grads), f"no gradient reached {must}"
    if backbone == "depthpro":
        assert any(n.startswith("scale_fov_proj") for n, p in model.named_parameters() if p.grad is not None)
    assert clip_gradients(optimizer, 1.0)
    optimizer.step()
    ema.update(model)
    scheduler.step()
    return loss.item(), sup.terms


if __name__ == "__main__":
    for backbone in ("unidepthv2", "depthpro"):
        loss, terms = test_training_step(backbone)
        print(f"[{backbone}] training step ok: loss {loss:.4f}",
              {k: round(v.item(), 4) for k, v in terms.items()})
