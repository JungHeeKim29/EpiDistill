"""The EpiDistill training objective (Eq. 7).

Consumes the dict returned by ``EpiDistill.forward_multiview``.  Weights are the
released recipe and are module constants, not options.
"""

import os
import sys
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

LOSS_WEIGHTS = {
    # L_rel
    "structure": 1.0,
    "edge": 1.0,
    # L_scale
    "scale": 1.0,
    "dense_scale": 1.0,
    "student_scale": 1.0,           # further divided by the number of decoded views
    # the three explicitly weighted terms of Eq. 7
    "ray": 1.0,
    "distill": 0.5,                 # multiplies (scale-token distill + cls_distill * CLS)
    "cls_distill": 1.0,
    "token_distill_cap": 0.05,      # reached linearly over TOKEN_DISTILL_WARMUP steps
    "attention": 0.1,
    # auxiliary
    "confidence": 0.1,
}
TOKEN_DISTILL_WARMUP = 4000
MIN_VALID_PIXELS = 100              # a sample contributes only with more GT pixels than this
LOG_EPS = 1e-4

_THIRD_PARTY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "UniDepth")
)


def _safe_log(x):
    return torch.log(torch.clamp(x, min=LOG_EPS))


def _edge_guided_ssi():
    if _THIRD_PARTY not in sys.path:
        sys.path.insert(0, _THIRD_PARTY)
    from unidepth.ops.losses.local_ssi import EdgeGuidedLocalSSI  # noqa: E402

    return EdgeGuidedLocalSSI(
        weight=1.0, output_fn="sqrt", min_samples=6, input_fn="log1i", use_global=True, eps=1e-5
    )


class ConfidenceLoss(nn.Module):
    """L1 between the predicted confidence and the absolute depth error, per pixel."""

    def forward(self, confidence, pred, gt, mask):
        B, C = gt.shape[:2]
        gt = gt.float().reshape(B, C, -1)
        pred = pred.float().reshape(B, C, -1)
        confidence = confidence.float().reshape(B, -1)
        mask = mask.reshape(B, -1).float()
        error = torch.abs((pred - gt).norm(dim=1) - confidence)
        per_sample = (error * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1e-6)
        return per_sample.mean()


@dataclass
class ObjectiveTerms:
    total: torch.Tensor
    terms: dict = field(default_factory=dict)


class SupervisedLosses(nn.Module):
    """Depth, scale, focal, edge and confidence terms (part 1 above)."""

    def __init__(self):
        super().__init__()
        self.edge = _edge_guided_ssi()
        self.confidence = ConfidenceLoss()

    def forward(self, out, batch):
        """
        Args:
            out:   ``EpiDistill.forward_multiview`` output.
            batch: dict with ref_image (B,3,H,W), src_images (B,S,3,H,W), ref_depth (B,1,H,W),
                   src_depths (B,S,1,H,W), K_ref (B,3,3), K_src (B,S,3,3).
        Returns:
            ObjectiveTerms, the per-sample log-scale target for the student term (B,),
            and the number of views that produced a scale term.
        """
        S = batch["src_images"].shape[1]
        images = [batch["ref_image"]] + [batch["src_images"][:, i] for i in range(S)]
        gt_depths = [batch["ref_depth"]] + [batch["src_depths"][:, i] for i in range(S)]
        B, _, H, W = images[0].shape
        device = images[0].device

        zero = torch.zeros((), device=device)
        structure = scale = dense = zero
        n_depth_terms = n_scale_terms = 0
        log_scale_target = torch.full((B,), float("nan"), device=device)

        for k, (pred, gt) in enumerate(zip(out["depth"], gt_depths)):
            if pred.shape[-2:] != gt.shape[-2:]:
                pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
            mask = gt > 1e-4
            s_hat = out["s_hat"][k]

            d_acc = s_acc = ds_acc = zero
            n = 0
            for b in range(B):
                m = mask[b]
                if m.sum() <= MIN_VALID_PIXELS:
                    continue
                p, g = pred[b][m], gt[b][m]
                p_med = p.median().clamp(min=0.01)
                g_med = g.median().clamp(min=0.01)
                d_acc = d_acc + (_safe_log(p / p_med) - _safe_log(g / g_med)).abs().mean()

                # Scale terms see the structure as a constant: only s_hat receives gradient.
                mf = s_hat[b].reshape(1)
                structural = (pred[b] / mf.clamp(min=0.01)).detach()
                p_scale = structural[m] * mf
                residual = _safe_log(p_scale.median().clamp(min=0.01)) - _safe_log(g_med)
                s_acc = s_acc + residual.abs()
                ds_acc = ds_acc + (
                    _safe_log(p_scale.clamp(min=0.01)) - _safe_log(g.clamp(min=0.01))
                ).abs().mean()
                if k == 0:
                    # The log-scale that would have been exact for this reference sample.
                    log_scale_target[b] = (_safe_log(mf.reshape(())) - residual).detach()
                n += 1

            if n > 0:
                structure = structure + LOSS_WEIGHTS["structure"] * d_acc / n
                scale = scale + LOSS_WEIGHTS["scale"] * s_acc / n
                dense = dense + LOSS_WEIGHTS["dense_scale"] * ds_acc / n
                n_depth_terms += 1
                n_scale_terms += 1

            if k == 0:
                K_pred, K_gt = out["intrinsics"][0], batch["K_ref"]
                ray = (
                    (K_pred[:, 0, 0] / W - K_gt[:, 0, 0] / W).abs()
                    + (K_pred[:, 1, 1] / H - K_gt[:, 1, 1] / H).abs()
                ).mean()
                pred_c, gt_c = torch.clamp(pred, min=0.01), torch.clamp(gt, min=0.01)
                edge = self.edge(pred_c.float(), gt_c.float(), mask.float(), images[0].detach(), mask.float()).mean()
                conf = self.confidence(out["confidence"][0], pred_c, gt_c, mask)

        structure = structure / max(1, n_depth_terms)
        scale = scale / max(1, n_scale_terms)
        dense = dense / max(1, n_scale_terms)
        ray = LOSS_WEIGHTS["ray"] * ray
        edge = LOSS_WEIGHTS["edge"] * edge
        conf = LOSS_WEIGHTS["confidence"] * conf

        total = structure + ray + edge + conf + scale + dense
        terms = {
            "structure": structure, "scale": scale, "dense_scale": dense,
            "ray": ray, "edge": edge, "confidence": conf,
        }
        return ObjectiveTerms(total, terms), log_scale_target, n_scale_terms


def token_distillation_weight(step: int) -> float:
    cap = LOSS_WEIGHTS["token_distill_cap"]
    return min(cap, step / TOKEN_DISTILL_WARMUP * cap)


def token_distillation_loss(teacher_tokens, student_tokens):
    """Reference-view patch tokens, per level: (1 - cos) + 0.1 |‖t‖ - ‖s‖|, teacher detached.

    Tokens are ``[CLS, patches]``; the CLS slot is left to ``distillation_losses``
    and only the patches enter here, normalised by the number of tokens of
    non-zero norm.  Returns the sum over levels divided by the number of levels.
    """
    total = torch.zeros((), device=student_tokens[0].device)
    for t, s in zip(teacher_tokens, student_tokens):
        t = t[:, 0].detach()
        s = s[:, 0]
        valid = t.norm(dim=-1) > 1e-6                                     # (B, 1+N)
        t_sp, s_sp, valid_sp = t[:, 1:], s[:, 1:], valid[:, 1:].float()
        direction = (1.0 - F.cosine_similarity(s_sp, t_sp, dim=-1)) * valid_sp
        magnitude = (t_sp.norm(dim=-1) - s_sp.norm(dim=-1)).abs() * valid_sp
        count = valid.sum()
        if count > 0:
            total = total + (direction + 0.1 * magnitude).sum() / count
    return total / len(teacher_tokens)


def _fold_teacher_slots(teacher, student):
    """The teacher carries 1 + S scale slots, the student two (reference, T_rst-ref):
    the teacher's source slots are averaged into one so the target has the student's shape."""
    if teacher.dim() == 3 and teacher.shape[1] > student.shape[1] == 2:
        return torch.cat([teacher[:, :1], teacher[:, 1:].mean(dim=1, keepdim=True)], dim=1)
    return teacher


def _cosine_norm_ratio(student, teacher):
    """(1 - mean cos) + 0.1 * L1(‖s‖/‖t‖, 1) with the teacher detached."""
    teacher = _fold_teacher_slots(teacher, student).detach()
    cos = 1.0 - F.cosine_similarity(student, teacher, dim=-1).mean()
    ratio = student.norm(dim=-1) / (teacher.norm(dim=-1) + 1e-6)
    return cos + 0.1 * F.l1_loss(ratio, torch.ones_like(ratio))


def distillation_losses(out):
    """Scale-token and CLS distillation (part 2 above), already combined with their weights.

    Returns ``distill * (scale_token_term + cls_distill * cls_term)`` and the two raw terms.
    """
    t_hist, s_hist = out["teacher_scale_history"], out["student_scale_history"]
    scale_term = torch.zeros((), device=s_hist[0].device)
    for t, s in zip(t_hist, s_hist):
        scale_term = scale_term + _cosine_norm_ratio(s, t)
    scale_term = scale_term + _cosine_norm_ratio(out["student_scale"], out["teacher_scale"])
    scale_term = scale_term / (len(t_hist) + 1)

    cls_term = torch.zeros((), device=s_hist[0].device)
    for t, s in zip(out["teacher_tokens"], out["student_tokens"]):
        cls_term = cls_term + _cosine_norm_ratio(s[:, 0, 0, :], t[:, 0, 0, :])
    cls_term = cls_term / len(out["teacher_tokens"])

    combined = LOSS_WEIGHTS["distill"] * (scale_term + LOSS_WEIGHTS["cls_distill"] * cls_term)
    return combined, scale_term, cls_term


def attention_loss(out):
    """L_attn (Eq. 5) with its weight."""
    return LOSS_WEIGHTS["attention"] * out["attn_loss"]


def student_scale_loss(out, log_scale_target, n_scale_terms):
    """|log s_hat_student - target| over the samples that produced a target."""
    log_s = _safe_log(out["s_hat_student"].reshape(-1).float())
    ok = torch.isfinite(log_scale_target)
    if not ok.any():
        return torch.zeros((), device=log_s.device)
    loss = (log_s[ok] - log_scale_target[ok]).abs().mean()
    return LOSS_WEIGHTS["student_scale"] / max(1, n_scale_terms) * loss
