import math

import torch


def build_optimizer(model, lr: float):
    """AdamW in two groups: the scale head without weight decay, everything else with 0.01."""
    scale_head, others = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (scale_head if name.startswith("scale_head.") else others).append(p)
    groups = [{"params": others, "lr": lr, "weight_decay": 0.01, "eps": 1e-8}]
    if scale_head:                       # absent on the DepthPro variant
        groups.append({"params": scale_head, "lr": lr, "weight_decay": 0.0, "eps": 1e-8})
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer, lr: float, warmup_steps: int, total_steps: int, lr_min: float = 1e-6):
    """Linear warm-up to ``lr`` over ``warmup_steps``, then cosine decay to ``lr_min``
    at ``total_steps``; stepped once per optimiser step."""
    floor = lr_min / lr

    def factor(step):
        if step < warmup_steps:
            return max(step / max(1, warmup_steps), floor)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return floor + 0.5 * (1.0 - floor) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def clip_gradients(optimizer, max_norm: float = 1.0):
    """Clip each parameter group separately.  Returns True when every gradient is finite."""
    finite = True
    for group in optimizer.param_groups:
        norm = torch.nn.utils.clip_grad_norm_(group["params"], max_norm)
        finite = finite and bool(torch.isfinite(norm))
    return finite


class WeightEMA:
    """Exponential moving average of the trainable weights (decay 0.999).

    The shadow is seeded with the initial weights and updated after every
    optimiser step.  ``state_dict`` returns the averaged weights in the model's
    own key names, so a saved EMA checkpoint loads like any other.
    """

    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def state_dict(self, model):
        out = model.state_dict()
        for name, value in self.shadow.items():
            out[name] = value.clone()
        return out
