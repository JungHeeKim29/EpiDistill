"""Checkpoint loading, including conversion from earlier key names.

The released weights use the paper's names.  ``convert_legacy_state_dict``
rewrites the key names a checkpoint may carry from an earlier internal naming, so
either form loads unchanged.
"""

from collections import OrderedDict

import torch

# Longest prefixes first -- the rewrite is applied greedily.
KEY_RENAMES = (
    ("viewwise_blocks.", "frame_blocks."),
    ("focal_fov_proj.", "scale_fov_proj."),
    ("focal_fov_attn.", "scale_fov_attn."),
    ("focal_camera_head.", "scale_focal_head."),
    ("virtual_img_conditioning.", "reference_guided_attn."),
    ("virtual_tokens", "rectified_stereo_tokens"),
    ("focal_cross_view_attn.", "scale_cross_view_attn."),
    ("focal_level_self_attn.", "scale_level_attn."),
    ("focal_level_proj.", "scale_level_proj."),
    ("focal_level_seed", "scale_level_seed"),
    ("focal_src_token", "scale_src_token"),
    ("focal_token", "scale_token"),
    ("light_dpt.", "structure_head."),
    ("camera_head.", "focal_head."),
)

# Modules with no counterpart in the released architecture.
DROP_PREFIXES = (
    "model.",                    # the frozen backbone (loaded from its own weights)
    "pixel_decoder.",            # UniDepth decoder copy, used by an alternative DPT mode
    "decoder_adapters.",
    "cls_cross_view_attn.",      # CLS cross-view enrichment ablation
    "scale_head.cls_proj.",      # CLS input channel to the scale head, held at zero
    "focal_tokens_per_level",    # superseded by scale_level_seed
    "focal_extra_tokens",        # multi-register scale read-out ablation
    "focal_extra_merge.",
    "focal_cls_seed_proj.",      # CLS-conditioned level-0 query ablation
    "depthpro_feat_projs.",
    "depthpro_tok_projs.",
    "depthpro_light_dpt.",
)


def convert_legacy_state_dict(state_dict, verbose: bool = False):
    """Rewrite parameter names from an earlier naming into the released one."""
    out = OrderedDict()
    dropped = []
    for key, value in state_dict.items():
        name = key[len("module."):] if key.startswith("module.") else key

        if any(name == p.rstrip(".") or name.startswith(p) for p in DROP_PREFIXES):
            dropped.append(name)
            continue

        for old, new in KEY_RENAMES:
            if name.startswith(old):
                name = new + name[len(old):]
                break
        out[name] = value

    if verbose and dropped:
        print(f"[checkpoint] dropped {len(dropped)} tensors not part of the model")
    return out


def load_checkpoint(model, path, strict: bool = True, map_location="cpu", verbose: bool = True):
    """Load a checkpoint into an :class:`EpiDistill`.

    Accepts a raw ``state_dict`` or any of the usual wrappers
    (``{'model_state_dict': ...}``, ``{'state_dict': ...}``, ``{'model': ...}``).
    """
    blob = torch.load(path, map_location=map_location, weights_only=False)
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(blob, dict) and key in blob and isinstance(blob[key], dict):
            blob = blob[key]
            break

    state = convert_legacy_state_dict(blob, verbose=verbose)

    # Heads the released variant does not have (DepthPro reads its focal residual
    # and scale from the scale token, so the CLS camera head and the scale head
    # are absent).  A checkpoint may still carry them, untrained.
    candidates = ("scale_head.", "focal_head.", "scale_focal_head.",
                  "scale_fov_proj.", "scale_fov_attn.")
    absent = tuple(p for p in candidates if not hasattr(model, p[:-1]))
    if absent:
        n_before = len(state)
        state = OrderedDict((k, v) for k, v in state.items() if not k.startswith(absent))
        if verbose:
            print(f"[checkpoint] dropped {n_before - len(state)} tensors of absent heads {absent}")

    missing, unexpected = model.load_state_dict(state, strict=False)

    if verbose:
        print(f"[checkpoint] loaded {len(state)} tensors from {path}")
        if missing:
            print(f"[checkpoint] missing keys ({len(missing)}): {missing[:8]}"
                  + (" ..." if len(missing) > 8 else ""))
        if unexpected:
            print(f"[checkpoint] unexpected keys ({len(unexpected)}): {unexpected[:8]}"
                  + (" ..." if len(unexpected) > 8 else ""))

    if strict and (missing or unexpected):
        raise RuntimeError(
            f"checkpoint does not match the model: {len(missing)} missing, "
            f"{len(unexpected)} unexpected keys (pass strict=False to ignore)"
        )
    return model
