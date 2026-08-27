from .base import DepthBackbone
from .depth_pro import DepthProBackbone
from .unidepth_v2 import UniDepthV2Backbone

BACKBONES = {
    "unidepthv2": UniDepthV2Backbone,
    "depthpro": DepthProBackbone,
}


def build_backbone(name: str, **kwargs) -> DepthBackbone:
    if name not in BACKBONES:
        raise ValueError(f"unknown backbone {name!r}; available: {sorted(BACKBONES)}")
    return BACKBONES[name](**kwargs)


__all__ = [
    "DepthBackbone",
    "DepthProBackbone",
    "UniDepthV2Backbone",
    "BACKBONES",
    "build_backbone",
]
