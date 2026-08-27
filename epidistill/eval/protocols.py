"""Per-dataset evaluation ranges (UniDepthV2 protocol)."""

DEPTH_RANGES = {
    "KITTI": (0.01, 80.0),
    "NYU": (0.01, 10.0),
    "BONN": (0.01, 10.0),
    "BOOSTER": (0.01, 10.0),
    "IBIMS": (0.01, 50.0),
    "ETH3D": (0.01, 50.0),
    "DIODE": (0.01, 80.0),
    "DIODEIndoor": (0.01, 25.0),
    "DIODEOutdoor": (0.01, 80.0),
    "DDAD": (1.0, 200.0),
    "SCANNETPP": (0.01, 40.0),
}

DEFAULT_RANGE = (0.01, 40.0)


def depth_range(dataset: str):
    """Longest matching name wins, so 'DIODEIndoor' does not fall back to 'DIODE'."""
    matches = [k for k in DEPTH_RANGES if k.upper() in dataset.upper()]
    if not matches:
        return DEFAULT_RANGE
    return DEPTH_RANGES[max(matches, key=len)]
