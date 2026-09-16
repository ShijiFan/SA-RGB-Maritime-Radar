"""SA-RGB: Slot-Addressable RGB Radar Grids for Multi-Feature Small-Target Detection.

This package provides:
- Core physical feature extraction (RAA, RDPH, RVE, Hurst, RI, NR, MS)
- Slot-addressable RGB radar grid encoding and rendering
- SAFR (Slot-Anchor Feature Readout) and RadarGrid-Lite neural architectures
"""

from .slot_anchor_readout import SlotAnchorReadoutBlock, register_with_ultralytics
from .slot_encoding import (
    FEATURE_ORDER,
    FEATURE_GROUPS,
    encode_sargb_cell,
    render_sargb_image,
    fit_training_percentiles,
    quantize_features,
)
from .feature_extraction import extract_radar_features

__all__ = [
    "SlotAnchorReadoutBlock",
    "register_with_ultralytics",
    "FEATURE_ORDER",
    "FEATURE_GROUPS",
    "encode_sargb_cell",
    "render_sargb_image",
    "fit_training_percentiles",
    "quantize_features",
    "extract_radar_features",
]
