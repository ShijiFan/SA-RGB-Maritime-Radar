"""Slot-anchor feature readout used by the SAFR and RadarGrid-Lite models.

The registration helper is required before loading a serialized Ultralytics
checkpoint that contains ``SlotAnchorReadoutBlock``.
"""

from __future__ import annotations

import torch
from torch import nn

from ultralytics.nn.modules.conv import Conv


class SlotAnchorReadoutBlock(nn.Module):
    """Read fixed slot-channel addresses relative to a neutral RGB anchor."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 2,
        anchor: float = 128.0 / 255.0,
    ) -> None:
        super().__init__()
        hidden = max(8, int(0.5 * c2))
        self.anchor = float(anchor)
        self.proj = Conv(3 * c1, hidden, 1, 1)
        self.local = Conv(hidden, hidden, 3, 1, g=hidden)
        self.slot_h = nn.Sequential(
            nn.Conv2d(hidden, hidden, (1, 5), 1, (0, 2), groups=hidden, bias=False),
            nn.BatchNorm2d(hidden, eps=0.001, momentum=0.03),
            nn.SiLU(inplace=True),
        )
        self.slot_v = nn.Sequential(
            nn.Conv2d(hidden, hidden, (5, 1), 1, (2, 0), groups=hidden, bias=False),
            nn.BatchNorm2d(hidden, eps=0.001, momentum=0.03),
            nn.SiLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, hidden, 1),
            nn.Sigmoid(),
        )
        self.out = Conv(3 * hidden, c2, k, s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        anchor_delta = x - self.anchor
        y0 = self.proj(torch.cat((x, anchor_delta, anchor_delta.abs()), dim=1))
        gate = self.gate(y0)
        local = self.local(y0)
        horizontal = self.slot_h(y0) * gate
        vertical = self.slot_v(y0) * gate
        return self.out(torch.cat((local, horizontal, vertical), dim=1))


def register_with_ultralytics() -> None:
    """Expose the custom class at the module paths used by saved checkpoints."""

    import ultralytics.nn.modules.block as block_module
    import ultralytics.nn.tasks as tasks_module

    SlotAnchorReadoutBlock.__module__ = block_module.__name__
    block_module.SlotAnchorReadoutBlock = SlotAnchorReadoutBlock
    tasks_module.SlotAnchorReadoutBlock = SlotAnchorReadoutBlock

