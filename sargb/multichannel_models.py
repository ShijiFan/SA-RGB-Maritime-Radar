"""Ultralytics-compatible models for Paper3 multichannel baselines."""

from __future__ import annotations

import torch
from ultralytics.nn.tasks import DetectionModel


class ProjectedDetectionModel(DetectionModel):
    """YOLO11 detector with a learned 1x1 seven-to-three input projection."""

    def __init__(self, cfg="yolo11n.yaml", ch=7, nc=None, verbose=True):
        self.source_channels = int(ch)
        super().__init__(cfg=cfg, ch=3, nc=nc, verbose=verbose)
        self.input_projection = torch.nn.Conv2d(self.source_channels, 3, kernel_size=1, bias=False)
        torch.nn.init.xavier_uniform_(self.input_projection.weight)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        if x.shape[1] == self.source_channels:
            x = self.input_projection(x)
        return super().predict(x, profile=profile, visualize=visualize, augment=augment, embed=embed)


def projection_parameters(model: torch.nn.Module) -> int:
    projection = getattr(model, "input_projection", None)
    return 0 if projection is None else sum(parameter.numel() for parameter in projection.parameters())
