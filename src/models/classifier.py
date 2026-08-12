from __future__ import annotations

import torch
from torch import nn


class VideoClassifier(nn.Module):
    """Lightweight baseline classifier for fixed-length RGB clips."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, padding=1), nn.BatchNorm3d(16), nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=3, padding=1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=3, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, num_classes))

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # Input: [B,T,C,H,W] or [B,C,T,H,W]
        if video.ndim != 5:
            raise ValueError(f"Expected 5D video tensor, got {video.shape}")
        if video.shape[1] == 3:
            x = video
        else:
            x = video.permute(0, 2, 1, 3, 4)
        return self.head(self.encoder(x))
