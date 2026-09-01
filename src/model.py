"""Model definition for the Split-MNIST continual-learning grid."""

from __future__ import annotations

from torch import nn


class MLP(nn.Module):
    """Two-hidden-layer single-head MLP used for all reported runs."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):  # type: ignore[no-untyped-def]
        return self.net(x)
