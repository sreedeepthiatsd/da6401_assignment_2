"""Localization modules
"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
        """
        super(VGG11Localizer, self).__init__()

        self.encoder = VGG11Encoder(in_channels)

        self.regressor = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),

            nn.Linear(512, 4)   # Output: [x_center, y_center, w, h]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format in original image pixel space(not normalized values).
        """
        # TODO: Implement forward pass.
        x = self.encoder(x)        # [B, 512, 7, 7]
        x = torch.flatten(x, 1)    # [B, 25088]
        x = self.regressor(x)     # [B, 4]
        x = torch.sigmoid(x) 

        return x
        # raise NotImplementedError("Implement VGG11Localizer.forward")
