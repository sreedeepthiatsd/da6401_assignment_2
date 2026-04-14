"""Segmentation model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .vgg11 import VGG11Encoder


class VGG11UNet(nn.Module):
    """U-Net style segmentation network."""

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        super(VGG11UNet, self).__init__()

        self.encoder = VGG11Encoder(in_channels)

        # Decoder (upsampling)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p)
        )

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p)
        )

        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p)
        )

        self.up4 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p)
        )

        self.up5 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.conv5 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Final segmentation layer
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x = self.encoder(x)   # [B, 512, 7, 7]

        # Decoder
        x = self.up1(x)       # 7 → 14
        x = self.conv1(x)

        x = self.up2(x)       # 14 → 28
        x = self.conv2(x)

        x = self.up3(x)       # 28 → 56
        x = self.conv3(x)

        x = self.up4(x)       # 56 → 112
        x = self.conv4(x)

        x = self.up5(x)       # 112 → 224
        x = self.conv5(x)

        x = self.final(x)     # [B, num_classes, 224, 224]

        return x
