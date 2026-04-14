"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn


class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns.
    """

    def __init__(self, in_channels: int = 3):

        # it initialises the parent class nn.Module
        super(VGG11Encoder, self).__init__()

        self.block1 = nn.Sequential(
            # 64 filters each of size 3x3 is used
            nn.Conv2d(in_channels, 64, 3, padding = 1), # stride = 1 is ddefault in pytorch
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), # inplace = True --> modify the input tensor directly instead of creating a new one # memory efficiency
            nn.MaxPool2d(2, 2) # reduces the spatial dimension
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding = 1), # 64 in channels, 128 out channels(128 filters) each of size 3x3
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2,2) # stride = 2, kernel_size = 2
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )
        
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2,2)
        )



    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor.
            - if return_features=True: (bottleneck, feature_dict).
        """
        # TODO: Implement forward pass.
        f1 = self.block1(x)  # we get feature o/p size of 112x112 after this
        f2 = self.block2(f1)  # we get 56x56
        f3 = self.block3(f2)  # we get 28x28
        f4 = self.block4(f3)  # we get 14x14
        f5 = self.block5(f4)  # we get 7x7

        if return_features:
            features = {
                "f1": f1,
                "f2": f2,
                "f3": f3,
                "f4": f4
            }
            return f5, features

        return f5
    
        # raise NotImplementedError("Implement VGG11Encoder.forward")
