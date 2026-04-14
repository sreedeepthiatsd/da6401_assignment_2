"""Classification components
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        super(VGG11Classifier, self).__init__()
        self.encoder = VGG11Encoder(in_channels)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(4096, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.
        Returns:
            Classification logits [B, num_classes].
        """
        # TODO: Implement forward pass.
        x = self.encoder(x)        # [B, 512, 7, 7]
        x = torch.flatten(x, 1)    # [B, 25088]
        x = self.classifier(x)     # [B, 37]
        return x
        #raise NotImplementedError("Implement VGG11Classifier.forward")
