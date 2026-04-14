"""Unified multi-task model
"""

import torch
import torch.nn as nn
from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet



class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 2, in_channels: int = 3, classifier_path: str = "checkpoints/classifier.pth", localizer_path: str = "checkpoints/localizer.pth", unet_path: str = "checkpoints/unet.pth"):
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """
        import gdown
        gdown.download(id="<classifier.pth 1dP5Oq_JBvM2HvPDhpYyA1rD0Uus-nNhj>", output=classifier_path, quiet=False)
        gdown.download(id="<localizer.pth 1h15oL4s0neaBQaQUicBzvHJxQcav8bRv>", output=localizer_path, quiet=False)
        gdown.download(id="<unet.pth 10aFoN41DW6kwvxGyGlksvcvwFViPvTAL>", output=unet_path, quiet=False)
        super().__init__()
        # -------- CLASSIFIER --------
        self.classifier = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)

        classifier_ckpt = torch.load(classifier_path, map_location="cpu")
        self.classifier.load_state_dict(classifier_ckpt["state_dict"])

        # -------- LOCALIZER --------
        self.localizer = VGG11Localizer(in_channels=in_channels)

        localizer_ckpt = torch.load(localizer_path, map_location="cpu")
        self.localizer.load_state_dict(localizer_ckpt["state_dict"])

        # -------- SEGMENTATION (UNET) --------
        self.segmenter = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        unet_ckpt = torch.load(unet_path, map_location="cpu")
        self.segmenter.load_state_dict(unet_ckpt["state_dict"])




    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        # TODO: Implement forward pass.
        classification_logits = self.classifier(x)
        localization = self.localizer(x)
        segmentation = self.segmenter(x)

        return {
            "classification": classification_logits,
            "localization": localization,
            "segmentation": segmentation
        }
        #raise NotImplementedError("Implement MultiTaskPerceptionModel.forward")
