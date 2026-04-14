"""Inference and evaluation
"""

import torch
from models.multitask import MultiTaskPerceptionModel


def load_model(device="cpu"):
    model = MultiTaskPerceptionModel()
    model.to(device)
    model.eval()
    return model


def predict(model, images):
    """
    Args:
        model: MultiTaskPerceptionModel
        images: tensor [B, 3, 224, 224]

    Returns:
        dict with:
            classification: [B, num_classes]
            localization: [B, 4]
            segmentation: [B, seg_classes, H, W]
    """
    with torch.no_grad():
        outputs = model(images)

    return outputs


# Optional: simple test
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(device)

    # dummy input
    x = torch.randn(2, 3, 224, 224).to(device)

    outputs = predict(model, x)

    print("Classification:", outputs["classification"].shape)
    print("Localization:", outputs["localization"].shape)
    print("Segmentation:", outputs["segmentation"].shape)
