"""Custom IoU loss 
"""

import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.
        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format."""
        pred_half_wh = pred_boxes[:, 2:].clamp_min(0) * 0.5
        target_half_wh = target_boxes[:, 2:].clamp_min(0) * 0.5

        pred_mins = pred_boxes[:, :2] - pred_half_wh
        pred_maxs = pred_boxes[:, :2] + pred_half_wh
        target_mins = target_boxes[:, :2] - target_half_wh
        target_maxs = target_boxes[:, :2] + target_half_wh

        inter_mins = torch.maximum(pred_mins, target_mins)
        inter_maxs = torch.minimum(pred_maxs, target_maxs)
        inter_wh = (inter_maxs - inter_mins).clamp_min(0)
        inter_area = inter_wh[:, 0] * inter_wh[:, 1]

        pred_area = pred_boxes[:, 2].clamp_min(0) * pred_boxes[:, 3].clamp_min(0)
        target_area = target_boxes[:, 2].clamp_min(0) * target_boxes[:, 3].clamp_min(0)
        union_area = pred_area + target_area - inter_area + self.eps

        iou = inter_area / union_area 
        iou = iou.clamp(0, 1)
        loss = 1.0 - iou

        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()
