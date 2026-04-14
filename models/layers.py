"""Reusable custom layers 
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer.
    """

    def __init__(self, p: float = 0.5):
        super(CustomDropout, self).__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement dropout.
        if not self.training or self.p == 0.0:
            return x
        if self.p == 1.0:
            return torch.zeros_like(x)

        keep_prob = 1.0 - self.p
        #mask generation
        mask = torch.rand_like(x) < keep_prob
        return x * mask.to(dtype=x.dtype) / keep_prob
    
        #raise NotImplementedError("Implement CustomDropout.forward")

