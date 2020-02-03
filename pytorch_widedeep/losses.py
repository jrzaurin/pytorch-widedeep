import torch
import torch.nn as nn
import torch.nn.functional as F

from .wdtypes import *


use_cuda = torch.cuda.is_available()


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def get_weight(self, x: Tensor, t: Tensor) -> Tensor:
        p = x.sigmoid()
        pt = p * t + (1 - p) * (1 - t)  # type: ignore
        w = self.alpha * t + (1 - self.alpha) * (1 - t)  # type: ignore
        return (w * (1 - pt).pow(self.gamma)).detach()  # type: ignore

    def forward(self, input: Tensor, target: Tensor) -> Tensor:  # type: ignore
        if input.size(1) == 1:
            input = torch.cat([1 - input, input], axis=1)  # type: ignore
            num_class = 2
        else:
            num_class = input.size(1)
        binary_target = torch.eye(num_class)[target.long()]
        if use_cuda:
            binary_target = binary_target.cuda()
        binary_target = binary_target.contiguous()
        weight = self.get_weight(input, binary_target)
        return F.binary_cross_entropy_with_logits(
            input, binary_target, weight, reduction="mean"
        )
