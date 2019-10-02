import torch
import torch.nn as nn
import torch.nn.functional as F

from .wdtypes import *

class FocalLoss(nn.Module):
    def __init__(self, num_classes:int, alpha:float=0.25, gamma:float=1.):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma

    def get_weight(self, x:Tensor, t:Tensor) -> Tensor:
        self.alpha,self.gamma = 0.25,1
        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)
        w = self.alpha*t + (1-self.alpha)*(1-t)
        return w * (1-pt).pow(self.gamma)

    def forward(self, input:Tensor, target:Tensor) -> Tensor:
        binary_target = torch.eye(self.num_classes)[target.data.cpu()]
        binary_target = binary_target.contiguous()
        weight = self.get_weight(input, binary_target)
        return F.binary_cross_entropy_with_logits(input, binary_target, weight, reduction='sum')/self.num_classes