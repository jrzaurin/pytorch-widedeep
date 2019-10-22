import torch

from torch import nn
from ..wdtypes import *

class Wide(nn.Module):
    def __init__(self,wide_dim:int, output_dim:int=1):
        super(Wide, self).__init__()
        self.wide_linear = nn.Linear(wide_dim, output_dim)

    def forward(self, X:Tensor)->Tensor:
        out = self.wide_linear(X.float())
        return out
