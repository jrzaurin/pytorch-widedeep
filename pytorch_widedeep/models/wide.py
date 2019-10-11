import torch
import torch.nn as nn
from torch import Tensor

class Wide(nn.Module):
    def __init__(self,wide_dim:int, output_dim:int=1):
        super(Wide, self).__init__()
        # (Wide Linear, wlinear)
        self.wlinear = nn.Linear(wide_dim, output_dim)

    def forward(self, X:Tensor)->Tensor:
        out = self.wlinear(X.float())
        return out