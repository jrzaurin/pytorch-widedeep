"""
Most of the code here is a direct copy and paste from the fantastic tabnet
implementation here: https://github.com/dreamquark-ai/tabnet

Therefore, ALL CREDIT TO THE DREAMQUARK-AI TEAM
-----------------------------------------------

Here I simply adapted what I needed the TabNet to work within pytorch-widedeep
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from pytorch_widedeep.wdtypes import *  # noqa: F403


def initialize_non_glu(module, input_dim: int, output_dim: int):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(4 * input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    return


def initialize_glu(module, input_dim: int, output_dim: int):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    return


class GBN(torch.nn.Module):
    """
    Ghost Batch Normalization
    https://arxiv.org/abs/1705.08741
    """

    def __init__(
        self, input_dim: int, virtual_batch_size: int = 128, momentum: float = 0.01
    ):
        super(GBN, self).__init__()

        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]
        return torch.cat(res, dim=0)


class GLU_Block(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_glu: int = 2,
        first: bool = False,
        shared_layers: List = None,
        ghost_bn: bool = True,
        virtual_batch_size: int = 128,
        momentum: float = 0.02,
    ):
        super(GLU_Block, self).__init__()
        self.first = first
        self.shared_layers = shared_layers
        self.n_glu = n_glu
        self.glu_layers = nn.ModuleList()

        params = {
            "ghost_bn": ghost_bn,
            "virtual_batch_size": virtual_batch_size,
            "momentum": momentum,
        }

        glu_dim = [input_dim] + [output_dim] * self.n_glu
        for i in range(self.n_glu):
            fc = shared_layers[i] if shared_layers else None
            self.glu_layers.append(GLU_Layer(glu_dim[i], glu_dim[i + 1], fc=fc, **params))  # type: ignore[arg-type]

    def forward(self, x):
        scale = torch.sqrt(torch.FloatTensor([0.5]).to(x.device))

        if self.first:  # the first layer of the block has no scale multiplication
            x = self.glu_layers[0](x)
            layers_left = range(1, self.n_glu)
        else:
            layers_left = range(self.n_glu)

        for glu_id in layers_left:
            x = torch.add(x, self.glu_layers[glu_id](x)) * scale

        return x


class GLU_Layer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        fc: nn.Module = None,
        ghost_bn: bool = True,
        virtual_batch_size: int = 128,
        momentum: float = 0.02,
    ):
        super(GLU_Layer, self).__init__()

        self.output_dim = output_dim
        if fc:
            self.fc = fc
        else:
            self.fc = nn.Linear(input_dim, 2 * output_dim, bias=False)
        initialize_glu(self.fc, input_dim, 2 * output_dim)

        if ghost_bn:
            self.bn = GBN(
                2 * output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum
            )
        else:
            self.bn = nn.BatchNorm1d(2 * output_dim, momentum=momentum)  # type: ignore[assignment]

    def forward(self, x):
        return F.glu(self.bn(self.fc(x)))
