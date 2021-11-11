from collections import OrderedDict

from torch import nn
from torch.nn import Module

from pytorch_widedeep.wdtypes import *  # noqa: F403


class BasicBlock(nn.Module):
    # inspired by https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L37
    def __init__(self, inp: int, out: int, dropout: float = 0.0, resize: Module = None):
        super(BasicBlock, self).__init__()

        self.lin1 = nn.Linear(inp, out)
        self.bn1 = nn.BatchNorm1d(out)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        if dropout > 0.0:
            self.dropout = True
            self.dp = nn.Dropout(dropout)
        else:
            self.dropout = False
        self.lin2 = nn.Linear(out, out)
        self.bn2 = nn.BatchNorm1d(out)
        self.resize = resize

    def forward(self, x):

        identity = x

        out = self.lin1(x)
        out = self.bn1(out)
        out = self.leaky_relu(out)
        if self.dropout:
            out = self.dp(out)

        out = self.lin2(out)
        out = self.bn2(out)

        if self.resize is not None:
            identity = self.resize(x)

        out += identity
        out = self.leaky_relu(out)

        return out


class DenseResnet(nn.Module):
    def __init__(self, input_dim: int, blocks_dims: List[int], dropout: float):
        super(DenseResnet, self).__init__()

        self.input_dim = input_dim
        self.blocks_dims = blocks_dims
        self.dropout = dropout

        if input_dim != blocks_dims[0]:
            self.dense_resnet = nn.Sequential(
                OrderedDict(
                    [
                        ("lin1", nn.Linear(input_dim, blocks_dims[0])),
                        ("bn1", nn.BatchNorm1d(blocks_dims[0])),
                    ]
                )
            )
        else:
            self.dense_resnet = nn.Sequential()
        for i in range(1, len(blocks_dims)):
            resize = None
            if blocks_dims[i - 1] != blocks_dims[i]:
                resize = nn.Sequential(
                    nn.Linear(blocks_dims[i - 1], blocks_dims[i]),
                    nn.BatchNorm1d(blocks_dims[i]),
                )
            self.dense_resnet.add_module(
                "block_{}".format(i - 1),
                BasicBlock(blocks_dims[i - 1], blocks_dims[i], dropout, resize),
            )

    def forward(self, X: Tensor) -> Tensor:
        return self.dense_resnet(X)
