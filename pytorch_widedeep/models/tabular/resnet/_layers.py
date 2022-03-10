from collections import OrderedDict

from torch import nn
from torch.nn import Module

from pytorch_widedeep.wdtypes import *  # noqa: F403


class BasicBlock(nn.Module):
    # inspired by https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L37
    def __init__(
        self,
        inp: int,
        out: int,
        dropout: float = 0.0,
        simplify: bool = False,
        resize: Module = None,
    ):
        super(BasicBlock, self).__init__()

        self.simplify = simplify
        self.resize = resize

        self.lin1 = nn.Linear(inp, out, bias=False)
        self.bn1 = nn.BatchNorm1d(out)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        if dropout > 0.0:
            self.dropout = True
            self.dp = nn.Dropout(dropout)
        else:
            self.dropout = False

        if not self.simplify:
            self.lin2 = nn.Linear(out, out, bias=False)
            self.bn2 = nn.BatchNorm1d(out)

    def forward(self, X: Tensor) -> Tensor:

        identity = X

        out = self.lin1(X)
        out = self.bn1(out)
        out = self.leaky_relu(out)
        if self.dropout:
            out = self.dp(out)

        if not self.simplify:
            out = self.lin2(out)
            out = self.bn2(out)

        if self.resize is not None:
            identity = self.resize(X)

        out += identity
        out = self.leaky_relu(out)

        return out


class DenseResnet(nn.Module):
    def __init__(
        self, input_dim: int, blocks_dims: List[int], dropout: float, simplify: bool
    ):
        super(DenseResnet, self).__init__()

        if input_dim != blocks_dims[0]:
            self.dense_resnet = nn.Sequential(
                OrderedDict(
                    [
                        ("lin_inp", nn.Linear(input_dim, blocks_dims[0], bias=False)),
                        ("bn_inp", nn.BatchNorm1d(blocks_dims[0])),
                    ]
                )
            )
        else:
            self.dense_resnet = nn.Sequential()
        for i in range(1, len(blocks_dims)):
            resize = None
            if blocks_dims[i - 1] != blocks_dims[i]:
                resize = nn.Sequential(
                    nn.Linear(blocks_dims[i - 1], blocks_dims[i], bias=False),
                    nn.BatchNorm1d(blocks_dims[i]),
                )
            self.dense_resnet.add_module(
                "block_{}".format(i - 1),
                BasicBlock(
                    blocks_dims[i - 1], blocks_dims[i], dropout, simplify, resize
                ),
            )

    def forward(self, X: Tensor) -> Tensor:
        return self.dense_resnet(X)
