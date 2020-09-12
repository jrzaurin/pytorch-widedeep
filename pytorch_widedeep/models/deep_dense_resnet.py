from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import Module

from ..wdtypes import *


class BasicBlock(nn.Module):
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


class DeepDenseResnet(nn.Module):
    r"""Dense branch of the deep side of the model receiving continuous
    columns and the embeddings from categorical columns.

    This class is an alternative to
    :class:`pytorch_widedeep.models.deep_dense.DeepDense`. Combines embedding
    representations of the categorical features with numerical (aka
    continuous) features. Then, instead of being passed through a series of
    dense layers, the embeddings plus continuous features are passed through a
    series of  Resnet blocks. See
    ``pytorch_widedeep.models.deep_dense_resnet.BasicBlock`` for details on
    the structure of each block.

    Parameters
    ----------
    deep_column_idx: Dict
        Dict containing the index of the columns that will be passed through
        the DeepDense model. Required to slice the tensors. e.g. {'education':
        0, 'relationship': 1, 'workclass': 2, ...}
    blocks: List
        List of integers that define the input and output units of each block.
        For example: ``[128, 64, 32]`` will generate 2 blocks. The first will
        receive a tensor of size 128 and output a tensor of size 64, and the
        second will receive a tensor of size 64 and output a tensor of size
        32. See ``pytorch_widedeep.models.deep_dense_resnet.BasicBlock`` for
        details on the structure of each block.
    dropout: float, default = 0.0
       Block's `"internal"` dropout. This dropout is applied to the first of
       the two dense layers that comprise each ``BasicBlock``
    embeddings_input: List, Optional
        List of Tuples with the column name, number of unique values and
        embedding dimension. e.g. [(education, 11, 32), ...]
    embed_dropout: float
        embeddings dropout
    continuous_cols: List, Optional
        List with the name of the numeric (aka continuous) columns

        .. note:: Either ``embeddings_input`` or ``continuous_cols`` (or both) should be passed to the
            model

    Attributes
    ----------
    dense_resnet: :obj:`nn.Sequential`
        deep dense Resnet model that will receive the concatenation of the
        embeddings and the continuous columns
    embed_layers: :obj:`nn.ModuleDict`
        :obj:`ModuleDict` with the embedding layers
    output_dim: :obj:`int`
        The output dimension of the model. This is a required attribute
        neccesary to build the WideDeep class

    Example
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import DeepDenseResnet
    >>> X_deep = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
    >>> colnames = ['a', 'b', 'c', 'd', 'e']
    >>> embed_input = [(u,i,j) for u,i,j in zip(colnames[:4], [4]*4, [8]*4)]
    >>> deep_column_idx = {k:v for v,k in enumerate(colnames)}
    >>> model = DeepDenseResnet(blocks=[16,4], deep_column_idx=deep_column_idx, embed_input=embed_input)
    >>> out = model(X_deep)
    """

    def __init__(
        self,
        deep_column_idx: Dict[str, int],
        blocks: List[int],
        dropout: float = 0.0,
        embed_dropout: float = 0.0,
        embed_input: Optional[List[Tuple[str, int, int]]] = None,
        continuous_cols: Optional[List[str]] = None,
    ):
        super(DeepDenseResnet, self).__init__()

        if len(blocks) < 2:
            raise ValueError(
                "'blocks' must contain at least two elements, e.g. [256, 128]"
            )

        self.embed_input = embed_input
        self.continuous_cols = continuous_cols
        self.deep_column_idx = deep_column_idx

        # Embeddings
        if self.embed_input is not None:
            self.embed_layers = nn.ModuleDict(
                {
                    "emb_layer_" + col: nn.Embedding(val, dim)
                    for col, val, dim in self.embed_input
                }
            )
            self.embed_dropout = nn.Dropout(embed_dropout)
            emb_inp_dim = np.sum([embed[2] for embed in self.embed_input])
        else:
            emb_inp_dim = 0

        # Continuous
        if self.continuous_cols is not None:
            cont_inp_dim = len(self.continuous_cols)
        else:
            cont_inp_dim = 0

        # Dense Resnet
        input_dim = emb_inp_dim + cont_inp_dim
        if input_dim != blocks[0]:
            self.dense_resnet = nn.Sequential(
                OrderedDict(
                    [
                        ("lin1", nn.Linear(input_dim, blocks[0])),
                        ("bn1", nn.BatchNorm1d(blocks[0])),
                    ]
                )
            )
        else:
            self.dense_resnet = nn.Sequential()
        for i in range(1, len(blocks)):
            resize = None
            if blocks[i - 1] != blocks[i]:
                resize = nn.Sequential(
                    nn.Linear(blocks[i - 1], blocks[i]), nn.BatchNorm1d(blocks[i])
                )
            self.dense_resnet.add_module(
                "block_{}".format(i - 1),
                BasicBlock(blocks[i - 1], blocks[i], dropout, resize),
            )

        # the output_dim attribute will be used as input_dim when "merging" the models
        self.output_dim = blocks[-1]

    def forward(self, X: Tensor) -> Tensor:  # type: ignore
        r"""Forward pass that concatenates the continuous features with the
        embeddings. The result is then passed through a series of dense Resnet
        blocks"""
        if self.embed_input is not None:
            x = [
                self.embed_layers["emb_layer_" + col](
                    X[:, self.deep_column_idx[col]].long()
                )
                for col, _, _ in self.embed_input
            ]
            x = torch.cat(x, 1)  # type: ignore
            x = self.embed_dropout(x)  # type: ignore
        if self.continuous_cols is not None:
            cont_idx = [self.deep_column_idx[col] for col in self.continuous_cols]
            x_cont = X[:, cont_idx].float()
            x = torch.cat([x, x_cont], 1) if self.embed_input is not None else x_cont  # type: ignore
        return self.dense_resnet(x)  # type: ignore
