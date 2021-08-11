from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import Module

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models.tab_mlp import MLP


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


class TabResnet(nn.Module):
    r"""Defines a so-called ``TabResnet`` model that can be used as the
    ``deeptabular`` component of a Wide & Deep model.

    This class combines embedding representations of the categorical
    features with numerical (aka continuous) features. These are then
    passed through a series of Resnet blocks. See
    ``pytorch_widedeep.models.tab_resnet.BasicBlock`` for details
    on the structure of each block.

    .. note:: Unlike ``TabMlp``, ``TabResnet`` assumes that there are always
        categorical columns

    Parameters
    ----------
    column_idx: Dict
        Dict containing the index of the columns that will be passed through
        the TabMlp model. Required to slice the tensors. e.g. {'education':
        0, 'relationship': 1, 'workclass': 2, ...}
    embed_input: List
        List of Tuples with the column name, number of unique values and
        embedding dimension. e.g. [(education, 11, 32), ...].
    embed_dropout: float, default = 0.1
        embeddings dropout
    continuous_cols: List, Optional, default = None
        List with the name of the numeric (aka continuous) columns
    cont_norm_layer: str, default =  "batchnorm"
        Type of normalization layer applied to the continuous features. Options
        are: 'layernorm', 'batchnorm' or None.
    concat_cont_first: bool, default = True
        Boolean indicating whether the continuum columns will be
        concatenated with the Embeddings and then passed through the
        Resnet blocks (``True``) or, alternatively, will be concatenated
        with the result of passing the embeddings through the Resnet
        Blocks (``False``)
    blocks_dims: List, default = [200, 100, 100]
        List of integers that define the input and output units of each block.
        For example: [200, 100, 100] will generate 2 blocks. The first will
        receive a tensor of size 200 and output a tensor of size 100, and the
        second will receive a tensor of size 100 and output a tensor of size
        100. See :obj:`pytorch_widedeep.models.tab_resnet.BasicBlock` for
        details on the structure of each block.
    blocks_dropout: float, default =  0.1
       Block's `"internal"` dropout. This dropout is applied to the first
       of the two dense layers that comprise each ``BasicBlock``.
    mlp_hidden_dims: List, Optional, default = None
        List with the number of neurons per dense layer in the mlp. e.g:
        [64, 32]. If ``None`` the  output of the Resnet Blocks will be
        connected directly to the output neuron(s), i.e. using a MLP is
        optional.
    mlp_activation: str, default = "relu"
        MLP activation function. 'relu', 'leaky_relu' and 'gelu' are supported
    mlp_dropout: float, default = 0.1
        float with the dropout between the dense layers of the MLP.
    mlp_batchnorm: bool, default = False
        Boolean indicating whether or not batch normalization will be applied
        to the dense layers
    mlp_batchnorm_last: bool, default = False
        Boolean indicating whether or not batch normalization will be applied
        to the last of the dense layers
    mlp_linear_first: bool, default = False
        Boolean indicating the order of the operations in the dense
        layer. If ``True: [LIN -> ACT -> BN -> DP]``. If ``False: [BN -> DP ->
        LIN -> ACT]``

    Attributes
    ----------
    embed_layers: ``nn.ModuleDict``
        ``ModuleDict`` with the embeddings setup
    dense_resnet: ``nn.Sequential``
        deep dense Resnet model that will receive the concatenation of the
        embeddings and the continuous columns
    cont_norm: ``nn.Module``
        continuous normalization layer
    tab_resnet_mlp: ``nn.Sequential``
        if ``mlp_hidden_dims`` is ``True``, this attribute will be an mlp
        model that will receive:

        - the results of the concatenation of the embeddings and the
          continuous columns -- if present -- and then passed it through
          the ``dense_resnet`` (``concat_cont_first = True``), or

        - the result of passing the embeddings through the ``dense_resnet``
          and then concatenating the results with the continuous columns --
          if present -- (``concat_cont_first = False``)

    output_dim: `int`
        The output dimension of the model. This is a required attribute
        neccesary to build the WideDeep class

    Example
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import TabResnet
    >>> X_deep = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
    >>> colnames = ['a', 'b', 'c', 'd', 'e']
    >>> embed_input = [(u,i,j) for u,i,j in zip(colnames[:4], [4]*4, [8]*4)]
    >>> column_idx = {k:v for v,k in enumerate(colnames)}
    >>> model = TabResnet(blocks_dims=[16,4], column_idx=column_idx, embed_input=embed_input,
    ... continuous_cols = ['e'])
    >>> out = model(X_deep)
    """

    def __init__(
        self,
        column_idx: Dict[str, int],
        embed_input: List[Tuple[str, int, int]],
        embed_dropout: float = 0.1,
        continuous_cols: Optional[List[str]] = None,
        cont_norm_layer: str = "batchnorm",
        concat_cont_first: bool = True,
        blocks_dims: List[int] = [200, 100, 100],
        blocks_dropout: float = 0.1,
        mlp_hidden_dims: Optional[List[int]] = None,
        mlp_activation: str = "relu",
        mlp_dropout: float = 0.1,
        mlp_batchnorm: bool = False,
        mlp_batchnorm_last: bool = False,
        mlp_linear_first: bool = False,
    ):
        super(TabResnet, self).__init__()

        self.column_idx = column_idx
        self.embed_input = embed_input
        self.embed_dropout = embed_dropout
        self.continuous_cols = continuous_cols
        self.cont_norm_layer = cont_norm_layer
        self.concat_cont_first = concat_cont_first
        self.blocks_dims = blocks_dims
        self.blocks_dropout = blocks_dropout
        self.mlp_activation = mlp_activation
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.mlp_linear_first = mlp_linear_first

        if len(self.blocks_dims) < 2:
            raise ValueError(
                "'blocks' must contain at least two elements, e.g. [256, 128]"
            )

        # Embeddings: val + 1 because 0 is reserved for padding/unseen cateogories.
        self.embed_layers = nn.ModuleDict(
            {
                "emb_layer_" + col: nn.Embedding(val + 1, dim, padding_idx=0)
                for col, val, dim in self.embed_input
            }
        )
        self.embedding_dropout = nn.Dropout(embed_dropout)
        emb_inp_dim = np.sum([embed[2] for embed in self.embed_input])

        # Continuous
        if self.continuous_cols is not None:
            self.cont_idx = [self.column_idx[col] for col in self.continuous_cols]
            cont_inp_dim = len(self.continuous_cols)
            if self.cont_norm_layer == "batchnorm":
                self.cont_norm: NormLayers = nn.BatchNorm1d(cont_inp_dim)
            elif self.cont_norm_layer == "layernorm":
                self.cont_norm = nn.LayerNorm(cont_inp_dim)
            else:
                self.cont_norm = nn.Identity()
        else:
            cont_inp_dim = 0

        # DenseResnet
        if self.concat_cont_first:
            dense_resnet_input_dim = emb_inp_dim + cont_inp_dim
            self.output_dim = blocks_dims[-1]
        else:
            dense_resnet_input_dim = emb_inp_dim
            self.output_dim = cont_inp_dim + blocks_dims[-1]
        self.tab_resnet_blks = DenseResnet(
            dense_resnet_input_dim, blocks_dims, blocks_dropout  # type: ignore[arg-type]
        )

        # MLP
        if self.mlp_hidden_dims is not None:
            if self.concat_cont_first:
                mlp_input_dim = blocks_dims[-1]
            else:
                mlp_input_dim = cont_inp_dim + blocks_dims[-1]
            mlp_hidden_dims = [mlp_input_dim] + mlp_hidden_dims
            self.tab_resnet_mlp = MLP(
                mlp_hidden_dims,
                mlp_activation,
                mlp_dropout,
                mlp_batchnorm,
                mlp_batchnorm_last,
                mlp_linear_first,
            )
            self.output_dim = mlp_hidden_dims[-1]

    def forward(self, X: Tensor) -> Tensor:  # type: ignore
        r"""Forward pass that concatenates the continuous features with the
        embeddings. The result is then passed through a series of dense Resnet
        blocks"""
        embed = [
            self.embed_layers["emb_layer_" + col](X[:, self.column_idx[col]].long())
            for col, _, _ in self.embed_input
        ]
        x = torch.cat(embed, 1)
        x = self.embedding_dropout(x)

        if self.continuous_cols is not None:
            x_cont = self.cont_norm((X[:, self.cont_idx].float()))
            if self.concat_cont_first:
                x = torch.cat([x, x_cont], 1)
                out = self.tab_resnet_blks(x)
            else:
                out = torch.cat([self.tab_resnet_blks(x), x_cont], 1)
        else:
            out = self.tab_resnet_blks(x)

        if self.mlp_hidden_dims is not None:
            out = self.tab_resnet_mlp(out)

        return out
