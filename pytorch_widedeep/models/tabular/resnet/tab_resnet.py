import torch
from torch import nn

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models.embeddings_layers import (
    DiffSizeCatAndContEmbeddings,
)
from pytorch_widedeep.models.tabular.mlp._layers import MLP
from pytorch_widedeep.models.tabular.resnet._layers import DenseResnet


class TabResnet(nn.Module):
    r"""Defines a so-called ``TabResnet`` model that can be used as the
    ``deeptabular`` component of a Wide & Deep model.

    This class combines embedding representations of the categorical features
    with numerical (aka continuous) features. These are then passed through a
    series of Resnet blocks. See
    :obj:`pytorch_widedeep.models.tab_resnet.BasicBlock` for details on the
    structure of each block.

    Parameters
    ----------
    column_idx: Dict
        Dict containing the index of the columns that will be passed through
        the ``Resnet`` model. Required to slice the tensors. e.g. {'education':
        0, 'relationship': 1, 'workclass': 2, ...}
    cat_embed_input: List
        List of Tuples with the column name, number of unique values and
        embedding dimension. e.g. [(education, 11, 32), ...].
    cat_embed_dropout: float, default = 0.1
        embeddings dropout
    embed_continuous: bool, default = False,
        Boolean indicating if the continuous columns will be embedded
        (i.e. passed each through a linear layer with or without activation)
    cont_embed_dim: int, default = 32,
        Size of the continuous embeddings
    cont_embed_dropout: float, default = 0.1,
        Dropout for the continuous embeddings
    cont_embed_activation: Optional, str, default = None,
        Activation function for the continuous embeddings
    use_cont_bias: bool, default = True,
        Boolean indicating in bias will be used for the continuous embeddings
    cont_norm_layer: str, default =  "batchnorm"
        Type of normalization layer applied to the continuous features. Options
        are: 'layernorm', 'batchnorm' or None.
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
    simplify_blocks: bool, default = False,
        Boolean indicating if the simples possible residual blocks (X ->
        [LIN1, BN1, ACT1] + X) will be used instead of a standard one (X ->
        [LIN1, BN1, ACT1] -> [LIN2, BN2] + X).
    mlp_hidden_dims: List, Optional, default = None
        List with the number of neurons per dense layer in the MLP. e.g:
        [64, 32]. If ``None`` the  output of the Resnet Blocks will be
        connected directly to the output neuron(s), i.e. using a MLP is
        optional.
    mlp_activation: str, default = "relu"
        Activation function for the dense layers of the MLP. Currently
        ``tanh``, ``relu``, ``leaky_relu`` and ``gelu`` are supported
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
    cat_and_cont_embed: ``nn.Module``
        This is the module that processes the categorical and continuous columns
    tab_resnet_blks: ``nn.Sequential``
        deep dense Resnet model that will receive the concatenation of the
        embeddings and the continuous columns
    tab_resnet_mlp: ``nn.Sequential``
        if ``mlp_hidden_dims`` is ``True``, this attribute will be an mlp
        model that will receive the results of the concatenation of the
        embeddings and the continuous columns -- if present --.
    output_dim: `int`
        The output dimension of the model. This is a required attribute
        neccesary to build the WideDeep class

    Example
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import TabResnet
    >>> X_deep = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
    >>> colnames = ['a', 'b', 'c', 'd', 'e']
    >>> cat_embed_input = [(u,i,j) for u,i,j in zip(colnames[:4], [4]*4, [8]*4)]
    >>> column_idx = {k:v for v,k in enumerate(colnames)}
    >>> model = TabResnet(blocks_dims=[16,4], column_idx=column_idx, cat_embed_input=cat_embed_input,
    ... continuous_cols = ['e'])
    >>> out = model(X_deep)
    """

    def __init__(
        self,
        column_idx: Dict[str, int],
        cat_embed_input: Optional[List[Tuple[str, int, int]]] = None,
        cat_embed_dropout: float = 0.1,
        continuous_cols: Optional[List[str]] = None,
        embed_continuous: bool = False,
        cont_embed_dim: int = 32,
        cont_embed_dropout: float = 0.1,
        cont_embed_activation: Optional[str] = None,
        use_cont_bias: bool = True,
        cont_norm_layer: str = "batchnorm",
        blocks_dims: List[int] = [200, 100, 100],
        blocks_dropout: float = 0.1,
        simplify_blocks: bool = False,
        mlp_hidden_dims: Optional[List[int]] = None,
        mlp_activation: str = "relu",
        mlp_dropout: float = 0.1,
        mlp_batchnorm: bool = False,
        mlp_batchnorm_last: bool = False,
        mlp_linear_first: bool = False,
    ):
        super(TabResnet, self).__init__()

        if len(blocks_dims) < 2:
            raise ValueError(
                "'blocks' must contain at least two elements, e.g. [256, 128]"
            )

        self.column_idx = column_idx
        self.cat_embed_input = cat_embed_input
        self.cat_embed_dropout = cat_embed_dropout

        self.continuous_cols = continuous_cols
        self.embed_continuous = embed_continuous
        self.cont_embed_dim = cont_embed_dim
        self.cont_embed_dropout = cont_embed_dropout
        self.cont_embed_activation = cont_embed_activation
        self.use_cont_bias = use_cont_bias
        self.cont_norm_layer = cont_norm_layer

        self.blocks_dims = blocks_dims
        self.blocks_dropout = blocks_dropout
        self.simplify_blocks = simplify_blocks

        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.mlp_dropout = mlp_dropout
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.mlp_linear_first = mlp_linear_first

        self.cat_and_cont_embed = DiffSizeCatAndContEmbeddings(
            column_idx,
            cat_embed_input,
            cat_embed_dropout,
            continuous_cols,
            embed_continuous,
            cont_embed_dim,
            cont_embed_dropout,
            cont_embed_activation,
            use_cont_bias,
            cont_norm_layer,
        )
        cat_out_dim = self.cat_and_cont_embed.cat_out_dim
        cont_out_dim = self.cat_and_cont_embed.cont_out_dim

        # DenseResnet
        dense_resnet_input_dim = cat_out_dim + cont_out_dim
        self.tab_resnet_blks = DenseResnet(
            dense_resnet_input_dim, blocks_dims, blocks_dropout, self.simplify_blocks
        )

        # Mlp
        if self.mlp_hidden_dims is not None:
            mlp_hidden_dims = [blocks_dims[-1]] + mlp_hidden_dims
            self.tab_resnet_mlp = MLP(
                mlp_hidden_dims,
                mlp_activation,
                mlp_dropout,
                mlp_batchnorm,
                mlp_batchnorm_last,
                mlp_linear_first,
            )
            self.output_dim = mlp_hidden_dims[-1]
        else:
            self.output_dim = blocks_dims[-1]

    def forward(self, X: Tensor) -> Tensor:
        r"""Forward pass that concatenates the continuous features with the
        embeddings. The result is then passed through a series of dense Resnet
        blocks"""

        x_emb, x_cont = self.cat_and_cont_embed(X)

        if x_emb is not None:
            x = x_emb
        if x_cont is not None:
            x = torch.cat([x, x_cont], 1) if x_emb is not None else x_cont

        out = self.tab_resnet_blks(x)

        if self.mlp_hidden_dims is not None:
            out = self.tab_resnet_mlp(out)

        return out
