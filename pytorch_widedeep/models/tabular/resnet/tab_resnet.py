import torch
from torch import nn

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models._embeddings_layers import CatEmbeddingsAndCont
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
        If ``True`` the continuum columns will be concatenated with the
        Categorical Embeddings and then passed through the Resnet blocks. If
        ``False``, the  Categorical Embeddings will be passed through the
        Resnet blocks and then the output of the Resnet blocks will be
        concatenated with the continuous features.
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
    cat_embed_and_cont: ``nn.Module``
        This is the module that processes the categorical and continuous columns
    dense_resnet: ``nn.Sequential``
        deep dense Resnet model that will receive the concatenation of the
        embeddings and the continuous columns
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
        embed_input: Optional[List[Tuple[str, int, int]]] = None,
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

        if len(blocks_dims) < 2:
            raise ValueError(
                "'blocks' must contain at least two elements, e.g. [256, 128]"
            )

        if not concat_cont_first and embed_input is None:
            raise ValueError(
                "If 'concat_cont_first = False' 'embed_input' must be not 'None'"
            )

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

        self.cat_embed_and_cont = CatEmbeddingsAndCont(
            column_idx,
            embed_input,
            embed_dropout,
            continuous_cols,
            cont_norm_layer,
        )

        emb_out_dim = self.cat_embed_and_cont.emb_out_dim
        cont_out_dim = self.cat_embed_and_cont.cont_out_dim

        # DenseResnet
        if self.concat_cont_first:
            dense_resnet_input_dim = emb_out_dim + cont_out_dim
            self.output_dim = blocks_dims[-1]
        else:
            dense_resnet_input_dim = emb_out_dim
            self.output_dim = cont_out_dim + blocks_dims[-1]
        self.tab_resnet_blks = DenseResnet(
            dense_resnet_input_dim, blocks_dims, blocks_dropout
        )

        # MLP
        if self.mlp_hidden_dims is not None:
            if self.concat_cont_first:
                mlp_input_dim = blocks_dims[-1]
            else:
                mlp_input_dim = cont_out_dim + blocks_dims[-1]
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

    def forward(self, X: Tensor) -> Tensor:
        r"""Forward pass that concatenates the continuous features with the
        embeddings. The result is then passed through a series of dense Resnet
        blocks"""

        x_emb, x_cont = self.cat_embed_and_cont(X)

        if x_cont is not None:
            if self.concat_cont_first:
                x = torch.cat([x_emb, x_cont], 1) if x_emb is not None else x_cont
                out = self.tab_resnet_blks(x)
            else:
                out = torch.cat([self.tab_resnet_blks(x_emb), x_cont], 1)
        else:
            out = self.tab_resnet_blks(x_emb)

        if self.mlp_hidden_dims is not None:
            out = self.tab_resnet_mlp(out)

        return out
