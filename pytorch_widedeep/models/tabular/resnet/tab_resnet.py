from torch import nn

from pytorch_widedeep.wdtypes import Dict, List, Tuple, Tensor, Optional
from pytorch_widedeep.models.tabular.mlp._layers import MLP
from pytorch_widedeep.models.tabular.resnet._layers import DenseResnet
from pytorch_widedeep.models.tabular._base_tabular_model import (
    BaseTabularModelWithoutAttention,
)


class TabResnet(BaseTabularModelWithoutAttention):
    r"""Defines a `TabResnet` model that can be used as the `deeptabular`
    component of a Wide & Deep model or independently by itself.

    This class combines embedding representations of the categorical features
    with numerical (aka continuous) features, embedded or not. These are then
    passed through a series of Resnet blocks. See
    `pytorch_widedeep.models.tab_resnet._layers` for details on the
    structure of each block.

    Parameters
    ----------
    column_idx: Dict
        Dict containing the index of the columns that will be passed through
        the model. Required to slice the tensors. e.g.
        _{'education': 0, 'relationship': 1, 'workclass': 2, ...}_
    cat_embed_input: List
        List of Tuples with the column name, number of unique values and
        embedding dimension. e.g. _[(education, 11, 32), ...]_.
    cat_embed_dropout: float, default = 0.1
        Categorical embeddings dropout
    use_cat_bias: bool, default = False,
        Boolean indicating if bias will be used for the categorical embeddings
    cat_embed_activation: Optional, str, default = None,
        Activation function for the categorical embeddings, if any. Currently
        _'tanh'_, _'relu'_, _'leaky'_relu` and _'gelu'_ are supported
    continuous_cols: List, Optional, default = None
        List with the name of the numeric (aka continuous) columns
    cont_norm_layer: str, default =  "batchnorm"
        Type of normalization layer applied to the continuous features. Options
        are: _'layernorm'_, _'batchnorm'_ or `None`.
    embed_continuous: bool, default = False,
        Boolean indicating if the continuous columns will be embedded
        (i.e. passed each through a linear layer with or without activation)
    cont_embed_dim: int, default = 32,
        Size of the continuous embeddings
    cont_embed_dropout: float, default = 0.1,
        Continuous embeddings dropout
    use_cont_bias: bool, default = True,
        Boolean indicating if bias will be used for the continuous embeddings
    cont_embed_activation: Optional, str, default = None,
        Activation function for the continuous embeddings, if any. Currently
        _'tanh'_, _'relu'_, _'leaky'_relu` and _'gelu'_ are supported
    blocks_dims: List, default = [200, 100, 100]
        List of integers that define the input and output units of each block.
        For example: _[200, 100, 100]_ will generate 2 blocks. The first will
        receive a tensor of size 200 and output a tensor of size 100, and the
        second will receive a tensor of size 100 and output a tensor of size
        100. See `pytorch_widedeep.models.tab_resnet._layers` for
        details on the structure of each block.
    blocks_dropout: float, default =  0.1
        Block's internal dropout.
    simplify_blocks: bool, default = False,
        Boolean indicating if the simplest possible residual blocks (`X -> [
        [LIN, BN, ACT]  + X ]`) will be used instead of a standard one
        (`X -> [ [LIN1, BN1, ACT1] -> [LIN2, BN2]  + X ]`).
    mlp_hidden_dims: List, Optional, default = None
        List with the number of neurons per dense layer in the MLP. e.g:
        _[64, 32]_. If `None` the  output of the Resnet Blocks will be
        connected directly to the output neuron(s).
    mlp_activation: str, default = "relu"
        Activation function for the dense layers of the MLP. Currently
        _'tanh'_, _'relu'_, _'leaky'_relu` and _'gelu'_ are supported
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
        layer. If `True: [LIN -> ACT -> BN -> DP]`. If `False: [BN -> DP ->
        LIN -> ACT]`

    Attributes
    ----------
    cat_and_cont_embed: nn.Module
        This is the module that processes the categorical and continuous columns
    encoder: nn.Module
        deep dense Resnet model that will receive the concatenation of the
        embeddings and the continuous columns
    mlp: nn.Module
        if `mlp_hidden_dims` is `True`, this attribute will be an mlp
        model that will receive the results of the concatenation of the
        embeddings and the continuous columns -- if present --.

    Examples
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
        use_cat_bias: bool = False,
        cat_embed_activation: Optional[str] = None,
        continuous_cols: Optional[List[str]] = None,
        cont_norm_layer: str = "batchnorm",
        embed_continuous: bool = False,
        cont_embed_dim: int = 32,
        cont_embed_dropout: float = 0.1,
        use_cont_bias: bool = True,
        cont_embed_activation: Optional[str] = None,
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
        super(TabResnet, self).__init__(
            column_idx=column_idx,
            cat_embed_input=cat_embed_input,
            cat_embed_dropout=cat_embed_dropout,
            use_cat_bias=use_cat_bias,
            cat_embed_activation=cat_embed_activation,
            continuous_cols=continuous_cols,
            cont_norm_layer=cont_norm_layer,
            embed_continuous=embed_continuous,
            cont_embed_dim=cont_embed_dim,
            cont_embed_dropout=cont_embed_dropout,
            use_cont_bias=use_cont_bias,
            cont_embed_activation=cont_embed_activation,
        )

        if len(blocks_dims) < 2:
            raise ValueError(
                "'blocks' must contain at least two elements, e.g. [256, 128]"
            )

        self.blocks_dims = blocks_dims
        self.blocks_dropout = blocks_dropout
        self.simplify_blocks = simplify_blocks

        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.mlp_dropout = mlp_dropout
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.mlp_linear_first = mlp_linear_first

        # Embeddings are instantiated at the base model
        cat_out_dim = self.cat_and_cont_embed.cat_out_dim
        cont_out_dim = self.cat_and_cont_embed.cont_out_dim

        # Resnet
        dense_resnet_input_dim = cat_out_dim + cont_out_dim
        self.encoder = DenseResnet(
            dense_resnet_input_dim, blocks_dims, blocks_dropout, self.simplify_blocks
        )

        # Mlp
        if self.mlp_hidden_dims is not None:
            mlp_hidden_dims = [self.blocks_dims[-1]] + mlp_hidden_dims
            self.mlp = MLP(
                mlp_hidden_dims,
                mlp_activation,
                mlp_dropout,
                mlp_batchnorm,
                mlp_batchnorm_last,
                mlp_linear_first,
            )
        else:
            self.mlp = None

    def forward(self, X: Tensor) -> Tensor:
        x = self._get_embeddings(X)
        x = self.encoder(x)
        if self.mlp is not None:
            x = self.mlp(x)
        return x

    @property
    def output_dim(self) -> int:
        r"""The output dimension of the model. This is a required property
        neccesary to build the `WideDeep` class
        """
        return (
            self.mlp_hidden_dims[-1]
            if self.mlp_hidden_dims is not None
            else self.blocks_dims[-1]
        )


class TabResnetDecoder(nn.Module):
    r"""Companion decoder model for the `TabResnet` model (which can be
    considered an encoder itself)

    This class is designed to be used with the `EncoderDecoderTrainer` when
    using self-supervised pre-training (see the corresponding section in the
    docs). This class will receive the output from the ResNet blocks or the
    MLP(if present) and '_reconstruct_' the embeddings.

    Parameters
    ----------
    embed_dim: int
        Size of the embeddings tensor to be reconstructed.
    blocks_dims: List, default = [200, 100, 100]
        List of integers that define the input and output units of each block.
        For example: _[200, 100, 100]_ will generate 2 blocks. The first will
        receive a tensor of size 200 and output a tensor of size 100, and the
        second will receive a tensor of size 100 and output a tensor of size
        100. See `pytorch_widedeep.models.tab_resnet._layers` for
        details on the structure of each block.
    blocks_dropout: float, default =  0.1
        Block's internal dropout.
    simplify_blocks: bool, default = False,
        Boolean indicating if the simplest possible residual blocks (`X -> [
        [LIN, BN, ACT]  + X ]`) will be used instead of a standard one
        (`X -> [ [LIN1, BN1, ACT1] -> [LIN2, BN2]  + X ]`).
    mlp_hidden_dims: List, Optional, default = None
        List with the number of neurons per dense layer in the MLP. e.g:
        _[64, 32]_. If `None` the  output of the Resnet Blocks will be
        connected directly to the output neuron(s).
    mlp_activation: str, default = "relu"
        Activation function for the dense layers of the MLP. Currently
        _'tanh'_, _'relu'_, _'leaky'_relu` and _'gelu'_ are supported
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
        layer. If `True: [LIN -> ACT -> BN -> DP]`. If `False: [BN -> DP ->
        LIN -> ACT]`

    Attributes
    ----------
    decoder: nn.Module
        deep dense Resnet model that will receive the output of the encoder IF
        `mlp_hidden_dims` is None
    mlp: nn.Module
        if `mlp_hidden_dims` is not None, the overall decoder will consist
        in an MLP that will receive the output of the encoder followed by the
        deep dense Resnet.

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import TabResnetDecoder
    >>> x_inp = torch.rand(3, 8)
    >>> decoder = TabResnetDecoder(embed_dim=32, blocks_dims=[8, 16, 16])
    >>> res = decoder(x_inp)
    >>> res.shape
    torch.Size([3, 32])
    """

    def __init__(
        self,
        embed_dim: int,
        blocks_dims: List[int] = [100, 100, 200],
        blocks_dropout: float = 0.1,
        simplify_blocks: bool = False,
        mlp_hidden_dims: Optional[List[int]] = None,
        mlp_activation: str = "relu",
        mlp_dropout: float = 0.1,
        mlp_batchnorm: bool = False,
        mlp_batchnorm_last: bool = False,
        mlp_linear_first: bool = False,
    ):
        super(TabResnetDecoder, self).__init__()

        if len(blocks_dims) < 2:
            raise ValueError(
                "'blocks' must contain at least two elements, e.g. [256, 128]"
            )

        self.embed_dim = embed_dim

        self.blocks_dims = blocks_dims
        self.blocks_dropout = blocks_dropout
        self.simplify_blocks = simplify_blocks

        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.mlp_dropout = mlp_dropout
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.mlp_linear_first = mlp_linear_first

        if self.mlp_hidden_dims is not None:
            self.mlp = MLP(
                mlp_hidden_dims,
                mlp_activation,
                mlp_dropout,
                mlp_batchnorm,
                mlp_batchnorm_last,
                mlp_linear_first,
            )
        else:
            self.mlp = None

        if self.mlp is not None:
            self.decoder = DenseResnet(
                mlp_hidden_dims[-1], blocks_dims, blocks_dropout, self.simplify_blocks
            )
        else:
            self.decoder = DenseResnet(
                blocks_dims[0], blocks_dims, blocks_dropout, self.simplify_blocks
            )

        self.reconstruction_layer = nn.Linear(blocks_dims[-1], embed_dim, bias=False)

    def forward(self, X: Tensor) -> Tensor:
        x = self.mlp(X) if self.mlp is not None else X
        return self.reconstruction_layer(self.decoder(x))
