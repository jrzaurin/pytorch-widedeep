from torch import nn

from pytorch_widedeep.wdtypes import (
    Dict,
    List,
    Tuple,
    Tensor,
    Literal,
    Optional,
)
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

    Most of the parameters for this class are `Optional` since the use of
    categorical or continuous is in fact optional (i.e. one can use
    categorical features only, continuous features only or both).

    Parameters
    ----------
    column_idx: Dict
        Dict containing the index of the columns that will be passed through
        the `TabMlp` model. Required to slice the tensors. e.g. _{'education':
        0, 'relationship': 1, 'workclass': 2, ...}_.
    cat_embed_input: List, Optional, default = None
        List of Tuples with the column name, number of unique values and
        embedding dimension. e.g. _[(education, 11, 32), ...]_
    cat_embed_dropout: float, Optional, default = None
        Categorical embeddings dropout. If `None`, it will default
        to 0.
    use_cat_bias: bool, Optional, default = None,
        Boolean indicating if bias will be used for the categorical embeddings.
        If `None`, it will default to 'False'.
    cat_embed_activation: Optional, str, default = None,
        Activation function for the categorical embeddings, if any. Currently
        _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported
    continuous_cols: List, Optional, default = None
        List with the name of the numeric (aka continuous) columns
    cont_norm_layer: str, Optional, default =  None
        Type of normalization layer applied to the continuous features.
        Options are: _'layernorm'_ and _'batchnorm'_. if `None`, no
        normalization layer will be used.
    embed_continuous: bool, Optional, default = None,
        Boolean indicating if the continuous columns will be embedded using
        one of the available methods: _'standard'_, _'periodic'_
        or _'piecewise'_. If `None`, it will default to 'False'.<br/>
        :information_source: **NOTE**: This parameter is deprecated and it
         will be removed in future releases. Please, use the
         `embed_continuous_method` parameter instead.
    embed_continuous_method: Optional, str, default = None,
        Method to use to embed the continuous features. Options are:
        _'standard'_, _'periodic'_ or _'piecewise'_. The _'standard'_
        embedding method is based on the FT-Transformer implementation
        presented in the paper: [Revisiting Deep Learning Models for
        Tabular Data](https://arxiv.org/abs/2106.11959v5). The _'periodic'_
        and_'piecewise'_ methods were presented in the paper: [On Embeddings for
        Numerical Features in Tabular Deep Learning](https://arxiv.org/abs/2203.05556).
        Please, read the papers for details.
    cont_embed_dim: int, Optional, default = None,
        Size of the continuous embeddings. If the continuous columns are
        embedded, `cont_embed_dim` must be passed.
    cont_embed_dropout: float, Optional, default = None,
        Dropout for the continuous embeddings. If `None`, it will default to 0.0
    cont_embed_activation: Optional, str, default = None,
        Activation function for the continuous embeddings if any. Currently
        _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported.
        If `None`, no activation function will be applied.
    quantization_setup: Dict[str, List[float]], Optional, default = None,
        This parameter is used when the _'piecewise'_ method is used to embed
        the continuous cols. It is a dict where keys are the name of the continuous
        columns and values are lists with the boundaries for the quantization
        of the continuous_cols. See the examples for details. If
        If the _'piecewise'_ method is used, this parameter is required.
    n_frequencies: int, Optional, default = None,
        This is the so called _'k'_ in their paper [On Embeddings for
        Numerical Features in Tabular Deep Learning](https://arxiv.org/abs/2203.05556),
        and is the number of 'frequencies' that will be used to represent each
        continuous column. See their Eq 2 in the paper for details. If
        the _'periodic'_ method is used, this parameter is required.
    sigma: float, Optional, default = None,
        This is the sigma parameter in the paper mentioned when describing the
        previous parameters and it is used to initialise the 'frequency
        weights'. See their Eq 2 in the paper for details. If
        the _'periodic'_ method is used, this parameter is required.
    share_last_layer: bool, Optional, default = None,
        This parameter is not present in the before mentioned paper but it is implemented in
        the [official repo](https://github.com/yandex-research/rtdl-num-embeddings/tree/main).
        If `True` the linear layer that turns the frequencies into embeddings
        will be shared across the continuous columns. If `False` a different
        linear layer will be used for each continuous column.
        If the _'periodic'_ method is used, this parameter is required.
    full_embed_dropout: bool, Optional, default = None,
        If `True`, the full embedding corresponding to a column will be masked
        out/dropout. If `None`, it will default to `False`.
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
    mlp_activation: str, Optional, default = None
        Activation function for the dense layers of the MLP. Currently
        _'tanh'_, _'relu'_, _'leaky'_relu' and _'gelu'_ are supported.
        If 'mlp_hidden_dims' is not `None` and this parameter is `None`, it
        will default to _'relu'_.
    mlp_dropout: float, Optional, default = None
        float with the dropout between the dense layers of the MLP.
        If 'mlp_hidden_dims' is not `None` and this parameter is `None`, it
        will default to 0.0.
    mlp_batchnorm: bool, Optional, default = None
        Boolean indicating whether or not batch normalization will be applied
        to the dense layers
        If 'mlp_hidden_dims' is not `None` and this parameter is `None`, it
        will default to False.
    mlp_batchnorm_last: bool, Optional, default = None
        Boolean indicating whether or not batch normalization will be applied
        to the last of the dense layers
        If 'mlp_hidden_dims' is not `None` and this parameter is `None`, it
        will default to False.
    mlp_linear_first: bool, Optional, default = None
        Boolean indicating the order of the operations in the dense
        layer. If `True: [LIN -> ACT -> BN -> DP]`. If `False: [BN -> DP ->
        LIN -> ACT]`
        If 'mlp_hidden_dims' is not `None` and this parameter is `None`, it
        will default to `True`.

    Attributes
    ----------
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
        *,
        cat_embed_input: Optional[List[Tuple[str, int, int]]] = None,
        cat_embed_dropout: Optional[float] = None,
        use_cat_bias: Optional[bool] = None,
        cat_embed_activation: Optional[str] = None,
        continuous_cols: Optional[List[str]] = None,
        cont_norm_layer: Optional[Literal["batchnorm", "layernorm"]] = None,
        embed_continuous: Optional[bool] = None,
        embed_continuous_method: Optional[
            Literal["standard", "piecewise", "periodic"]
        ] = None,
        cont_embed_dim: Optional[int] = None,
        cont_embed_dropout: Optional[float] = None,
        cont_embed_activation: Optional[str] = None,
        quantization_setup: Optional[Dict[str, List[float]]] = None,
        n_frequencies: Optional[int] = None,
        sigma: Optional[float] = None,
        share_last_layer: Optional[bool] = None,
        full_embed_dropout: Optional[bool] = None,
        blocks_dims: List[int] = [200, 100, 100],
        blocks_dropout: float = 0.1,
        simplify_blocks: bool = False,
        mlp_hidden_dims: Optional[List[int]] = None,
        mlp_activation: Optional[str] = None,
        mlp_dropout: Optional[float] = None,
        mlp_batchnorm: Optional[bool] = None,
        mlp_batchnorm_last: Optional[bool] = None,
        mlp_linear_first: Optional[bool] = None,
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
            embed_continuous_method=embed_continuous_method,
            cont_embed_dim=cont_embed_dim,
            cont_embed_dropout=cont_embed_dropout,
            cont_embed_activation=cont_embed_activation,
            quantization_setup=quantization_setup,
            n_frequencies=n_frequencies,
            sigma=sigma,
            share_last_layer=share_last_layer,
            full_embed_dropout=full_embed_dropout,
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

        # Resnet
        dense_resnet_input_dim = self.cat_out_dim + self.cont_out_dim
        self.encoder = DenseResnet(
            dense_resnet_input_dim, blocks_dims, blocks_dropout, self.simplify_blocks
        )

        # Mlp: adding an MLP on top of the Resnet blocks is optional and
        # therefore all related params are optional
        if self.mlp_hidden_dims is not None:
            self.mlp = MLP(
                d_hidden=[self.blocks_dims[-1]] + self.mlp_hidden_dims,
                activation=(
                    "relu" if self.mlp_activation is None else self.mlp_activation
                ),
                dropout=0.0 if self.mlp_dropout is None else self.mlp_dropout,
                batchnorm=False if self.mlp_batchnorm is None else self.mlp_batchnorm,
                batchnorm_last=(
                    False
                    if self.mlp_batchnorm_last is None
                    else self.mlp_batchnorm_last
                ),
                linear_first=(
                    True if self.mlp_linear_first is None else self.mlp_linear_first
                ),
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
    mlp_activation: str, Optional, default = None
        Activation function for the dense layers of the MLP. Currently
        _'tanh'_, _'relu'_, _'leaky'_relu' and _'gelu'_ are supported.
        If 'mlp_hidden_dims' is not `None` and this parameter is `None`, it
        will default to _'relu'_.
    mlp_dropout: float, Optional, default = None
        float with the dropout between the dense layers of the MLP.
        If 'mlp_hidden_dims' is not `None` and this parameter is `None`, it
        will default to 0.0.
    mlp_batchnorm: bool, Optional, default = None
        Boolean indicating whether or not batch normalization will be applied
        to the dense layers
        If 'mlp_hidden_dims' is not `None` and this parameter is `None`, it
        will default to False.
    mlp_batchnorm_last: bool, Optional, default = None
        Boolean indicating whether or not batch normalization will be applied
        to the last of the dense layers
        If 'mlp_hidden_dims' is not `None` and this parameter is `None`, it
        will default to False.
    mlp_linear_first: bool, Optional, default = None
        Boolean indicating the order of the operations in the dense
        layer. If `True: [LIN -> ACT -> BN -> DP]`. If `False: [BN -> DP ->
        LIN -> ACT]`
        If 'mlp_hidden_dims' is not `None` and this parameter is `None`, it
        will default to `True`.

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
        mlp_activation: Optional[str] = None,
        mlp_dropout: Optional[float] = None,
        mlp_batchnorm: Optional[bool] = None,
        mlp_batchnorm_last: Optional[bool] = None,
        mlp_linear_first: Optional[bool] = None,
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
                d_hidden=[self.mlp_first_hidden_dim] + self.mlp_hidden_dims,
                activation=(
                    "relu" if self.mlp_activation is None else self.mlp_activation
                ),
                dropout=0.0 if self.mlp_dropout is None else self.mlp_dropout,
                batchnorm=False if self.mlp_batchnorm is None else self.mlp_batchnorm,
                batchnorm_last=(
                    False
                    if self.mlp_batchnorm_last is None
                    else self.mlp_batchnorm_last
                ),
                linear_first=(
                    True if self.mlp_linear_first is None else self.mlp_linear_first
                ),
            )
            self.decoder = DenseResnet(
                self.mlp_hidden_dims[-1],
                blocks_dims,
                blocks_dropout,
                self.simplify_blocks,
            )
        else:
            self.mlp = None
            self.decoder = DenseResnet(
                blocks_dims[0], blocks_dims, blocks_dropout, self.simplify_blocks
            )

        self.reconstruction_layer = nn.Linear(blocks_dims[-1], embed_dim, bias=False)

    def forward(self, X: Tensor) -> Tensor:
        x = self.mlp(X) if self.mlp is not None else X
        return self.reconstruction_layer(self.decoder(x))
