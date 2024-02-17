from torch import nn

from pytorch_widedeep.wdtypes import (
    Dict,
    List,
    Tuple,
    Union,
    Tensor,
    Literal,
    Optional,
)
from pytorch_widedeep.models.tabular.mlp._layers import MLP
from pytorch_widedeep.models.tabular._base_tabular_model import (
    BaseTabularModelWithoutAttention,
)


class TabMlp(BaseTabularModelWithoutAttention):
    r"""Defines a `TabMlp` model that can be used as the `deeptabular`
    component of a Wide & Deep model or independently by itself.

    This class combines embedding representations of the categorical features
    with numerical (aka continuous) features, embedded or not. These are then
    passed through a series of dense layers (i.e. a MLP).

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
    mlp_hidden_dims: List, default = [200, 100]
        List with the number of neurons per dense layer in the mlp.
    mlp_activation: str, default = "relu"
        Activation function for the dense layers of the MLP. Currently
        _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported
    mlp_dropout: float or List, default = 0.1
        float or List of floats with the dropout between the dense layers.
        e.g: _[0.5,0.5]_
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
    encoder: nn.Module
        mlp model that will receive the concatenation of the embeddings and
        the continuous columns

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import TabMlp
    >>> X_tab = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
    >>> colnames = ["a", "b", "c", "d", "e"]
    >>> cat_embed_input = [(u, i, j) for u, i, j in zip(colnames[:4], [4] * 4, [8] * 4)]
    >>> column_idx = {k: v for v, k in enumerate(colnames)}
    >>> model = TabMlp(mlp_hidden_dims=[8, 4], column_idx=column_idx, cat_embed_input=cat_embed_input,
    ... continuous_cols=["e"])
    >>> out = model(X_tab)
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
        mlp_hidden_dims: List[int] = [200, 100],
        mlp_activation: str = "relu",
        mlp_dropout: Union[float, List[float]] = 0.1,
        mlp_batchnorm: bool = False,
        mlp_batchnorm_last: bool = False,
        mlp_linear_first: bool = True,
    ):
        super(TabMlp, self).__init__(
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

        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.mlp_dropout = mlp_dropout
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.mlp_linear_first = mlp_linear_first

        # Embeddings are instantiated at the base model
        # Mlp
        mlp_input_dim = self.cat_out_dim + self.cont_out_dim
        mlp_hidden_dims = [mlp_input_dim] + mlp_hidden_dims
        self.encoder = MLP(
            mlp_hidden_dims,
            mlp_activation,
            mlp_dropout,
            mlp_batchnorm,
            mlp_batchnorm_last,
            mlp_linear_first,
        )

    def forward(self, X: Tensor) -> Tensor:
        x = self._get_embeddings(X)
        return self.encoder(x)

    @property
    def output_dim(self) -> int:
        r"""The output dimension of the model. This is a required property
        neccesary to build the `WideDeep` class"""
        return self.mlp_hidden_dims[-1]


# This is a companion Decoder for the TabMlp. We prefer not to refer to the
# 'TabMlp' model as 'TabMlpEncoder' (despite the fact that is indeed an
#  encoder) for two reasons: 1. For convenience accross the library and 2.
#  Because decoders are only going to be used in one of our implementations
#  of Self Supervised pretraining, and we prefer to keep the names of
#  the 'general' DL models as they are (e.g. TabMlp) as opposed as carry
#  the 'Encoder' description (e.g. TabMlpEncoder) throughout the library
class TabMlpDecoder(nn.Module):
    r"""Companion decoder model for the `TabMlp` model (which can be considered
    an encoder itself).

    This class is designed to be used with the `EncoderDecoderTrainer` when
    using self-supervised pre-training (see the corresponding section in the
    docs). The `TabMlpDecoder` will receive the output from the MLP
    and '_reconstruct_' the embeddings.

    Parameters
    ----------
    embed_dim: int
        Size of the embeddings tensor that needs to be reconstructed.
    mlp_hidden_dims: List, default = [200, 100]
        List with the number of neurons per dense layer in the mlp.
    mlp_activation: str, default = "relu"
        Activation function for the dense layers of the MLP. Currently
        _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported
    mlp_dropout: float or List, default = 0.1
        float or List of floats with the dropout between the dense layers.
        e.g: _[0.5,0.5]_
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
        mlp model that will receive the output of the encoder

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import TabMlpDecoder
    >>> x_inp = torch.rand(3, 8)
    >>> decoder = TabMlpDecoder(embed_dim=32, mlp_hidden_dims=[8,16])
    >>> res = decoder(x_inp)
    >>> res.shape
    torch.Size([3, 32])
    """

    def __init__(
        self,
        embed_dim: int,
        mlp_hidden_dims: List[int] = [100, 200],
        mlp_activation: str = "relu",
        mlp_dropout: Union[float, List[float]] = 0.1,
        mlp_batchnorm: bool = False,
        mlp_batchnorm_last: bool = False,
        mlp_linear_first: bool = True,
    ):
        super(TabMlpDecoder, self).__init__()

        self.embed_dim = embed_dim

        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.mlp_dropout = mlp_dropout
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.mlp_linear_first = mlp_linear_first

        self.decoder = MLP(
            mlp_hidden_dims + [self.embed_dim],
            mlp_activation,
            mlp_dropout,
            mlp_batchnorm,
            mlp_batchnorm_last,
            mlp_linear_first,
        )

    def forward(self, X: Tensor) -> Tensor:
        return self.decoder(X)
