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
from pytorch_widedeep.models.tabular._base_tabular_model import (
    BaseTabularModelWithAttention,
)
from pytorch_widedeep.models.tabular.transformers._encoders import (
    FTTransformerEncoder,
)


class FTTransformer(BaseTabularModelWithAttention):
    r"""Defines a [FTTransformer model](https://arxiv.org/abs/2106.11959) that
    can be used as the `deeptabular` component of a Wide & Deep model or
    independently by itself.

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
        List of Tuples with the column name and number of unique values and
        embedding dimension. e.g. _[(education, 11), ...]_
    cat_embed_dropout: float, Optional, default = None
        Categorical embeddings dropout. If `None`, it will default
        to 0.
    use_cat_bias: bool, Optional, default = None,
        Boolean indicating if bias will be used for the categorical embeddings.
        If `None`, it will default to 'False'.
    cat_embed_activation: Optional, str, default = None,
        Activation function for the categorical embeddings, if any. Currently
        _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported
    shared_embed: bool, Optional, default = None
        Boolean indicating if the embeddings will be "shared". The idea behind `shared_embed` is
        described in the Appendix A in the [TabTransformer paper](https://arxiv.org/abs/2012.06678):
        _'The goal of having column embedding is to enable the model to
        distinguish the classes in one column from those in the other
        columns'_. In other words, the idea is to let the model learn which
        column is embedded at the time. See: `pytorch_widedeep.models.transformers._layers.SharedEmbeddings`.
    add_shared_embed: bool, Optional, default = None
        The two embedding sharing strategies are: 1) add the shared embeddings
        to the column embeddings or 2) to replace the first
        `frac_shared_embed` with the shared embeddings.
        See `pytorch_widedeep.models.embeddings_layers.SharedEmbeddings`
        If 'None' is passed, it will default to 'False'.
    frac_shared_embed: float, Optional, default = None
        The fraction of embeddings that will be shared (if `add_shared_embed
        = False`) by all the different categories for one particular
        column. If 'None' is passed, it will default to 0.0.
    continuous_cols: List, Optional, default = None
        List with the name of the numeric (aka continuous) columns
    cont_norm_layer: str, Optional, default =  None
        Type of normalization layer applied to the continuous features.
        Options are: _'layernorm'_ and _'batchnorm'_. if `None`, no
        normalization layer will be used.
    embed_continuous_method: Optional, str, default = None,
        Method to use to embed the continuous features. Options are:
        _'standard'_, _'periodic'_ or _'piecewise'_. The _'standard'_
        embedding method is based on the FT-Transformer implementation
        presented in the paper: [Revisiting Deep Learning Models for
        Tabular Data](https://arxiv.org/abs/2106.11959v5). The _'periodic'_
        and_'piecewise'_ methods were presented in the paper: [On Embeddings for
        Numerical Features in Tabular Deep Learning](https://arxiv.org/abs/2203.05556).
        Please, read the papers for details.
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
    input_dim: int, default = 64
        The so-called *dimension of the model*. Is the number of embeddings used to encode
        the categorical and/or continuous columns.
    kv_compression_factor: int, default = 0.5
        By default, the FTTransformer uses Linear Attention
        (See [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768>) ).
        The compression factor that will be used to reduce the input sequence
        length. If we denote the resulting sequence length as
        $k = int(kv_{compression \space factor} \times s)$
        where $s$ is the input sequence length.
    kv_sharing: bool, default = False
        Boolean indicating if the $E$ and $F$ projection matrices
        will share weights.  See [Linformer: Self-Attention with Linear
        Complexity](https://arxiv.org/abs/2006.04768) for details
    n_heads: int, default = 8
        Number of attention heads per FTTransformer block
    use_qkv_bias: bool, default = False
        Boolean indicating whether or not to use bias in the Q, K, and V
        projection layers
    n_blocks: int, default = 4
        Number of FTTransformer blocks
    attn_dropout: float, default = 0.2
        Dropout that will be applied to the Linear-Attention layers
    ff_dropout: float, default = 0.1
        Dropout that will be applied to the FeedForward network
    ff_factor: float, default = 4 / 3
        Multiplicative factor applied to the first layer of the FF network in
        each Transformer block, This is normally set to 4, but they use 4/3
        in the paper.
    transformer_activation: str, default = "gelu"
        Transformer Encoder activation function. _'tanh'_, _'relu'_,
        _'leaky_relu'_, _'gelu'_, _'geglu'_ and _'reglu'_ are supported
    mlp_hidden_dims: List, Optional, default = None
        List with the number of neurons per dense layer in the MLP. e.g:
        _[64, 32]_. If not provided no MLP on top of the final
        FTTransformer block will be used.
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
        Sequence of FTTransformer blocks
    mlp: nn.Module
        MLP component in the model

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import FTTransformer
    >>> X_tab = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
    >>> colnames = ['a', 'b', 'c', 'd', 'e']
    >>> cat_embed_input = [(u,i) for u,i in zip(colnames[:4], [4]*4)]
    >>> continuous_cols = ['e']
    >>> column_idx = {k:v for v,k in enumerate(colnames)}
    >>> model = FTTransformer(column_idx=column_idx, cat_embed_input=cat_embed_input, continuous_cols=continuous_cols)
    >>> out = model(X_tab)
    """

    def __init__(
        self,
        column_idx: Dict[str, int],
        *,
        cat_embed_input: Optional[List[Tuple[str, int]]] = None,
        cat_embed_dropout: Optional[float] = None,
        use_cat_bias: Optional[bool] = None,
        cat_embed_activation: Optional[str] = None,
        shared_embed: Optional[bool] = None,
        add_shared_embed: Optional[bool] = None,
        frac_shared_embed: Optional[float] = None,
        continuous_cols: Optional[List[str]] = None,
        cont_norm_layer: Optional[Literal["batchnorm", "layernorm"]] = None,
        embed_continuous_method: Optional[
            Literal["standard", "piecewise", "periodic"]
        ] = "standard",
        cont_embed_dropout: Optional[float] = None,
        cont_embed_activation: Optional[str] = None,
        quantization_setup: Optional[Dict[str, List[float]]] = None,
        n_frequencies: Optional[int] = None,
        sigma: Optional[float] = None,
        share_last_layer: Optional[bool] = None,
        full_embed_dropout: Optional[bool] = None,
        input_dim: int = 64,
        kv_compression_factor: float = 0.5,
        kv_sharing: bool = False,
        use_qkv_bias: bool = False,
        n_heads: int = 8,
        n_blocks: int = 4,
        attn_dropout: float = 0.2,
        ff_dropout: float = 0.1,
        ff_factor: float = 1.33,
        transformer_activation: str = "reglu",
        mlp_hidden_dims: Optional[List[int]] = None,
        mlp_activation: Optional[str] = None,
        mlp_dropout: Optional[float] = None,
        mlp_batchnorm: Optional[bool] = None,
        mlp_batchnorm_last: Optional[bool] = None,
        mlp_linear_first: Optional[bool] = None,
    ):
        super(FTTransformer, self).__init__(
            column_idx=column_idx,
            cat_embed_input=cat_embed_input,
            cat_embed_dropout=cat_embed_dropout,
            use_cat_bias=use_cat_bias,
            cat_embed_activation=cat_embed_activation,
            shared_embed=shared_embed,
            add_shared_embed=add_shared_embed,
            frac_shared_embed=frac_shared_embed,
            continuous_cols=continuous_cols,
            cont_norm_layer=cont_norm_layer,
            embed_continuous=None,
            embed_continuous_method=embed_continuous_method,
            cont_embed_dropout=cont_embed_dropout,
            cont_embed_activation=cont_embed_activation,
            input_dim=input_dim,
            quantization_setup=quantization_setup,
            n_frequencies=n_frequencies,
            sigma=sigma,
            share_last_layer=share_last_layer,
            full_embed_dropout=full_embed_dropout,
        )

        self.kv_compression_factor = kv_compression_factor
        self.kv_sharing = kv_sharing
        self.use_qkv_bias = use_qkv_bias
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.ff_factor = ff_factor
        self.transformer_activation = transformer_activation

        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.mlp_dropout = mlp_dropout
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.mlp_linear_first = mlp_linear_first

        self.with_cls_token = "cls_token" in column_idx
        self.n_cat = len(cat_embed_input) if cat_embed_input is not None else 0
        self.n_cont = len(continuous_cols) if continuous_cols is not None else 0
        self.n_feats = self.n_cat + self.n_cont

        # Embeddings are instantiated at the base model
        # Transformer blocks
        is_first = True
        self.encoder = nn.Sequential()
        for i in range(n_blocks):
            self.encoder.add_module(
                "fttransformer_block" + str(i),
                FTTransformerEncoder(
                    input_dim,
                    self.n_feats,
                    n_heads,
                    use_qkv_bias,
                    attn_dropout,
                    ff_dropout,
                    ff_factor,
                    kv_compression_factor,
                    kv_sharing,
                    transformer_activation,
                    is_first,
                ),
            )
            is_first = False

        self.mlp_first_hidden_dim = (
            self.input_dim if self.with_cls_token else (self.n_feats * self.input_dim)
        )

        # Mlp: adding an MLP on top of the Resnet blocks is optional and
        # therefore all related params are optional
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
                    False if self.mlp_linear_first is None else self.mlp_linear_first
                ),
            )
        else:
            self.mlp = None

    def forward(self, X: Tensor) -> Tensor:
        x = self._get_embeddings(X)
        x = self.encoder(x)
        if self.with_cls_token:
            x = x[:, 0, :]
        else:
            x = x.flatten(1)
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
            else self.mlp_first_hidden_dim
        )

    @property
    def attention_weights(self) -> List[Tensor]:
        r"""List with the attention weights per block

        The shape of the attention weights is: $(N, H, F, k)$, where $N$ is
        the batch size, $H$ is the number of attention heads, $F$ is the
        number of features/columns and $k$ is the reduced sequence length or
        dimension, i.e. $k = int(kv_{compression \space factor} \times s)$
        """
        return [blk.attn.attn_weights for blk in self.encoder]
