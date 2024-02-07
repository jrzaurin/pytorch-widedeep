from torch import nn

from pytorch_widedeep.wdtypes import (
    Dict,
    List,
    Tuple,
    Tensor,
    Literal,
    Optional,
)
from pytorch_widedeep.models.tabular.mlp._encoders import (
    ContextAttentionEncoder,
)
from pytorch_widedeep.models.tabular._base_tabular_model import (
    BaseTabularModelWithAttention,
)


class ContextAttentionMLP(BaseTabularModelWithAttention):
    r"""Defines a `ContextAttentionMLP` model that can be used as the
    `deeptabular` component of a Wide & Deep model or independently by
    itself.

    This class combines embedding representations of the categorical features
    with numerical (aka continuous) features that are also embedded. These
    are then passed through a series of attention blocks. Each attention
    block is comprised by a `ContextAttentionEncoder`. Such encoder is in
    part inspired by the attention mechanism described in
    [Hierarchical Attention Networks for Document
    Classification](https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf).
    See `pytorch_widedeep.models.tabular.mlp._attention_layers` for details.

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
    input_dim: int, default = 32
        The so-called *dimension of the model*. Is the number of embeddings
        used to encode the categorical and/or continuous columns
    attn_dropout: float, default = 0.2
        Dropout for each attention block
    with_addnorm: bool = False,
        Boolean indicating if residual connections will be used in the
        attention blocks
    attn_activation: str, default = "leaky_relu"
        String indicating the activation function to be applied to the dense
        layer in each attention encoder. _'tanh'_, _'relu'_, _'leaky_relu'_
        and _'gelu'_ are supported.
    n_blocks: int, default = 3
        Number of attention blocks

    Attributes
    ----------
    encoder: nn.Module
        Sequence of attention encoders.

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import ContextAttentionMLP
    >>> X_tab = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
    >>> colnames = ['a', 'b', 'c', 'd', 'e']
    >>> cat_embed_input = [(u,i,j) for u,i,j in zip(colnames[:4], [4]*4, [8]*4)]
    >>> column_idx = {k:v for v,k in enumerate(colnames)}
    >>> model = ContextAttentionMLP(column_idx=column_idx, cat_embed_input=cat_embed_input, continuous_cols = ['e'])
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
        input_dim: int = 32,
        attn_dropout: float = 0.2,
        with_addnorm: bool = False,
        attn_activation: str = "leaky_relu",
        n_blocks: int = 3,
    ):
        super(ContextAttentionMLP, self).__init__(
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

        self.attn_dropout = attn_dropout
        self.with_addnorm = with_addnorm
        self.attn_activation = attn_activation
        self.n_blocks = n_blocks

        self.with_cls_token = "cls_token" in column_idx
        self.n_cat = len(cat_embed_input) if cat_embed_input is not None else 0
        self.n_cont = len(continuous_cols) if continuous_cols is not None else 0

        # Embeddings are instantiated at the base model
        # Attention Blocks
        self.encoder = nn.Sequential()
        for i in range(n_blocks):
            self.encoder.add_module(
                "attention_block" + str(i),
                ContextAttentionEncoder(
                    input_dim,
                    attn_dropout,
                    with_addnorm,
                    attn_activation,
                ),
            )

    def forward(self, X: Tensor) -> Tensor:
        x = self._get_embeddings(X)
        x = self.encoder(x)
        if self.with_cls_token:
            out = x[:, 0, :]
        else:
            out = x.flatten(1)
        return out

    @property
    def output_dim(self) -> int:
        r"""The output dimension of the model. This is a required property
        neccesary to build the `WideDeep` class
        """
        return (
            self.input_dim
            if self.with_cls_token
            else ((self.n_cat + self.n_cont) * self.input_dim)
        )

    @property
    def attention_weights(self) -> List[Tensor]:
        r"""List with the attention weights per block

        The shape of the attention weights is $(N, F)$, where $N$ is the batch
        size and $F$ is the number of features/columns in the dataset
        """
        return [blk.attn.attn_weights for blk in self.encoder]
