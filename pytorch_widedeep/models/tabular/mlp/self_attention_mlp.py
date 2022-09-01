from torch import nn

from pytorch_widedeep.wdtypes import Dict, List, Tuple, Tensor, Optional
from pytorch_widedeep.models.tabular.mlp._encoders import SelfAttentionEncoder
from pytorch_widedeep.models.tabular._base_tabular_model import (
    BaseTabularModelWithAttention,
)


class SelfAttentionMLP(BaseTabularModelWithAttention):
    r"""Defines a `SelfAttentionMLP` model that can be used as the
    deeptabular component of a Wide & Deep model or independently by
    itself.

    This class combines embedding representations of the categorical features
    with numerical (aka continuous) features that are also embedded. These
    are then passed through a series of attention blocks. Each attention
    block is comprised by what we would refer as a simplified
    `SelfAttentionEncoder`. See
    `pytorch_widedeep.models.tabular.mlp._attention_layers` for details. The
    reason to use a simplified version of self attention is because we
    observed that the '_standard_' attention mechanism used in the
    TabTransformer has a notable tendency to overfit.

    Parameters
    ----------
    column_idx: Dict
        Dict containing the index of the columns that will be passed through
        the model. Required to slice the tensors. e.g.
        _{'education': 0, 'relationship': 1, 'workclass': 2, ...}_
    cat_embed_input: List
        List of Tuples with the column name and number of unique values per
        categorical column e.g. _[(education, 11), ...]_.
    cat_embed_dropout: float, default = 0.1
        Categorical embeddings dropout
    use_cat_bias: bool, default = False,
        Boolean indicating if bias will be used for the categorical embeddings
    cat_embed_activation: Optional, str, default = None,
        Activation function for the categorical embeddings, if any. Currently
        _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported.
    full_embed_dropout: bool, default = False
        Boolean indicating if an entire embedding (i.e. the representation of
        one column) will be dropped in the batch. See:
        `pytorch_widedeep.models.embeddings_layers.FullEmbeddingDropout`.
        If full_embed_dropout = True, `cat_embed_dropout` is ignored.
    shared_embed: bool, default = False
        The of sharing part of the embeddings per column is to enable the
        model to distinguish the classes in one column from those in the
        other columns. In other words, the idea is to let the model learn
        which column is embedded at the time.
    add_shared_embed: bool, default = False,
        The two embedding sharing strategies are: 1) add the shared embeddings
        to the column embeddings or 2) to replace the first
        frac_shared_embed with the shared embeddings.
        See `pytorch_widedeep.models.embeddings_layers.SharedEmbeddings`
    frac_shared_embed: float, default = 0.25
        The fraction of embeddings that will be shared (if add_shared_embed
        = False) by all the different categories for one particular
        column.
    continuous_cols: List, Optional, default = None
        List with the name of the numeric (aka continuous) columns
    cont_norm_layer: str, default =  "batchnorm"
        Type of normalization layer applied to the continuous features. Options
        are: _'layernorm'_, _'batchnorm'_ or None.
    cont_embed_dropout: float, default = 0.1,
        Continuous embeddings dropout
    use_cont_bias: bool, default = True,
        Boolean indicating if bias will be used for the continuous embeddings
    cont_embed_activation: str, default = None
        Activation function to be applied to the continuous embeddings, if
        any. _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported.
    input_dim: int, default = 32
        The so-called *dimension of the model*. Is the number of
        embeddings used to encode the categorical and/or continuous columns
    attn_dropout: float, default = 0.2
        Dropout for each attention block
    n_heads: int, default = 8
        Number of attention heads per attention block.
    use_bias: bool, default = False
        Boolean indicating whether or not to use bias in the Q, K projection
        layers.
    with_addnorm: bool = False,
        Boolean indicating if residual connections will be used in the attention blocks
    attn_activation: str, default = "leaky_relu"
        String indicating the activation function to be applied to the dense
        layer in each attention encoder. _'tanh'_, _'relu'_, _'leaky_relu'_
        and _'gelu'_ are supported.
    n_blocks: int, default = 3
        Number of attention blocks

    Attributes
    ----------
    cat_and_cont_embed: nn.Module
        This is the module that processes the categorical and continuous columns
    encoder: nn.Module
        Sequence of attention encoders.

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import SelfAttentionMLP
    >>> X_tab = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
    >>> colnames = ['a', 'b', 'c', 'd', 'e']
    >>> cat_embed_input = [(u,i,j) for u,i,j in zip(colnames[:4], [4]*4, [8]*4)]
    >>> column_idx = {k:v for v,k in enumerate(colnames)}
    >>> model = SelfAttentionMLP(column_idx=column_idx, cat_embed_input=cat_embed_input, continuous_cols = ['e'])
    >>> out = model(X_tab)
    """

    def __init__(
        self,
        column_idx: Dict[str, int],
        cat_embed_input: Optional[List[Tuple[str, int]]] = None,
        cat_embed_dropout: float = 0.1,
        use_cat_bias: bool = False,
        cat_embed_activation: Optional[str] = None,
        full_embed_dropout: bool = False,
        shared_embed: bool = False,
        add_shared_embed: bool = False,
        frac_shared_embed: float = 0.25,
        continuous_cols: Optional[List[str]] = None,
        cont_norm_layer: str = None,
        cont_embed_dropout: float = 0.1,
        use_cont_bias: bool = True,
        cont_embed_activation: Optional[str] = None,
        input_dim: int = 32,
        attn_dropout: float = 0.2,
        n_heads: int = 8,
        use_bias: bool = False,
        with_addnorm: bool = False,
        attn_activation: str = "leaky_relu",
        n_blocks: int = 3,
    ):
        super(SelfAttentionMLP, self).__init__(
            column_idx=column_idx,
            cat_embed_input=cat_embed_input,
            cat_embed_dropout=cat_embed_dropout,
            use_cat_bias=use_cat_bias,
            cat_embed_activation=cat_embed_activation,
            full_embed_dropout=full_embed_dropout,
            shared_embed=shared_embed,
            add_shared_embed=add_shared_embed,
            frac_shared_embed=frac_shared_embed,
            continuous_cols=continuous_cols,
            cont_norm_layer=cont_norm_layer,
            embed_continuous=True,
            cont_embed_dropout=cont_embed_dropout,
            use_cont_bias=use_cont_bias,
            cont_embed_activation=cont_embed_activation,
            input_dim=input_dim,
        )

        self.attn_dropout = attn_dropout
        self.n_heads = n_heads
        self.use_bias = use_bias
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
                SelfAttentionEncoder(
                    input_dim,
                    attn_dropout,
                    use_bias,
                    n_heads,
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
        neccesary to build the WideDeep class
        """
        return (
            self.input_dim
            if self.with_cls_token
            else ((self.n_cat + self.n_cont) * self.input_dim)
        )

    @property
    def attention_weights(self) -> List:
        r"""List with the attention weights per block

        The shape of the attention weights is $(N, H, F, F)$, where $N$ is the
        batch size, $H$ is the number of attention heads and $F$ is the
        number of features/columns in the dataset
        """
        return [blk.attn.attn_weights for blk in self.encoder]
