from torch import nn

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models.tabular.mlp._encoders import (
    ContextAttentionEncoder,
)
from pytorch_widedeep.models.tabular._base_tabular_model import (
    BaseTabularModelWithAttention,
)


class ContextAttentionMLP(BaseTabularModelWithAttention):
    r"""Defines a ``ContextAttentionMLP`` model that can be used as the
    ``deeptabular`` component of a Wide & Deep model or independently by
    itself.

    This class combines embedding representations of the categorical features
    with numerical (aka continuous) features that are also embedded. These
    are then passed through a series of attention blocks. Each attention
    block is comprised by a ``ContextAttentionEncoder``.
    See :obj:`pytorch_widedeep.models.tabular.mlp._attention_layers` for
    details


    Parameters
    ----------
    column_idx: Dict
        Dict containing the index of the columns that will be passed through
        the model. Required to slice the tensors. e.g.
        {'education': 0, 'relationship': 1, 'workclass': 2, ...}
    cat_embed_input: List
        List of Tuples with the column name and number of unique values per
        categorical columns. e.g. [(education, 11), ...]
    cat_embed_dropout: float, default = 0.1
        Categorical embeddings dropout
    use_cat_bias: bool, default = False,
        Boolean indicating if bias will be used for the categorical embeddings
    cat_embed_activation: Optional, str, default = None,
        Activation function for the categorical embeddings, if any. `'tanh'`,
        `'relu'`, `'leaky_relu'` and `'gelu'` are supported.
    full_embed_dropout: bool, default = False
        Boolean indicating if an entire embedding (i.e. the representation of
        one column) will be dropped in the batch. See:
        :obj:`pytorch_widedeep.models.embeddings_layers.FullEmbeddingDropout`.
        If ``full_embed_dropout = True``, ``cat_embed_dropout`` is ignored.
    shared_embed: bool, default = False
        The idea behind sharing part of the embeddings per column is to let
        the model learn which column is embedded at the time.
    add_shared_embed: bool, default = False,
        The two embedding sharing strategies are: 1) add the shared embeddings
        to the column embeddings or 2) to replace the first
        ``frac_shared_embed`` with the shared embeddings.
        See :obj:`pytorch_widedeep.models.embeddings_layers.SharedEmbeddings`
    frac_shared_embed: float, default = 0.25
        The fraction of embeddings that will be shared (if ``add_shared_embed
        = False``) by all the different categories for one particular
        column.
    continuous_cols: List, Optional, default = None
        List with the name of the numeric (aka continuous) columns
    cont_norm_layer: str, default =  "batchnorm"
        Type of normalization layer applied to the continuous features. Options
        are: 'layernorm', 'batchnorm' or None.
    cont_embed_dropout: float, default = 0.1,
        Continuous embeddings dropout
    use_cont_bias: bool, default = True,
        Boolean indicating if bias will be used for the continuous embeddings
    cont_embed_activation: str, default = None
        Activation function to be applied to the continuous embeddings, if
        any. `'tanh'`, `'relu'`, `'leaky_relu'` and `'gelu'` are supported.
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
        layer in each attention encoder. `'tanh'`, `'relu'`, `'leaky_relu'`
        and `'gelu'` are supported.
    n_blocks: int, default = 3
        Number of attention blocks

    Attributes
    ----------
    cat_and_cont_embed: ``nn.Module``
        This is the module that processes the categorical and continuous columns
    attention_blks: ``nn.Sequential``
        Sequence of attention encoders.
    output_dim: int
        The output dimension of the model. This is a required attribute
        neccesary to build the ``WideDeep`` class

    Example
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
        self.with_addnorm = with_addnorm
        self.attn_activation = attn_activation
        self.n_blocks = n_blocks

        self.with_cls_token = "cls_token" in column_idx
        self.n_cat = len(cat_embed_input) if cat_embed_input is not None else 0
        self.n_cont = len(continuous_cols) if continuous_cols is not None else 0

        # Embeddings are instantiated at the base model
        # Attention Blocks
        self.attention_blks = nn.Sequential()
        for i in range(n_blocks):
            self.attention_blks.add_module(
                "attention_block" + str(i),
                ContextAttentionEncoder(
                    input_dim,
                    attn_dropout,
                    with_addnorm,
                    attn_activation,
                ),
            )

        self.output_dim: int = (
            input_dim
            if self.with_cls_token
            else ((self.n_cat + self.n_cont) * input_dim)
        )

    def forward(self, X: Tensor) -> Tensor:
        x = self._get_embeddings(X)
        x = self.attention_blks(x)
        if self.with_cls_token:
            out = x[:, 0, :]
        else:
            out = x.flatten(1)
        return out

    @property
    def attention_weights(self) -> List:
        r"""List with the attention weights per block

        The shape of the attention weights is:

        :math:`(N, F)`

        Where *N* is the batch size and *F* is the number of features/columns
        in the dataset
        """
        return [blk.attn.attn_weights for blk in self.attention_blks]
