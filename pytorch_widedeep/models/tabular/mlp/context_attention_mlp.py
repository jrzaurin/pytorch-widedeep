import torch
from torch import nn

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models.embeddings_layers import (
    SameSizeCatAndContEmbeddings,
)
from pytorch_widedeep.models.tabular.mlp._encoders import (
    ContextAttentionEncoder,
)


class ContextAttentionMLP(nn.Module):
    r"""Defines a ``ContextAttentionMLP`` model.

    Parameters
    ----------
    column_idx: Dict
        Dict containing the index of the columns that will be passed through
        the ``TabMlp`` model. Required to slice the tensors. e.g. {'education':
        0, 'relationship': 1, 'workclass': 2, ...}
    cat_embed_input: List, Optional, default = None
        List of Tuples with the column name, number of unique values and
        embedding dimension. e.g. [(education, 11, 32), ...]
    cat_embed_dropout: float, default = 0.1
        embeddings dropout
    full_embed_dropout: bool, default = False
        Boolean indicating if an entire embedding (i.e. the representation of
        one column) will be dropped in the batch. See:
        :obj:`pytorch_widedeep.models.embeddings_layers.FullEmbeddingDropout`.
        If ``full_embed_dropout = True``, ``cat_embed_dropout`` is ignored.
    shared_embed: bool, default = False
        The of sharing part of the embeddings per column is to enable the
        model to distinguish the classes in one column from those in the
        other columns. In other words, the idea is to let the model learn
        which column is embedded at the time.
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
    embed_continuous_activation: str, default = None
        String indicating the activation function to be applied to the
        continuous embeddings, if any. ``tanh``, ``relu``, ``leaky_relu`` and
        ``gelu`` are supported.
    cont_embed_dropout: float, default = 0.1,
        Dropout for the continuous embeddings
    cont_embed_activation: Optional, str, default = None,
        Activation function for the continuous embeddings
    cont_norm_layer: str, default =  "batchnorm"
        Type of normalization layer applied to the continuous features. Options
        are: 'layernorm', 'batchnorm' or None.
    input_dim: int, default = 32
        The so-called *dimension of the model*. In general is the number of
        embeddings used to encode the categorical and/or continuous columns
    attn_dropout: float, default = 0.2
        Dropout for each attention block
    with_addnorm: bool = False,
        Boolean indicating if residual connections will be used in the attention blocks
    attn_activation: str, default = "leaky_relu"
        String indicating the activation function to be applied to the dense
        layer in each attention encoder. ``tanh``, ``relu``, ``leaky_relu``
        and ``gelu`` are supported.
    n_blocks: int, default = 3
        Number of attention block

    Attributes
    ----------
    cat_and_cont_embed: ``nn.Module``
        This is the module that processes the categorical and continuous columns
    attention_blks: ``nn.Sequential``
        Sequence of attention encoders.
    output_dim: int
        The output dimension of the model. This is a required attribute
        neccesary to build the WideDeep class

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
        full_embed_dropout: bool = False,
        shared_embed: bool = False,
        add_shared_embed: bool = False,
        frac_shared_embed: float = 0.25,
        continuous_cols: Optional[List[str]] = None,
        embed_continuous_activation: str = None,
        cont_embed_dropout: float = 0.0,
        cont_embed_activation: str = None,
        cont_norm_layer: str = None,
        input_dim: int = 32,
        attn_dropout: float = 0.2,
        with_addnorm: bool = False,
        attn_activation: str = "leaky_relu",
        n_blocks: int = 3,
    ):
        super(ContextAttentionMLP, self).__init__()

        self.column_idx = column_idx
        self.cat_embed_input = cat_embed_input
        self.cat_embed_dropout = cat_embed_dropout
        self.full_embed_dropout = full_embed_dropout
        self.shared_embed = shared_embed
        self.add_shared_embed = add_shared_embed
        self.frac_shared_embed = frac_shared_embed

        self.continuous_cols = continuous_cols
        self.embed_continuous_activation = embed_continuous_activation
        self.cont_embed_dropout = cont_embed_dropout
        self.cont_embed_activation = cont_embed_activation
        self.cont_norm_layer = cont_norm_layer

        self.input_dim = input_dim
        self.attn_dropout = attn_dropout
        self.with_addnorm = with_addnorm
        self.attn_activation = attn_activation
        self.n_blocks = n_blocks

        self._check_activations()

        self.with_cls_token = "cls_token" in column_idx
        self.n_cat = len(cat_embed_input) if cat_embed_input is not None else 0
        self.n_cont = len(continuous_cols) if continuous_cols is not None else 0

        self.cat_and_cont_embed = SameSizeCatAndContEmbeddings(
            input_dim,
            column_idx,
            cat_embed_input,
            cat_embed_dropout,
            full_embed_dropout,
            shared_embed,
            add_shared_embed,
            frac_shared_embed,
            False,  # use_embed_bias
            continuous_cols,
            True,  # embed_continuous,
            cont_embed_dropout,
            embed_continuous_activation,
            True,  # use_cont_bias
            cont_norm_layer,
        )

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

        self.output_dim = (
            input_dim
            if self.with_cls_token
            else ((self.n_cat + self.n_cont) * input_dim)
        )

    def forward(self, X: Tensor) -> Tensor:

        x_cat, x_cont = self.cat_and_cont_embed(X)

        if x_cat is not None:
            x = x_cat
        if x_cont is not None:
            x = torch.cat([x, x_cont], 1) if x_cat is not None else x_cont

        x = self.attention_blks(x)

        if self.with_cls_token:
            out = x[:, 0, :]
        else:
            out = x.flatten(1)

        return out

    @property
    def attention_weights(self) -> List:
        r"""List with the attention weights

        The shape of the attention weights is:

        :math:`(N, F)`

        Where *N* is the batch size and *F* is the number of features/columns
        in the dataset
        """
        return [blk.attn.attn_weights for blk in self.attention_blks]

    def _check_activations(self):

        allowed_activations = [
            "relu",
            "leaky_relu",
            "tanh",
            "gelu",
        ]

        allowed = []
        allowed.append(
            self.embed_continuous_activation in allowed_activations
            if self.embed_continuous_activation is not None
            else True
        )
        allowed.append(self.attn_activation in allowed_activations)

        all_allowed = all(allowed)

        if not all_allowed:
            raise ValueError(
                "Currently, only the following activation functions are supported for "
                "the AttentiveTabMlp: {}.".format(", ".join(allowed_activations))
            )
