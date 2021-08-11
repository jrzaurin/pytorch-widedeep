import torch
from torch import nn

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models.tab_mlp import MLP
from pytorch_widedeep.models.transformers.layers import (
    SharedEmbeddings,
    TransformerEncoder,
    ContinuousEmbeddings,
    FullEmbeddingDropout,
)


class TabTransformer(nn.Module):
    r"""Adaptation of TabTransformer model
    (https://arxiv.org/abs/2012.06678) model that can be used as the
    deeptabular component of a Wide & Deep model.

    Parameters
    ----------
    column_idx: Dict
        Dict containing the index of the columns that will be passed through
        the DeepDense model. Required to slice the tensors. e.g. {'education':
        0, 'relationship': 1, 'workclass': 2, ...}
    embed_input: List
        List of Tuples with the column name and number of unique values
        e.g. [(education, 11), ...]
    embed_dropout: float, default = 0.1
        Dropout to be applied to the embeddings matrix
    full_embed_dropout: bool, default = False
        Boolean indicating if an entire embedding (i.e. the representation of
        one column) will be dropped in the batch. See:
        :obj:`pytorch_widedeep.models.transformers.layers.FullEmbeddingDropout`.
        If ``full_embed_dropout = True``, ``embed_dropout`` is ignored.
    shared_embed: bool, default = False
        The idea behind ``shared_embed`` is described in the Appendix A in the paper:
        `'The goal of having column embedding is to enable the model to distinguish the
        classes in one column from those in the other columns'`. In other words, the idea
        is to let the model learn which column is embedding at the time.
    add_shared_embed: bool, default = False,
        The two embedding sharing strategies are: 1) add the shared embeddings to the column
        embeddings or 2) to replace the first ``frac_shared_embed`` with the shared
        embeddings. See :obj:`pytorch_widedeep.models.transformers.layers.SharedEmbeddings`
    frac_shared_embed: float, default = 0.25
        The fraction of embeddings that will be shared by all the different categories for
        one particular column.
    continuous_cols: List, Optional, default = None
        List with the name of the numeric (aka continuous) columns
    embed_continuous: bool, default = False,
        Boolean indicating if the continuous features will be "embedded". See
        ``pytorch_widedeep.models.transformers.layers.ContinuousEmbeddings``
        Note that setting this to true is equivalent to the so called
        `FT-Transformer <https://arxiv.org/abs/2106.11959>`_
        (Feature Tokenizer + Transformer). The only difference is that this
        implementation does not consider using bias for the categorical
        embeddings.
    embed_continuous_activation: str, default = None
        String indicating the activation function to be applied to the
        continuous embeddings, if any.
        'relu', 'leaky_relu' and 'gelu' are supported.
    cont_norm_layer: str, default =  None,
        Type of normalization layer applied to the continuous features. Options
        are: 'layernorm', 'batchnorm' or None.
    input_dim: int, default = 32
        The so-called *dimension of the model*. Is the number of embeddings used to encode
        the categorical columns
    n_heads: int, default = 8
        Number of attention heads per Transformer block
    n_blocks: int, default = 6
        Number of Transformer blocks
    dropout: float, default = 0.1
        Dropout that will be applied internally to the ``TransformerEncoder``
        (see :obj:`pytorch_widedeep.models.transformers.layers.TransformerEncoder`)
        and the output MLP
    keep_attn_weights: bool, default = False
        If set to ``True`` the model will store the attention weights in the ``attention_weights``
        attribute.
    ff_hidden_dim: int, default = 128
        Hidden dimension of the ``FeedForward`` Layer. See
        :obj:`pytorch_widedeep.models.transformers.layers.FeedForward`.
    transformer_activation: str, default = "gelu"
        Transformer Encoder activation function. 'relu', 'leaky_relu', 'gelu'
        and 'geglu' are supported
    mlp_hidden_dims: List, Optional, default = None
        MLP hidden dimensions. If not provided it will default to ``[4*l,
        2*l]`` where ``l`` is the mlp input dimension
    mlp_activation: str, default = "relu"
        MLP activation function. 'relu', 'leaky_relu' and 'gelu' are supported
    mlp_batchnorm: bool, default = False
        Boolean indicating whether or not to apply batch normalization to the
        dense layers
    mlp_batchnorm_last: bool, default = False
        Boolean indicating whether or not to apply batch normalization to the
        last of the dense layers
    mlp_linear_first: bool, default = False
        Boolean indicating whether the order of the operations in the dense
        layer. If ``True: [LIN -> ACT -> BN -> DP]``. If ``False: [BN -> DP ->
        LIN -> ACT]``

    Attributes
    ----------
    cat_embed_layers: ``nn.ModuleDict``
        Dict with the embeddings per column
    cont_embed: ``nn.Module``
        Continuous embeddings layer if ``embed_continuous=True``. See
        ``pytorch_widedeep.models.transformers.layers.ContinuousEmbeddings``
    cont_norm: ``nn.Module``
        continuous normalization layer
    transformer_blks: ``nn.Sequential``
        Sequence of Transformer blocks
    attention_weights: List
        List with the attention weights per block if ``keep_attn_weights = True``.
    transformer_mlp: ``nn.Module``
        MLP component in the model
    output_dim: int
        The output dimension of the model. This is a required attribute
        neccesary to build the WideDeep class

    Example
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import TabTransformer
    >>> X_tab = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
    >>> colnames = ['a', 'b', 'c', 'd', 'e']
    >>> embed_input = [(u,i) for u,i in zip(colnames[:4], [4]*4)]
    >>> continuous_cols = ['e']
    >>> column_idx = {k:v for v,k in enumerate(colnames)}
    >>> model = TabTransformer(column_idx=column_idx, embed_input=embed_input, continuous_cols=continuous_cols)
    >>> out = model(X_tab)
    """

    def __init__(
        self,
        column_idx: Dict[str, int],
        embed_input: List[Tuple[str, int]],
        embed_dropout: float = 0.1,
        full_embed_dropout: bool = False,
        shared_embed: bool = False,
        add_shared_embed: bool = False,
        frac_shared_embed: float = 0.25,
        continuous_cols: Optional[List[str]] = None,
        embed_continuous: bool = False,
        embed_continuous_activation: str = None,
        cont_norm_layer: str = None,
        input_dim: int = 32,
        n_heads: int = 8,
        n_blocks: int = 6,
        dropout: float = 0.1,
        keep_attn_weights: bool = False,
        ff_hidden_dim: int = 32 * 4,
        transformer_activation: str = "gelu",
        mlp_hidden_dims: Optional[List[int]] = None,
        mlp_activation: str = "relu",
        mlp_batchnorm: bool = False,
        mlp_batchnorm_last: bool = False,
        mlp_linear_first: bool = True,
    ):
        super(TabTransformer, self).__init__()

        self.column_idx = column_idx
        self.embed_input = embed_input
        self.embed_dropout = embed_dropout
        self.full_embed_dropout = full_embed_dropout
        self.shared_embed = shared_embed
        self.add_shared_embed = add_shared_embed
        self.frac_shared_embed = frac_shared_embed
        self.continuous_cols = continuous_cols
        self.embed_continuous = embed_continuous
        self.embed_continuous_activation = embed_continuous_activation
        self.cont_norm_layer = cont_norm_layer
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.keep_attn_weights = keep_attn_weights
        self.ff_hidden_dim = ff_hidden_dim
        self.transformer_activation = transformer_activation
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.mlp_linear_first = mlp_linear_first

        self.with_cls_token = "cls_token" in self.column_idx
        self.categorical_cols = [ei[0] for ei in self.embed_input]
        self.n_tokens = sum([ei[1] for ei in self.embed_input])

        self._set_categ_embeddings()

        self._set_cont_cols()

        self.transformer_blks = nn.Sequential()
        for i in range(n_blocks):
            self.transformer_blks.add_module(
                "block" + str(i),
                TransformerEncoder(
                    input_dim,
                    n_heads,
                    keep_attn_weights,
                    ff_hidden_dim,
                    dropout,
                    transformer_activation,
                ),
            )
        if keep_attn_weights:
            self.attention_weights: List[Any] = [None] * n_blocks

        if not mlp_hidden_dims:
            mlp_hidden_dims = self._set_mlp_hidden_dims()
        self.transformer_mlp = MLP(
            mlp_hidden_dims,
            mlp_activation,
            dropout,
            mlp_batchnorm,
            mlp_batchnorm_last,
            mlp_linear_first,
        )

        # the output_dim attribute will be used as input_dim when "merging" the models
        self.output_dim = mlp_hidden_dims[-1]

    def forward(self, X: Tensor) -> Tensor:

        if self.shared_embed:
            x_cat_embed = [
                self.cat_embed["emb_layer_" + col](
                    X[:, self.column_idx[col]].long()
                ).unsqueeze(1)
                for col, _ in self.embed_input
            ]
            x = torch.cat(x_cat_embed, 1)
        else:
            x = self.cat_embed(X[:, self.cat_idx].long())

        if not self.shared_embed and self.embedding_dropout is not None:
            x = self.embedding_dropout(x)

        if self.continuous_cols is not None and self.embed_continuous:
            x_cont = self.cont_norm((X[:, self.cont_idx].float()))
            x_cont_embed = self.cont_embed(x_cont)
            x = torch.cat([x, x_cont_embed], 1)

        for i, blk in enumerate(self.transformer_blks):
            x = blk(x)
            if self.keep_attn_weights:
                if hasattr(blk, "row_attn"):
                    self.attention_weights[i] = (
                        blk.self_attn.attn_weights,
                        blk.row_attn.attn_weights,
                    )
                else:
                    self.attention_weights[i] = blk.self_attn.attn_weights

        if self.with_cls_token:
            x = x[:, 0, :]
        else:
            x = x.flatten(1)

        if self.continuous_cols is not None and not self.embed_continuous:
            x_cont = self.cont_norm((X[:, self.cont_idx].float()))
            x = torch.cat([x, x_cont], 1)

        return self.transformer_mlp(x)

    def _set_categ_embeddings(self):
        self.cat_idx = [self.column_idx[col] for col in self.categorical_cols]
        # Categorical: val + 1 because 0 is reserved for padding/unseen cateogories.
        if self.shared_embed:
            self.cat_embed = nn.ModuleDict(
                {
                    "emb_layer_"
                    + col: SharedEmbeddings(
                        val if col == "cls_token" else val + 1,
                        self.input_dim,
                        self.embed_dropout,
                        self.full_embed_dropout,
                        self.add_shared_embed,
                        self.frac_shared_embed,
                    )
                    for col, val in self.embed_input
                }
            )
        else:
            self.cat_embed = nn.Embedding(
                self.n_tokens + 1, self.input_dim, padding_idx=0
            )
            if self.full_embed_dropout:
                self.embedding_dropout: DropoutLayers = FullEmbeddingDropout(
                    self.embed_dropout
                )
            else:
                self.embedding_dropout = nn.Dropout(self.embed_dropout)

    def _set_cont_cols(self):
        if self.continuous_cols is not None:
            self.cont_idx = [self.column_idx[col] for col in self.continuous_cols]
            if self.cont_norm_layer == "layernorm":
                self.cont_norm: NormLayers = nn.LayerNorm(len(self.continuous_cols))
            elif self.cont_norm_layer == "batchnorm":
                self.cont_norm = nn.BatchNorm1d(len(self.continuous_cols))
            else:
                self.cont_norm = nn.Identity()
            if self.embed_continuous:
                self.cont_embed = ContinuousEmbeddings(
                    len(self.continuous_cols),
                    self.input_dim,
                    self.embed_continuous_activation,
                )

    def _set_mlp_hidden_dims(self) -> List[int]:
        if self.continuous_cols is not None:
            if self.with_cls_token:
                if self.embed_continuous:
                    mlp_hidden_dims = [
                        self.input_dim,
                        self.input_dim * 4,
                        self.input_dim * 2,
                    ]
                else:
                    mlp_inp_l = self.input_dim + len(self.continuous_cols)
                    mlp_hidden_dims = [mlp_inp_l, mlp_inp_l * 4, mlp_inp_l * 2]
            elif self.embed_continuous:
                mlp_inp_l = (
                    len(self.embed_input) + len(self.continuous_cols)
                ) * self.input_dim
                mlp_hidden_dims = [mlp_inp_l, mlp_inp_l * 4, mlp_inp_l * 2]
            else:
                mlp_inp_l = len(self.embed_input) * self.input_dim + len(
                    self.continuous_cols
                )
                mlp_hidden_dims = [mlp_inp_l, mlp_inp_l * 4, mlp_inp_l * 2]
        else:
            mlp_inp_l = len(self.embed_input) * self.input_dim
            mlp_hidden_dims = [mlp_inp_l, mlp_inp_l * 4, mlp_inp_l * 2]
        return mlp_hidden_dims
