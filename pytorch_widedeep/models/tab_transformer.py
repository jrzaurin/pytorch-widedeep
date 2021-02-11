"""
The code in this module is inspired by a number of implementations:

Classes PositionwiseFF and AddNorm are 'stolen' with much gratitude from the fantastic d2l.ai book:
https://d2l.ai/chapter_attention-mechanisms/transformer.html

MultiHeadedAttention is inspired by the TabTransformer implementation here:
https://github.com/lucidrains/tab-transformer-pytorch. General comment: just go and have a look to
https://github.com/lucidrains

The fixed attention implementation and SharedEmbeddings are inspired by the
TabTransformer available in AutoGluon:
https://github.com/awslabs/autogluon/tree/master/tabular/src/autogluon/tabular/models/tab_transformer
If you have not checked that library, you should.
"""

import math

import torch
import einops
from torch import nn, einsum

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models.tab_mlp import MLP, _get_activation_fn


class PositionwiseFF(nn.Module):
    def __init__(
        self,
        input_dim: int,
        ff_hidden_dim: int,
        dropout: float,
        activation: str,
    ):
        super(PositionwiseFF, self).__init__()
        self.w_1 = nn.Linear(input_dim, ff_hidden_dim)
        self.w_2 = nn.Linear(ff_hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, X: Tensor) -> Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(X))))


class AddNorm(nn.Module):
    def __init__(self, input_dim: int, dropout: float):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(input_dim)

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        return self.ln(self.dropout(Y) + X)


class MultiHeadedAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        keep_attn_weights: bool,
        dropout: float,
        fixed_attention: bool,
        num_cat_columns: int,
    ):
        super(MultiHeadedAttention, self).__init__()

        assert (
            input_dim % num_heads == 0
        ), "'input_dim' must be divisible by 'num_heads'"
        if fixed_attention and not num_cat_columns:
            raise ValueError(
                "if 'fixed_attention' is 'True' the number of categorical "
                "columns 'num_cat_columns' must be specified"
            )
        # Consistent with other implementations I assume d_v = d_k
        self.d_k = input_dim // num_heads
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.fixed_attention = fixed_attention
        if fixed_attention:
            self.inp_proj = nn.Linear(input_dim, input_dim)
            self.fixed_key = nn.init.xavier_normal_(
                nn.Parameter(torch.empty(num_cat_columns, input_dim))
            )
            self.fixed_query = nn.init.xavier_normal_(
                nn.Parameter(torch.empty(num_cat_columns, input_dim))
            )
        else:
            self.inp_proj = nn.Linear(input_dim, input_dim * 3)
        self.out_proj = nn.Linear(input_dim, input_dim)
        self.keep_attn_weights = keep_attn_weights

    def forward(self, X: Tensor) -> Tensor:
        # b: batch size, s: src seq length (num of categorical features
        # encoded as embeddings), l: target sequence (l = s), e: embeddings
        # dimensions, h: number of attention heads, d: d_k
        if self.fixed_attention:
            v = self.inp_proj(X)
            k = einops.repeat(
                self.fixed_key.unsqueeze(0), "b s e -> (b copy) s e", copy=X.shape[0]
            )
            q = einops.repeat(
                self.fixed_query.unsqueeze(0), "b s e -> (b copy) s e", copy=X.shape[0]
            )
        else:
            q, k, v = self.inp_proj(X).chunk(3, dim=2)
        q, k, v = map(
            lambda t: einops.rearrange(t, "b s (h d) -> b h s d", h=self.num_heads),
            (q, k, v),
        )
        scores = einsum("b h s d, b h l d -> b h s l", q, k) / math.sqrt(self.d_k)
        attn_weights = self.dropout(scores.softmax(dim=-1))
        if self.keep_attn_weights:
            self.attn_weights = attn_weights
        attn_output = einsum("b h s l, b h l d -> b h s d", attn_weights, v)
        output = einops.rearrange(attn_output, "b h s d -> b s (h d)", h=self.num_heads)

        return self.out_proj(output)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        keep_attn_weights: bool,
        ff_hidden_dim: int,
        dropout: float,
        activation: str,
        fixed_attention,
        num_cat_columns,
    ):
        super(TransformerEncoder, self).__init__()
        self.self_attn = MultiHeadedAttention(
            input_dim,
            num_heads,
            keep_attn_weights,
            dropout,
            fixed_attention,
            num_cat_columns,
        )
        self.feed_forward = PositionwiseFF(
            input_dim, ff_hidden_dim, dropout, activation
        )
        self.attn_addnorm = AddNorm(input_dim, dropout)
        self.ff_addnorm = AddNorm(input_dim, dropout)

    def forward(self, X: Tensor) -> Tensor:
        Y = self.attn_addnorm(X, self.self_attn(X))
        return self.ff_addnorm(Y, self.feed_forward(Y))


class FullEmbeddingDropout(nn.Module):
    def __init__(self, dropout: float):
        super(FullEmbeddingDropout, self).__init__()
        self.dropout = dropout

    def forward(self, X: Tensor) -> Tensor:
        mask = X.new().resize_((X.size(1), 1)).bernoulli_(1 - self.dropout).expand_as(
            X
        ) / (1 - self.dropout)
        return mask * X


class SharedEmbeddings(nn.Module):
    def __init__(
        self,
        num_embed: int,
        embed_dim: int,
        embed_dropout: float,
        full_embed_dropout: bool = False,
        add_shared_embed: bool = False,
        frac_shared_embed=8,
    ):
        super(SharedEmbeddings, self).__init__()
        assert (
            embed_dim % frac_shared_embed == 0
        ), "'embed_dim' must be divisible by 'frac_shared_embed'"

        self.add_shared_embed = add_shared_embed
        self.embed = nn.Embedding(num_embed, embed_dim, padding_idx=0)
        self.embed.weight.data.clamp_(-2, 2)
        if add_shared_embed:
            col_embed_dim = embed_dim
        else:
            col_embed_dim = embed_dim // frac_shared_embed
        self.shared_embed = nn.Parameter(torch.empty(1, col_embed_dim).uniform_(-1, 1))

        if full_embed_dropout:
            self.dropout = FullEmbeddingDropout(embed_dropout)
        else:
            self.dropout = nn.Dropout(embed_dropout)  # type: ignore[assignment]

    def forward(self, X: Tensor) -> Tensor:
        out = self.dropout(self.embed(X))
        shared_embed = self.shared_embed.expand(out.shape[0], -1)
        if self.add_shared_embed:
            out += shared_embed
        else:
            out[:, : shared_embed.shape[1]] = shared_embed
        return out


class TabTransformer(nn.Module):
    def __init__(
        self,
        column_idx: Dict[str, int],
        embed_input: List[Tuple[str, int]],
        continuous_cols: Optional[List[str]] = None,
        embed_dropout: float = 0.1,
        full_embed_dropout: bool = False,
        shared_embed: bool = False,
        add_shared_embed: bool = False,
        frac_shared_embed: int = 8,
        input_dim: int = 32,
        num_heads: int = 8,
        num_blocks: int = 6,
        dropout: float = 0.1,
        keep_attn_weights: bool = False,
        fixed_attention: bool = False,
        num_cat_columns: Optional[int] = None,
        ff_hidden_dim: int = 32 * 4,
        transformer_activation: str = "gelu",
        mlp_hidden_dims: Optional[List[int]] = None,
        mlp_activation: str = "relu",
        mlp_batchnorm: bool = False,
        mlp_batchnorm_last: bool = False,
        mlp_linear_first: bool = True,
    ):

        r"""TabTransformer model (https://arxiv.org/pdf/2012.06678.pdf) model that
        can be used as the deeptabular component of a Wide & Deep model.


        Parameters
        ----------
        column_idx: Dict
            Dict containing the index of the columns that will be passed through
            the DeepDense model. Required to slice the tensors. e.g. {'education':
            0, 'relationship': 1, 'workclass': 2, ...}
        embed_input: List
            List of Tuples with the column name and number of unique values
            e.g. [(education, 11, 32), ...]
        continuous_cols: List, Optional, default = None
            List with the name of the numeric (aka continuous) columns
        embed_dropout: float, default = 0.1
            Dropout to be applied to the embeddings matrix
        full_embed_dropout: bool, default = False
            Boolean indicating if an entire embedding (i.e. the representation
            for one categorical column) will be dropped in the batch. See:
            ``pytorch_widedeep.model.tab_transformer.FullEmbeddingDropout``.
            If ``full_embed_dropout = True``, ``embed_dropout`` is ignored.
        shared_embed: bool, default = False
            The idea behind ``shared_embed`` is described in the Appendix A in the paper:
            `'The goal of having column embedding is to enable the model to distinguish the
            classes in one column from those in the other columns'`. In other words, the idea
            is to let the model learn which column is embedding at the time.
        add_shared_embed: bool, default = False,
            The two embedding sharing strategies are: 1) add the shared embeddings to the column
            embeddings or 2) to replace the first ``frac_shared_embed`` with the shared
            embeddings. See ``pytorch_widedeep.models.tab_transformer.SharedEmbeddings``
        frac_shared_embed: int, default = 8
            The fraction of embeddings that will be shared by all the different categories for
            one particular column.
        input_dim: int, default = 32
            The so-called *dimension of the model*. Is the number of embeddings used to encode
            the categorical columns
        num_heads: int, default = 8
            Number of attention heads per Transformer block
        num_blocks: int, default = 6
            Number of Transformer blocks
        dropout: float, default = 0.1
            Dropout that will be applied internally to the
            ``TransformerEncoder`` (see
            ``pytorch_widedeep.model.tab_transformer.TransformerEncoder``) and the
            output MLP
        keep_attn_weights: bool, default = False
            If set to ``True`` the model will store the attention weights in the ``attention_weights``
            attribute.
        fixed_attention: bool, default = False
            If set to ``True`` all the observations in a batch will have the
            same Key and Query. This implementation is inspired by the one
            available at the `Autogluon tabular library
            <https://github.com/awslabs/autogluon/blob/master/tabular/src/autogluon/tabular/models/tab_transformer/modified_transformer.py>`_.
        num_cat_columns: int, Optional, default = None
            If `fixed_attention` is set to ``True`` the number of categorical
            columns that will be encoded as embeddings must be specified
        ff_hidden_dim: int, default = 128
            Hidden dimension of the ``FeedForward`` Layer. See
            ``pytorch_widedeep.model.tab_transformer.FeedForward``.
        transformer_activation: str, default = "gelu"
            Transformer Encoder activation function
        mlp_hidden_dims: List, Optional, default = None
            MLP hidden dimensions. If not provided it will default to ``[4*l,
            2*l]`` where ``l`` is the mlp input dimension
        mlp_activation: str, default = "gelu"
            MLP activation function
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
        embed_layers: ``nn.ModuleDict``
            Dict with the embeddings per column
        blks: ``nn.Sequential``
            Sequence of Transformer blocks
        attention_weights: List
            List with the attention weights per block
        mlp: ``nn.Module``
            MLP component in the TabTransformer model
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
        super(TabTransformer, self).__init__()

        self.column_idx = column_idx
        self.embed_input = embed_input
        self.continuous_cols = continuous_cols
        self.embed_dropout = embed_dropout
        self.full_embed_dropout = full_embed_dropout
        self.shared_embed = shared_embed
        self.add_shared_embed = add_shared_embed
        self.frac_shared_embed = frac_shared_embed
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.keep_attn_weights = keep_attn_weights
        self.fixed_attention = fixed_attention
        self.num_cat_columns = num_cat_columns
        self.ff_hidden_dim = ff_hidden_dim
        self.transformer_activation = transformer_activation
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.mlp_linear_first = mlp_linear_first

        # Embeddings: val + 1 because 0 is reserved for padding/unseen cateogories.
        if shared_embed:
            self.embed_layers = nn.ModuleDict(
                {
                    "emb_layer_"
                    + col: SharedEmbeddings(
                        val + 1,
                        input_dim,
                        embed_dropout,
                        full_embed_dropout,
                        add_shared_embed,
                        frac_shared_embed,
                    )
                    for col, val in self.embed_input
                }
            )
        else:
            self.embed_layers = nn.ModuleDict(
                {
                    "emb_layer_" + col: nn.Embedding(val + 1, input_dim, padding_idx=0)
                    for col, val in self.embed_input
                }
            )
            if full_embed_dropout:
                self.embedding_dropout = FullEmbeddingDropout(embed_dropout)
            else:
                self.embedding_dropout = nn.Dropout(embed_dropout)  # type: ignore[assignment]
        # Continuous
        if self.continuous_cols is not None:
            cont_inp_dim = len(self.continuous_cols)
        else:
            cont_inp_dim = 0

        self.blks = nn.Sequential()
        for i in range(num_blocks):
            self.blks.add_module(
                "block" + str(i),
                TransformerEncoder(
                    input_dim,
                    num_heads,
                    keep_attn_weights,
                    ff_hidden_dim,
                    dropout,
                    transformer_activation,
                    fixed_attention,
                    num_cat_columns,
                ),
            )

        if keep_attn_weights:
            self.attention_weights = [None] * num_blocks

        if not mlp_hidden_dims:
            mlp_inp_l = len(embed_input) * input_dim + cont_inp_dim
            mlp_hidden_dims = [mlp_inp_l, mlp_inp_l * 4, mlp_inp_l * 2]

        self.tab_transformer_mlp = MLP(
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

        embed = [
            self.embed_layers["emb_layer_" + col](
                X[:, self.column_idx[col]].long()
            ).unsqueeze(1)
            for col, _ in self.embed_input
        ]
        x = torch.cat(embed, 1)
        if not self.shared_embed and self.embedding_dropout is not None:
            x = self.embedding_dropout(x)

        for i, blk in enumerate(self.blks):
            x = blk(x)
            if self.keep_attn_weights:
                self.attention_weights[i] = blk.self_attn.attn_weights
        x = x.flatten(1)

        if self.continuous_cols is not None:
            cont_idx = [self.column_idx[col] for col in self.continuous_cols]
            x_cont = X[:, cont_idx].float()
            x = torch.cat([x, x_cont], 1)

        return self.tab_transformer_mlp(x)
