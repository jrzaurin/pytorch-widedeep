"""
The code in this module is inspired by a number of implementations:

Classes PositionwiseFF and AddNorm are 'stolen' with much gratitude from the fantastic d2l.ai book:
https://d2l.ai/chapter_attention-mechanisms/transformer.html

MultiHeadedAttention is inspired by the TabTransformer implementation here:
https://github.com/lucidrains/tab-transformer-pytorch. General comment: just go and have a look to
https://github.com/lucidrains

SharedEmbeddings is inspired by the TabTransformer available in AutoGluon:
https://github.com/awslabs/autogluon/tree/master/tabular/src/autogluon/tabular/models/tab_transformer
If you have not checked that library, you should.

The forward pass of the SaintEncoder is based on the original code release:
https://github.com/somepago/saint
"""

import math

import torch
import einops
from torch import nn, einsum

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models.tab_mlp import get_activation_fn


class PositionwiseFF(nn.Module):
    def __init__(
        self,
        input_dim: int,
        dropout: float,
        activation: str,
        mult: int = 4,
    ):
        super(PositionwiseFF, self).__init__()
        ff_hidden_dim = input_dim * mult
        self.w_1 = nn.Linear(
            input_dim, ff_hidden_dim * 2 if activation == "geglu" else ff_hidden_dim
        )
        self.w_2 = nn.Linear(ff_hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)

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
        n_heads: int,
        use_bias: bool,
        dropout: float,
        query_dim: Optional[int] = None,
    ):
        super(MultiHeadedAttention, self).__init__()

        assert input_dim % n_heads == 0, "'input_dim' must be divisible by 'n_heads'"

        self.head_dim = input_dim // n_heads
        self.n_heads = n_heads

        self.dropout = nn.Dropout(dropout)

        query_dim = query_dim if query_dim is not None else input_dim
        self.q_proj = nn.Linear(query_dim, input_dim, bias=use_bias)
        self.kv_proj = nn.Linear(input_dim, input_dim * 2, bias=use_bias)
        self.out_proj = nn.Linear(input_dim, query_dim, bias=use_bias)

    def forward(self, X: Tensor, Y: Optional[Tensor] = None) -> Tensor:
        # b: batch size
        # s: seq length
        # l: target sequence length
        # h: number of attention heads,
        # d: head_dim
        q = self.q_proj(X)
        Y = Y if Y is not None else X
        k, v = self.kv_proj(Y).chunk(2, dim=-1)
        q, k, v = map(
            lambda t: einops.rearrange(t, "b s (h d) -> b h s d", h=self.n_heads),
            (q, k, v),
        )
        scores = einsum("b h s d, b h l d -> b h s l", q, k) / math.sqrt(self.head_dim)
        attn_weights = scores.softmax(dim=-1)
        self.attn_weights = attn_weights
        attn_output = einsum("b h s l, b h l d -> b h s d", attn_weights, v)
        output = einops.rearrange(attn_output, "b h s d -> b s (h d)", h=self.n_heads)

        return self.dropout(self.out_proj(output))


class AdditiveAttention(nn.Module):
    r"""To be honest this is a convoluted FF network with a residual
    connection...I am not sure this can be called Attention, or added to the
    transformer family. However, the fact that is a weird residual MLP might
    make it work for tabular data.
    """

    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        use_bias: bool,
        dropout: float,
        share_qv_weights: bool,
    ):
        super(AdditiveAttention, self).__init__()

        assert input_dim % n_heads == 0, "'input_dim' must be divisible by 'n_heads'"

        self.head_dim = input_dim // n_heads
        self.n_heads = n_heads
        self.share_qv_weights = share_qv_weights

        self.dropout = nn.Dropout(dropout)

        # In the paper: " [...] we share the value and query transformation
        # parameters to reduce the memory cost [...]"
        if share_qv_weights:
            self.qv_proj = nn.Linear(input_dim, input_dim, bias=use_bias)
        else:
            self.q_proj = nn.Linear(input_dim, input_dim, bias=use_bias)
            self.v_proj = nn.Linear(input_dim, input_dim, bias=use_bias)
        self.k_proj = nn.Linear(input_dim, input_dim, bias=use_bias)

        self.global_query_logits = nn.Linear(self.head_dim, 1, bias=use_bias)
        self.global_key_logits = nn.Linear(self.head_dim, 1, bias=use_bias)

        self.r_out = nn.Linear(self.head_dim, self.head_dim)

    def forward(self, X: Tensor) -> Tensor:

        q = self.qv_proj(X) if self.share_qv_weights else self.q_proj(X)
        v = self.qv_proj(X) if self.share_qv_weights else self.v_proj(X)
        k = self.k_proj(X)

        q, k, v = map(
            lambda t: einops.rearrange(t, "b s (h d) -> b h s d", h=self.n_heads),
            (q, k, v),
        )

        alphas = (
            einops.rearrange(self.global_query_logits(q), "b h s () -> b h s")
            / math.sqrt(self.head_dim)
        ).softmax(dim=-1)
        global_query = einsum("b h s, b h s d -> b h d", alphas, q)
        global_query = einops.rearrange(global_query, "b h d -> b h () d")

        p = k * global_query

        betas = (
            einops.rearrange(self.global_key_logits(p), "b h s () -> b h s")
            / math.sqrt(self.head_dim)
        ).softmax(dim=-1)
        global_key = einsum("b h s, b h s d -> b h d", betas, p)
        global_key = einops.rearrange(global_key, "b h d -> b h () d")

        u = v * global_key

        self.attn_weights = (alphas, betas)

        # the "magical" residual connection
        output = q + self.dropout(self.r_out(u))

        return einops.rearrange(output, "b h s d -> b s (h d)", h=self.n_heads)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        use_bias: bool,
        dropout: float,
        activation: str,
        query_dim: Optional[int] = None,
    ):
        super(TransformerEncoder, self).__init__()

        self.attn = MultiHeadedAttention(
            input_dim,
            n_heads,
            use_bias,
            dropout,
            query_dim,
        )

        attn_dim_out = query_dim if query_dim is not None else input_dim
        self.ff = PositionwiseFF(attn_dim_out, dropout, activation)
        self.attn_addnorm = AddNorm(attn_dim_out, dropout)
        self.ff_addnorm = AddNorm(attn_dim_out, dropout)

    def forward(self, X: Tensor, Y: Optional[Tensor] = None) -> Tensor:
        x = self.attn_addnorm(X, self.attn(X, Y))
        return self.ff_addnorm(x, self.ff(x))


class SaintEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        use_bias: bool,
        dropout: float,
        activation: str,
        n_feat: int,
    ):
        super(SaintEncoder, self).__init__()

        self.n_feat = n_feat

        self.col_attn = MultiHeadedAttention(
            input_dim,
            n_heads,
            use_bias,
            dropout,
        )
        self.col_attn_ff = PositionwiseFF(input_dim, dropout, activation)
        self.col_attn_addnorm = AddNorm(input_dim, dropout)
        self.col_attn_ff_addnorm = AddNorm(input_dim, dropout)

        self.row_attn = MultiHeadedAttention(
            n_feat * input_dim,
            n_heads,
            use_bias,
            dropout,
        )
        self.row_attn_ff = PositionwiseFF(n_feat * input_dim, dropout, activation)
        self.row_attn_addnorm = AddNorm(n_feat * input_dim, dropout)
        self.row_attn_ff_addnorm = AddNorm(n_feat * input_dim, dropout)

    def forward(self, X: Tensor) -> Tensor:
        x = self.col_attn_addnorm(X, self.col_attn(X))
        x = self.col_attn_ff_addnorm(x, self.col_attn_ff(x))
        x = einops.rearrange(x, "b n d -> 1 b (n d)")
        x = self.row_attn_addnorm(x, self.row_attn(x))
        x = self.row_attn_ff_addnorm(x, self.row_attn_ff(x))
        x = einops.rearrange(x, "1 b (n d) -> b n d", n=self.n_feat)
        return x


class FastFormerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        use_bias: bool,
        dropout: float,
        share_qv_weights: bool,
        activation: str,
    ):
        super(FastFormerEncoder, self).__init__()

        self.attn = AdditiveAttention(
            input_dim,
            n_heads,
            use_bias,
            dropout,
            share_qv_weights,
        )

        self.ff = PositionwiseFF(input_dim, dropout, activation)
        self.attn_addnorm = AddNorm(input_dim, dropout)
        self.ff_addnorm = AddNorm(input_dim, dropout)

    def forward(self, X: Tensor) -> Tensor:
        x = self.attn_addnorm(X, self.attn(X))
        return self.ff_addnorm(x, self.ff(x))


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
        n_embed: int,
        embed_dim: int,
        embed_dropout: float,
        full_embed_dropout: bool = False,
        add_shared_embed: bool = False,
        frac_shared_embed=0.25,
    ):
        super(SharedEmbeddings, self).__init__()

        assert frac_shared_embed < 1, "'frac_shared_embed' must be less than 1"
        self.add_shared_embed = add_shared_embed
        self.embed = nn.Embedding(n_embed, embed_dim, padding_idx=0)
        self.embed.weight.data.clamp_(-2, 2)
        if add_shared_embed:
            col_embed_dim = embed_dim
        else:
            col_embed_dim = int(embed_dim * frac_shared_embed)
        self.shared_embed = nn.Parameter(torch.empty(1, col_embed_dim).uniform_(-1, 1))

        if full_embed_dropout:
            self.dropout: DropoutLayers = FullEmbeddingDropout(embed_dropout)
        else:
            self.dropout = nn.Dropout(embed_dropout)

    def forward(self, X: Tensor) -> Tensor:
        out = self.dropout(self.embed(X))
        shared_embed = self.shared_embed.expand(out.shape[0], -1)
        if self.add_shared_embed:
            out += shared_embed
        else:
            out[:, : shared_embed.shape[1]] = shared_embed
        return out


class ContinuousEmbeddings(nn.Module):
    def __init__(
        self,
        n_cont_cols: int,
        embed_dim: int,
        activation: str = None,
        bias: bool = True,
    ):
        super(ContinuousEmbeddings, self).__init__()
        self.n_cont_cols = n_cont_cols
        self.embed_dim = embed_dim
        self.activation = activation

        self.weight = nn.Parameter(torch.Tensor(n_cont_cols, embed_dim))
        self.bias = nn.Parameter(torch.Tensor(n_cont_cols, embed_dim)) if bias else None
        self._reset_parameters()

        self.act_fn = get_activation_fn(activation) if activation else None

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, X: Tensor) -> Tensor:
        x = self.weight.unsqueeze(0) * X.unsqueeze(2)

        if self.bias is not None:
            x = x + self.bias.unsqueeze(0)

        if self.act_fn is not None:
            x = self.act_fn(x)

        return x


class CatAndContEmbeddings(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        column_idx: Dict[str, int],
        embed_input: Optional[List[Tuple[str, int]]],
        embed_dropout: float,
        full_embed_dropout: bool,
        shared_embed: bool,
        add_shared_embed: bool,
        frac_shared_embed: float,
        continuous_cols: Optional[List[str]],
        embed_continuous: bool,
        embed_continuous_activation: str,
        cont_norm_layer: str,
    ):
        super(CatAndContEmbeddings, self).__init__()

        self.embed_dim = embed_dim
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

        # Categorical
        if self.embed_input is not None:
            self.categorical_cols = [ei[0] for ei in self.embed_input]
            self.n_tokens = sum([ei[1] for ei in self.embed_input])
            self.cat_idx = [self.column_idx[col] for col in self.categorical_cols]
            # Categorical: val + 1 because 0 is reserved for padding/unseen cateogories.
            if self.shared_embed:
                self.cat_embed: Union[nn.ModuleDict, nn.Embedding] = nn.ModuleDict(
                    {
                        "emb_layer_"
                        + col: SharedEmbeddings(
                            val if col == "cls_token" else val + 1,
                            self.embed_dim,
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
                    self.n_tokens + 1, self.embed_dim, padding_idx=0
                )
                if self.full_embed_dropout:
                    self.embedding_dropout: DropoutLayers = FullEmbeddingDropout(
                        self.embed_dropout
                    )
                else:
                    self.embedding_dropout = nn.Dropout(self.embed_dropout)

        # Continuous
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
                    self.embed_dim,
                    self.embed_continuous_activation,
                )

    def forward(self, X: Tensor) -> Tuple[Tensor, Any]:

        if self.embed_input is not None:
            if self.shared_embed:
                cat_embed = [
                    self.cat_embed["emb_layer_" + col](  # type: ignore[index]
                        X[:, self.column_idx[col]].long()
                    ).unsqueeze(1)
                    for col, _ in self.embed_input
                ]
                x_cat = torch.cat(cat_embed, 1)
            else:
                x_cat = self.cat_embed(X[:, self.cat_idx].long())

            if not self.shared_embed and self.embedding_dropout is not None:
                x_cat = self.embedding_dropout(x_cat)
        else:
            x_cat = None

        if self.continuous_cols is not None:
            x_cont = self.cont_norm((X[:, self.cont_idx].float()))
            if self.embed_continuous:
                x_cont = self.cont_embed(x_cont)
        else:
            x_cont = None

        return x_cat, x_cont
