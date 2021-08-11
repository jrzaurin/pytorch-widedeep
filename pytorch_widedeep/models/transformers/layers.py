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
from pytorch_widedeep.models.tab_mlp import _get_activation_fn


class PositionwiseFF(nn.Module):
    def __init__(
        self,
        input_dim: int,
        ff_hidden_dim: int,
        dropout: float,
        activation: str,
    ):
        super(PositionwiseFF, self).__init__()
        self.w_1 = nn.Linear(
            input_dim, ff_hidden_dim * 2 if activation == "geglu" else ff_hidden_dim
        )
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
        n_heads: int,
        keep_attn_weights: bool,
        dropout: float,
    ):
        super(MultiHeadedAttention, self).__init__()

        assert input_dim % n_heads == 0, "'input_dim' must be divisible by 'n_heads'"
        # Consistent with other implementations I assume d_v = d_k
        self.d_k = input_dim // n_heads
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.inp_proj = nn.Linear(input_dim, input_dim * 3)
        self.out_proj = nn.Linear(input_dim, input_dim)
        self.keep_attn_weights = keep_attn_weights

    def forward(self, X: Tensor) -> Tensor:
        # b: batch size, s: src seq length (num of categorical features
        # encoded as embeddings), l: target sequence (l = s), e: embeddings
        # dimensions, h: number of attention heads, d: d_k
        q, k, v = self.inp_proj(X).chunk(3, dim=2)
        q, k, v = map(
            lambda t: einops.rearrange(t, "b s (h d) -> b h s d", h=self.n_heads),
            (q, k, v),
        )
        scores = einsum("b h s d, b h l d -> b h s l", q, k) / math.sqrt(self.d_k)
        attn_weights = self.dropout(scores.softmax(dim=-1))
        if self.keep_attn_weights:
            self.attn_weights = attn_weights
        attn_output = einsum("b h s l, b h l d -> b h s d", attn_weights, v)
        output = einops.rearrange(attn_output, "b h s d -> b s (h d)", h=self.n_heads)

        return self.out_proj(output)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        keep_attn_weights: bool,
        ff_hidden_dim: int,
        dropout: float,
        activation: str,
    ):
        super(TransformerEncoder, self).__init__()

        self.self_attn = MultiHeadedAttention(
            input_dim,
            n_heads,
            keep_attn_weights,
            dropout,
        )
        self.ff = PositionwiseFF(input_dim, ff_hidden_dim, dropout, activation)
        self.attn_addnorm = AddNorm(input_dim, dropout)
        self.ff_addnorm = AddNorm(input_dim, dropout)

    def forward(self, X: Tensor) -> Tensor:
        x = self.attn_addnorm(X, self.self_attn(X))
        return self.ff_addnorm(x, self.ff(x))


class SaintEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        keep_attn_weights: bool,
        ff_hidden_dim: int,
        dropout: float,
        activation: str,
        n_feat: int,
    ):
        super(SaintEncoder, self).__init__()

        self.n_feat = n_feat

        self.self_attn = MultiHeadedAttention(
            input_dim,
            n_heads,
            keep_attn_weights,
            dropout,
        )
        self.self_attn_ff = PositionwiseFF(
            input_dim, ff_hidden_dim, dropout, activation
        )
        self.self_attn_addnorm = AddNorm(input_dim, dropout)
        self.self_attn_ff_addnorm = AddNorm(input_dim, dropout)

        self.row_attn = MultiHeadedAttention(
            n_feat * input_dim,
            n_heads,
            keep_attn_weights,
            dropout,
        )
        self.row_attn_ff = PositionwiseFF(
            n_feat * input_dim, n_feat * ff_hidden_dim, dropout, activation
        )
        self.row_attn_addnorm = AddNorm(n_feat * input_dim, dropout)
        self.row_attn_ff_addnorm = AddNorm(n_feat * input_dim, dropout)

    def forward(self, X: Tensor) -> Tensor:
        x = self.self_attn_addnorm(X, self.self_attn(X))
        x = self.self_attn_ff_addnorm(x, self.self_attn_ff(x))
        x = einops.rearrange(x, "b n d -> 1 b (n d)")
        x = self.row_attn_addnorm(x, self.row_attn(x))
        x = self.row_attn_ff_addnorm(x, self.row_attn_ff(x))
        x = einops.rearrange(x, "1 b (n d) -> b n d", n=self.n_feat)
        return x


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

        self.act_fn = _get_activation_fn(activation) if activation else None

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
