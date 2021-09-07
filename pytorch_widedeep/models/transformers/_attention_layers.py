"""
MultiHeadedAttention is inspired by the implementation at
https://github.com/lucidrains
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
        mult: float = 4.0,
    ):
        super(PositionwiseFF, self).__init__()
        ff_hidden_dim = int(input_dim * mult)
        self.w_1 = nn.Linear(
            input_dim,
            ff_hidden_dim * 2 if activation.endswith("glu") else ff_hidden_dim,
        )
        self.w_2 = nn.Linear(ff_hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)

    def forward(self, X: Tensor) -> Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(X))))


class NormAdd(nn.Module):
    """aka PreNorm"""

    def __init__(self, input_dim: int, dropout: float):
        super(NormAdd, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(input_dim)

    def forward(self, X: Tensor, sublayer: nn.Module) -> Tensor:
        return X + self.dropout(sublayer(self.ln(X)))


class AddNorm(nn.Module):
    """aka PosNorm"""

    def __init__(self, input_dim: int, dropout: float):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(input_dim)

    def forward(self, X: Tensor, sublayer: nn.Module) -> Tensor:
        return self.ln(X + self.dropout(sublayer(X)))


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
        self.out_proj = (
            nn.Linear(input_dim, query_dim, bias=use_bias) if n_heads > 1 else None
        )

    def forward(self, X_Q: Tensor, X_KV: Optional[Tensor] = None) -> Tensor:
        # b: batch size
        # s: seq length
        # l: target sequence length
        # m: used to refer indistinctively to s or l
        # h: number of attention heads,
        # d: head_dim
        q = self.q_proj(X_Q)
        X_KV = X_KV if X_KV is not None else X_Q
        k, v = self.kv_proj(X_KV).chunk(2, dim=-1)
        q, k, v = map(
            lambda t: einops.rearrange(t, "b m (h d) -> b h m d", h=self.n_heads),
            (q, k, v),
        )
        scores = einsum("b h s d, b h l d -> b h s l", q, k) / math.sqrt(self.head_dim)
        attn_weights = scores.softmax(dim=-1)
        self.attn_weights = attn_weights
        attn_weights = self.dropout(attn_weights)
        attn_output = einsum("b h s l, b h l d -> b h s d", attn_weights, v)
        output = einops.rearrange(attn_output, "b h s d -> b s (h d)", h=self.n_heads)

        if self.out_proj is not None:
            output = self.out_proj(output)

        return output


class LinearAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_feats: int,
        n_heads: int,
        use_bias: bool,
        dropout: float,
        kv_compression_factor: float,
        kv_sharing: bool,
    ):
        super(LinearAttention, self).__init__()
        assert input_dim % n_heads == 0, "'input_dim' must be divisible by 'n_heads'"

        self.n_feats = n_feats
        self.head_dim = input_dim // n_heads
        self.n_heads = n_heads
        self.kv_compression_factor = kv_compression_factor
        self.share_kv = kv_sharing

        dim_k = int(self.kv_compression_factor * self.n_feats)

        self.dropout = nn.Dropout(dropout)
        self.qkv_proj = nn.Linear(input_dim, input_dim * 3, bias=use_bias)

        self.E = nn.init.xavier_uniform_(nn.Parameter(torch.zeros(n_feats, dim_k)))
        if not kv_sharing:
            self.F = nn.init.xavier_uniform_(nn.Parameter(torch.zeros(n_feats, dim_k)))
        else:
            self.F = self.E

        self.out_proj = (
            nn.Linear(input_dim, input_dim, bias=use_bias) if n_heads > 1 else None
        )

    def forward(self, X: Tensor) -> Tensor:
        # b: batch size
        # s: seq length
        # h: number of attention heads,
        # i: input dim
        # k: k dim
        # d: head dim
        q, k, v = self.qkv_proj(X).chunk(3, dim=-1)

        q = einops.rearrange(q, "b s (h d) -> b h s d", h=self.n_heads)
        k = einsum("b s i, s k -> b k i", k, self.E)
        v = einsum("b s i, s k -> b k i", v, self.F)

        k = einops.rearrange(k, "b k (h d) -> b h k d", d=self.head_dim)
        v = einops.rearrange(v, "b k (h d) -> b h k d", d=self.head_dim)

        scores = einsum("b h s d, b h k d -> b h s k", q, k) / math.sqrt(self.head_dim)
        attn_weights = scores.softmax(dim=-1)
        self.attn_weights = attn_weights
        attn_weights = self.dropout(attn_weights)
        output = einsum("b h s k, b h k d -> b h s d", attn_weights, v)
        output = einops.rearrange(output, "b h s d -> b s (h d)")

        if self.out_proj is not None:
            output = self.out_proj(output)

        return output


class AdditiveAttention(nn.Module):
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

        self.W_q = nn.Linear(input_dim, n_heads)
        self.W_k = nn.Linear(input_dim, n_heads)

        self.r_out = nn.Linear(input_dim, input_dim)

    def forward(self, X: Tensor) -> Tensor:
        # b: batch size
        # s: seq length
        # h: number of attention heads,
        # d: head_dim
        q = self.qv_proj(X) if self.share_qv_weights else self.q_proj(X)
        v = self.qv_proj(X) if self.share_qv_weights else self.v_proj(X)
        k = self.k_proj(X)

        alphas = (self.W_q(q) / math.sqrt(self.head_dim)).softmax(dim=-1)
        q_r = einops.rearrange(q, "b s (h d) -> b s h d", h=self.n_heads)
        global_query = einsum(" b s h, b s h d -> b h d", alphas, q_r)
        global_query = einops.rearrange(global_query, "b h d -> b () (h d)")

        p = k * global_query

        betas = (self.W_k(p) / math.sqrt(self.head_dim)).softmax(dim=-1)
        p_r = einops.rearrange(p, "b s (h d) -> b s h d", h=self.n_heads)
        global_key = einsum(" b s h, b s h d -> b h d", betas, p_r)
        global_key = einops.rearrange(global_key, "b h d -> b () (h d)")

        u = v * global_key

        # for consistency with all other transformer-based models, rearrange
        # the attn_weights
        self.attn_weights = (
            einops.rearrange(alphas, "b s h -> b h s"),
            einops.rearrange(betas, "b s h -> b h s"),
        )

        output = q + self.dropout(self.r_out(u))

        return output
