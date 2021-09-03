import einops
from torch import nn

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models.transformers._attention_layers import (
    AddNorm,
    NormAdd,
    PositionwiseFF,
    LinearAttention,
    AdditiveAttention,
    MultiHeadedAttention,
)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        use_bias: bool,
        attn_dropout: float,
        ff_dropout: float,
        activation: str,
    ):
        super(TransformerEncoder, self).__init__()

        self.attn = MultiHeadedAttention(
            input_dim,
            n_heads,
            use_bias,
            attn_dropout,
        )
        self.ff = PositionwiseFF(input_dim, ff_dropout, activation)

        self.attn_addnorm = AddNorm(input_dim, attn_dropout)
        self.ff_addnorm = AddNorm(input_dim, ff_dropout)

    def forward(self, X: Tensor) -> Tensor:
        x = self.attn_addnorm(X, self.attn)
        return self.ff_addnorm(x, self.ff)


class SaintEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        use_bias: bool,
        attn_dropout: float,
        ff_dropout: float,
        activation: str,
        n_feat: int,
    ):
        super(SaintEncoder, self).__init__()

        self.n_feat = n_feat

        self.col_attn = MultiHeadedAttention(
            input_dim,
            n_heads,
            use_bias,
            attn_dropout,
        )
        self.col_attn_ff = PositionwiseFF(input_dim, ff_dropout, activation)
        self.col_attn_addnorm = AddNorm(input_dim, attn_dropout)
        self.col_attn_ff_addnorm = AddNorm(input_dim, ff_dropout)

        self.row_attn = MultiHeadedAttention(
            n_feat * input_dim,
            n_heads,
            use_bias,
            attn_dropout,
        )
        self.row_attn_ff = PositionwiseFF(n_feat * input_dim, ff_dropout, activation)
        self.row_attn_addnorm = AddNorm(n_feat * input_dim, attn_dropout)
        self.row_attn_ff_addnorm = AddNorm(n_feat * input_dim, ff_dropout)

    def forward(self, X: Tensor) -> Tensor:
        x = self.col_attn_addnorm(X, self.col_attn)
        x = self.col_attn_ff_addnorm(x, self.col_attn_ff)
        x = einops.rearrange(x, "b n d -> 1 b (n d)")
        x = self.row_attn_addnorm(x, self.row_attn)
        x = self.row_attn_ff_addnorm(x, self.row_attn_ff)
        x = einops.rearrange(x, "1 b (n d) -> b n d", n=self.n_feat)
        return x


class FTTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_feats: int,
        n_heads: int,
        use_bias: bool,
        attn_dropout: float,
        ff_dropout: float,
        kv_compression_factor: float,
        kv_sharing: bool,
        activation: str,
        ff_factor: float,
        first_block: bool,
    ):
        super(FTTransformerEncoder, self).__init__()

        self.first_block = first_block

        self.attn = LinearAttention(
            input_dim,
            n_feats,
            n_heads,
            use_bias,
            attn_dropout,
            kv_compression_factor,
            kv_sharing,
        )
        self.ff = PositionwiseFF(input_dim, ff_dropout, activation, ff_factor)

        self.attn_normadd = NormAdd(input_dim, attn_dropout)
        self.ff_normadd = NormAdd(input_dim, ff_dropout)

    def forward(self, X: Tensor) -> Tensor:
        if self.first_block:
            x = X + self.attn(X)
        else:
            x = self.attn_normadd(X, self.attn)
        return self.ff_normadd(x, self.ff)


class PerceiverEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        use_bias: bool,
        attn_dropout: float,
        ff_dropout: float,
        activation: str,
        query_dim: Optional[int] = None,
    ):
        super(PerceiverEncoder, self).__init__()

        self.attn = MultiHeadedAttention(
            input_dim,
            n_heads,
            use_bias,
            attn_dropout,
            query_dim,
        )
        attn_dim_out = query_dim if query_dim is not None else input_dim
        self.ff = PositionwiseFF(attn_dim_out, ff_dropout, activation)

        self.ln_q = nn.LayerNorm(attn_dim_out)
        self.ln_kv = nn.LayerNorm(input_dim)
        self.norm_attn_dropout = nn.Dropout(attn_dropout)

        self.ff_norm = nn.LayerNorm(attn_dim_out)
        self.norm_ff_dropout = nn.Dropout(ff_dropout)

    def forward(self, X_Q: Tensor, X_KV: Optional[Tensor] = None) -> Tensor:
        x = self.ln_q(X_Q)
        y = None if X_KV is None else self.ln_kv(X_KV)
        x = x + self.norm_attn_dropout(self.attn(x, y))
        return x + self.norm_ff_dropout(self.ff(self.ff_norm(x)))


class FastFormerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        use_bias: bool,
        attn_dropout: float,
        ff_dropout: float,
        share_qv_weights: bool,
        activation: str,
    ):
        super(FastFormerEncoder, self).__init__()

        self.attn = AdditiveAttention(
            input_dim,
            n_heads,
            use_bias,
            attn_dropout,
            share_qv_weights,
        )

        self.ff = PositionwiseFF(input_dim, ff_dropout, activation)
        self.attn_addnorm = AddNorm(input_dim, attn_dropout)
        self.ff_addnorm = AddNorm(input_dim, ff_dropout)

    def forward(self, X: Tensor) -> Tensor:
        x = self.attn_addnorm(X, self.attn)
        return self.ff_addnorm(x, self.ff)
