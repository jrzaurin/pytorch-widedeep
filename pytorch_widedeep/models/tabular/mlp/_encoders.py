from torch import nn

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models.tabular.mlp._layers import SLP
from pytorch_widedeep.models.tabular.mlp._attention_layers import (
    ContextAttention,
    QueryKeySelfAttention,
)
from pytorch_widedeep.models.tabular.transformers._attention_layers import (
    AddNorm,
)


class ContextAttentionEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        dropout: float,
        with_addnorm: bool,
        activation: str,
    ):
        super(ContextAttentionEncoder, self).__init__()

        self.with_addnorm = with_addnorm
        self.attn = ContextAttention(input_dim, dropout)
        if with_addnorm:
            self.attn_addnorm = AddNorm(input_dim, dropout)
            self.slp_addnorm = AddNorm(input_dim, dropout)

        self.slp = SLP(
            input_dim,
            dropout,
            activation,
            not with_addnorm,
        )

    def forward(self, X: Tensor) -> Tensor:
        if self.with_addnorm:
            x = self.attn_addnorm(X, self.attn)
            out = self.slp_addnorm(x, self.slp)
        else:
            out = self.slp(self.attn(X))
        return out


class SelfAttentionEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        dropout: float,
        use_bias: bool,
        n_heads: int,
        with_addnorm: bool,
        activation: str,
    ):
        super(SelfAttentionEncoder, self).__init__()

        self.with_addnorm = with_addnorm
        self.attn = QueryKeySelfAttention(input_dim, dropout, use_bias, n_heads)
        if with_addnorm:
            self.attn_addnorm = AddNorm(input_dim, dropout)
            self.slp_addnorm = AddNorm(input_dim, dropout)

        self.slp = SLP(
            input_dim,
            dropout,
            activation,
            not with_addnorm,
        )

    def forward(self, X: Tensor) -> Tensor:
        if self.with_addnorm:
            x = self.attn_addnorm(X, self.attn)
            out = self.slp_addnorm(x, self.slp)
        else:
            out = self.slp(self.attn(X))
        return out
