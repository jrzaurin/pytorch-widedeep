from torch import nn

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models.tabular.mlp._attention_layers import (
    ContextAttention,
    QueryKeySelfAttention,
)
from pytorch_widedeep.models.tabular.transformers._attention_layers import (
    AddNorm,
)


class AttentionEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        dropout: float,
        with_residual: bool,
        attention_name: str,
        use_bias: bool,
        n_heads: Optional[int],
    ):
        super(AttentionEncoder, self).__init__()

        self.with_residual = with_residual

        if attention_name == "context_attention":
            self.attn: Union[
                ContextAttention, QueryKeySelfAttention
            ] = ContextAttention(input_dim, dropout)
        if attention_name == "self_attention":
            self.attn = QueryKeySelfAttention(input_dim, use_bias, dropout, n_heads)

        if with_residual:
            self.attn_addnorm = AddNorm(input_dim, dropout)

    def forward(self, X: Tensor) -> Tensor:
        if self.with_residual:
            return self.attn_addnorm(X, self.attn)
        else:
            return self.attn(X)
