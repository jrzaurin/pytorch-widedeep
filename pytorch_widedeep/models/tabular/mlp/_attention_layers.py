import math

import einops
from torch import nn, einsum

from pytorch_widedeep.wdtypes import *  # noqa: F403


class ContextAttention(nn.Module):
    r"""Attention mechanism inspired by `Hierarchical Attention Networks for
    Document Classification
    <https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf>`_
    """

    def __init__(self, input_dim: int, dropout: float, sum_along_seq: bool = False):
        super(ContextAttention, self).__init__()

        self.inp_proj = nn.Linear(input_dim, input_dim)
        self.context = nn.Linear(input_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.sum_along_seq = sum_along_seq

    def forward(self, X: Tensor) -> Tensor:
        scores = torch.tanh_(self.inp_proj(X))
        attn_weights = self.context(scores).softmax(dim=1)
        self.attn_weights = attn_weights.squeeze(2)
        attn_weights = self.dropout(attn_weights)
        output = (attn_weights * X).sum(1) if self.sum_along_seq else (attn_weights * X)
        return output


class QueryKeySelfAttention(nn.Module):
    r"""Attention mechanism inspired by the well known multi-head attention. Here,
    rather than learning a value projection matrix that will be multiplied by
    the attention weights, we multiply such weights directly by the input
    tensor.

    The rationale behind this implementation comes, among other
    considerations, from the fact that Transformer based models tend to
    heavily overfit tabular. Therefore, by reducing the number of trainable
    parameters and multiply directly by the incoming tensor we help
    mitigating such overfitting
    """

    def __init__(
        self,
        input_dim: int,
        dropout: float,
        use_bias: bool,
        n_heads: int,
    ):
        super(QueryKeySelfAttention, self).__init__()

        assert input_dim % n_heads == 0, "'input_dim' must be divisible by 'n_heads'"

        self.head_dim = input_dim // n_heads
        self.n_heads = n_heads
        self.qk_proj = nn.Linear(input_dim, input_dim * 2, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: Tensor) -> Tensor:
        # b: batch size
        # s: seq length
        # l: target sequence length. Here l = s
        # m: used to refer indistinctively to s or l
        # h: number of attention heads,
        # d: head_dim
        q, k = self.qk_proj(X).chunk(2, dim=-1)
        q, k, x_rearr = map(
            lambda t: einops.rearrange(t, "b m (h d) -> b h m d", h=self.n_heads),
            (q, k, X),
        )
        scores = einsum("b h s d, b h l d -> b h s l", q, k) / math.sqrt(self.head_dim)
        attn_weights = scores.softmax(dim=-1)
        self.attn_weights = attn_weights
        attn_weights = self.dropout(attn_weights)
        attn_output = einsum("b h s l, b h l d -> b h s d", attn_weights, x_rearr)
        output = einops.rearrange(attn_output, "b h s d -> b s (h d)", h=self.n_heads)
        return output
