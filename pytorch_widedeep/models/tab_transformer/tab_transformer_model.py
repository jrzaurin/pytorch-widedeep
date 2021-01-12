import copy
import math

from torch import nn, einsum
from einops import rearrange


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        keep_attn_weights: bool,
        dropout: float = 0.0,
    ):
        super(MultiHeadedAttention, self).__init__()

        assert d_model % n_heads == 0

        # Consistent with other implementations I assume d_v = d_k
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.inp_proj = nn.Linear(d_model, self.d_k * 3)
        self.out_proj = nn.Linear(self.d_k, d_model)
        self.keep_attn_weights = keep_attn_weights

    def forward(self, x):

        q, k, v = self.inp_proj(x).chunk(3, dim=-1)

        # b: batch size, s: src seq length (num of categorical features
        # encoded as embeddings), h: number of attention heads, d: d_k
        q, k, v = map(
            lambda t: rearrange(t, "b s (h d) -> b h s d", h=self.n_heads), (q, k, v)
        )

        # l: target sequence (l = s)
        scores = einsum("b h s d, b h l d -> b h s l", q, k) / math.sqrt(self.d_k)

        attn_weights = self.dropout(scores.softmax(dim=-1))
        if self.keep_attn_weights:
            self.attn_weights = attn_weights

        attn_output = einsum("b h s l, b h l d -> b h s d", attn_weights, v)
        output = rearrange(attn_output, "b h s d -> b s (h d)", h=self.n_heads)

        return self.out_proj(output)
