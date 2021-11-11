import math

import einops
from torch import nn, einsum

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models._get_activation_fn import get_activation_fn


def dense_layer(
    inp: int,
    out: int,
    activation: str,
    p: float,
    bn: bool,
    linear_first: bool,
):
    # This is basically the LinBnDrop class at the fastai library
    if activation == "geglu":
        raise ValueError(
            "'geglu' activation is only used as 'transformer_activation' "
            "in transformer-based models"
        )
    act_fn = get_activation_fn(activation)
    layers = [nn.BatchNorm1d(out if linear_first else inp)] if bn else []
    if p != 0:
        layers.append(nn.Dropout(p))  # type: ignore[arg-type]
    lin = [nn.Linear(inp, out, bias=not bn), act_fn]
    layers = lin + layers if linear_first else layers + lin
    return nn.Sequential(*layers)


class MLP(nn.Module):
    def __init__(
        self,
        d_hidden: List[int],
        activation: str,
        dropout: Optional[Union[float, List[float]]],
        batchnorm: bool,
        batchnorm_last: bool,
        linear_first: bool,
    ):
        super(MLP, self).__init__()

        if not dropout:
            dropout = [0.0] * len(d_hidden)
        elif isinstance(dropout, float):
            dropout = [dropout] * len(d_hidden)

        self.mlp = nn.Sequential()
        for i in range(1, len(d_hidden)):
            self.mlp.add_module(
                "dense_layer_{}".format(i - 1),
                dense_layer(
                    d_hidden[i - 1],
                    d_hidden[i],
                    activation,
                    dropout[i - 1],
                    batchnorm and (i != len(d_hidden) - 1 or batchnorm_last),
                    linear_first,
                ),
            )

    def forward(self, X: Tensor) -> Tensor:
        return self.mlp(X)


class ContextAttention(nn.Module):
    """
    Attention Mechanism with context inspired by the work descibed at
    Hierarchical Attention Networks for Document Classification:
    https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf
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
    def __init__(
        self,
        input_dim: int,
        use_bias: bool,
        dropout: float,
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
