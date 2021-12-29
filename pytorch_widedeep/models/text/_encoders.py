import torch
from torch import nn

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models.tabular.mlp._attention_layers import (
    ContextAttention,
)
from pytorch_widedeep.models.tabular.transformers._attention_layers import (
    AddNorm,
)


class ContextAttentionEncoder(nn.Module):
    def __init__(
        self,
        rnn: nn.Module,
        input_dim: int,
        attn_dropout: float,
        attn_concatenate: bool,
        with_addnorm: bool,
        sum_along_seq: bool,
    ):
        super(ContextAttentionEncoder, self).__init__()

        self.rnn = rnn
        self.bidirectional = self.rnn.bidirectional
        self.attn_concatenate = attn_concatenate

        self.with_addnorm = with_addnorm
        if with_addnorm:
            self.attn_addnorm = AddNorm(input_dim, attn_dropout)

        self.attn = ContextAttention(input_dim, attn_dropout, sum_along_seq)

    def forward(self, X: Tensor, h: Tensor, c: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

        if isinstance(self.rnn, nn.LSTM):
            o, (h, c) = self.rnn(X, (h, c))
        elif isinstance(self.rnn, nn.GRU):
            o, h = self.rnn(X, h)

        attn_inp = self._process_rnn_outputs(o, h)

        if self.with_addnorm:
            out = self.attn_addnorm(attn_inp, self.attn)
        else:
            out = self.attn(attn_inp)

        return out, c, h

    def _process_rnn_outputs(self, output: Tensor, hidden: Tensor) -> Tensor:

        if self.attn_concatenate:
            if self.bidirectional:
                bi_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
                attn_inp = torch.cat(
                    [output, bi_hidden.unsqueeze(1).expand_as(output)], dim=2
                )
            else:
                attn_inp = torch.cat(
                    [output, hidden[-1].unsqueeze(1).expand_as(output)], dim=2
                )
        else:
            attn_inp = output

        return attn_inp
