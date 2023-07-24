import math

import torch
from torch import nn

from pytorch_widedeep.wdtypes import Union, Tensor, Optional
from pytorch_widedeep.models.tabular.transformers._encoders import (
    TransformerEncoder,
)


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        n_heads: int,
        n_blocks: int,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        activation: str = "gelu",
        ff_dim_multiplier: float = 1.0,
        *,
        with_pos_encoding: bool = True,
        pos_encoding_dropout: float = 0.1,
        seq_length: Optional[int] = None,
        pos_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.activation = activation
        self.ff_dim_multiplier = ff_dim_multiplier
        self.with_pos_encoding = with_pos_encoding
        self.pos_encoding_dropout = pos_encoding_dropout
        self.seq_length = seq_length

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        if with_pos_encoding:
            if pos_encoder is not None:
                self.pos_encoder: Union[
                    nn.Module, nn.Identity, PositionalEncoding
                ] = self.pos_encoder
            else:
                assert (
                    seq_length is not None
                ), "If positional encoding is used 'seq_length' must be passed to the model"
                self.pos_encoder = PositionalEncoding(
                    embed_dim, pos_encoding_dropout, seq_length
                )
        else:
            self.pos_encoder = nn.Identity()

        self.encoder = nn.Sequential()
        for i in range(n_blocks):
            self.encoder.add_module(
                "transformer_block" + str(i),
                TransformerEncoder(
                    embed_dim,
                    n_heads,
                    False,  # use_qkv_bias
                    attn_dropout,
                    ff_dropout,
                    activation,
                ),
            )

    def forward(self, X: Tensor) -> Tensor:
        x = self.embedding(X)
        x = self.pos_encoder(x)
        out = self.encoder(x)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, dropout: float, seq_length: int):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(seq_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(seq_length, 1, embed_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, X: Tensor) -> Tensor:
        return self.dropout(X + self.pe)
