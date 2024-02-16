import math

import torch
from torch import nn

from pytorch_widedeep.wdtypes import Union, Tensor, Optional
from pytorch_widedeep.utils.general_utils import alias
from pytorch_widedeep.models.tabular.transformers._encoders import (
    TransformerEncoder,
)


class Transformer(nn.Module):
    r"""Basic Encoder-Only Transformer Model for text classification/regression.
    As all other models in the library this model can be used as the
    `deeptext` component of a Wide & Deep model or independently by itself.

    :information_source: **NOTE**:
    This model is introduced in the context of recommendation systems and
    thought for sequences of any nature (e.g. items). It can, of course,
    still be used for text. However, at this stage, we have decided to not
    include the possibility of loading pretrained word vectors since we aim
    to integrate the library wit Huggingface in the (hopefully) near future

    Parameters
    ----------
    vocab_size: int
        Number of words in the vocabulary
    input_dim: int
        Dimension of the token embeddings

        Param aliases: `embed_dim`, `d_model`. <br/>

    seq_length: int, Optional, default = None
        Input sequence length
    n_heads: int, default = 8
        Number of attention heads per Transformer block
    n_blocks: int, default = 4
        Number of Transformer blocks
    attn_dropout: float, default = 0.2
        Dropout that will be applied to the Multi-Head Attention layers
    ff_dropout: float, default = 0.1
        Dropout that will be applied to the FeedForward network
    ff_factor: float, default = 4
        Multiplicative factor applied to the first layer of the FF network in
        each Transformer block, This is normally set to 4.
    activation: str, default = "gelu"
        Transformer Encoder activation function. _'tanh'_, _'relu'_,
        _'leaky_relu'_, _'gelu'_, _'geglu'_ and _'reglu'_ are supported
    padding_idx: int, default = 0
        index of the padding token in the padded-tokenised sequences.
    with_cls_token: bool, default = False
        Boolean indicating if a `'[CLS]'` token is included in the tokenized
        sequences. If present, the final hidden state corresponding to this
        token is used as the aggregated representation for classification and
        regression tasks. **NOTE**: if included in the tokenized sequences it
        must be inserted as the first token in the sequences.
    with_pos_encoding: bool, default = True
        Boolean indicating if positional encoding will be used
    pos_encoding_dropout: float, default = 0.1
        Positional encoding dropout
    pos_encoder: nn.Module, Optional, default = None
        This model uses by default a standard positional encoding approach.
        However, any custom positional encoder can also be used and pass to
        the Transformer model via the 'pos_encoder' parameter

    Attributes
    ----------
    embedding: nn.Module
        Standard token embedding layer
    pos_encoder: nn.Module
        Positional Encoder
    encoder: nn.Module
        Sequence of Transformer blocks

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import Transformer
    >>> X_text = torch.cat((torch.zeros([5,1]), torch.empty(5, 4).random_(1,4)), axis=1)
    >>> model = Transformer(vocab_size=4, seq_length=5, input_dim=8, n_heads=1, n_blocks=1)
    >>> out = model(X_text)
    """

    @alias("input_dim", ["embed_dim", "d_model"])
    @alias("seq_length", ["max_length", "maxlen"])
    def __init__(
        self,
        vocab_size: int,
        seq_length: int,
        input_dim: int,
        n_heads: int,
        n_blocks: int,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        ff_factor: int = 4,
        activation: str = "gelu",
        use_linear_attention: bool = False,
        use_flash_attention: bool = False,
        padding_idx: int = 0,
        with_cls_token: bool = False,
        *,  # from here on pos encoding args
        with_pos_encoding: bool = True,
        pos_encoding_dropout: float = 0.1,
        pos_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.ff_factor = ff_factor
        self.activation = activation
        self.use_linear_attention = use_linear_attention
        self.use_flash_attention = use_flash_attention
        self.padding_idx = padding_idx
        self.with_cls_token = with_cls_token
        self.with_pos_encoding = with_pos_encoding
        self.pos_encoding_dropout = pos_encoding_dropout

        self.embedding = nn.Embedding(
            vocab_size, input_dim, padding_idx=self.padding_idx
        )

        if with_pos_encoding:
            if pos_encoder is not None:
                self.pos_encoder: Union[nn.Module, nn.Identity, PositionalEncoding] = (
                    pos_encoder
                )
            else:
                self.pos_encoder = PositionalEncoding(
                    input_dim, pos_encoding_dropout, seq_length
                )
        else:
            self.pos_encoder = nn.Identity()

        self.encoder = nn.Sequential()
        for i in range(n_blocks):
            self.encoder.add_module(
                "transformer_block" + str(i),
                TransformerEncoder(
                    input_dim,
                    n_heads,
                    False,  # use_qkv_bias
                    attn_dropout,
                    ff_dropout,
                    ff_factor,
                    activation,
                    use_linear_attention,
                    use_flash_attention,
                ),
            )

    def forward(self, X: Tensor) -> Tensor:
        x = self.embedding(X.long())
        x = self.pos_encoder(x)
        x = self.encoder(x)
        if self.with_cls_token:
            x = x[:, 0, :]
        else:
            x = x.flatten(1)
        return x

    @property
    def output_dim(self) -> int:
        if self.with_cls_token:
            output_dim = self.input_dim
        else:
            output_dim = self.input_dim * self.seq_length
        return output_dim


class PositionalEncoding(nn.Module):
    """Positional Encoding copied and pasted directly from [The Beginners'
    Tutorial]
    (https://pytorch.org/tutorials/beginner/transformer_tutorial.html) at the
    Pytorch site. Here is simply adapated so that the input sequence length
    must be specified and in our implementation the input tensor dimensions
    are arranged as `[batch_size, seq_len, embedding_dim]` instead of `
    [seq_len, batch_size, embedding_dim]` , as in the before mentioned
    tutorial

    Parameters
    ----------
    input_dim: int
        Dimension of the token embeddings
    dropout: float
        Positional encoding dropout
    seq_length: int
        input sequence length

    """

    def __init__(self, input_dim: int, dropout: float, seq_length: int):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(seq_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, input_dim, 2) * (-math.log(10000.0) / input_dim)
        )
        pe = torch.zeros(1, seq_length, input_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, X: Tensor) -> Tensor:
        return self.dropout(X + self.pe)
