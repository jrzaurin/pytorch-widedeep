import warnings

import numpy as np
import torch
from torch import nn

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models.text._encoders import ContextAttentionEncoder
from pytorch_widedeep.models.tabular.mlp._layers import MLP


class StackedAttentiveRNN(nn.Module):
    r"""Text classifier/regressor comprised by a stack of blocks:
    ``[RNN + Attention]``. This can be used as the ``deeptext`` component of a
    Wide & Deep model or independently by itself.

    In addition, there is the option to add a Fully Connected (FC) set of
    dense layers on top of the attentiob blocks

    Parameters
    ----------
    vocab_size: int
        Number of words in the vocabulary
    embed_dim: int, Optional, default = None
        Dimension of the word embeddings if non-pretained word vectors are
        used
    embed_matrix: np.ndarray, Optional, default = None
        Pretrained word embeddings
    embed_trainable: bool, default = True
        Boolean indicating if the pretrained embeddings are trainable
    rnn_type: str, default = 'lstm'
        String indicating the type of RNN to use. One of 'lstm' or 'gru'
    hidden_dim: int, default = 64
        Hidden dim of the RNN
    bidirectional: bool, default = True
        Boolean indicating whether the staked RNNs are bidirectional
    padding_idx: int, default = 1
        index of the padding token in the padded-tokenised sequences. The
        ``TextPreprocessor`` class within this library uses ``fastai``'s
        tokenizer where the token index 0 is reserved for the `'unknown'`
        word token. Therefore, the default value is set to 1.
    n_blocks: int, default = 3
        Number of attention blocks. Each block is comprised by an RNN and a
        Context Attention Encoder
    attn_concatenate: bool, default = True
        Boolean indicating if the input to the attention mechanism will be the
        output of the RNN or the output of the RNN concatenated with the last
        hidden state or simply
    attn_dropout: float, default = 0.1
        Internal dropout for the attention mechanism
    with_addnorm: bool, default = False
        Boolean indicating if the output of each block will be added to the
        input and normalised
    head_hidden_dims: List, Optional, default = None
        List with the sizes of the dense layers in the head e.g: [128, 64]
    head_activation: str, default = "relu"
        Activation function for the dense layers in the head. Currently
        `'tanh'`, `'relu'`, `'leaky_relu'` and `'gelu'` are supported
    head_dropout: float, Optional, default = None
        Dropout of the dense layers in the head
    head_batchnorm: bool, default = False
        Boolean indicating whether or not to include batch normalization in
        the dense layers that form the `'rnn_mlp'`
    head_batchnorm_last: bool, default = False
        Boolean indicating whether or not to apply batch normalization to the
        last of the dense layers in the head
    head_linear_first: bool, default = False
        Boolean indicating whether the order of the operations in the dense
        layer. If ``True: [LIN -> ACT -> BN -> DP]``. If ``False: [BN -> DP ->
        LIN -> ACT]``

    Attributes
    ----------
    word_embed: ``nn.Module``
        word embedding matrix
    rnn: ``nn.Module``
        Stack of RNNs
    rnn_mlp: ``nn.Sequential``
        Stack of dense layers on top of the RNN. This will only exists if
        ``head_layers_dim`` is not ``None``
    output_dim: int
        The output dimension of the model. This is a required attribute
        neccesary to build the ``WideDeep`` class

    Example
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import StackedAttentiveRNN
    >>> X_text = torch.cat((torch.zeros([5,1]), torch.empty(5, 4).random_(1,4)), axis=1)
    >>> model = StackedAttentiveRNN(vocab_size=4, hidden_dim=4, padding_idx=0, embed_dim=4)
    >>> out = model(X_text)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: Optional[int] = None,
        embed_matrix: Optional[np.ndarray] = None,
        embed_trainable: bool = True,
        rnn_type: str = "lstm",
        hidden_dim: int = 64,
        bidirectional: bool = False,
        padding_idx: int = 1,
        n_blocks: int = 3,
        attn_concatenate: bool = False,
        attn_dropout: float = 0.1,
        with_addnorm: bool = False,
        head_hidden_dims: Optional[List[int]] = None,
        head_activation: str = "relu",
        head_dropout: Optional[float] = None,
        head_batchnorm: bool = False,
        head_batchnorm_last: bool = False,
        head_linear_first: bool = False,
    ):
        super(StackedAttentiveRNN, self).__init__()

        if (
            embed_dim is not None
            and embed_matrix is not None
            and not embed_dim == embed_matrix.shape[1]
        ):
            warnings.warn(
                "the input embedding dimension {} and the dimension of the "
                "pretrained embeddings {} do not match. The pretrained embeddings "
                "dimension ({}) will be used".format(
                    embed_dim, embed_matrix.shape[1], embed_matrix.shape[1]
                ),
                UserWarning,
            )

        if rnn_type.lower() not in ["lstm", "gru"]:
            raise ValueError(
                f"'rnn_type' must be 'lstm' or 'gru', got {rnn_type} instead"
            )

        self.vocab_size = vocab_size
        self.embed_trainable = embed_trainable
        self.embed_dim = embed_dim

        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.padding_idx = padding_idx

        self.n_blocks = n_blocks
        self.attn_concatenate = attn_concatenate
        self.attn_dropout = attn_dropout
        self.with_addnorm = with_addnorm

        self.head_hidden_dims = head_hidden_dims
        self.head_activation = head_activation
        self.head_dropout = head_dropout
        self.head_batchnorm = head_batchnorm
        self.head_batchnorm_last = head_batchnorm_last
        self.head_linear_first = head_linear_first

        # Embeddings
        self.word_embed, self.embed_dim = self._set_embeddings(embed_matrix)

        # Linear Projection: if embed_dim is different that the input of the
        # attention blocks we add a linear projection
        if bidirectional and attn_concatenate:
            attn_input_dim = hidden_dim * 4
        elif bidirectional or attn_concatenate:
            attn_input_dim = hidden_dim * 2
        else:
            attn_input_dim = hidden_dim
        self.output_dim = attn_input_dim

        if attn_input_dim != self.embed_dim:
            self.embed_proj: Union[nn.Linear, nn.Identity] = nn.Linear(
                self.embed_dim, attn_input_dim
            )
        else:
            self.embed_proj = nn.Identity()

        # RNN
        rnn_params = {
            "input_size": attn_input_dim,
            "hidden_size": hidden_dim,
            "bidirectional": bidirectional,
            "batch_first": True,
        }
        if self.rnn_type.lower() == "lstm":
            self.rnn: Union[nn.LSTM, nn.GRU] = nn.LSTM(**rnn_params)
        elif self.rnn_type.lower() == "gru":
            self.rnn = nn.GRU(**rnn_params)

        # FC-Head (Mlp)
        self.attention_blks = nn.ModuleList()
        for i in range(n_blocks):
            self.attention_blks.append(
                ContextAttentionEncoder(
                    self.rnn,
                    attn_input_dim,
                    attn_dropout,
                    attn_concatenate,
                    with_addnorm=with_addnorm if i != n_blocks - 1 else False,
                    sum_along_seq=i == n_blocks - 1,
                )
            )

        # Mlp
        if self.head_hidden_dims is not None:
            head_hidden_dims = [self.output_dim] + head_hidden_dims
            self.rnn_mlp: Union[MLP, nn.Identity] = MLP(
                head_hidden_dims,
                head_activation,
                head_dropout,
                head_batchnorm,
                head_batchnorm_last,
                head_linear_first,
            )
            self.output_dim = head_hidden_dims[-1]
        else:
            # simple hack to add readability in the forward pass
            self.rnn_mlp = nn.Identity()

    def forward(self, X: Tensor) -> Tensor:  # type: ignore
        x = self.embed_proj(self.word_embed(X.long()))

        h = nn.init.zeros_(
            torch.Tensor(2 if self.bidirectional else 1, X.shape[0], self.hidden_dim)
        ).to(x.device)
        if self.rnn_type == "lstm":
            c = nn.init.zeros_(
                torch.Tensor(
                    2 if self.bidirectional else 1, X.shape[0], self.hidden_dim
                )
            ).to(x.device)
        else:
            c = None

        for blk in self.attention_blks:
            x, h, c = blk(x, h, c)

        return self.rnn_mlp(x)

    @property
    def attention_weights(self) -> List:
        r"""List with the attention weights

        The shape of the attention weights is:

        :math:`(N, S)`

        Where *N* is the batch size and *S* is the length of the sequence
        """
        return [blk.attn.attn_weights for blk in self.attention_blks]

    def _set_embeddings(
        self, embed_matrix: Union[Any, np.ndarray]
    ) -> Tuple[nn.Module, int]:
        if isinstance(embed_matrix, np.ndarray):
            assert (
                embed_matrix.dtype == "float32"
            ), "'embed_matrix' must be of dtype 'float32', got dtype '{}'".format(
                str(embed_matrix.dtype)
            )
            word_embed = nn.Embedding(
                self.vocab_size, embed_matrix.shape[1], padding_idx=self.padding_idx
            )
            if self.embed_trainable:
                word_embed.weight = nn.Parameter(
                    torch.tensor(embed_matrix), requires_grad=True
                )
            else:
                word_embed.weight = nn.Parameter(
                    torch.tensor(embed_matrix), requires_grad=False
                )
            embed_dim = embed_matrix.shape[1]
        else:
            word_embed = nn.Embedding(
                self.vocab_size, self.embed_dim, padding_idx=self.padding_idx
            )
            embed_dim = self.embed_dim

        return word_embed, embed_dim
