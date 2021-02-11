import warnings

import numpy as np
import torch
from torch import nn

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models.tab_mlp import MLP


class DeepText(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 64,
        n_layers: int = 3,
        rnn_dropout: float = 0.1,
        bidirectional: bool = False,
        padding_idx: int = 1,
        embed_dim: Optional[int] = None,
        embed_matrix: Optional[np.ndarray] = None,
        embed_trainable: bool = True,
        head_hidden_dims: Optional[List[int]] = None,
        head_activation: str = "relu",
        head_dropout: Optional[float] = None,
        head_batchnorm: bool = False,
        head_batchnorm_last: bool = False,
        head_linear_first: bool = False,
    ):
        r"""Standard text classifier/regressor comprised by a stack of RNNs
        (in particular LSTMs).

        In addition, there is the option to add a Fully Connected (FC) set of dense
        layers (referred as `texthead`) on top of the stack of RNNs

        Parameters
        ----------
        vocab_size: int
            number of words in the vocabulary
        hidden_dim: int, default = 64
            Hidden dim of the LSTM
        n_layers: int, default = 3
            number of recurrent layers
        rnn_dropout: float, default = 0.1
            dropout for the dropout layer on the outputs of each LSTM layer except
            the last layer
        bidirectional: bool, default = True
            indicates whether the staked RNNs are bidirectional
        padding_idx: int, default = 1
            index of the padding token in the padded-tokenised sequences. I
            use the ``fastai`` tokenizer where the token index 0 is reserved
            for the `'unknown'` word token
        embed_dim: int, Optional, default = None
            Dimension of the word embedding matrix if non-pretained word
            vectors are used
        embed_matrix: np.ndarray, Optional, default = None
             Pretrained word embeddings
        embed_trainable: bool, default = True
            Boolean indicating if the pretrained embeddings are trainable
        head_hidden_dims: List, Optional, default = None
            List with the sizes of the stacked dense layers in the head
            e.g: [128, 64]
        head_activation: str, default = "relu"
            Activation function for the dense layers in the head
        head_dropout: float, Optional, default = None
            dropout between the dense layers in the head
        head_batchnorm: bool, default = False
            Whether or not to include batch normalization in the dense layers that
            form the `'texthead'`
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
            Stack of LSTMs
        texthead: ``nn.Sequential``
            Stack of dense layers on top of the RNN. This will only exists if
            ``head_layers_dim`` is not ``None``
        output_dim: int
            The output dimension of the model. This is a required attribute
            neccesary to build the WideDeep class

        Example
        --------
        >>> import torch
        >>> from pytorch_widedeep.models import DeepText
        >>> X_text = torch.cat((torch.zeros([5,1]), torch.empty(5, 4).random_(1,4)), axis=1)
        >>> model = DeepText(vocab_size=4, hidden_dim=4, n_layers=1, padding_idx=0, embed_dim=4)
        >>> out = model(X_text)
        """
        super(DeepText, self).__init__()

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

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn_dropout = rnn_dropout
        self.bidirectional = bidirectional
        self.padding_idx = padding_idx
        self.embed_dim = embed_dim
        self.embed_trainable = embed_trainable
        self.head_hidden_dims = head_hidden_dims
        self.head_activation = head_activation
        self.head_dropout = head_dropout
        self.head_batchnorm = head_batchnorm
        self.head_batchnorm_last = head_batchnorm_last
        self.head_linear_first = head_linear_first

        # Pre-trained Embeddings
        if isinstance(embed_matrix, np.ndarray):
            assert (
                embed_matrix.dtype == "float32"
            ), "'embed_matrix' must be of dtype 'float32', got dtype '{}'".format(
                str(embed_matrix.dtype)
            )
            self.word_embed = nn.Embedding(
                vocab_size, embed_matrix.shape[1], padding_idx=padding_idx
            )
            if embed_trainable:
                self.word_embed.weight = nn.Parameter(
                    torch.tensor(embed_matrix), requires_grad=True
                )
            else:
                self.word_embed.weight = nn.Parameter(
                    torch.tensor(embed_matrix), requires_grad=False
                )
            embed_dim = embed_matrix.shape[1]
        else:
            self.word_embed = nn.Embedding(
                vocab_size, embed_dim, padding_idx=padding_idx
            )

        # stack of RNNs (LSTMs)
        self.rnn = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=rnn_dropout,
            batch_first=True,
        )

        # the output_dim attribute will be used as input_dim when "merging" the models
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        if self.head_hidden_dims is not None:
            assert self.head_hidden_dims[0] == self.output_dim, (
                "The hidden dimension from the stack or RNNs ({}) is not consistent with "
                "the expected input dimension ({}) of the fc-head".format(
                    self.output_dim, self.head_hidden_dims[0]
                )
            )
            self.texthead = MLP(
                head_hidden_dims,
                head_activation,
                head_dropout,
                head_batchnorm,
                head_batchnorm_last,
                head_linear_first,
            )
            self.output_dim = head_hidden_dims[-1]

    def forward(self, X: Tensor) -> Tensor:  # type: ignore
        r"""Forward pass that is simply a standard RNN-based
        classifier/regressor with an optional `'Fully Connected head'`
        """
        embed = self.word_embed(X.long())
        o, (h, c) = self.rnn(embed)
        if self.bidirectional:
            last_h = torch.cat((h[-2], h[-1]), dim=1)
        else:
            last_h = h[-1]
        if self.head_hidden_dims is not None:
            out = self.texthead(last_h)
            return out
        else:
            return last_h
