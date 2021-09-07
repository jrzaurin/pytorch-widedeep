import warnings

import numpy as np
import torch
from torch import nn

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models.tab_mlp import MLP


class DeepText(nn.Module):
    r"""Standard text classifier/regressor comprised by a stack of RNNs
    (LSTMs or GRUs).

    In addition, there is the option to add a Fully Connected (FC) set of dense
    layers (referred as `texthead`) on top of the stack of RNNs

    Parameters
    ----------
    vocab_size: int
        number of words in the vocabulary
    rnn_type: str, default = 'lstm'
        String indicating the type of RNN to use. One of ``lstm`` or ``gru``
    hidden_dim: int, default = 64
        Hidden dim of the RNN
    n_layers: int, default = 3
        number of recurrent layers
    rnn_dropout: float, default = 0.1
        dropout for the dropout layer on the outputs of each RNN layer except
        the last layer
    bidirectional: bool, default = True
        indicates whether the staked RNNs are bidirectional
    use_hidden_state: str, default = True
        Boolean indicating whether to use the final hidden state or the
        RNN output as predicting features
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
        Activation function for the dense layers in the head. Currently
        ``tanh``, ``relu``, ``leaky_relu`` and ``gelu`` are supported
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
        Stack of RNNs
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

    def __init__(
        self,
        vocab_size: int,
        rnn_type: str = "lstm",
        hidden_dim: int = 64,
        n_layers: int = 3,
        rnn_dropout: float = 0.1,
        bidirectional: bool = False,
        use_hidden_state: bool = True,
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

        if rnn_type.lower() not in ["lstm", "gru"]:
            raise ValueError(
                f"'rnn_type' must be 'lstm' or 'gru', got {rnn_type} instead"
            )

        self.vocab_size = vocab_size
        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn_dropout = rnn_dropout
        self.bidirectional = bidirectional
        self.use_hidden_state = use_hidden_state
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
        rnn_params = {
            "input_size": embed_dim,
            "hidden_size": hidden_dim,
            "num_layers": n_layers,
            "bidirectional": bidirectional,
            "dropout": rnn_dropout,
            "batch_first": True,
        }
        if self.rnn_type.lower() == "lstm":
            self.rnn: Union[nn.LSTM, nn.GRU] = nn.LSTM(**rnn_params)
        elif self.rnn_type.lower() == "gru":
            self.rnn = nn.GRU(**rnn_params)

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

        if self.rnn_type.lower() == "lstm":
            o, (h, c) = self.rnn(embed)
        elif self.rnn_type.lower() == "gru":
            o, h = self.rnn(embed)

        o = o.permute(1, 0, 2)

        if self.bidirectional:
            rnn_out = (
                torch.cat((h[-2], h[-1]), dim=1) if self.use_hidden_state else o[-1]
            )
        else:
            rnn_out = h[-1] if self.use_hidden_state else o[-1]

        if self.head_hidden_dims is not None:
            head_out = self.texthead(rnn_out)
            return head_out
        else:
            return rnn_out
