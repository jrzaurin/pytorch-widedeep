import numpy as np
import torch

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models.text.basic_rnn import BasicRNN
from pytorch_widedeep.models.tabular.mlp._layers import MLP
from pytorch_widedeep.models.tabular.mlp._attention_layers import (
    ContextAttention,
)


class AttentiveRNN(BasicRNN):
    r"""Text classifier/regressor comprised by a stack of RNNs
    (LSTMs or GRUs) plus an attention layer that can be used as the
    ``deeptext`` component of a Wide & Deep model or independently by
    itself.

    In addition, there is the option to add a Fully Connected (FC) set of dense
    layers on top of attention layer

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
    n_layers: int, default = 3
        Number of recurrent layers
    rnn_dropout: float, default = 0.1
        Dropout for each RNN layer except the last layer
    bidirectional: bool, default = True
        Boolean indicating whether the staked RNNs are bidirectional
    use_hidden_state: str, default = True
        Boolean indicating whether to use the final hidden state or the RNN's
        output as predicting features. Typically the former is used.
    padding_idx: int, default = 1
        index of the padding token in the padded-tokenised sequences. The
        ``TextPreprocessor`` class within this library uses ``fastai``'s
        tokenizer where the token index 0 is reserved for the `'unknown'`
        word token. Therefore, the default value is set to 1.
    attn_concatenate: bool, default = True
        Boolean indicating if the input to the attention mechanism will be the
        output of the RNN or the output of the RNN concatenated with the last
        hidden state.
    attn_dropout: float, default = 0.1
        Internal dropout for the attention mechanism
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
    >>> from pytorch_widedeep.models import AttentiveRNN
    >>> X_text = torch.cat((torch.zeros([5,1]), torch.empty(5, 4).random_(1,4)), axis=1)
    >>> model = AttentiveRNN(vocab_size=4, hidden_dim=4, n_layers=2, padding_idx=0, embed_dim=4)
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
        n_layers: int = 3,
        rnn_dropout: float = 0.1,
        bidirectional: bool = False,
        use_hidden_state: bool = True,
        padding_idx: int = 1,
        attn_concatenate: bool = True,
        attn_dropout: float = 0.1,
        head_hidden_dims: Optional[List[int]] = None,
        head_activation: str = "relu",
        head_dropout: Optional[float] = None,
        head_batchnorm: bool = False,
        head_batchnorm_last: bool = False,
        head_linear_first: bool = False,
    ):
        super(AttentiveRNN, self).__init__(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            embed_matrix=embed_matrix,
            embed_trainable=embed_trainable,
            rnn_type=rnn_type,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            rnn_dropout=rnn_dropout,
            bidirectional=bidirectional,
            use_hidden_state=use_hidden_state,
            padding_idx=padding_idx,
            head_hidden_dims=head_hidden_dims,
            head_activation=head_activation,
            head_dropout=head_dropout,
            head_batchnorm=head_batchnorm,
            head_batchnorm_last=head_batchnorm_last,
            head_linear_first=head_linear_first,
        )

        # Embeddings and RNN defined in the BasicRNN inherited class

        # Attention
        self.attn_concatenate = attn_concatenate
        self.attn_dropout = attn_dropout

        if bidirectional and attn_concatenate:
            attn_input_dim = hidden_dim * 4
        elif bidirectional or attn_concatenate:
            attn_input_dim = hidden_dim * 2
        else:
            attn_input_dim = hidden_dim
        self.attn = ContextAttention(attn_input_dim, attn_dropout, sum_along_seq=True)
        self.output_dim = attn_input_dim

        # FC-Head (Mlp)
        if self.head_hidden_dims is not None:
            head_hidden_dims = [self.output_dim] + head_hidden_dims
            self.rnn_mlp = MLP(
                head_hidden_dims,
                head_activation,
                head_dropout,
                head_batchnorm,
                head_batchnorm_last,
                head_linear_first,
            )
            self.output_dim = head_hidden_dims[-1]

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

        return self.attn(attn_inp)

    @property
    def attention_weights(self) -> List:
        r"""List with the attention weights

        The shape of the attention weights is:

        :math:`(N, S)`

        Where *N* is the batch size and *S* is the length of the sequence
        """
        return self.attn.attn_weights
