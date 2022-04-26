#


## AttentiveRNN
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/text/attentive_rnn.py/#L12)
```python 
AttentiveRNN(
   vocab_size: int, embed_dim: Optional[int] = None,
   embed_matrix: Optional[np.ndarray] = None, embed_trainable: bool = True,
   rnn_type: str = 'lstm', hidden_dim: int = 64, n_layers: int = 3,
   rnn_dropout: float = 0.1, bidirectional: bool = False, use_hidden_state: bool = True,
   padding_idx: int = 1, attn_concatenate: bool = True, attn_dropout: float = 0.1,
   head_hidden_dims: Optional[List[int]] = None, head_activation: str = 'relu',
   head_dropout: Optional[float] = None, head_batchnorm: bool = False,
   head_batchnorm_last: bool = False, head_linear_first: bool = False
)
```


---
Text classifier/regressor comprised by a stack of RNNs
(LSTMs or GRUs) plus an attention layer that can be used as the
``deeptext`` component of a Wide & Deep model or independently by
itself.

In addition, there is the option to add a Fully Connected (FC) set of dense
layers on top of attention layer

Parameters
----------
vocab_size: int
Number of words in the vocabulary
---
    used
    Pretrained word embeddings
    Boolean indicating if the pretrained embeddings are trainable
    String indicating the type of RNN to use. One of 'lstm' or 'gru'
    Hidden dim of the RNN
    Number of recurrent layers
    Dropout for each RNN layer except the last layer
    Boolean indicating whether the staked RNNs are bidirectional
    output as predicting features. Typically the former is used.
    word token. Therefore, the default value is set to 1.
    hidden state.
    Internal dropout for the attention mechanism
    List with the sizes of the dense layers in the head e.g: [128, 64]
    `'tanh'`, `'relu'`, `'leaky_relu'` and `'gelu'` are supported
    Dropout of the dense layers in the head
    the dense layers that form the `'rnn_mlp'`
    last of the dense layers in the head
    LIN -> ACT]``

Attributes
----------
    word embedding matrix
    Stack of RNNs
    ``head_layers_dim`` is not ``None``
    neccesary to build the ``WideDeep`` class

Example
--------

```python

>>> from pytorch_widedeep.models import AttentiveRNN
>>> X_text = torch.cat((torch.zeros([5,1]), torch.empty(5, 4).random_(1,4)), axis=1)
>>> model = AttentiveRNN(vocab_size=4, hidden_dim=4, n_layers=2, padding_idx=0, embed_dim=4)
>>> out = model(X_text)
```


**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/text/attentive_rnn.py/#L96)
```python
.__init__(
   vocab_size: int, embed_dim: Optional[int] = None,
   embed_matrix: Optional[np.ndarray] = None, embed_trainable: bool = True,
   rnn_type: str = 'lstm', hidden_dim: int = 64, n_layers: int = 3,
   rnn_dropout: float = 0.1, bidirectional: bool = False, use_hidden_state: bool = True,
   padding_idx: int = 1, attn_concatenate: bool = True, attn_dropout: float = 0.1,
   head_hidden_dims: Optional[List[int]] = None, head_activation: str = 'relu',
   head_dropout: Optional[float] = None, head_batchnorm: bool = False,
   head_batchnorm_last: bool = False, head_linear_first: bool = False
)
```


### .attention_weights
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/text/attentive_rnn.py/#L183)
```python
.attention_weights()
```

---
List with the attention weights

The shape of the attention weights is:

:math:`(N, S)`

Where *N* is the batch size and *S* is the length of the sequence

### ._process_rnn_outputs
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/text/attentive_rnn.py/#L166)
```python
._process_rnn_outputs(
   output: Tensor, hidden: Tensor
)
```

