#


## StackedAttentiveRNN
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/text/stacked_attentive_rnn.py/#L12)
```python 
StackedAttentiveRNN(
   vocab_size: int, embed_dim: Optional[int] = None,
   embed_matrix: Optional[np.ndarray] = None, embed_trainable: bool = True,
   rnn_type: str = 'lstm', hidden_dim: int = 64, bidirectional: bool = False,
   padding_idx: int = 1, n_blocks: int = 3, attn_concatenate: bool = False,
   attn_dropout: float = 0.1, with_addnorm: bool = False,
   head_hidden_dims: Optional[List[int]] = None, head_activation: str = 'relu',
   head_dropout: Optional[float] = None, head_batchnorm: bool = False,
   head_batchnorm_last: bool = False, head_linear_first: bool = False
)
```


---
Text classifier/regressor comprised by a stack of blocks:
``[RNN + Attention]``. This can be used as the ``deeptext`` component of a
Wide & Deep model or independently by itself.

In addition, there is the option to add a Fully Connected (FC) set of
dense layers on top of the attentiob blocks

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
    Boolean indicating whether the staked RNNs are bidirectional
    word token. Therefore, the default value is set to 1.
    Context Attention Encoder
    hidden state or simply
    Internal dropout for the attention mechanism
    input and normalised
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

>>> from pytorch_widedeep.models import StackedAttentiveRNN
>>> X_text = torch.cat((torch.zeros([5,1]), torch.empty(5, 4).random_(1,4)), axis=1)
>>> model = StackedAttentiveRNN(vocab_size=4, hidden_dim=4, padding_idx=0, embed_dim=4)
>>> out = model(X_text)
```


**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/text/stacked_attentive_rnn.py/#L94)
```python
.__init__(
   vocab_size: int, embed_dim: Optional[int] = None,
   embed_matrix: Optional[np.ndarray] = None, embed_trainable: bool = True,
   rnn_type: str = 'lstm', hidden_dim: int = 64, bidirectional: bool = False,
   padding_idx: int = 1, n_blocks: int = 3, attn_concatenate: bool = False,
   attn_dropout: float = 0.1, with_addnorm: bool = False,
   head_hidden_dims: Optional[List[int]] = None, head_activation: str = 'relu',
   head_dropout: Optional[float] = None, head_batchnorm: bool = False,
   head_batchnorm_last: bool = False, head_linear_first: bool = False
)
```


### ._set_embeddings
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/text/stacked_attentive_rnn.py/#L251)
```python
._set_embeddings(
   embed_matrix: Union[Any, np.ndarray]
)
```


### .attention_weights
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/text/stacked_attentive_rnn.py/#L240)
```python
.attention_weights()
```

---
List with the attention weights

The shape of the attention weights is:

:math:`(N, S)`

Where *N* is the batch size and *S* is the length of the sequence

### .forward
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/text/stacked_attentive_rnn.py/#L219)
```python
.forward(
   X: Tensor
)
```

