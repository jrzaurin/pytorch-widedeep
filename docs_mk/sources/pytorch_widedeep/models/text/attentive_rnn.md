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


**Args**

* **vocab_size** (int) : Number of words in the vocabulary
* **embed_dim** (int, Optional, default = None) : Dimension of the word embeddings if non-pretained word vectors are
    used
* **embed_matrix** (np.ndarray, Optional, default = None) : Pretrained word embeddings
* **embed_trainable** (bool, default = True) : Boolean indicating if the pretrained embeddings are trainable
* **rnn_type** (str, default = 'lstm') : String indicating the type of RNN to use. One of 'lstm' or 'gru'
* **hidden_dim** (int, default = 64) : Hidden dim of the RNN
* **n_layers** (int, default = 3) : Number of recurrent layers
* **rnn_dropout** (float, default = 0.1) : Dropout for each RNN layer except the last layer
* **bidirectional** (bool, default = True) : Boolean indicating whether the staked RNNs are bidirectional
* **use_hidden_state** (str, default = True) : Boolean indicating whether to use the final hidden state or the RNN's
    output as predicting features. Typically the former is used.
* **padding_idx** (int, default = 1) : index of the padding token in the padded-tokenised sequences. The
    ``TextPreprocessor`` class within this library uses ``fastai``'s
    tokenizer where the token index 0 is reserved for the `'unknown'`
    word token. Therefore, the default value is set to 1.
* **attn_concatenate** (bool, default = True) : Boolean indicating if the input to the attention mechanism will be the
    output of the RNN or the output of the RNN concatenated with the last
    hidden state.
* **attn_dropout** (float, default = 0.1) : Internal dropout for the attention mechanism
* **head_hidden_dims** (List, Optional, default = None) : List with the sizes of the dense layers in the head e.g: [128, 64]
* **head_activation** (str, default = "relu") : Activation function for the dense layers in the head. Currently
    `'tanh'`, `'relu'`, `'leaky_relu'` and `'gelu'` are supported
* **head_dropout** (float, Optional, default = None) : Dropout of the dense layers in the head
* **head_batchnorm** (bool, default = False) : Boolean indicating whether or not to include batch normalization in
    the dense layers that form the `'rnn_mlp'`
* **head_batchnorm_last** (bool, default = False) : Boolean indicating whether or not to apply batch normalization to the
    last of the dense layers in the head
* **head_linear_first** (bool, default = False) : Boolean indicating whether the order of the operations in the dense
    layer. If ``True: [LIN -> ACT -> BN -> DP]``. If ``False: [BN -> DP ->
    LIN -> ACT]``


**Attributes**

* **word_embed** (nn.Module) : word embedding matrix
* **rnn** (nn.Module) : Stack of RNNs
* **rnn_mlp** (nn.Sequential) : Stack of dense layers on top of the RNN. This will only exists if
    ``head_layers_dim`` is not ``None``
* **output_dim** (int) : The output dimension of the model. This is a required attribute
    neccesary to build the ``WideDeep`` class


**Example**


```python

>>> from pytorch_widedeep.models import AttentiveRNN
>>> X_text = torch.cat((torch.zeros([5,1]), torch.empty(5, 4).random_(1,4)), axis=1)
>>> model = AttentiveRNN(vocab_size=4, hidden_dim=4, n_layers=2, padding_idx=0, embed_dim=4)
>>> out = model(X_text)
```


**Methods:**


### .attention_weights
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/text/attentive_rnn.py/#L157)
```python
.attention_weights()
```

---
List with the attention weights

The shape of the attention weights is:

:math:`(N, S)`

Where *N* is the batch size and *S* is the length of the sequence
