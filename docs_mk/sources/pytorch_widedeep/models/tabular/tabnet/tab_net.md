#


## TabNetPredLayer
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/tabnet/tab_net.py/#L204)
```python 
TabNetPredLayer(
   inp, out
)
```




**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/tabnet/tab_net.py/#L205)
```python
.__init__(
   inp, out
)
```

---
This class is a 'hack' required because TabNet is a very particular
model within ``WideDeep``.

TabNet's forward method within ``WideDeep`` outputs two tensors, one
with the last layer's activations and the sparse regularization
factor. Since the output needs to be collected by ``WideDeep`` to then
Sequentially build the output layer (connection to the output
neuron(s)) I need to code a custom TabNetPredLayer that accepts two
inputs. This will be used by the ``WideDeep`` class.

### .forward
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/tabnet/tab_net.py/#L220)
```python
.forward(
   tabnet_tuple: Tuple[Tensor, Tensor]
)
```

