#


## DataLoaderImbalanced
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/dataloaders.py/#L35)
```python 
DataLoaderImbalanced(
   dataset: WideDeepDataset, batch_size: int, num_workers: int, **kwargs
)
```


---
Class to load and shuffle batches with adjusted weights for imbalanced
datasets. If the classes do not begin from 0 remapping is necessary. See
`here
<https://towardsdatascience.com/pytorch-tabular-multiclass-classification-9f8211a123ab>`_
.

Parameters
----------
dataset: ``WideDeepDataset``
see ``pytorch_widedeep.training._wd_dataset``
---
    size of batch
    number of workers


**Methods:**


### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/dataloaders.py/#L52)
```python
.__init__(
   dataset: WideDeepDataset, batch_size: int, num_workers: int, **kwargs
)
```


----


### get_class_weights
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/dataloaders.py/#L8)
```python
.get_class_weights(
   dataset: WideDeepDataset
)
```

---
Helper function to get weights of classes in the imbalanced dataset.


**Args**

* **dataset** (WideDeepDataset) : dataset containing target classes in dataset.Y


**Returns**

* **weights** (array) : numpy array with weights
* **minor_class_count** (int) : count of samples in the smallest class for undersampling
* **num_classes** (int) : number of classes

