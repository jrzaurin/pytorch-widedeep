#


## Vocab
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/fastai_transforms.py/#L314)
```python 
Vocab(
   itos: Collection[str]
)
```


---
Contains the correspondence between numbers and tokens.

Parameters
----------
itos: Collection
`index to str`. Collection of strings that are the tokens of the
vocabulary

---
Attributes
----------
    their corresponding index


**Methods:**


### .textify
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/fastai_transforms.py/#L338)
```python
.textify(
   nums: Collection[int], sep = ''
)
```

---
Convert a list of ``nums`` (or indexes) to their tokens.

### .__init__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/fastai_transforms.py/#L330)
```python
.__init__(
   itos: Collection[str]
)
```


### .save
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/fastai_transforms.py/#L349)
```python
.save(
   path
)
```

---
Save the  attribute ``self.itos`` in ``path``

### .__setstate__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/fastai_transforms.py/#L345)
```python
.__setstate__(
   state: dict
)
```


### .create
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/fastai_transforms.py/#L354)
```python
.create(
   cls, tokens: Tokens, max_vocab: int, min_freq: int
)
```

---
Create a vocabulary object from a set of tokens.

Parameters
----------
tokens: Tokens
Custom type: ``Collection[Collection[str]]``  see
:obj:`pytorch_widedeep.wdtypes`. Collection of collection of
strings(e.g. list of tokenized sentences)
---
    maximum vocabulary size
    vocabulary

Examples
--------

```python

>>> texts = ['Machine learning is great', 'but building stuff is even better']
>>> tokens = Tokenizer().process_all(texts)
>>> vocab = Vocab.create(tokens, max_vocab=18, min_freq=1)
>>> vocab.numericalize(['machine', 'learning', 'is', 'great'])
[10, 11, 9, 12]
>>> vocab.textify([10, 11, 9, 12])
'machine learning is great'

```
    `docs <https://docs.fast.ai/text.transform.html#Tokenizer>`_.

### .__getstate__
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/fastai_transforms.py/#L342)
```python
.__getstate__()
```


### .load
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/fastai_transforms.py/#L400)
```python
.load(
   cls, path
)
```

---
Load an intance of :obj:`Vocab` contained in ``path``

### .numericalize
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/fastai_transforms.py/#L334)
```python
.numericalize(
   t: Collection[str]
)
```

---
Convert a list of tokens ``t`` to their ids.

----


### partition
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/fastai_transforms.py/#L23)
```python
.partition(
   a: Collection, sz: int
)
```

---
Split iterables `a` in equal parts of size `sz`

----


### partition_by_cores
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/fastai_transforms.py/#L28)
```python
.partition_by_cores(
   a: Collection, n_cpus: int
)
```

---
Split data in `a` equally among `n_cpus` cores

----


### ifnone
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/fastai_transforms.py/#L33)
```python
.ifnone(
   a: Any, b: Any
)
```

---
`a` if `a` is not None, otherwise `b`.

----


### num_cpus
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/fastai_transforms.py/#L38)
```python
.num_cpus()
```

---
Get number of cpus

----


### spec_add_spaces
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/fastai_transforms.py/#L127)
```python
.spec_add_spaces(
   t: str
)
```

---
Add spaces around / and # in `t`. 

----


### rm_useless_spaces
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/fastai_transforms.py/#L132)
```python
.rm_useless_spaces(
   t: str
)
```

---
Remove multiple spaces in `t`.

----


### replace_rep
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/fastai_transforms.py/#L137)
```python
.replace_rep(
   t: str
)
```

---
Replace repetitions at the character level in `t`.

----


### replace_wrep
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/fastai_transforms.py/#L148)
```python
.replace_wrep(
   t: str
)
```

---
Replace word repetitions in `t`.

----


### fix_html
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/fastai_transforms.py/#L159)
```python
.fix_html(
   x: str
)
```

---
List of replacements from html strings in `x`.

----


### replace_all_caps
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/fastai_transforms.py/#L181)
```python
.replace_all_caps(
   x: Collection[str]
)
```

---
Replace tokens in ALL CAPS in `x` by their lower version and add `TK_UP` before.

----


### deal_caps
[source](https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/utils/fastai_transforms.py/#L193)
```python
.deal_caps(
   x: Collection[str]
)
```

---
Replace all Capitalized tokens in `x` by their lower version and add `TK_MAJ` before.
