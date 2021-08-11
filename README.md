
<p align="center">
  <img width="300" src="docs/figures/widedeep_logo.png">
</p>

[![Build Status](https://travis-ci.org/jrzaurin/pytorch-widedeep.svg?branch=master)](https://travis-ci.org/jrzaurin/pytorch-widedeep)
[![Documentation Status](https://readthedocs.org/projects/pytorch-widedeep/badge/?version=latest)](https://pytorch-widedeep.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/pytorch-widedeep.svg)](https://badge.fury.io/py/pytorch-widedeep)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/jrzaurin/pytorch-widedeep/graphs/commit-activity)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/jrzaurin/pytorch-widedeep/issues)
[![codecov](https://codecov.io/gh/jrzaurin/pytorch-widedeep/branch/master/graph/badge.svg)](https://codecov.io/gh/jrzaurin/pytorch-widedeep)
[![Python 3.6 3.7 3.8 3.9](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue.svg)](https://www.python.org/)

# pytorch-widedeep

A flexible package to use Deep Learning with tabular data, text and images
using wide and deep models.

**Documentation:** [https://pytorch-widedeep.readthedocs.io](https://pytorch-widedeep.readthedocs.io/en/latest/index.html)

**Companion posts and tutorials:** [infinitoml](https://jrzaurin.github.io/infinitoml/)

**Experiments and comparisson with `LightGBM`**: [TabularDL vs LightGBM](https://github.com/jrzaurin/tabulardl-benchmark)

**Slack**: if you want to contribute or just want to chat with us, join [slack](https://join.slack.com/t/pytorch-widedeep/shared_invite/zt-soss7stf-iXpVuLeKZz8lGTnxxtHtTw)

### Introduction

``pytorch-widedeep`` is based on Google's [Wide and Deep Algorithm](https://arxiv.org/abs/1606.07792)

In general terms, `pytorch-widedeep` is a package to use deep learning with
tabular data. In particular, is intended to facilitate the combination of text
and images with corresponding tabular data using wide and deep models. With
that in mind there are a number of architectures that can be implemented with
just a few lines of code. The main components of those architectures are shown
in the Figure below:


<p align="center">
  <img width="750" src="docs/figures/widedeep_arch.png">
</p>

The dashed boxes in the figure represent optional, overall components, and the
dashed lines/arrows indicate the corresponding connections, depending on
whether or not certain components are present. For example, the dashed,
blue-lines indicate that the ``deeptabular``, ``deeptext`` and ``deepimage``
components are connected directly to the output neuron or neurons (depending
on whether we are performing a binary classification or regression, or a
multi-class classification) if the optional ``deephead`` is not present.
Finally, the components within the faded-pink rectangle are concatenated.

Note that it is not possible to illustrate the number of possible
architectures and components available in ``pytorch-widedeep`` in one Figure.
Therefore, for more details on possible architectures (and more) please, see
the
[documentation]((https://pytorch-widedeep.readthedocs.io/en/latest/index.html)),
or the Examples folders and the notebooks there.

In math terms, and following the notation in the
[paper](https://arxiv.org/abs/1606.07792), the expression for the architecture
without a ``deephead`` component can be formulated as:

<p align="center">
  <img width="500" src="docs/figures/architecture_1_math.png">
</p>


Where *'W'* are the weight matrices applied to the wide model and to the final
activations of the deep models, *'a'* are these final activations, and
&phi;(x) are the cross product transformations of the original features *'x'*.
In case you are wondering what are *"cross product transformations"*, here is
a quote taken directly from the paper: *"For binary features, a cross-product
transformation (e.g., “AND(gender=female, language=en)”) is 1 if and only if
the constituent features (“gender=female” and “language=en”) are all 1, and 0
otherwise".*


While if there is a ``deephead`` component, the previous expression turns
into:

<p align="center">
  <img width="300" src="docs/figures/architecture_2_math.png">
</p>

It is important to emphasize that **each individual component, `wide`,
`deeptabular`, `deeptext` and `deepimage`, can be used independently** and in
isolation. For example, one could use only `wide`, which is in simply a
linear model. In fact, one of the most interesting functionalities
in``pytorch-widedeep`` is the ``deeptabular`` component. Currently,
``pytorch-widedeep`` offers the following different models for that
component:

1. ``TabMlp``: this is almost identical to the [tabular
model](https://docs.fast.ai/tutorial.tabular.html) in the fantastic
[fastai](https://docs.fast.ai/) library, and consists simply in embeddings
representing the categorical features, concatenated with the continuous
features, and passed then through a MLP.

2. ``TabRenset``: This is similar to the previous model but the embeddings are
passed through a series of ResNet blocks built with dense layers.

3. ``Tabnet``: Details on TabNet can be found in:
[TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442)

4. ``TabTransformer``: Details on the TabTransformer can be found in:
[TabTransformer: Tabular Data Modeling Using Contextual Embeddings](https://arxiv.org/pdf/2012.06678.pdf).
Note that the TabTransformer implementation available at ``pytorch-widedeep``
is an adaptation of the original implementation.

5. ``FT-Transformer``: or Feature Tokenizer transformer. This is a relatively small
variation of the ``TabTransformer``. The variation itself was first
introduced in the ``SAINT`` paper, but the name "``FT-Transformer``" was first
used in
[Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959).
When using the ``FT-Transformer`` each continuous feature is "embedded"
(i.e. going through a 1-layer MLP with or without activation function) and
then passed through the attention blocks along with the categorical features.
This is available in ``pytorch-widedeep``'s ``TabTransformer`` by setting the
parameter ``embed_continuous = True``.


6. ``SAINT``: Details on SAINT can be found in:
[SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training](https://arxiv.org/abs/2106.01342).

For details on these models and their options please see the examples in the
Examples folder and the documentation.

Finally, while I recommend using the ``wide`` and ``deeptabular`` models in
``pytorch-widedeep`` it is very likely that users will want to use their own
models for the ``deeptext`` and ``deepimage`` components. That is perfectly
possible as long as the the custom models have an attribute called
``output_dim`` with the size of the last layer of activations, so that
``WideDeep`` can be constructed. Again, examples on how to use custom
components can be found in the Examples folder. Just in case
``pytorch-widedeep`` includes standard text (stack of LSTMs) and image
(pre-trained ResNets or stack of CNNs) models.


### Installation

Install using pip:

```bash
pip install pytorch-widedeep
```

Or install directly from github

```bash
pip install git+https://github.com/jrzaurin/pytorch-widedeep.git
```

#### Developer Install

```bash
# Clone the repository
git clone https://github.com/jrzaurin/pytorch-widedeep
cd pytorch-widedeep

# Install in dev mode
pip install -e .
```

**Important note for Mac users**: at the time of writing the latest `torch`
release is `1.9`. Some past [issues](https://stackoverflow.com/questions/64772335/pytorch-w-parallelnative-cpp206)
when running on Mac, present in previous versions, persist on this release
and the data-loaders will not run in parallel. In addition, since `python
3.8`, [the `multiprocessing` library start method changed from `'fork'` to`'spawn'`](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods).
This also affects the data-loaders (for any `torch` version) and they will
not run in parallel. Therefore, for Mac users I recommend using `python
3.6` or `3.7` and `torch <= 1.6` (with the corresponding, consistent
version of `torchvision`, e.g. `0.7.0` for `torch 1.6`). I do not want to
force this versioning in the `setup.py` file since I expect that all these
issues are fixed in the future. Therefore, after installing
`pytorch-widedeep` via pip or directly from github, downgrade `torch` and
`torchvision` manually:

```bash
pip install pytorch-widedeep
pip install torch==1.6.0 torchvision==0.7.0
```

None of these issues affect Linux users.

### Quick start

Binary classification with the [adult
dataset]([adult](https://www.kaggle.com/wenruliu/adult-income-dataset))
using `Wide` and `DeepDense` and defaults settings.

Building a wide (linear) and deep model with ``pytorch-widedeep``:

```python

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from pytorch_widedeep import Trainer
from pytorch_widedeep.preprocessing import WidePreprocessor, TabPreprocessor
from pytorch_widedeep.models import Wide, TabMlp, WideDeep
from pytorch_widedeep.metrics import Accuracy

# the following 4 lines are not directly related to ``pytorch-widedeep``. I
# assume you have downloaded the dataset and place it in a dir called
# data/adult/
df = pd.read_csv("data/adult/adult.csv.zip")
df["income_label"] = (df["income"].apply(lambda x: ">50K" in x)).astype(int)
df.drop("income", axis=1, inplace=True)
df_train, df_test = train_test_split(df, test_size=0.2, stratify=df.income_label)

# prepare wide, crossed, embedding and continuous columns
wide_cols = [
    "education",
    "relationship",
    "workclass",
    "occupation",
    "native-country",
    "gender",
]
cross_cols = [("education", "occupation"), ("native-country", "occupation")]
embed_cols = [
    ("education", 16),
    ("workclass", 16),
    ("occupation", 16),
    ("native-country", 32),
]
cont_cols = ["age", "hours-per-week"]
target_col = "income_label"

# target
target = df_train[target_col].values

# wide
wide_preprocessor = WidePreprocessor(wide_cols=wide_cols, crossed_cols=cross_cols)
X_wide = wide_preprocessor.fit_transform(df_train)
wide = Wide(wide_dim=np.unique(X_wide).shape[0], pred_dim=1)

# deeptabular
tab_preprocessor = TabPreprocessor(embed_cols=embed_cols, continuous_cols=cont_cols)
X_tab = tab_preprocessor.fit_transform(df_train)
deeptabular = TabMlp(
    mlp_hidden_dims=[64, 32],
    column_idx=tab_preprocessor.column_idx,
    embed_input=tab_preprocessor.embeddings_input,
    continuous_cols=cont_cols,
)

# wide and deep
model = WideDeep(wide=wide, deeptabular=deeptabular)

# train the model
trainer = Trainer(model, objective="binary", metrics=[Accuracy])
trainer.fit(
    X_wide=X_wide,
    X_tab=X_tab,
    target=target,
    n_epochs=5,
    batch_size=256,
    val_split=0.1,
)

# predict
X_wide_te = wide_preprocessor.transform(df_test)
X_tab_te = tab_preprocessor.transform(df_test)
preds = trainer.predict(X_wide=X_wide_te, X_tab=X_tab_te)

# Save and load

# Option 1: this will also save training history and lr history if the
# LRHistory callback is used
trainer.save(path="model_weights", save_state_dict=True)

# Option 2: save as any other torch model
torch.save(model.state_dict(), "model_weights/wd_model.pt")

# From here in advance, Option 1 or 2 are the same. I assume the user has
# prepared the data and defined the new model components:
# 1. Build the model
model_new = WideDeep(wide=wide, deeptabular=deeptabular)
model_new.load_state_dict(torch.load("model_weights/wd_model.pt"))

# 2. Instantiate the trainer
trainer_new = Trainer(
    model_new,
    objective="binary",
)

# 3. Either start the fit or directly predict
preds = trainer_new.predict(X_wide=X_wide, X_tab=X_tab)
```

Of course, one can do **much more**. See the Examples folder, the
documentation or the companion posts for a better understanding of the content
of the package and its functionalities.

### Testing

```
pytest tests
```

### Acknowledgments

This library takes from a series of other libraries, so I think it is just
fair to mention them here in the README (specific mentions are also included
in the code).

The `Callbacks` and `Initializers` structure and code is inspired by the
[`torchsample`](https://github.com/ncullen93/torchsample) library, which in
itself partially inspired by [`Keras`](https://keras.io/).

The `TextProcessor` class in this library uses the
[`fastai`](https://docs.fast.ai/text.transform.html#BaseTokenizer.tokenizer)'s
`Tokenizer` and `Vocab`. The code at `utils.fastai_transforms` is a minor
adaptation of their code so it functions within this library. To my experience
their `Tokenizer` is the best in class.

The `ImageProcessor` class in this library uses code from the fantastic [Deep
Learning for Computer
Vision](https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/)
(DL4CV) book by Adrian Rosebrock.