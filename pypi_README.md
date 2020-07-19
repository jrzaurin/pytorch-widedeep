# pytorch-widedeep

A flexible package to combine tabular data with text and images using wide and
deep models.

**Documentation:** [https://pytorch-widedeep.readthedocs.io](https://pytorch-widedeep.readthedocs.io/en/latest/index.html)

### Introduction

`pytorch-widedeep` is based on Google's Wide and Deep Algorithm. Details of
the original algorithm can be found
[here](https://www.tensorflow.org/tutorials/wide_and_deep), and the nice
research paper can be found [here](https://arxiv.org/abs/1606.07792).

In general terms, `pytorch-widedeep` is a package to use deep learning with
tabular data. In particular, is intended to facilitate the combination of text
and images with corresponding tabular data using wide and deep models. With
that in mind there are two architectures that can be implemented with just a
few lines of code. For details on these architectures please visit the
[repo](https://github.com/jrzaurin/pytorch-widedeep).

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

### Examples

There are a number of notebooks in the `examples` folder plus some additional
files. These notebooks cover most of the utilities of this package and can
also act as documentation. In the case that github does not render the
notebooks, or it renders them missing some parts, they are saved as markdown
files in the `docs` folder.

### Quick start

Binary classification with the [adult
dataset]([adult](https://www.kaggle.com/wenruliu/adult-income-dataset))
using `Wide` and `DeepDense` and defaults settings.

````python
import pandas as pd
from sklearn.model_selection import train_test_split

from pytorch_widedeep.preprocessing import WidePreprocessor, DensePreprocessor
from pytorch_widedeep.models import Wide, DeepDense, WideDeep
from pytorch_widedeep.metrics import Accuracy

# these next 4 lines are not directly related to pytorch-widedeep. I assume
# you have downloaded the dataset and place it in a dir called data/adult/
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
preprocess_wide = WidePreprocessor(wide_cols=wide_cols, crossed_cols=cross_cols)
X_wide = preprocess_wide.fit_transform(df_train)
wide = Wide(wide_dim=X_wide.shape[1], pred_dim=1)

# deepdense
preprocess_deep = DensePreprocessor(embed_cols=embed_cols, continuous_cols=cont_cols)
X_deep = preprocess_deep.fit_transform(df_train)
deepdense = DeepDense(
    hidden_layers=[64, 32],
    deep_column_idx=preprocess_deep.deep_column_idx,
    embed_input=preprocess_deep.embeddings_input,
    continuous_cols=cont_cols,
)

# build, compile and fit
model = WideDeep(wide=wide, deepdense=deepdense)
model.compile(method="binary", metrics=[Accuracy])
model.fit(
    X_wide=X_wide,
    X_deep=X_deep,
    target=target,
    n_epochs=5,
    batch_size=256,
    val_split=0.1,
)

# predict
X_wide_te = preprocess_wide.transform(df_test)
X_deep_te = preprocess_deep.transform(df_test)
preds = model.predict(X_wide=X_wide_te, X_deep=X_deep_te)
```

Of course, one can do much more, such as using different initializations,
optimizers or learning rate schedulers for each component of the overall
model. Adding FC-Heads to the Text and Image components. Using the [Focal
Loss](https://arxiv.org/abs/1708.02002), warming up individual components
before joined training, etc. See the `examples` or the `docs` folders for a
better understanding of the content of the package and its functionalities.

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