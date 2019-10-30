
<p align="center">
  <img width="450" src="docs/figures/widedeep_logo.png">
</p>

# pytorch-widedeep

A flexible package to combine tabular data with text and images using wide and
deep models.

### Introduction

`pytorch-widedeep` is based on Google's Wide and Deep Algorithm. Details of
the original algorithm can be found
[here](https://www.tensorflow.org/tutorials/wide_and_deep), and the nice
research paper can be found [here](https://arxiv.org/abs/1606.07792).

In general terms, `pytorch-widedeep` is a package to use deep learning with
tabular data. In particular, is intended to facilitate the combination of text
and images with corresponding tabular data using wide and deep models. With
that in mind there are two architectures that can be implemented with just a
few lines of code.

### Architectures

**Architecture 1**:

<p align="center">
  <img width="600" src="docs/figures/architecture_1.png">
</p>

Architecture 1 combines the `Wide`, one-hot encoded features with the outputs
from the `DeepDense`, `DeepText` and `DeepImage` components connected to a
final output neuron or neurons, depending on whether we are performing a
binary classification or regression, or a multi-class classification. The
components within the faded-pink rectangles are concatenated.

In math terms, and following the notation in the [paper](https://arxiv.org/abs/1606.07792), Architecture 1 can be formulated as:

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


**Architecture 2**

<p align="center">
  <img width="600" src="docs/figures/architecture_2.png">
</p>

Architecture 2 combines the `Wide` one-hot encoded features with the Deep
components of the model connected to the output neuron(s), after the different
Deep components have been themselves combined through a FC-Head (that I refer
as `deephead`).

In math terms, and following the notation in the
[paper](https://arxiv.org/abs/1606.07792), Architecture 2 can be formulated
as:

<p align="center">
  <img width="300" src="docs/figures/architecture_2_math.png">
</p>

When using `pytorch-widedeep`, the assumption is that the so called `Wide` and
`DeepDense` components in the figures are **always** present, while `DeepText`
and `DeepImage` are optional. `pytorch-widedeep` includes standard text (stack
of LSTMs) and image (pre-trained ResNets or stack of CNNs) models. However,
the user can use any custom model as long as it has an attribute called
`output_dim` with the size of the last layer of activations, so that
`WideDeep` can be constructed. See the examples folder for more information.


### Installation

#### OSX and Ubuntu (with and without CUDA)

Install directly from github

```bash
pip install git+https://github.com/jrzaurin/pytorch-widedeep.git
```

#### If you are using Conda with CUDA

The `Pytorch` installation command with `pip` is different than that of
`conda` with CUDA. The later installs the CUDA toolkit, see
[here](https://pytorch.org/). Therefore, if you are using `conda` and have
already installed `torch` and `torvision`, or do not want to use `pip`, I
recommend cloning the directory, removing the `torch` and `torchvision`
dependencies from the `setup.py` file and then `pip install .`:

```bash
# Clone the repository
git clone https://github.com/jrzaurin/pytorch-widedeep
cd pytorch-widedeep

# remove torch and torchvision dependencies from setup.py and the run:
pip install .

# or dev mode
pip install -e .
```

Note that installing `pytorch-widedeep` directly from github would still work
(moreover if you do not have CUDA), but the CUDA toolkit is recommended for a
more efficient installation/performance.


### Examples

There are 4 main notebooks in the `examples` folder plus some additional
files. These notebooks cover most of the utilities of this package and can
also act as documentation. In the likely case that github does not render the
notebooks, or it renders them missing some parts, they are saved as markdown
files in the docs folder.

### Quick start

Binary classification with the [adult
dataset]([adult](https://www.kaggle.com/wenruliu/adult-income-dataset/downloads/adult.csv/2))
using `Wide` and `DeepDense` and defaults settings.

```python
import pandas as pd
from pytorch_widedeep.preprocessing import WidePreprocessor, DeepPreprocessor
from pytorch_widedeep.models import Wide, DeepDense, WideDeep
from pytorch_widedeep.metrics import BinaryAccuracy

# these next 3 lines are not directly related to pytorch-widedeep. I assume
# you have downloaded the dataset and place it in a dir called data/adult/
df = pd.read_csv('data/adult/adult.csv.zip')
df['income_label'] = (df["income"].apply(lambda x: ">50K" in x)).astype(int)
df.drop('income', axis=1, inplace=True)

# prepare wide, crossed, embedding and continuous columns
wide_cols  = ['education', 'relationship', 'workclass', 'occupation','native-country', 'gender']
cross_cols = [('education', 'occupation'), ('native-country', 'occupation')]
embed_cols = [('education',16), ('workclass',16), ('occupation',16),('native-country',32)]
cont_cols  = ["age", "hours-per-week"]
target_col = 'income_label'

# target
target = df[target_col].values

# wide
preprocess_wide = WidePreprocessor(wide_cols=wide_cols, crossed_cols=cross_cols)
X_wide = preprocess_wide.fit_transform(df)
wide = Wide(wide_dim=X_wide.shape[1], output_dim=1)

# deepdense
preprocess_deep = DeepPreprocessor(embed_cols=embed_cols, continuous_cols=cont_cols)
X_deep = preprocess_deep.fit_transform(df)
deepdense = DeepDense(hidden_layers=[64,32],
                      deep_column_idx=preprocess_deep.deep_column_idx,
                      embed_input=preprocess_deep.embeddings_input,
                      continuous_cols=cont_cols)

# build, compile, fit and predict
model = WideDeep(wide=wide, deepdense=deepdense)
model.compile(method='binary', metrics=[BinaryAccuracy])
model.fit(X_wide=X_wide, X_deep=X_deep, target=target, n_epochs=5, batch_size=256, val_split=0.2)
model.predict(X_wide=X_wide_te, X_deep=X_deep_te)
```

Of course, one can do much more, such as using different initializations,
optimizers or learning rate schedulers for each component of the overall
model. Adding FC-Heads to the Text and Image components, etc. See the
`examples` folder for a better understanding of the content of the package and
its functionalities.

### Testing

```
cd test
pytest --ignore=test_data_utils/test_du_deep_image.py
cd test_data_utils
pytest test_du_deep_image.py
```