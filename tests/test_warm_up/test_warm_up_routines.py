import string

import numpy as np
import torch
import pytest
import torch.nn.functional as F
from torch import nn
from sklearn.utils import Bunch
from torch.utils.data import Dataset, DataLoader

from pytorch_widedeep.models import Wide, DeepDense
from pytorch_widedeep.metrics import Accuracy
from pytorch_widedeep.models._warmup import WarmUp
from pytorch_widedeep.models.deep_image import conv_layer

use_cuda = torch.cuda.is_available()


# Define a series of simple models to quickly test the WarmUp class
class TestDeepText(nn.Module):
    def __init__(self):
        super(TestDeepText, self).__init__()
        self.word_embed = nn.Embedding(5, 16, padding_idx=0)
        self.rnn = nn.LSTM(16, 8, batch_first=True)
        self.linear = nn.Linear(8, 1)

    def forward(self, X):
        embed = self.word_embed(X.long())
        o, (h, c) = self.rnn(embed)
        return self.linear(h).view(-1, 1)


class TestDeepImage(nn.Module):
    def __init__(self):
        super(TestDeepImage, self).__init__()

        self.conv_block = nn.Sequential(
            conv_layer(3, 64, 3),
            conv_layer(64, 128, 1, maxpool=False, adaptiveavgpool=True),
        )
        self.linear = nn.Linear(128, 1)

    def forward(self, X):
        x = self.conv_block(X)
        x = x.view(x.size(0), -1)
        return self.linear(x)


#  Define a simple WideDeep Dataset
class WDset(Dataset):
    def __init__(self, X_wide, X_deep, X_text, X_img, target):

        self.X_wide = X_wide
        self.X_deep = X_deep
        self.X_text = X_text
        self.X_img = X_img
        self.Y = target

    def __getitem__(self, idx: int):

        X = Bunch(wide=self.X_wide[idx])
        X.deepdense = self.X_deep[idx]
        X.deeptext = self.X_text[idx]
        X.deepimage = self.X_img[idx]
        y = self.Y[idx]
        return X, y

    def __len__(self):
        return len(self.X_deep)


# Remember that the WarmUp class will be instantiated inside the WideDeep and
# will take, among others, the activation_fn and the loss_fn of that class as
# parameters. Therefore, we define equivalent classes to replicate the
# scenario
# def activ_fn(inp):
#     return torch.sigmoid(inp)


def loss_fn(y_pred, y_true):
    return F.binary_cross_entropy_with_logits(y_pred, y_true.view(-1, 1))


#  Define the data components:

# target
target = torch.empty(100, 1).random_(0, 2)

# wide
X_wide = torch.empty(100, 4).random_(1, 20)

# deep
colnames = list(string.ascii_lowercase)[:10]
embed_cols = [np.random.choice(np.arange(5), 100) for _ in range(5)]
cont_cols = [np.random.rand(100) for _ in range(5)]
embed_input = [(u, i, j) for u, i, j in zip(colnames[:5], [5] * 5, [16] * 5)]
deep_column_idx = {k: v for v, k in enumerate(colnames[:10])}
continuous_cols = colnames[-5:]
X_deep = torch.from_numpy(np.vstack(embed_cols + cont_cols).transpose())

# text
X_text = torch.cat((torch.zeros([100, 1]), torch.empty(100, 4).random_(1, 4)), axis=1)  # type: ignore[call-overload]

# image
X_image = torch.rand(100, 3, 28, 28)

# Define the model components

# wide
wide = Wide(X_wide.unique().size(0), 1)
if use_cuda:
    wide.cuda()

# deep
deepdense = DeepDense(
    hidden_layers=[16, 8],
    dropout=[0.5, 0.2],
    deep_column_idx=deep_column_idx,
    embed_input=embed_input,
    continuous_cols=continuous_cols,
)
deepdense = nn.Sequential(deepdense, nn.Linear(8, 1))  # type: ignore[assignment]
if use_cuda:
    deepdense.cuda()

# text
deeptext = TestDeepText()
if use_cuda:
    deeptext.cuda()

# image
deepimage = TestDeepImage()
if use_cuda:
    deepimage.cuda()

# Define the loader
wdset = WDset(X_wide, X_deep, X_text, X_image, target)
wdloader = DataLoader(wdset, batch_size=10, shuffle=True)

# Instantiate the WarmUp class
warmer = WarmUp(loss_fn, Accuracy(), "binary", False)

# List the layers for the warm_gradual method
text_layers = [c for c in list(deeptext.children())[1:]][::-1]
image_layers = [c for c in list(deepimage.children())][::-1]


###############################################################################
#  Simply test that warm_all runs
###############################################################################
@pytest.mark.parametrize(
    "model, modelname, loader, n_epochs, max_lr",
    [
        (wide, "wide", wdloader, 1, 0.01),
        (deepdense, "deepdense", wdloader, 1, 0.01),
        (deeptext, "deeptext", wdloader, 1, 0.01),
        (deepimage, "deepimage", wdloader, 1, 0.01),
    ],
)
def test_warm_all(model, modelname, loader, n_epochs, max_lr):
    has_run = True
    try:
        warmer.warm_all(model, modelname, loader, n_epochs, max_lr)
    except:
        has_run = False
    assert has_run


###############################################################################
#  Simply test that warm_gradual runs
###############################################################################
@pytest.mark.parametrize(
    "model, modelname, loader, max_lr, layers, routine",
    [
        (deeptext, "deeptext", wdloader, 0.01, text_layers, "felbo"),
        (deeptext, "deeptext", wdloader, 0.01, text_layers, "howard"),
        (deepimage, "deepimage", wdloader, 0.01, image_layers, "felbo"),
        (deepimage, "deepimage", wdloader, 0.01, image_layers, "howard"),
    ],
)
def test_warm_gradual(model, modelname, loader, max_lr, layers, routine):
    has_run = True
    try:
        warmer.warm_gradual(model, modelname, loader, max_lr, layers, routine)
    except:
        has_run = False
    assert has_run
