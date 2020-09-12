import string
from copy import deepcopy as c

import numpy as np
import torch
import pytest

from pytorch_widedeep.models import (
    Wide,
    DeepText,
    WideDeep,
    DeepDense,
    DeepImage,
)
from pytorch_widedeep.initializers import (
    Normal,
    Uniform,
    Orthogonal,
    XavierNormal,
    KaimingNormal,
    XavierUniform,
    KaimingUniform,
    ConstantInitializer,
)

# Wide array
X_wide = np.random.choice(50, (100, 100))

# Deep Array
colnames = list(string.ascii_lowercase)[:10]
embed_cols = [np.random.choice(np.arange(5), 100) for _ in range(5)]
embed_input = [(u, i, j) for u, i, j in zip(colnames[:5], [5] * 5, [16] * 5)]
cont_cols = [np.random.rand(100) for _ in range(5)]
deep_column_idx = {k: v for v, k in enumerate(colnames)}
X_deep = np.vstack(embed_cols + cont_cols).transpose()

#  Text Array
padded_sequences = np.random.choice(np.arange(1, 100), (100, 48))
vocab_size = 1000
X_text = np.hstack((np.repeat(np.array([[0, 0]]), 100, axis=0), padded_sequences))

#  Image Array
X_img = np.random.choice(256, (100, 224, 224, 3))


###############################################################################
# Simply Testing that "something happens (i.e. in!=out)" since the stochastic
# Nature of the most initializers does not allow for more
###############################################################################
initializers_1 = {
    "wide": XavierNormal,
    "deepdense": XavierUniform,
    "deeptext": KaimingNormal,
    "deepimage": KaimingUniform,
}

initializers_2 = {
    "wide": Normal,
    "deepdense": Uniform,
    "deeptext": ConstantInitializer(value=1.0),
    "deepimage": Orthogonal,
}

test_layers = [
    "wide.wlinear.weight",
    "deepdense.dense.dense_layer_1.0.weight",
    "deeptext.rnn.weight_hh_l1",
    "deepimage.dilinear.0.weight",
]


@pytest.mark.parametrize(
    "initializers, test_layers",
    [
        (initializers_1, test_layers),
        (initializers_2, test_layers),
    ],
)
def test_initializers_1(initializers, test_layers):

    wide = Wide(np.unique(X_wide).shape[0], 1)
    deepdense = DeepDense(
        hidden_layers=[32, 16],
        dropout=[0.5, 0.5],
        deep_column_idx=deep_column_idx,
        embed_input=embed_input,
        continuous_cols=colnames[-5:],
    )
    deeptext = DeepText(vocab_size=vocab_size, embed_dim=32, padding_idx=0)
    deepimage = DeepImage(pretrained=True)
    model = WideDeep(
        wide=wide,
        deepdense=deepdense,
        deeptext=deeptext,
        deepimage=deepimage,
        pred_dim=1,
    )
    cmodel = c(model)

    org_weights = []
    for n, p in cmodel.named_parameters():
        if n in test_layers:
            org_weights.append(p)

    model.compile(method="binary", verbose=0, initializers=initializers)
    init_weights = []
    for n, p in model.named_parameters():
        if n in test_layers:
            init_weights.append(p)

    res = all(
        [
            torch.all((1 - (a == b).int()).bool())
            for a, b in zip(org_weights, init_weights)
        ]
    )
    assert res


###############################################################################
# Make Sure that the "patterns" parameter works
###############################################################################
initializers_2 = {
    "wide": XavierNormal,
    "deepdense": XavierUniform,
    "deeptext": KaimingNormal(pattern=r"^(?!.*word_embed).*$"),  # type: ignore[dict-item]
}


def test_initializers_with_pattern():

    wide = Wide(100, 1)
    deepdense = DeepDense(
        hidden_layers=[32, 16],
        dropout=[0.5, 0.5],
        deep_column_idx=deep_column_idx,
        embed_input=embed_input,
        continuous_cols=colnames[-5:],
    )
    deeptext = DeepText(vocab_size=vocab_size, embed_dim=32, padding_idx=0)
    model = WideDeep(wide=wide, deepdense=deepdense, deeptext=deeptext, pred_dim=1)
    cmodel = c(model)
    org_word_embed = []
    for n, p in cmodel.named_parameters():
        if "word_embed" in n:
            org_word_embed.append(p)
    model.compile(method="binary", verbose=0, initializers=initializers_2)
    init_word_embed = []
    for n, p in model.named_parameters():
        if "word_embed" in n:
            init_word_embed.append(p)

    assert torch.all(org_word_embed[0] == init_word_embed[0].cpu())
