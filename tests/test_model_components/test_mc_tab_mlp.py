import string

import numpy as np
import torch
import pytest

from pytorch_widedeep.models import TabMlp
from pytorch_widedeep.models.tab_mlp import CatEmbeddingsAndCont

colnames = list(string.ascii_lowercase)[:10]
embed_cols = [np.random.choice(np.arange(5), 10) for _ in range(5)]
cont_cols = [np.random.rand(10) for _ in range(5)]

X_deep = torch.from_numpy(np.vstack(embed_cols + cont_cols).transpose())
X_deep_emb = X_deep[:, :5]
X_deep_cont = X_deep[:, 5:]


###############################################################################
# Embeddings and NO continuous_cols
###############################################################################
embed_input = [(u, i, j) for u, i, j in zip(colnames[:5], [5] * 5, [16] * 5)]
model1 = TabMlp(
    mlp_hidden_dims=[32, 16],
    mlp_dropout=[0.5, 0.2],
    column_idx={k: v for v, k in enumerate(colnames[:5])},
    embed_input=embed_input,
)


def test_deep_dense_embed():
    out = model1(X_deep_emb)
    assert out.size(0) == 10 and out.size(1) == 16


###############################################################################
# Continous cols but NO embeddings
###############################################################################
continuous_cols = colnames[-5:]
model2 = TabMlp(
    mlp_hidden_dims=[32, 16],
    mlp_dropout=[0.5, 0.2],
    column_idx={k: v for v, k in enumerate(colnames[5:])},
    continuous_cols=continuous_cols,
)


def test_deep_dense_cont():
    out = model2(X_deep_cont)
    assert out.size(0) == 10 and out.size(1) == 16


###############################################################################
# All parameters and cont norm
###############################################################################


@pytest.mark.parametrize(
    "cont_norm_layer",
    [None, "batchnorm", "layernorm"],
)
def test_deep_dense(cont_norm_layer):
    model3 = TabMlp(
        column_idx={k: v for v, k in enumerate(colnames)},
        mlp_hidden_dims=[32, 16, 8],
        mlp_dropout=0.1,
        mlp_batchnorm=True,
        mlp_batchnorm_last=False,
        mlp_linear_first=False,
        embed_input=embed_input,
        embed_dropout=0.1,
        continuous_cols=continuous_cols,
        cont_norm_layer=cont_norm_layer,
    )
    out = model3(X_deep)
    assert out.size(0) == 10 and out.size(1) == 8


###############################################################################
# Test raise ValueError
###############################################################################


def test_act_fn_ValueError():
    with pytest.raises(ValueError):
        model4 = TabMlp(  # noqa: F841
            mlp_hidden_dims=[32, 16],
            mlp_dropout=[0.5, 0.2],
            mlp_activation="javier",
            column_idx={k: v for v, k in enumerate(colnames)},
            embed_input=embed_input,
            continuous_cols=continuous_cols,
        )


###############################################################################
# Test CatEmbeddingsAndCont
###############################################################################


@pytest.mark.parametrize(
    "setup, column_idx, embed_input, continuous_cols",
    [
        ("w_embed", {k: v for v, k in enumerate(colnames[:5])}, embed_input, None),
        ("w_cont", {k: v for v, k in enumerate(colnames[5:])}, None, continuous_cols),
        (
            "w_both",
            {k: v for v, k in enumerate(colnames)},
            embed_input,
            continuous_cols,
        ),
    ],
)
def test_cat_embeddings_and_cont(setup, column_idx, embed_input, continuous_cols):

    if setup == "w_embed":
        X = X_deep_emb
    if setup == "w_cont":
        X = X_deep_cont
    if setup == "w_both":
        X = X_deep

    cat_embed_and_cont = CatEmbeddingsAndCont(
        column_idx, embed_input, 0.1, continuous_cols, None
    )
    x_cat, x_cont = cat_embed_and_cont(X)

    if setup == "w_embed":
        assert (
            x_cat.size() == torch.Size([X.shape[0], len(column_idx) * 16])
            and x_cont is None
        )
    if setup == "w_cont":
        assert (
            x_cont.size() == torch.Size([X.shape[0], len(column_idx)]) and x_cat is None
        )
    if setup == "w_both":
        assert x_cat.size() == torch.Size(
            [X.shape[0], len(embed_cols) * 16]
        ) and x_cont.size() == torch.Size([X.shape[0], len(cont_cols)])
