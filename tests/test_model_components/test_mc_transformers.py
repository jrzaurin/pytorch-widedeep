import string
from copy import copy

import numpy as np
import torch
import pytest

from pytorch_widedeep.models import SAINT, TabTransformer
from pytorch_widedeep.models.transformers.layers import *  # noqa: F403

# I am going over test these models due to the number of components

n_embed = 5
n_cols = 2
batch_size = 10
colnames = list(string.ascii_lowercase)[: (n_cols * 2)]
embed_cols = [np.random.choice(np.arange(n_embed), batch_size) for _ in range(n_cols)]
embed_cols_with_cls_token = [[n_embed] * batch_size] + embed_cols
cont_cols = [np.random.rand(batch_size) for _ in range(n_cols)]

X_tab = torch.from_numpy(np.vstack(embed_cols + cont_cols).transpose())
X_tab_with_cls_token = torch.from_numpy(
    np.vstack(embed_cols_with_cls_token + cont_cols).transpose()
)


###############################################################################
# Test functioning using the defaults
###############################################################################

embed_input = [(u, i) for u, i in zip(colnames[:2], [n_embed] * 2)]
model1 = TabTransformer(
    column_idx={k: v for v, k in enumerate(colnames)},
    embed_input=embed_input,
    continuous_cols=colnames[n_cols:],
)


def test_embeddings_have_padding():
    res = []
    res.append(model1.cat_embed.weight.size(0) == model1.n_tokens + 1)
    res.append(not torch.all(model1.cat_embed.weight[0].bool()))
    assert all(res)


def test_tabtransformer_output():
    out = model1(X_tab)
    assert out.size(0) == 10 and out.size(1) == (n_cols * 32 + len(cont_cols)) * 2


###############################################################################
# Test SharedEmbeddings
###############################################################################

# all manually passed
def test_tabtransformer_shared_embeddings():

    res = []

    shared_embeddings = SharedEmbeddings(
        n_embed=5,
        embed_dim=16,
        embed_dropout=0.0,
        add_shared_embed=False,
        frac_shared_embed=0.25,
    )

    X_inp = X_tab[:, 0]
    se = shared_embeddings(X_inp.long())

    res.append((se[:, :2][0] == se[:, :2]).all())

    shared_embeddings = SharedEmbeddings(
        n_embed=5,
        embed_dim=16,
        embed_dropout=0.0,
        add_shared_embed=True,
        frac_shared_embed=0.25,
    )

    X_inp = X_tab[:, 0]
    se = shared_embeddings(X_inp.long())
    not_se = shared_embeddings.embed(X_inp.long())
    res.append((not_se + shared_embeddings.shared_embed).allclose(se))

    assert all(res)


model2 = TabTransformer(
    column_idx={k: v for v, k in enumerate(colnames)},
    embed_input=embed_input,
    continuous_cols=colnames[n_cols:],
    shared_embed=True,
)


def test_shared_embeddings_have_padding():
    res = []
    for k, v in model2.cat_embed.items():
        res.append(v.embed.weight.size(0) == n_embed + 1)
        res.append(not torch.all(v.embed.weight[0].bool()))
    assert all(res)


def test_tabtransformer_w_shared_emb_output():
    out = model2(X_tab)
    assert out.size(0) == 10 and out.size(1) == (n_cols * 32 + len(cont_cols)) * 2


###############################################################################
# Test ContinuousEmbeddings
###############################################################################


def test_continuous_embeddings():
    bsz = 2
    n_cont_cols = 2
    embed_dim = 6

    X = torch.rand(bsz, n_cont_cols)

    cont_embed = ContinuousEmbeddings(
        n_cont_cols=n_cont_cols, embed_dim=embed_dim, activation=None, bias=None
    )
    out = cont_embed(X)
    res = (
        out.shape[0] == bsz
        and out.shape[1] == n_cont_cols
        and out.shape[2] == embed_dim
    )

    assert res and torch.allclose(out[0, 0, :], X[0][0] * cont_embed.weight[0])


###############################################################################
# Sanity Check: Test w/o continuous features
###############################################################################

model3 = TabTransformer(
    column_idx={k: v for v, k in enumerate(colnames)},
    embed_input=embed_input,
    continuous_cols=None,
)


def test_tabtransformer_output_no_cont():
    out = model3(X_tab)
    assert out.size(0) == 10 and out.size(1) == (n_cols * 32) * 2


###############################################################################
# Test full embed dropout
###############################################################################


def test_full_embed_dropout():
    bsz = 1
    cat = 10
    esz = 4
    full_embedding_dropout = FullEmbeddingDropout(dropout=0.5)
    inp = torch.rand(bsz, cat, esz)
    out = full_embedding_dropout(inp)
    # simply check that at least 1 full row is all 0s
    assert torch.any(torch.sum(out[0] == 0, axis=1) == esz)


# ###############################################################################
# # Beginning of a 360 test of SAINT and TabTransformer
# ###############################################################################


@pytest.mark.parametrize(
    "embed_continuous, with_cls_token, model_name",
    [
        (True, True, "tabtransformer"),
        (True, False, "tabtransformer"),
        (False, True, "tabtransformer"),
        (False, False, "tabtransformer"),
        (True, True, "saint"),
        (True, False, "saint"),
        (False, True, "saint"),
        (False, False, "saint"),
    ],
)
def test_embed_continuous_and_with_cls_token(
    embed_continuous, with_cls_token, model_name
):
    if with_cls_token:
        X = X_tab_with_cls_token
        n_colnames = ["cls_token"] + copy(colnames)
        cont_idx = n_cols + 1
    else:
        X = X_tab
        n_colnames = copy(colnames)
        cont_idx = n_cols

    if model_name == "tabtransformer":
        model = TabTransformer(
            column_idx={k: v for v, k in enumerate(n_colnames)},
            embed_input=embed_input,
            continuous_cols=n_colnames[cont_idx:],
            embed_continuous=embed_continuous,
        )
    elif model_name == "saint":
        model = SAINT(
            column_idx={k: v for v, k in enumerate(n_colnames)},
            embed_input=embed_input,
            continuous_cols=n_colnames[cont_idx:],
            embed_continuous=embed_continuous,
        )
    out = model(X)
    res = [out.size(0) == 10]
    if with_cls_token:
        if embed_continuous:
            res.append(model._set_mlp_hidden_dims()[0] == model.input_dim)
        else:
            res.append(
                model._set_mlp_hidden_dims()[0] == model.input_dim + len(cont_cols)
            )
    elif embed_continuous:
        mlp_first_h = X.shape[1] * model.input_dim
        res.append(model._set_mlp_hidden_dims()[0] == mlp_first_h)
    else:
        mlp_first_h = len(embed_cols) * model.input_dim + 2
        res.append(model._set_mlp_hidden_dims()[0] == mlp_first_h)

    assert all(res)


@pytest.mark.parametrize(
    "activation, model_name",
    [
        ("relu", "tabtransformer"),
        ("leaky_relu", "tabtransformer"),
        ("gelu", "tabtransformer"),
        ("geglu", "tabtransformer"),
        ("relu", "saint"),
        ("leaky_relu", "saint"),
        ("gelu", "saint"),
        ("geglu", "saint"),
    ],
)
def test_transformer_activations(activation, model_name):

    if model_name == "tabtransformer":
        model = TabTransformer(
            column_idx={k: v for v, k in enumerate(colnames)},
            embed_input=embed_input,
            continuous_cols=colnames[n_cols:],
            transformer_activation=activation,
        )
    elif model_name == "saint":
        model = SAINT(
            column_idx={k: v for v, k in enumerate(colnames)},
            embed_input=embed_input,
            continuous_cols=colnames[n_cols:],
            transformer_activation=activation,
        )
    out = model(X_tab)
    assert out.size(0) == 10


###############################################################################
# Test keep attention weights
###############################################################################


@pytest.mark.parametrize(
    "model_name",
    [
        "tabtransformer",
        "saint",
    ],
)
def test_tabtransformer_keep_attn(model_name):
    if model_name == "tabtransformer":
        model = TabTransformer(
            column_idx={k: v for v, k in enumerate(colnames)},
            embed_input=embed_input,
            continuous_cols=colnames[n_cols:],
            n_blocks=4,
            keep_attn_weights=True,
        )
    elif model_name == "saint":
        model = SAINT(
            column_idx={k: v for v, k in enumerate(colnames)},
            embed_input=embed_input,
            continuous_cols=colnames[n_cols:],
            n_blocks=4,
            keep_attn_weights=True,
        )
    out = model(X_tab)

    res = [out.size(0) == 10]
    res.append(out.size(1) == model._set_mlp_hidden_dims()[-1])
    res.append(len(model.attention_weights) == model.n_blocks)

    if model_name == "tabtransformer":
        res.append(
            list(model.attention_weights[0].shape)
            == [10, model.n_heads, n_cols, n_cols]
        )
    elif model_name == "saint":
        res.append(
            list(model.attention_weights[0][0].shape)
            == [10, model.n_heads, n_cols, n_cols]
        )
        res.append(
            list(model.attention_weights[0][1].shape)
            == [1, model.n_heads, n_cols * n_embed, n_cols * n_embed]
        )
    assert all(res)
