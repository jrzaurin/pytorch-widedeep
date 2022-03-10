import string
from copy import copy

import numpy as np
import torch
import pytest

from pytorch_widedeep.models import (
    SAINT,
    TabPerceiver,
    FTTransformer,
    TabFastFormer,
    TabTransformer,
)
from pytorch_widedeep.models.tabular.embeddings_layers import *  # noqa: F403
from pytorch_widedeep.models.tabular.transformers._attention_layers import *  # noqa: F403

# I am going over test these models due to the number of components

n_embed = 5
n_cols = 2
batch_size = 10
colnames = list(string.ascii_lowercase)[: (n_cols * 2)]
embed_cols = [np.random.choice(np.arange(n_embed), batch_size) for _ in range(n_cols)]
embed_cols_with_cls_token = [[n_embed] * batch_size] + embed_cols  # type: ignore[operator]
cont_cols = [np.random.rand(batch_size) for _ in range(n_cols)]

X_tab = torch.from_numpy(np.vstack(embed_cols + cont_cols).transpose())
X_tab_with_cls_token = torch.from_numpy(
    np.vstack(embed_cols_with_cls_token + cont_cols).transpose()  # type: ignore[operator]
)


###############################################################################
# Test functioning using the defaults
###############################################################################

embed_input = [(u, i) for u, i in zip(colnames[:2], [n_embed] * 2)]
model1 = TabTransformer(
    column_idx={k: v for v, k in enumerate(colnames)},
    cat_embed_input=embed_input,
    continuous_cols=colnames[n_cols:],
)


def test_embeddings_have_padding():
    res = []
    res.append(
        model1.cat_and_cont_embed.cat_embed.embed.weight.size(0)
        == model1.cat_and_cont_embed.cat_embed.n_tokens + 1
    )
    res.append(
        not torch.all(model1.cat_and_cont_embed.cat_embed.embed.weight[0].bool())
    )
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
    cat_embed_input=embed_input,
    continuous_cols=colnames[n_cols:],
    use_cont_bias=True,
    shared_embed=True,
)


def test_shared_embeddings_have_padding():
    res = []
    for k, v in model2.cat_and_cont_embed.cat_embed.embed.items():
        res.append(v.embed.weight.size(0) == n_embed + 1)
        res.append(not torch.all(v.embed.weight[0].bool()))
    assert all(res)


def test_tabtransformer_w_shared_emb_output():
    out = model2(X_tab)
    assert out.size(0) == 10 and out.size(1) == (n_cols * 32 + len(cont_cols)) * 2


###############################################################################
# Test ContEmbeddings
###############################################################################


def test_continuous_embeddings():
    bsz = 2
    n_cont_cols = 2
    embed_dim = 6

    X = torch.rand(bsz, n_cont_cols)

    cont_embed = ContEmbeddings(
        n_cont_cols=n_cont_cols,
        embed_dim=embed_dim,
        embed_dropout=0.0,
        use_bias=False,
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
    cat_embed_input=embed_input,
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
    full_embedding_dropout = FullEmbeddingDropout(p=0.8)
    inp = torch.rand(bsz, cat, esz)
    out = full_embedding_dropout(inp)
    # simply check that at least 1 full row is all 0s
    assert (torch.sum(out[0] == 0, axis=1) == esz).sum() > 0


# ###############################################################################
# # Beginning of a 360 test of the Transformer family
# ###############################################################################


def _build_model(model_name, params):
    if model_name == "tabtransformer":
        return TabTransformer(n_blocks=2, n_heads=2, **params)
    if model_name == "saint":
        return SAINT(n_blocks=2, n_heads=2, **params)
    if model_name == "fttransformer":
        return FTTransformer(n_blocks=2, n_heads=2, kv_compression_factor=0.5, **params)
    if model_name == "tabfastformer":
        return TabFastFormer(n_blocks=2, n_heads=2, **params)
    if model_name == "tabperceiver":
        return TabPerceiver(n_perceiver_blocks=2, n_latents=2, latent_dim=16, **params)


@pytest.mark.parametrize(
    "embed_continuous, with_cls_token, model_name",
    [
        (True, True, "tabtransformer"),
        (True, False, "tabtransformer"),
        (False, True, "tabtransformer"),
        (False, False, "tabtransformer"),
    ],
)
def test_embed_continuous_and_with_cls_token_tabtransformer(
    embed_continuous, with_cls_token, model_name
):
    if with_cls_token:
        X = X_tab_with_cls_token
        n_colnames = ["cls_token"] + copy(colnames)
        cont_idx = n_cols + 1
        with_cls_token_embed_input = [("cls_token", 1)] + embed_input
    else:
        X = X_tab
        n_colnames = copy(colnames)
        cont_idx = n_cols

    params = {
        "column_idx": {k: v for v, k in enumerate(n_colnames)},
        "cat_embed_input": with_cls_token_embed_input
        if with_cls_token
        else embed_input,
        "continuous_cols": n_colnames[cont_idx:],
        "embed_continuous": embed_continuous,
    }

    model = _build_model(model_name, params)

    out = model(X)
    res = [out.size(0) == 10]
    if with_cls_token:
        if embed_continuous:
            res.append(model._compute_attn_output_dim() == model.input_dim)
        else:
            res.append(
                model._compute_attn_output_dim() == model.input_dim + len(cont_cols)
            )
    elif embed_continuous:
        mlp_first_h = X.shape[1] * model.input_dim
        res.append(model._compute_attn_output_dim() == mlp_first_h)
    else:
        mlp_first_h = len(embed_cols) * model.input_dim + 2
        res.append(model._compute_attn_output_dim() == mlp_first_h)

    assert all(res)


@pytest.mark.parametrize(
    "with_cls_token, model_name",
    [
        (True, "saint"),
        (False, "saint"),
        (True, "fttransformer"),
        (False, "fttransformer"),
        (True, "tabfastformer"),
        (False, "tabfastformer"),
    ],
)
def test_embed_continuous_and_with_cls_token_transformer_family(
    with_cls_token, model_name
):
    if with_cls_token:
        X = X_tab_with_cls_token
        n_colnames = ["cls_token"] + copy(colnames)
        cont_idx = n_cols + 1
        with_cls_token_embed_input = [("cls_token", 1)] + embed_input
    else:
        X = X_tab
        n_colnames = copy(colnames)
        cont_idx = n_cols

    params = {
        "column_idx": {k: v for v, k in enumerate(n_colnames)},
        "cat_embed_input": with_cls_token_embed_input
        if with_cls_token
        else embed_input,
        "continuous_cols": n_colnames[cont_idx:],
    }

    total_n_cols = n_cols * 2
    model = _build_model(model_name, params)

    out = model(X)
    res = [out.size(0) == 10]
    if with_cls_token:
        if model_name in ["saint", "tabfastformer"]:
            res.append(out.shape[1] == model.input_dim * 2)
        elif model_name == "fttransformer":
            res.append(out.shape[1] == model.input_dim)
    else:
        if model_name in ["saint", "tabfastformer"]:
            res.append(out.shape[1] == (total_n_cols * model.input_dim) * 2)
        elif model_name == "fttransformer":
            res.append(out.shape[1] == (total_n_cols * model.input_dim))

    assert all(res)


@pytest.mark.parametrize(
    "activation, model_name",
    [
        ("tanh", "tabtransformer"),
        ("leaky_relu", "tabtransformer"),
        ("geglu", "tabtransformer"),
        ("reglu", "tabtransformer"),
        ("tanh", "saint"),
        ("leaky_relu", "saint"),
        ("geglu", "saint"),
        ("reglu", "saint"),
        ("tanh", "fttransformer"),
        ("leaky_relu", "fttransformer"),
        ("geglu", "fttransformer"),
        ("reglu", "fttransformer"),
        ("tanh", "tabfastformer"),
        ("leaky_relu", "tabfastformer"),
        ("geglu", "tabfastformer"),
        ("reglu", "tabfastformer"),
        ("tanh", "tabperceiver"),
        ("leaky_relu", "tabperceiver"),
        ("geglu", "tabperceiver"),
        ("reglu", "tabperceiver"),
    ],
)
def test_transformer_activations(activation, model_name):

    params = {
        "column_idx": {k: v for v, k in enumerate(colnames)},
        "cat_embed_input": embed_input,
        "continuous_cols": colnames[n_cols:],
        "transformer_activation": activation,
    }

    model = _build_model(model_name, params)

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
        "fttransformer",
        "tabfastformer",
        "tabperceiver",
    ],
)
def test_transformers_keep_attn(model_name):

    params = {
        "column_idx": {k: v for v, k in enumerate(colnames)},
        "cat_embed_input": embed_input,
        "continuous_cols": colnames[n_cols:],
    }

    # n_cols is an unfortunate name I might change in the future. It refers to
    # the number of cat and cont cols, so the total number of cols is
    # n_cols * 2
    total_n_cols = n_cols * 2

    model = _build_model(model_name, params)

    out = model(X_tab)

    res = [out.size(0) == 10]
    if model_name != "tabperceiver":
        res.append(len(model.attention_weights) == model.n_blocks)
    else:
        res.append(len(model.attention_weights) == model.n_perceiver_blocks)

    if model_name == "tabtransformer":
        res.append(
            list(model.attention_weights[0].shape)
            == [10, model.n_heads, n_cols, n_cols]
        )
    elif model_name == "saint":
        res.append(
            list(model.attention_weights[0][0].shape)
            == [10, model.n_heads, total_n_cols, total_n_cols]
        )
        res.append(
            list(model.attention_weights[0][1].shape)
            == [1, model.n_heads, X_tab.shape[0], X_tab.shape[0]]
        )
    if model_name == "fttransformer":
        res.append(
            list(model.attention_weights[0].shape)
            == [
                10,
                model.n_heads,
                total_n_cols,
                int(model.n_feats * model.kv_compression_factor),
            ]
        )
    elif model_name == "tabperceiver":
        res.append(
            len(model.attention_weights[0])
            == model.n_cross_attns + model.n_latent_blocks
        )
        res.append(
            list(model.attention_weights[0][0].shape)
            == [10, model.n_cross_attn_heads, model.n_latents, X_tab.shape[1]]
        )
        res.append(
            list(model.attention_weights[0][1].shape)
            == [10, model.n_cross_attn_heads, model.n_latents, model.n_latents]
        )
    elif model_name == "tabfastformer":
        res.append(
            list(model.attention_weights[0][0].shape)
            == [10, model.n_heads, total_n_cols]
        )
        res.append(
            list(model.attention_weights[0][1].shape)
            == [10, model.n_heads, total_n_cols]
        )
    assert all(res)


###############################################################################
# Test transformers with only continuous cols
###############################################################################


X_tab_only_cont = torch.from_numpy(
    np.vstack([np.random.rand(10) for _ in range(4)]).transpose()
)
colnames_only_cont = list(string.ascii_lowercase)[:4]


@pytest.mark.parametrize(
    "model_name",
    [
        "fttransformer",
        "saint",
        "tabfastformer",
    ],
)
def test_transformers_only_cont(model_name):
    params = {
        "column_idx": {k: v for v, k in enumerate(colnames_only_cont)},
        "continuous_cols": colnames_only_cont,
    }

    model = _build_model(model_name, params)
    out = model(X_tab_only_cont)

    assert out.size(0) == 10 and out.size(1) == model.output_dim
