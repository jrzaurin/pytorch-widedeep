import string

import numpy as np
import torch
import pytest

from pytorch_widedeep.models import TabMlp, WideDeep
from pytorch_widedeep.training import Trainer
from pytorch_widedeep.models.tabular.embeddings_layers import (
    DiffSizeCatAndContEmbeddings,
)

colnames = list(string.ascii_lowercase)[:10]
embed_cols = [np.random.choice(np.arange(5), 10) for _ in range(5)]
embed_input = [(u, i, j) for u, i, j in zip(colnames[:5], [5] * 5, [16] * 5)]
cont_cols = [np.random.rand(10) for _ in range(5)]
continuous_cols = colnames[-5:]

X_deep = torch.from_numpy(np.vstack(embed_cols + cont_cols).transpose())
X_deep_emb = X_deep[:, :5]
X_deep_cont = X_deep[:, 5:]
target = np.random.choice(2, 32)

tabmlp = TabMlp(
    column_idx={k: v for v, k in enumerate(colnames)},
    cat_embed_input=embed_input,
    continuous_cols=colnames[-5:],
    mlp_hidden_dims=[32, 16],
    mlp_dropout=[0.5, 0.5],
)
###############################################################################
# Embeddings and NO continuous_cols
###############################################################################


def test_tab_mlp_only_cat_embed():
    model = TabMlp(
        column_idx={k: v for v, k in enumerate(colnames[:5])},
        cat_embed_input=embed_input,
        mlp_hidden_dims=[32, 16],
        mlp_dropout=[0.5, 0.2],
    )
    out = model(X_deep_emb)
    assert out.size(0) == 10 and out.size(1) == 16


###############################################################################
# Continous cols but NO embeddings
###############################################################################


@pytest.mark.parametrize(
    "embed_continuous",
    [True, False],
)
def test_tab_mlp_only_cont(embed_continuous):
    model = TabMlp(
        mlp_hidden_dims=[32, 16],
        mlp_dropout=[0.5, 0.2],
        column_idx={k: v for v, k in enumerate(colnames[5:])},
        continuous_cols=continuous_cols,
        embed_continuous=embed_continuous,
        cont_embed_dim=6,
    )
    out = model(X_deep_cont)
    assert out.size(0) == 10 and out.size(1) == 16


###############################################################################
# All parameters and cont norm
###############################################################################


@pytest.mark.parametrize(
    "cont_norm_layer",
    [None, "batchnorm", "layernorm"],
)
def test_cont_norm_layer(cont_norm_layer):
    model = TabMlp(
        column_idx={k: v for v, k in enumerate(colnames)},
        cat_embed_input=embed_input,
        cat_embed_dropout=0.1,
        cat_embed_activation="relu",
        continuous_cols=continuous_cols,
        cont_norm_layer=cont_norm_layer,
        cont_embed_activation="relu",
        mlp_hidden_dims=[32, 16, 8],
        mlp_dropout=0.1,
        mlp_batchnorm=True,
        mlp_batchnorm_last=False,
        mlp_linear_first=False,
    )
    out = model(X_deep)
    assert out.size(0) == 10 and out.size(1) == 8


###############################################################################
# Test raise ValueError
###############################################################################


def test_act_fn_ValueError():
    with pytest.raises(ValueError):
        model = TabMlp(  # noqa: F841
            column_idx={k: v for v, k in enumerate(colnames)},
            cat_embed_input=embed_input,
            continuous_cols=continuous_cols,
            mlp_hidden_dims=[32, 16],
            mlp_dropout=[0.5, 0.2],
            mlp_activation="javier",
        )


###############################################################################
# Test DiffSizeCatAndContEmbeddings
###############################################################################


@pytest.mark.parametrize(
    "setup, column_idx, cat_embed_input, continuous_cols, embed_continuous",
    [
        ("w_cat", {k: v for v, k in enumerate(colnames[:5])}, embed_input, None, False),
        (
            "w_cont",
            {k: v for v, k in enumerate(colnames[5:])},
            None,
            continuous_cols,
            False,
        ),
        (
            "w_both",
            {k: v for v, k in enumerate(colnames)},
            embed_input,
            continuous_cols,
            False,
        ),
        (
            "w_both_and_embed_cont",
            {k: v for v, k in enumerate(colnames)},
            embed_input,
            continuous_cols,
            True,
        ),
    ],
)
def test_embedddings_class(
    setup, column_idx, cat_embed_input, continuous_cols, embed_continuous
):

    if setup == "w_cat":
        X = X_deep_emb
    elif setup == "w_cont":
        X = X_deep_cont
    else:
        X = X_deep

    cat_and_cont_embed = DiffSizeCatAndContEmbeddings(
        column_idx=column_idx,
        cat_embed_input=cat_embed_input,
        cat_embed_dropout=0.1,
        use_cat_bias=setup == "w_both_and_embed_cont",
        continuous_cols=continuous_cols,
        embed_continuous=embed_continuous,
        cont_embed_dim=16,
        cont_embed_dropout=0.1,
        use_cont_bias=setup == "w_both_and_embed_cont",
        cont_norm_layer=None,
    )
    x_cat, x_cont = cat_and_cont_embed(X)

    if setup == "w_cat":
        s1 = X.shape[0]
        s2 = sum([el[2] for el in cat_and_cont_embed.cat_embed_input])
        assert x_cat.size() == torch.Size((s1, s2)) and x_cont is None
    if setup == "w_cont":
        s1 = X.shape[0]
        s2 = len(continuous_cols)
        assert x_cont.size() == torch.Size((s1, s2)) and x_cat is None
    if setup == "w_both":
        s1 = X.shape[0]
        s2_cat = sum([el[2] for el in cat_and_cont_embed.cat_embed_input])
        s2_cont = len(continuous_cols)
        assert x_cat.size() == torch.Size((s1, s2_cat)) and x_cont.size() == torch.Size(
            (s1, s2_cont)
        )
    if setup == "w_both_and_embed_cont":
        s1 = X.shape[0]
        s2_cat = sum([el[2] for el in cat_and_cont_embed.cat_embed_input])
        s2_cont = len(continuous_cols) * cat_and_cont_embed.cont_embed_dim
        assert x_cat.size() == torch.Size((s1, s2_cat)) and x_cont.size() == torch.Size(
            (s1, s2_cont)
        )


###############################################################################
# Test Feature Dsitribution Smoothing
###############################################################################


@pytest.mark.parametrize(
    "with_lds",
    [True, False],
)
def test_fds(with_lds):
    # lds with model
    model = WideDeep(
        deeptabular=tabmlp,
        with_fds=True,
        momentum=None,
        clip_min=0,
        clip_max=10,
    )
    trainer = Trainer(model, objective="regression", everbose=0)
    # n_epochs=2 to run self._calibrate_mean_var
    trainer.fit(X_tab=X_deep, target=target, n_epochs=3, with_lds=with_lds)
    # simply checking that runs and produces outputs
    preds = trainer.predict(X_tab=X_deep)
    module_names = list(model.named_modules())
    assert module_names[-2][0] == "fds_layer"
    assert module_names[-1][0] == "fds_layer.pred_layer"
    assert preds.shape[0] == 10 and "train_loss" in trainer.history

    trainer.model.fds_layer.reset()
    assert float(trainer.model.fds_layer.num_samples_tracked.sum()) == 0
