import string
import warnings

import numpy as np
import pytest
from torch import nn

from pytorch_widedeep.models import (
    Wide,
    TabMlp,
    TabNet,
    WideDeep,
    TabTransformer,
)
from pytorch_widedeep.metrics import R2Score
from pytorch_widedeep.training import Trainer
from pytorch_widedeep.dataloaders import DataLoaderImbalanced

# Wide array
X_wide = np.random.choice(50, (32, 10))

# Deep Array
colnames = list(string.ascii_lowercase)[:10]
embed_cols = [np.random.choice(np.arange(5), 32) for _ in range(5)]
embed_input = [(u, i, j) for u, i, j in zip(colnames[:5], [5] * 5, [16] * 5)]
embed_input_tt = [(u, i) for u, i in zip(colnames[:5], [5] * 5)]
cont_cols = [np.random.rand(32) for _ in range(5)]
column_idx = {k: v for v, k in enumerate(colnames)}
X_tab = np.vstack(embed_cols + cont_cols).transpose()

# Target
target_regres = np.random.random(32)
target_binary = np.random.choice(2, 32)
target_binary_imbalanced = np.random.choice(2, 32, p=[0.75, 0.25])
target_multic = np.random.choice(3, 32)

# Test dictionary
X_test = {"X_wide": X_wide, "X_tab": X_tab}


##############################################################################
# Test that the three possible methods (regression, binary and mutliclass)
# work well
##############################################################################
@pytest.mark.parametrize(
    "X_wide, X_tab, target, objective, X_test, pred_dim, probs_dim, uncertainties_pred_dim",
    [
        (X_wide, X_tab, target_regres, "regression", None, 1, None, 4),
        (X_wide, X_tab, target_binary, "binary", None, 1, 2, 3),
        (X_wide, X_tab, target_multic, "multiclass", None, 3, 3, 4),
        (X_wide, X_tab, target_regres, "regression", X_test, 1, None, 4),
        (X_wide, X_tab, target_binary, "binary", X_test, 1, 2, 3),
        (X_wide, X_tab, target_multic, "multiclass", X_test, 3, 3, 4),
    ],
)
def test_fit_objectives(
    X_wide,
    X_tab,
    target,
    objective,
    X_test,
    pred_dim,
    probs_dim,
    uncertainties_pred_dim,
):
    wide = Wide(np.unique(X_wide).shape[0], pred_dim)
    deeptabular = TabMlp(
        column_idx=column_idx,
        cat_embed_input=embed_input,
        continuous_cols=colnames[-5:],
        mlp_hidden_dims=[32, 16],
        mlp_dropout=[0.5, 0.5],
    )
    model = WideDeep(wide=wide, deeptabular=deeptabular, pred_dim=pred_dim)
    trainer = Trainer(model, objective=objective, verbose=0)
    trainer.fit(X_wide=X_wide, X_tab=X_tab, target=target, batch_size=16)
    preds = trainer.predict(X_wide=X_wide, X_tab=X_tab, X_test=X_test)
    probs = trainer.predict_proba(X_wide=X_wide, X_tab=X_tab, X_test=X_test)
    unc_preds = trainer.predict_uncertainty(
        X_wide=X_wide, X_tab=X_tab, X_test=X_test, uncertainty_granularity=5
    )
    if objective == "regression":
        assert (preds.shape[0], probs, unc_preds.shape[1]) == (
            32,
            probs_dim,
            uncertainties_pred_dim,
        )
    else:
        assert (preds.shape[0], probs.shape[1], unc_preds.shape[1]) == (
            32,
            probs_dim,
            uncertainties_pred_dim,
        )


##############################################################################
# Simply Test that runs with the deephead parameter
##############################################################################
def test_fit_with_deephead():
    wide = Wide(np.unique(X_wide).shape[0], 1)
    deeptabular = TabMlp(
        column_idx=column_idx,
        cat_embed_input=embed_input,
        continuous_cols=colnames[-5:],
        mlp_hidden_dims=[32, 16],
    )
    deephead = nn.Sequential(nn.Linear(16, 8), nn.Linear(8, 4))
    model = WideDeep(wide=wide, deeptabular=deeptabular, pred_dim=1, deephead=deephead)
    trainer = Trainer(model, objective="binary", verbose=0)
    trainer.fit(X_wide=X_wide, X_tab=X_tab, target=target_binary, batch_size=16)
    preds = trainer.predict(X_wide=X_wide, X_tab=X_tab, X_test=X_test)
    probs = trainer.predict_proba(X_wide=X_wide, X_tab=X_tab, X_test=X_test)
    unc_preds = trainer.predict_uncertainty(
        X_wide=X_wide, X_tab=X_tab, X_test=X_test, uncertainty_granularity=5
    )
    assert (preds.shape[0], probs.shape[1], unc_preds.shape[1]) == (32, 2, 3)


##############################################################################
# Repeat 1st set of tests with the TabTransformer
##############################################################################


@pytest.mark.parametrize(
    "X_wide, X_tab, target, objective, X_wide_test, X_tab_test, X_test, pred_dim, probs_dim, uncertainties_pred_dim",
    [
        (X_wide, X_tab, target_regres, "regression", X_wide, X_tab, None, 1, None, 4),
        (X_wide, X_tab, target_binary, "binary", X_wide, X_tab, None, 1, 2, 3),
        (X_wide, X_tab, target_multic, "multiclass", X_wide, X_tab, None, 3, 3, 4),
        (X_wide, X_tab, target_regres, "regression", None, None, X_test, 1, None, 4),
        (X_wide, X_tab, target_binary, "binary", None, None, X_test, 1, 2, 3),
        (X_wide, X_tab, target_multic, "multiclass", None, None, X_test, 3, 3, 4),
    ],
)
def test_fit_objectives_tab_transformer(
    X_wide,
    X_tab,
    target,
    objective,
    X_wide_test,
    X_tab_test,
    X_test,
    pred_dim,
    probs_dim,
    uncertainties_pred_dim,
):
    wide = Wide(np.unique(X_wide).shape[0], pred_dim)
    tab_transformer = TabTransformer(
        column_idx={k: v for v, k in enumerate(colnames)},
        cat_embed_input=embed_input_tt,
        continuous_cols=colnames[5:],
    )
    model = WideDeep(wide=wide, deeptabular=tab_transformer, pred_dim=pred_dim)
    trainer = Trainer(model, objective=objective, verbose=0)
    trainer.fit(X_wide=X_wide, X_tab=X_tab, target=target, batch_size=16)
    preds = trainer.predict(X_wide=X_wide, X_tab=X_tab, X_test=X_test)
    probs = trainer.predict_proba(X_wide=X_wide, X_tab=X_tab, X_test=X_test)
    unc_preds = trainer.predict_uncertainty(
        X_wide=X_wide, X_tab=X_tab, X_test=X_test, uncertainty_granularity=5
    )
    if objective == "regression":
        assert (preds.shape[0], probs, unc_preds.shape[1]) == (
            32,
            probs_dim,
            uncertainties_pred_dim,
        )
    else:
        assert (preds.shape[0], probs.shape[1], unc_preds.shape[1]) == (
            32,
            probs_dim,
            uncertainties_pred_dim,
        )


##############################################################################
# Repeat 1st set of tests with TabNet
##############################################################################


@pytest.mark.parametrize(
    "X_wide, X_tab, target, objective, X_wide_test, X_tab_test, X_test, pred_dim, probs_dim, uncertainties_pred_dim",
    [
        (X_wide, X_tab, target_regres, "regression", X_wide, X_tab, None, 1, None, 4),
        (X_wide, X_tab, target_binary, "binary", X_wide, X_tab, None, 1, 2, 3),
        (X_wide, X_tab, target_multic, "multiclass", X_wide, X_tab, None, 3, 3, 4),
        (X_wide, X_tab, target_regres, "regression", None, None, X_test, 1, None, 4),
        (X_wide, X_tab, target_binary, "binary", None, None, X_test, 1, 2, 3),
        (X_wide, X_tab, target_multic, "multiclass", None, None, X_test, 3, 3, 4),
    ],
)
def test_fit_objectives_tabnet(
    X_wide,
    X_tab,
    target,
    objective,
    X_wide_test,
    X_tab_test,
    X_test,
    pred_dim,
    probs_dim,
    uncertainties_pred_dim,
):
    warnings.filterwarnings("ignore")
    wide = Wide(np.unique(X_wide).shape[0], pred_dim)
    tab_transformer = TabNet(
        column_idx={k: v for v, k in enumerate(colnames)},
        cat_embed_input=embed_input,
        continuous_cols=colnames[5:],
    )
    model = WideDeep(wide=wide, deeptabular=tab_transformer, pred_dim=pred_dim)
    trainer = Trainer(model, objective=objective, verbose=0)
    trainer.fit(X_wide=X_wide, X_tab=X_tab, target=target, batch_size=16)
    preds = trainer.predict(X_wide=X_wide, X_tab=X_tab, X_test=X_test)
    probs = trainer.predict_proba(X_wide=X_wide, X_tab=X_tab, X_test=X_test)
    unc_preds = trainer.predict_uncertainty(
        X_wide=X_wide, X_tab=X_tab, X_test=X_test, uncertainty_granularity=5
    )
    if objective == "regression":
        assert (preds.shape[0], probs, unc_preds.shape[1]) == (
            32,
            probs_dim,
            uncertainties_pred_dim,
        )
    else:
        assert (preds.shape[0], probs.shape[1], unc_preds.shape[1]) == (
            32,
            probs_dim,
            uncertainties_pred_dim,
        )


##############################################################################
# Test fit with R2 for regression
##############################################################################


def test_fit_with_regression_and_metric():
    wide = Wide(np.unique(X_wide).shape[0], 1)
    deeptabular = TabMlp(
        column_idx=column_idx,
        cat_embed_input=embed_input,
        continuous_cols=colnames[-5:],
        mlp_hidden_dims=[32, 16],
        mlp_dropout=[0.5, 0.5],
    )
    model = WideDeep(wide=wide, deeptabular=deeptabular, pred_dim=1)
    trainer = Trainer(model, objective="regression", metrics=[R2Score], verbose=0)
    trainer.fit(X_wide=X_wide, X_tab=X_tab, target=target_regres, batch_size=16)
    assert "train_r2" in trainer.history.keys()


##############################################################################
# Test aliases
##############################################################################


def test_aliases():
    wide = Wide(np.unique(X_wide).shape[0], 1)
    deeptabular = TabMlp(
        column_idx=column_idx,
        cat_embed_input=embed_input,
        continuous_cols=colnames[-5:],
        mlp_hidden_dims=[32, 16],
        mlp_dropout=[0.5, 0.5],
    )
    model = WideDeep(wide=wide, deeptabular=deeptabular, pred_dim=1)
    trainer = Trainer(model, loss="regression", verbose=0)
    trainer.fit(
        X_wide=X_wide, X_tab=X_tab, target=target_regres, batch_size=16, warmup=True
    )
    assert (
        "train_loss" in trainer.history.keys()
        and trainer.__wd_aliases_used["objective"] == "loss"
        and trainer.__wd_aliases_used["finetune"] == "warmup"
    )


##############################################################################
# Test custom dataloader
##############################################################################


def test_custom_dataloader():
    wide = Wide(np.unique(X_wide).shape[0], 1)
    deeptabular = TabMlp(
        column_idx=column_idx,
        cat_embed_input=embed_input,
        continuous_cols=colnames[-5:],
        mlp_hidden_dims=[32, 16],
        mlp_dropout=[0.5, 0.5],
    )
    model = WideDeep(wide=wide, deeptabular=deeptabular)
    trainer = Trainer(model, loss="binary", verbose=0)
    trainer.fit(
        X_wide=X_wide,
        X_tab=X_tab,
        target=target_binary_imbalanced,
        batch_size=16,
        custom_dataloader=DataLoaderImbalanced,
    )
    # simply checking that runs with DataLoaderImbalanced
    assert "train_loss" in trainer.history.keys()


##############################################################################
# Test raise warning for multiclass classification
##############################################################################


def test_multiclass_warning():
    wide = Wide(np.unique(X_wide).shape[0], 1)
    deeptabular = TabMlp(
        column_idx=column_idx,
        cat_embed_input=embed_input,
        continuous_cols=colnames[-5:],
        mlp_hidden_dims=[32, 16],
        mlp_dropout=[0.5, 0.5],
    )
    model = WideDeep(wide=wide, deeptabular=deeptabular)

    with pytest.raises(ValueError):
        trainer = Trainer(model, loss="multiclass", verbose=0)  # noqa: F841
