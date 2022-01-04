import string

import numpy as np
import torch
import pytest
from scipy import stats
from numpy.testing import assert_almost_equal
from sklearn.metrics import mean_squared_error, mean_squared_log_error

from pytorch_widedeep.losses import MSLELoss, RMSELoss, ZILNLoss
from pytorch_widedeep.models import Wide, TabMlp, WideDeep
from pytorch_widedeep.training import Trainer
from pytorch_widedeep.training._loss_and_obj_aliases import (
    _LossAliases,
    _ObjectiveToMethod,
)

# Wide array
X_wide = np.random.choice(50, (100, 10))

# Deep Array
colnames = list(string.ascii_lowercase)[:10]
embed_cols = [np.random.choice(np.arange(5), 100) for _ in range(5)]
embed_input = [(u, i, j) for u, i, j in zip(colnames[:5], [5] * 5, [16] * 5)]
cont_cols = [np.random.rand(100) for _ in range(5)]
column_idx = {k: v for v, k in enumerate(colnames)}
X_tab = np.vstack(embed_cols + cont_cols).transpose()

# Target
target_regres = np.random.rand(100)
target_binary = np.random.choice(2, 100)
target_multic = np.random.choice(3, 100)


##############################################################################
# Test that the model runs with the focal loss
##############################################################################
@pytest.mark.parametrize(
    "X_wide, X_tab, target, objective, pred_dim, probs_dim",
    [
        (X_wide, X_tab, target_binary, "binary", 1, 2),
        (X_wide, X_tab, target_multic, "multiclass", 3, 3),
    ],
)
def test_focal_loss(X_wide, X_tab, target, objective, pred_dim, probs_dim):
    objective = "_".join([objective, "focal_loss"])
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
    trainer.fit(X_wide=X_wide, X_tab=X_tab, target=target)
    probs = trainer.predict_proba(X_wide=X_wide, X_tab=X_tab)
    assert probs.shape[1] == probs_dim


##############################################################################
# Test RMSELoss and MSLELoss implementation
##############################################################################
def test_mse_based_losses():
    y_true = np.array([3, 5, 2.5, 7]).reshape(-1, 1)
    y_pred = np.array([2.5, 5, 4, 8]).reshape(-1, 1)

    t_true = torch.from_numpy(y_true)
    t_pred = torch.from_numpy(y_pred)

    out = []

    out.append(
        np.isclose(
            np.sqrt(mean_squared_error(y_true, y_pred)),
            RMSELoss()(t_pred, t_true).item(),
        )
    )

    out.append(
        np.isclose(
            mean_squared_log_error(y_true, y_pred),
            MSLELoss()(t_pred, t_true).item(),
        )
    )

    assert all(out)


##############################################################################
# Test ZILNloss implementation
##############################################################################

# adjusted test of original authors
# https://github.com/google/lifetime_value/blob/master/lifetime_value/zero_inflated_lognormal_test.py


def test_ziln_loss():
    # softplus function that calculates log(1+exp(x))
    def _softplus(x):
        return np.log(1.0 + np.exp(x))

    # _softplus = lambda x: np.log(1.0 + np.exp(x))

    def zero_inflated_lognormal_np(labels, logits):
        positive_logits = logits[..., :1]
        loss_zero = _softplus(positive_logits)
        loc = logits[..., 1:2]
        scale = np.maximum(
            _softplus(logits[..., 2:]), np.sqrt(torch.finfo(torch.float32).eps)
        )
        log_prob_non_zero = stats.lognorm.logpdf(
            x=labels, s=scale, loc=0, scale=np.exp(loc)
        )
        loss_non_zero = _softplus(-positive_logits) - log_prob_non_zero
        return np.mean(
            np.mean(np.where(labels == 0.0, loss_zero, loss_non_zero), axis=-1)
        )

    logits = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    labels = np.array([[0.0], [1.5]])

    expected_loss = zero_inflated_lognormal_np(labels, logits)
    zilnloss = ZILNLoss()
    loss = zilnloss.forward(torch.tensor(logits), torch.tensor(labels))
    assert_almost_equal(loss.numpy(), expected_loss, decimal=6)
    return loss.numpy(), expected_loss


##############################################################################
# Test all possible objectives
##############################################################################
method_to_objec = {
    "binary": [
        "binary",
        "logistic",
        "binary_logloss",
        "binary_cross_entropy",
        "binary_focal_loss",
    ],
    "multiclass": [
        "multiclass",
        "multi_logloss",
        "cross_entropy",
        "categorical_cross_entropy",
        "multiclass_focal_loss",
    ],
    "regression": [
        "regression",
        "mse",
        "l2",
        "mean_squared_error",
        "mean_absolute_error",
        "mae",
        "l1",
        "mean_squared_log_error",
        "msle",
        "root_mean_squared_error",
        "rmse",
        "root_mean_squared_log_error",
        "rmsle",
        "zero_inflated_lognormal",
        "ziln",
        "tweedie",
        "focalr_mse",
        "focalr_rmse",
        "focalr_l1",
        "huber",
    ],
    "qregression": [
        "quantile",
    ],
}


@pytest.mark.parametrize(
    "X_wide, X_tab, target, method, objective, pred_dim, probs_dim, enforce_positive",
    [
        (X_wide, X_tab, target_regres, "regression", "regression", 1, 1, False),
        (X_wide, X_tab, target_regres, "regression", "mse", 1, 1, False),
        (X_wide, X_tab, target_regres, "regression", "l2", 1, 1, False),
        (X_wide, X_tab, target_regres, "regression", "mean_squared_error", 1, 1, False),
        (
            X_wide,
            X_tab,
            target_regres,
            "regression",
            "mean_absolute_error",
            1,
            1,
            False,
        ),
        (X_wide, X_tab, target_regres, "regression", "mae", 1, 1, False),
        (X_wide, X_tab, target_regres, "regression", "l1", 1, 1, False),
        (
            X_wide,
            X_tab,
            target_regres,
            "regression",
            "mean_squared_log_error",
            1,
            1,
            True,
        ),
        (X_wide, X_tab, target_regres, "regression", "msle", 1, 1, True),
        (
            X_wide,
            X_tab,
            target_regres,
            "regression",
            "root_mean_squared_error",
            1,
            1,
            False,
        ),
        (X_wide, X_tab, target_regres, "regression", "rmse", 1, 1, False),
        (
            X_wide,
            X_tab,
            target_regres,
            "regression",
            "root_mean_squared_log_error",
            1,
            1,
            True,
        ),
        (X_wide, X_tab, target_regres, "regression", "rmsle", 1, 1, True),
        (X_wide, X_tab, target_regres, "regression", "focalr_mse", 1, 1, False),
        (X_wide, X_tab, target_regres, "regression", "focalr_rmse", 1, 1, False),
        (X_wide, X_tab, target_regres, "regression", "focalr_l1", 1, 1, False),
        (X_wide, X_tab, target_regres, "regression", "huber", 1, 1, False),
        (
            X_wide,
            X_tab,
            target_regres,
            "regression",
            "zero_inflated_lognormal",
            3,
            1,
            False,
        ),
        (X_wide, X_tab, target_regres, "regression", "ziln", 3, 1, False),
        (X_wide, X_tab, target_regres, "qregression", "quantile", 7, 7, False),
        (X_wide, X_tab, target_regres, "regression", "tweedie", 1, 1, True),
        (X_wide, X_tab, target_binary, "binary", "binary", 1, 2, False),
        (X_wide, X_tab, target_binary, "binary", "logistic", 1, 2, False),
        (X_wide, X_tab, target_binary, "binary", "binary_logloss", 1, 2, False),
        (X_wide, X_tab, target_binary, "binary", "binary_cross_entropy", 1, 2, False),
        (X_wide, X_tab, target_binary, "binary", "binary_focal_loss", 1, 2, False),
        (X_wide, X_tab, target_multic, "multiclass", "multiclass", 3, 3, False),
        (X_wide, X_tab, target_multic, "multiclass", "multi_logloss", 3, 3, False),
        (X_wide, X_tab, target_multic, "multiclass", "cross_entropy", 3, 3, False),
        (
            X_wide,
            X_tab,
            target_multic,
            "multiclass",
            "categorical_cross_entropy",
            3,
            3,
            False,
        ),
        (
            X_wide,
            X_tab,
            target_multic,
            "multiclass",
            "multiclass_focal_loss",
            3,
            3,
            False,
        ),
    ],
)
def test_all_possible_objectives(
    X_wide,
    X_tab,
    target,
    method,
    objective,
    pred_dim,
    probs_dim,
    enforce_positive,
):
    wide = Wide(np.unique(X_wide).shape[0], pred_dim)
    deeptabular = TabMlp(
        column_idx=column_idx,
        cat_embed_input=embed_input,
        continuous_cols=colnames[-5:],
        mlp_hidden_dims=[32, 16],
        mlp_dropout=[0.5, 0.5],
    )
    model = WideDeep(
        wide=wide,
        deeptabular=deeptabular,
        pred_dim=pred_dim,
        enforce_positive=enforce_positive,
    )
    trainer = Trainer(model, objective=objective, verbose=0)
    trainer.fit(X_wide=X_wide, X_tab=X_tab, target=target)
    out = []
    if method == "regression":
        preds = trainer.predict(X_wide=X_wide, X_tab=X_tab)
        out.append(preds.ndim == probs_dim)
    elif method == "qregression":
        preds = trainer.predict(X_wide=X_wide, X_tab=X_tab)
        out.append(preds.shape[1] == probs_dim)
    else:
        preds = trainer.predict_proba(X_wide=X_wide, X_tab=X_tab)
        out.append(preds.shape[1] == probs_dim)
    assert all(out)


##############################################################################
# Test inverse mappings
##############################################################################
def test_inverse_maps():
    out = []
    out.append(_LossAliases.alias_to_loss["binary_logloss"] == "binary")
    out.append(_LossAliases.alias_to_loss["multi_logloss"] == "multiclass")
    out.append(_LossAliases.alias_to_loss["l2"] == "regression")
    out.append(_LossAliases.alias_to_loss["mae"] == "mean_absolute_error")
    out.append(_LossAliases.alias_to_loss["msle"] == "mean_squared_log_error")
    out.append(_LossAliases.alias_to_loss["rmse"] == "root_mean_squared_error")
    out.append(_LossAliases.alias_to_loss["rmsle"] == "root_mean_squared_log_error")
    out.append(_LossAliases.alias_to_loss["ziln"] == "zero_inflated_lognormal")

    out.append("binary_logloss" in _ObjectiveToMethod.method_to_objecive["binary"])
    out.append("multi_logloss" in _ObjectiveToMethod.method_to_objecive["multiclass"])
    out.append(
        "root_mean_squared_error" in _ObjectiveToMethod.method_to_objecive["regression"]
    )
    out.append(
        "zero_inflated_lognormal" in _ObjectiveToMethod.method_to_objecive["regression"]
    )
    out.append("quantile" in _ObjectiveToMethod.method_to_objecive["qregression"])
    out.append("tweedie" in _ObjectiveToMethod.method_to_objecive["regression"])
    assert all(out)
