import string

import numpy as np
import pytest

from pytorch_widedeep.training import BayesianTrainer
from pytorch_widedeep.bayesian_models import BayesianWide, BayesianTabMlp

# Wide array
X_wide = np.random.choice(50, (32, 10))

# Deep Array
colnames = list(string.ascii_lowercase)[:10]
embed_cols = [np.random.choice(np.arange(5), 32) for _ in range(5)]
embed_input = [(u, i, j) for u, i, j in zip(colnames[:5], [5] * 5, [16] * 5)]
embed_input_tt = [(u, i) for u, i in zip(colnames[:5], [5] * 5)]
cont_cols = [np.random.rand(32) for _ in range(5)]
column_idx = {k: v for v, k in enumerate(colnames)}
X_tabmlp = np.vstack(embed_cols + cont_cols).transpose()

# Target
target_regres = np.random.random(32)
target_binary = np.random.choice(2, 32)
target_multic = np.random.choice(3, 32)

##############################################################################
# Test that the three possible methods (regression, binary and mutliclass)
# work well
##############################################################################


@pytest.mark.parametrize("model_name", ["wide", "tabmlp"])
@pytest.mark.parametrize("objective", ["binary", "multiclass"])
@pytest.mark.parametrize("return_samples", [True, False])
@pytest.mark.parametrize("embed_continuous", [True, False])
def test_classification(model_name, objective, return_samples, embed_continuous):
    bsz = 32
    n_samples = 5

    pred_dim = 1 if objective == "binary" else 3
    target = target_binary if objective == "binary" else target_multic

    if model_name == "wide":
        X_tab = X_wide
        model = BayesianWide(np.unique(X_wide).shape[0], pred_dim)
    elif model_name == "tabmlp":
        X_tab = X_tabmlp
        model = BayesianTabMlp(
            column_idx=column_idx,
            cat_embed_input=embed_input,
            continuous_cols=colnames[-5:],
            embed_continuous=embed_continuous,
            mlp_hidden_dims=[32, 16],
            pred_dim=pred_dim,
        )

    trainer = BayesianTrainer(model, objective=objective, verbose=0)
    trainer.fit(X_tab=X_tab, target=target, batch_size=16)
    preds = trainer.predict(X_tab=X_tab, return_samples=return_samples, batch_size=16)
    probs = trainer.predict_proba(
        X_tab=X_tab, return_samples=return_samples, batch_size=16
    )

    out = []

    if return_samples:
        out.append(preds.shape[0] == n_samples)
        out.append(preds.shape[1] == bsz)
        out.append(probs.shape[0] == n_samples)
        out.append(probs.shape[1] == bsz)
        out.append(probs.shape[2] == 2 if objective == "binary" else 3)

    else:
        out.append(preds.shape[0] == bsz)
        out.append(probs.shape[0] == bsz)
        out.append(probs.shape[1] == 2 if objective == "binary" else 3)

    assert all(out)


@pytest.mark.parametrize("model_name", ["wide", "tabmlp"])
@pytest.mark.parametrize("return_samples", [True, False])
def test_regression(model_name, return_samples):
    bsz = 32
    n_samples = 5

    if model_name == "wide":
        X_tab = X_wide
        model = BayesianWide(np.unique(X_wide).shape[0], 1)
    elif model_name == "tabmlp":
        X_tab = X_tabmlp
        model = BayesianTabMlp(
            column_idx=column_idx,
            cat_embed_input=embed_input,
            continuous_cols=colnames[-5:],
            mlp_hidden_dims=[32, 16],
            pred_dim=1,
        )

    trainer = BayesianTrainer(model, objective="regression", verbose=0)
    trainer.fit(X_tab=X_tab, target=target_regres, batch_size=16)
    preds = trainer.predict(X_tab=X_tab, return_samples=return_samples, batch_size=16)

    out = []

    if return_samples:
        out.append(preds.shape[0] == n_samples)
        out.append(preds.shape[1] == bsz)
    else:
        out.append(preds.shape[0] == bsz)

    assert all(out)
