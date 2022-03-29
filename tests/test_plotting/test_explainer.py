import string
import warnings

import numpy as np
import pytest

from pytorch_widedeep.models import TabMlp, WideDeep
from pytorch_widedeep.training import Trainer
from pytorch_widedeep.plotting.explainer import ShapExplainer

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
target_multic = np.random.choice(3, 32)


##############################################################################
# Test that the three possible methods (regression, binary and mutliclass)
# work well with 3 possible explainer types
##############################################################################
@pytest.mark.parametrize(
    "X_tab, target, objective, pred_dim, explainer_type",
    [
        (X_tab, target_regres, "regression", 1, "kernel"),
        (X_tab, target_regres, "regression", 1, "deep"),
        (X_tab, target_regres, "regression", 1, "gradient"),
        (X_tab, target_binary, "binary", 1, "kernel"),
        (X_tab, target_binary, "binary", 1, "deep"),
        (X_tab, target_binary, "binary", 1, "gradient"),
        (X_tab, target_multic, "multiclass", 3, "kernel"),
        (X_tab, target_multic, "multiclass", 3, "deep"),
        (X_tab, target_multic, "multiclass", 3, "gradient"),
    ],
)
def test_fit_objectives(
    X_tab,
    target,
    objective,
    pred_dim,
    explainer_type,
):
    deeptabular = TabMlp(
        column_idx=column_idx,
        cat_embed_input=embed_input,
        continuous_cols=colnames[-5:],
        mlp_hidden_dims=[32, 16],
        mlp_dropout=[0.5, 0.5],
    )
    model = WideDeep(deeptabular=deeptabular, pred_dim=pred_dim)
    trainer = Trainer(model, objective=objective, verbose=0)
    trainer.fit(X_tab=X_tab, target=target, batch_size=16)
    shap_explainer = ShapExplainer()
    shap_explainer.fit(
        tab_trainer=trainer,
        X_tab_train=X_tab,
        explainer_type=explainer_type,
        background_sample_count=5,
    )
    shap_explainer.explain_decision_plot(X_tab_explain=X_tab[0], feature_names=colnames)
