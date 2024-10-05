import os
import shutil
import string

import numpy as np
import torch
import pytest

from pytorch_widedeep.training import BayesianTrainer
from pytorch_widedeep.bayesian_models import BayesianWide, BayesianTabMlp

# Wide array
X_wide = np.random.choice(50, (32, 10))

# Deep Array
colnames = list(string.ascii_lowercase)[:10]
embed_cols = [np.random.choice(np.arange(5), 32) for _ in range(5)]
embed_input = [(u, i, j) for u, i, j in zip(colnames[:5], [5] * 5, [16] * 5)]
cont_cols = [np.random.rand(32) for _ in range(5)]
column_idx = {k: v for v, k in enumerate(colnames)}
X_tab = np.vstack(embed_cols + cont_cols).transpose()

# target
target = np.random.choice(2, 32)


###############################################################################
# Test that history saves the information adequately
###############################################################################
bwide = BayesianWide(np.unique(X_wide).shape[0], 1)

btabmlp = BayesianTabMlp(
    column_idx=column_idx,
    cat_embed_input=embed_input,
    continuous_cols=colnames[-5:],
    mlp_hidden_dims=[32, 16],
)


@pytest.mark.parametrize("model", [bwide, btabmlp])
def test_save_and_load(model):
    btrainer = BayesianTrainer(
        model=model,
        objective="binary",
        verbose=0,
    )

    X = X_wide if model.__class__.__name__ == "BayesianWide" else X_tab
    btrainer.fit(
        X_tab=X,
        target=target,
        n_epochs=5,
        batch_size=16,
    )

    if model.__class__.__name__ == "BayesianWide":
        weight_mu = model.bayesian_wide_linear.weight_mu.data
        weight_rho = model.bayesian_wide_linear.weight_rho.data
    elif model.__class__.__name__ == "BayesianTabMlp":
        weight_mu = model.cat_embed.embed_layers["emb_layer_a"].weight_mu.data
        weight_rho = model.cat_embed.embed_layers["emb_layer_a"].weight_rho.data

    btrainer.save(
        "tests/test_bayesian_models/test_bayes_model_functioning/model_dir/",
        model_filename="bayesian_model.pt",
    )

    new_model = torch.load(
        "tests/test_bayesian_models/test_bayes_model_functioning/model_dir/bayesian_model.pt",
        weights_only=False,
    )

    if model.__class__.__name__ == "BayesianWide":
        new_weight_mu = new_model.bayesian_wide_linear.weight_mu.data
        new_weight_rho = new_model.bayesian_wide_linear.weight_rho.data
    elif model.__class__.__name__ == "BayesianTabMlp":
        new_weight_mu = new_model.cat_embed.embed_layers["emb_layer_a"].weight_mu.data
        new_weight_rho = new_model.cat_embed.embed_layers["emb_layer_a"].weight_rho.data

    shutil.rmtree("tests/test_bayesian_models/test_bayes_model_functioning/model_dir/")

    assert torch.allclose(weight_mu, new_weight_mu) and torch.allclose(
        weight_rho, new_weight_rho
    )


@pytest.mark.parametrize("model_name", ["wide", "tabmlp"])
def test_save_and_load_dict(model_name):
    model1, btrainer1 = _build_model_and_trainer(model_name)

    X = X_wide if model_name == "wide" else X_tab

    btrainer1.fit(
        X_tab=X,
        target=target,
        n_epochs=5,
        batch_size=16,
    )

    btrainer1.save(
        "tests/test_bayesian_models/test_bayes_model_functioning/model_dir/",
        model_filename="bayesian_model.pt",
        save_state_dict=True,
    )

    if model_name == "wide":
        weight_mu = model1.bayesian_wide_linear.weight_mu.data
        weight_rho = model1.bayesian_wide_linear.weight_rho.data
    elif model_name == "tabmlp":
        weight_mu = model1.cat_embed.embed_layers["emb_layer_a"].weight_mu.data
        weight_rho = model1.cat_embed.embed_layers["emb_layer_a"].weight_rho.data

    model2, btrainer2 = _build_model_and_trainer(model_name)

    btrainer2.model.load_state_dict(
        torch.load(
            "tests/test_bayesian_models/test_bayes_model_functioning/model_dir/bayesian_model.pt",
            weights_only=False,
        )
    )

    if model_name == "wide":
        new_weight_mu = model1.bayesian_wide_linear.weight_mu.data
        new_weight_rho = model1.bayesian_wide_linear.weight_rho.data
    elif model_name == "tabmlp":
        new_weight_mu = model1.cat_embed.embed_layers["emb_layer_a"].weight_mu.data
        new_weight_rho = model1.cat_embed.embed_layers["emb_layer_a"].weight_rho.data

    same_weights = torch.allclose(weight_mu, new_weight_mu) and torch.allclose(
        weight_rho, new_weight_rho
    )

    if os.path.isfile(
        "tests/test_bayesian_models/test_bayes_model_functioning/model_dir/history/train_eval_history.json"
    ):
        history_saved = True
    else:
        history_saved = False
    shutil.rmtree("tests/test_bayesian_models/test_bayes_model_functioning/model_dir/")
    assert same_weights and history_saved


def _build_model_and_trainer(model_name):
    if model_name == "wide":
        model = BayesianWide(np.unique(X_wide).shape[0], 1)
    elif model_name == "tabmlp":
        model = BayesianTabMlp(
            column_idx=column_idx,
            cat_embed_input=embed_input,
            continuous_cols=colnames[-5:],
            mlp_hidden_dims=[32, 16],
        )
    trainer = BayesianTrainer(
        model=model,
        objective="binary",
        verbose=0,
    )
    return model, trainer
