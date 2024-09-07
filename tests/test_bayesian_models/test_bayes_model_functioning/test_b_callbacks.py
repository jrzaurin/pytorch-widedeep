import os
import shutil
import string

import numpy as np
import torch
import pytest
from torch.optim.lr_scheduler import StepLR, CyclicLR

from pytorch_widedeep.training import BayesianTrainer
from pytorch_widedeep.callbacks import LRHistory, EarlyStopping, ModelCheckpoint
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
@pytest.mark.parametrize("scheduler_name", ["step", "cyclic"])
def test_history_callback(model, scheduler_name):
    init_lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    if scheduler_name == "cyclic":
        scheduler = CyclicLR(
            optimizer,
            base_lr=init_lr,
            max_lr=0.01,
            step_size_up=5,
            cycle_momentum=False,
        )
        len_lr_output = 11
    elif scheduler_name == "step":
        scheduler = StepLR(optimizer, step_size=4)
        len_lr_output = 5

    btrainer = BayesianTrainer(
        model=model,
        objective="binary",
        optimizer=optimizer,
        lr_scheduler=scheduler,
        callbacks=[LRHistory(n_epochs=5)],
        verbose=0,
    )

    btrainer.fit(
        X_tab=X_tab,
        target=target,
        n_epochs=5,
        batch_size=16,
    )

    out = []
    out.append(len(btrainer.history["train_loss"]) == 5)

    lr_list = btrainer.lr_history["lr_0"]
    out.append(len(lr_list) == len_lr_output)

    if scheduler_name == "step":
        out.append(lr_list[-1] == init_lr / 10)
    elif scheduler_name == "cyclic":
        out.append(lr_list[-1] == init_lr)

    assert all(out)


###############################################################################
# Test that EarlyStopping stops as expected
###############################################################################
@pytest.mark.parametrize("model", [bwide, btabmlp])
def test_early_stop(model):
    btrainer = BayesianTrainer(
        model=model,
        objective="binary",
        callbacks=[
            EarlyStopping(
                min_delta=1e4, patience=3, restore_best_weights=True, verbose=1
            )
        ],
        verbose=0,
    )
    btrainer.fit(
        X_tab=X_tab,
        target=target,
        val_split=0.2,
        n_epochs=5,
        batch_size=16,
    )
    # length of history = patience+1
    assert len(btrainer.history["train_loss"]) == 3 + 1


###############################################################################
# Test that ModelCheckpoint behaves as expected
###############################################################################
@pytest.mark.parametrize(
    "fpath, save_best_only, max_save, n_files",
    [
        (
            "tests/test_bayesian_models/test_model_functioning/weights/test_weights",
            True,
            2,
            2,
        ),
        (
            "tests/test_bayesian_models/test_model_functioning/weights/test_weights",
            False,
            2,
            2,
        ),
        (
            "tests/test_bayesian_models/test_model_functioning/weights/test_weights",
            False,
            0,
            5,
        ),
        (None, False, 0, 0),
    ],
)
def test_model_checkpoint(fpath, save_best_only, max_save, n_files):
    trainer = BayesianTrainer(
        model=btabmlp,
        objective="binary",
        callbacks=[
            ModelCheckpoint(
                filepath=fpath,
                save_best_only=save_best_only,
                max_save=max_save,
            )
        ],
        verbose=0,
    )
    trainer.fit(X_tab=X_tab, target=target, n_epochs=5, val_split=0.2)
    if fpath:
        n_saved = len(
            os.listdir("tests/test_bayesian_models/test_model_functioning/weights/")
        )
        shutil.rmtree("tests/test_bayesian_models/test_model_functioning/weights/")
    else:
        n_saved = 0
    assert n_saved <= n_files


def test_filepath_error():
    btabmlp = BayesianTabMlp(
        column_idx=column_idx,
        cat_embed_input=embed_input,
        continuous_cols=colnames[-5:],
        mlp_hidden_dims=[16, 4],
    )
    with pytest.raises(ValueError):
        trainer = BayesianTrainer(  # noqa: F841
            model=btabmlp,
            objective="binary",
            callbacks=[ModelCheckpoint(filepath="wrong_file_path")],
            verbose=0,
        )
