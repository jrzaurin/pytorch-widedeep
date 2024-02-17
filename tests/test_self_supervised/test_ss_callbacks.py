import os
import shutil
import string

import numpy as np
import torch
import pandas as pd
import pytest
from torch.optim.lr_scheduler import StepLR, CyclicLR, ReduceLROnPlateau

from pytorch_widedeep.models import TabMlp, TabTransformer
from pytorch_widedeep.callbacks import (
    LRHistory,
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.self_supervised_training import (
    EncoderDecoderTrainer,
    ContrastiveDenoisingTrainer,
)

some_letters = list(string.ascii_lowercase)
some_numbers = range(10)
full_df = pd.DataFrame(
    {
        "col1": list(np.random.choice(some_letters, 64)),
        "col2": list(np.random.choice(some_letters, 64)),
        "col3": list(np.random.choice(some_numbers, 64)),
        "col4": list(np.random.choice(some_numbers, 64)),
    }
)

non_transf_preprocessor = TabPreprocessor(
    cat_embed_cols=["col1", "col2"],
    continuous_cols=["col3", "col4"],
)
X_tab = non_transf_preprocessor.fit_transform(full_df[:32])
X_tab_valid = non_transf_preprocessor.transform(full_df[32:])


# for contrastive denoising training all categorical values in a validation
# set must have been seen in the training set, since each col is used as a
# target
transf_df = pd.DataFrame(
    {
        "col1": list(np.random.choice(some_letters, 32)),
        "col2": list(np.random.choice(some_letters, 32)),
        "col3": list(np.random.choice(some_numbers, 32)),
        "col4": list(np.random.choice(some_numbers, 32)),
    }
)

transf_preprocessor = TabPreprocessor(
    cat_embed_cols=["col1", "col2"],
    continuous_cols=["col3", "col4"],
    with_attention=True,
)
X_tab_transf = transf_preprocessor.fit_transform(transf_df)
X_tab_valid_transf = transf_preprocessor.transform(transf_df.sample(frac=1))


###############################################################################
# Test that history saves the information adequately
###############################################################################
@pytest.mark.parametrize(
    "model_type",
    ["mlp", "transformer"],
)
@pytest.mark.parametrize(
    "schedulers_type, len_loss_output, len_lr_output, init_lr",
    [
        ("step", 5, 5, 0.001),
        ("cyclic", 5, 11, 0.001),
        ("reducelronplateau", 5, 5, 0.001),
    ],
)
def test_lr_history(  # noqa: C901
    model_type, schedulers_type, len_loss_output, len_lr_output, init_lr
):
    if model_type == "mlp":
        model = TabMlp(
            column_idx=non_transf_preprocessor.column_idx,
            cat_embed_input=non_transf_preprocessor.cat_embed_input,
            continuous_cols=non_transf_preprocessor.continuous_cols,
            mlp_hidden_dims=[16, 8],
        )
        X, X_valid = X_tab, X_tab_valid
    elif model_type == "transformer":
        model = TabTransformer(
            column_idx=transf_preprocessor.column_idx,
            cat_embed_input=transf_preprocessor.cat_embed_input,
            continuous_cols=transf_preprocessor.continuous_cols,
            embed_continuous=True,
            embed_continuous_method="standard",
            n_heads=2,
            n_blocks=2,
        )
        X, X_valid = X_tab_transf, X_tab_valid_transf

    optimizer = torch.optim.Adam(model.parameters())

    if schedulers_type == "step":
        lr_scheduler = StepLR(optimizer, step_size=4)
    elif schedulers_type == "cyclic":
        lr_scheduler = CyclicLR(
            optimizer, base_lr=0.001, max_lr=0.01, step_size_up=5, cycle_momentum=False
        )
    elif schedulers_type == "reducelronplateau":
        lr_scheduler = ReduceLROnPlateau(optimizer, patience=2, threshold=0.5)

    if model_type == "mlp":
        trainer = EncoderDecoderTrainer(
            encoder=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            callbacks=[LRHistory(n_epochs=5)],
            masked_prob=0.2,
            verbose=0,
        )
    elif model_type == "transformer":
        trainer = ContrastiveDenoisingTrainer(
            model=model,
            preprocessor=transf_preprocessor,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            callbacks=[LRHistory(n_epochs=5)],
            verbose=0,
        )

    if schedulers_type == "reducelronplateau":
        trainer.pretrain(X, X_tab_val=X_valid, n_epochs=5, batch_size=16)
    else:
        trainer.pretrain(X, n_epochs=5, batch_size=16)

    if schedulers_type == "step":
        history_assert = len(trainer.history["train_loss"]) == len_loss_output
        lr_history_assert = len(trainer.lr_history["lr_0"]) == len_lr_output
        lr_assert = trainer.lr_history["lr_0"][-1] == init_lr / 10.0

    if schedulers_type == "cyclic":
        history_assert = len(trainer.history["train_loss"]) == len_loss_output
        lr_history_assert = len(trainer.lr_history["lr_0"]) == len_lr_output
        lr_assert = trainer.lr_history["lr_0"][-1] == init_lr

    if schedulers_type == "reducelronplateau":
        history_assert = len(trainer.history["train_loss"]) == len_loss_output
        lr_history_assert = len(trainer.lr_history["lr_0"]) == len_lr_output
        lr_assert = trainer.lr_history["lr_0"][-1] == init_lr * lr_scheduler.factor

    assert all([history_assert, lr_history_assert, lr_assert])


# ###############################################################################
# # Test that EarlyStopping stops as expected
# ###############################################################################


@pytest.mark.parametrize(
    "model_type",
    ["mlp", "transformer"],
)
def test_early_stop(model_type):
    if model_type == "mlp":
        model = TabMlp(
            column_idx=non_transf_preprocessor.column_idx,
            cat_embed_input=non_transf_preprocessor.cat_embed_input,
            continuous_cols=non_transf_preprocessor.continuous_cols,
            mlp_hidden_dims=[16, 8],
        )
        trainer = EncoderDecoderTrainer(
            encoder=model,
            masked_prob=0.2,
            callbacks=[
                EarlyStopping(
                    min_delta=5.0, patience=3, restore_best_weights=True, verbose=1
                )
            ],
            verbose=0,
        )
        X, X_valid = X_tab, X_tab_valid
    elif model_type == "transformer":
        model = TabTransformer(
            column_idx=transf_preprocessor.column_idx,
            cat_embed_input=transf_preprocessor.cat_embed_input,
            continuous_cols=transf_preprocessor.continuous_cols,
            embed_continuous=True,
            embed_continuous_method="standard",
            n_heads=2,
            n_blocks=2,
        )

        trainer = ContrastiveDenoisingTrainer(
            model=model,
            preprocessor=transf_preprocessor,
            callbacks=[
                EarlyStopping(
                    min_delta=100.0, patience=3, restore_best_weights=True, verbose=1
                )
            ],
            verbose=0,
        )
        X, X_valid = X_tab_transf, X_tab_valid_transf

    trainer.pretrain(X_tab=X, X_tab_val=X_valid, n_epochs=5, batch_size=16)
    # length of history = patience+1
    assert len(trainer.history["train_loss"]) == 3 + 1


###############################################################################
# Test that ModelCheckpoint behaves as expected
###############################################################################
@pytest.mark.parametrize(
    "model_type",
    ["mlp", "transformer"],
)
@pytest.mark.parametrize(
    "fpath, save_best_only, max_save, n_files",
    [
        ("tests/test_self_supervised/weights/test_weights", True, 2, 2),
        ("tests/test_self_supervised/weights/test_weights", False, 2, 2),
        ("tests/test_self_supervised/weights/test_weights", False, 0, 5),
        (None, False, 0, 0),
    ],
)
def test_checkpoint(model_type, fpath, save_best_only, max_save, n_files):
    if model_type == "mlp":
        model = TabMlp(
            column_idx=non_transf_preprocessor.column_idx,
            cat_embed_input=non_transf_preprocessor.cat_embed_input,
            continuous_cols=non_transf_preprocessor.continuous_cols,
            mlp_hidden_dims=[16, 8],
        )

        trainer = EncoderDecoderTrainer(
            encoder=model,
            masked_prob=0.2,
            callbacks=[
                ModelCheckpoint(
                    filepath=fpath,
                    save_best_only=save_best_only,
                    max_save=max_save,
                )
            ],
            verbose=0,
        )
        X, X_valid = X_tab, X_tab_valid
    elif model_type == "transformer":
        model = TabTransformer(
            column_idx=transf_preprocessor.column_idx,
            cat_embed_input=transf_preprocessor.cat_embed_input,
            continuous_cols=transf_preprocessor.continuous_cols,
            embed_continuous=True,
            embed_continuous_method="standard",
            n_heads=2,
            n_blocks=2,
        )

        trainer = ContrastiveDenoisingTrainer(
            model=model,
            preprocessor=transf_preprocessor,
            callbacks=[
                ModelCheckpoint(
                    filepath=fpath,
                    save_best_only=save_best_only,
                    max_save=max_save,
                )
            ],
            verbose=0,
        )
        X, X_valid = X_tab_transf, X_tab_valid_transf

    trainer.pretrain(X_tab=X, X_tab_val=X_valid, n_epochs=5, batch_size=16)

    if fpath:
        n_saved = len(os.listdir("tests/test_self_supervised/weights/"))
        shutil.rmtree("tests/test_self_supervised/weights/")
    else:
        n_saved = 0
    assert n_saved <= n_files
