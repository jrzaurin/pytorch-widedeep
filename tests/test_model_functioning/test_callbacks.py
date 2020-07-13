import os
import string
from itertools import chain

import numpy as np
import torch
import pytest
from torch.optim.lr_scheduler import StepLR, CyclicLR

from pytorch_widedeep.optim import RAdam
from pytorch_widedeep.models import Wide, WideDeep, DeepDense
from pytorch_widedeep.callbacks import (
    LRHistory,
    EarlyStopping,
    ModelCheckpoint,
)

# Wide array
X_wide = np.random.choice(2, (100, 100), p=[0.8, 0.2])

# Deep Array
colnames = list(string.ascii_lowercase)[:10]
embed_cols = [np.random.choice(np.arange(5), 100) for _ in range(5)]
embed_input = [(u, i, j) for u, i, j in zip(colnames[:5], [5] * 5, [16] * 5)]
cont_cols = [np.random.rand(100) for _ in range(5)]
deep_column_idx = {k: v for v, k in enumerate(colnames)}
X_deep = np.vstack(embed_cols + cont_cols).transpose()

#  Text Array
padded_sequences = np.random.choice(np.arange(1, 100), (100, 48))
vocab_size = 1000
X_text = np.hstack((np.repeat(np.array([[0, 0]]), 100, axis=0), padded_sequences))

# target
target = np.random.choice(2, 100)


###############################################################################
# Test that history saves the information adequately
###############################################################################
wide = Wide(100, 1)
deepdense = DeepDense(
    hidden_layers=[32, 16],
    dropout=[0.5, 0.5],
    deep_column_idx=deep_column_idx,
    embed_input=embed_input,
    continuous_cols=colnames[-5:],
)
model = WideDeep(wide=wide, deepdense=deepdense)

wide_opt_1 = torch.optim.Adam(model.wide.parameters())
deep_opt_1 = torch.optim.Adam(model.deepdense.parameters())
wide_sch_1 = StepLR(wide_opt_1, step_size=4)
deep_sch_1 = CyclicLR(
    deep_opt_1, base_lr=0.001, max_lr=0.01, step_size_up=10, cycle_momentum=False
)
optimizers_1 = {"wide": wide_opt_1, "deepdense": deep_opt_1}
lr_schedulers_1 = {"wide": wide_sch_1, "deepdense": deep_sch_1}

wide_opt_2 = torch.optim.Adam(model.wide.parameters())
deep_opt_2 = RAdam(model.deepdense.parameters())
wide_sch_2 = StepLR(wide_opt_2, step_size=4)
deep_sch_2 = StepLR(deep_opt_2, step_size=4)
optimizers_2 = {"wide": wide_opt_2, "deepdense": deep_opt_2}
lr_schedulers_2 = {"wide": wide_sch_2, "deepdense": deep_sch_2}


@pytest.mark.parametrize(
    "optimizers, schedulers, len_loss_output, len_lr_output",
    [(optimizers_1, lr_schedulers_1, 5, 21), (optimizers_2, lr_schedulers_2, 5, 5)],
)
def test_history_callback(optimizers, schedulers, len_loss_output, len_lr_output):
    model.compile(
        method="binary",
        optimizers=optimizers,
        lr_schedulers=schedulers,
        callbacks=[LRHistory(n_epochs=5)],
        verbose=0,
    )
    model.fit(X_wide=X_wide, X_deep=X_deep, X_text=X_text, target=target, n_epochs=5)
    out = []
    out.append(len(model.history._history["train_loss"]) == len_loss_output)
    try:
        lr_list = list(chain.from_iterable(model.lr_history["lr_deepdense_0"]))
    except TypeError:
        lr_list = model.lr_history["lr_deepdense_0"]
    out.append(len(lr_list) == len_lr_output)
    assert all(out)


###############################################################################
# Test that EarlyStopping stops as expected
###############################################################################
def test_early_stop():
    wide = Wide(100, 1)
    deepdense = DeepDense(
        hidden_layers=[32, 16],
        dropout=[0.5, 0.5],
        deep_column_idx=deep_column_idx,
        embed_input=embed_input,
        continuous_cols=colnames[-5:],
    )
    model = WideDeep(wide=wide, deepdense=deepdense)
    model.compile(
        method="binary",
        callbacks=[
            EarlyStopping(
                min_delta=0.1, patience=3, restore_best_weights=True, verbose=1
            )
        ],
        verbose=1,
    )
    model.fit(X_wide=X_wide, X_deep=X_deep, target=target, val_split=0.2, n_epochs=5)
    # length of history = patience+1
    assert len(model.history._history["train_loss"]) == 3 + 1


###############################################################################
# Test that ModelCheckpoint behaves as expected
###############################################################################
@pytest.mark.parametrize(
    "save_best_only, max_save, n_files", [(True, 2, 2), (False, 2, 2), (False, 0, 5)]
)
def test_model_checkpoint(save_best_only, max_save, n_files):
    wide = Wide(100, 1)
    deepdense = DeepDense(
        hidden_layers=[32, 16],
        dropout=[0.5, 0.5],
        deep_column_idx=deep_column_idx,
        embed_input=embed_input,
        continuous_cols=colnames[-5:],
    )
    model = WideDeep(wide=wide, deepdense=deepdense)
    model.compile(
        method="binary",
        callbacks=[
            ModelCheckpoint(
                "weights/test_weights", save_best_only=save_best_only, max_save=max_save
            )
        ],
        verbose=0,
    )
    model.fit(X_wide=X_wide, X_deep=X_deep, target=target, n_epochs=5, val_split=0.2)
    n_saved = len(os.listdir("weights/"))
    for f in os.listdir("weights/"):
        os.remove("weights/" + f)
    assert n_saved <= n_files
