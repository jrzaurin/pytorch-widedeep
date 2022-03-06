import os
import pickle
import shutil
import string
from pathlib import Path
from itertools import chain

import numpy as np
import torch
import pytest
from ray import tune
from torch.optim.lr_scheduler import StepLR, CyclicLR, ReduceLROnPlateau

from pytorch_widedeep.models import Wide, TabMlp, WideDeep, TabTransformer
from pytorch_widedeep.training import Trainer
from pytorch_widedeep.callbacks import (
    LRHistory,
    EarlyStopping,
    ModelCheckpoint,
    RayTuneReporter,
)

# Wide array
X_wide = np.random.choice(50, (32, 10))
X_wide_val = np.random.choice(50, (4, 10))

# Deep Array
colnames = list(string.ascii_lowercase)[:10]
embed_cols = [np.random.choice(np.arange(5), 32) for _ in range(5)]
embed_input = [(u, i, j) for u, i, j in zip(colnames[:5], [5] * 5, [16] * 5)]
cont_cols = [np.random.rand(32) for _ in range(5)]
column_idx = {k: v for v, k in enumerate(colnames)}
X_tab = np.vstack(embed_cols + cont_cols).transpose()

embed_cols_val = [np.random.choice(np.arange(5), 4) for _ in range(5)]
cont_cols_val = [np.random.rand(4) for _ in range(5)]
X_tab_val = np.vstack(embed_cols + cont_cols).transpose()

# target
target = np.random.choice(2, 32)
target_val = np.random.choice(2, 4)


###############################################################################
# Test that history saves the information adequately
###############################################################################
wide = Wide(np.unique(X_wide).shape[0], 1)
deeptabular = TabMlp(
    column_idx=column_idx,
    cat_embed_input=embed_input,
    continuous_cols=colnames[-5:],
    mlp_hidden_dims=[32, 16],
    mlp_dropout=[0.5, 0.5],
)
model = WideDeep(wide=wide, deeptabular=deeptabular)

# 1. Single optimizers_1, single scheduler, not cyclic and both passed directly
optimizers_1 = torch.optim.Adam(model.parameters())
lr_schedulers_1 = StepLR(optimizers_1, step_size=4)

# 2. Multiple optimizers, single scheduler, cyclic and pass via a 1 item
# dictionary
wide_opt_2 = torch.optim.Adam(model.wide.parameters())
deep_opt_2 = torch.optim.AdamW(model.deeptabular.parameters())
deep_sch_2 = CyclicLR(
    deep_opt_2, base_lr=0.001, max_lr=0.01, step_size_up=5, cycle_momentum=False
)
optimizers_2 = {"wide": wide_opt_2, "deeptabular": deep_opt_2}
lr_schedulers_2 = {"deeptabular": deep_sch_2}

# 3. Multiple schedulers no cyclic
wide_opt_3 = torch.optim.Adam(model.wide.parameters())
deep_opt_3 = torch.optim.AdamW(model.deeptabular.parameters())
wide_sch_3 = StepLR(wide_opt_3, step_size=4)
deep_sch_3 = StepLR(deep_opt_3, step_size=4)
optimizers_3 = {"wide": wide_opt_3, "deeptabular": deep_opt_3}
lr_schedulers_3 = {"wide": wide_sch_3, "deeptabular": deep_sch_3}

# 4. Multiple schedulers with cyclic
wide_opt_4 = torch.optim.Adam(model.wide.parameters())
deep_opt_4 = torch.optim.AdamW(model.deeptabular.parameters())
wide_sch_4 = StepLR(wide_opt_4, step_size=4)
deep_sch_4 = CyclicLR(
    deep_opt_4, base_lr=0.001, max_lr=0.01, step_size_up=5, cycle_momentum=False
)
optimizers_4 = {"wide": wide_opt_4, "deeptabular": deep_opt_4}
lr_schedulers_4 = {"wide": wide_sch_4, "deeptabular": deep_sch_4}

# 5. Single optimizers_5, single scheduler, cyclic and both passed directly
optimizers_5 = torch.optim.Adam(model.parameters())
lr_schedulers_5 = CyclicLR(
    optimizers_5, base_lr=0.001, max_lr=0.01, step_size_up=5, cycle_momentum=False
)

# 6. Single optimizers_6, single ReduceLROnPlateau lr_schedulers_6
optimizers_6 = torch.optim.Adam(model.parameters())
lr_schedulers_6 = ReduceLROnPlateau(optimizers_6)


@pytest.mark.parametrize(
    "optimizers, schedulers, len_loss_output, len_lr_output, init_lr, schedulers_type",
    [
        (optimizers_1, lr_schedulers_1, 5, 5, 0.001, "step"),
        (optimizers_2, lr_schedulers_2, 5, 11, 0.001, "cyclic"),
        (optimizers_3, lr_schedulers_3, 5, 5, None, None),
        (optimizers_4, lr_schedulers_4, 5, 11, None, None),
        (optimizers_5, lr_schedulers_5, 5, 11, 0.001, "cyclic"),
        (optimizers_6, lr_schedulers_6, 5, 5, None, None),
    ],
)
def test_history_callback(
    optimizers, schedulers, len_loss_output, len_lr_output, init_lr, schedulers_type
):

    trainer = Trainer(
        model=model,
        objective="binary",
        optimizers=optimizers,
        lr_schedulers=schedulers,
        callbacks=[LRHistory(n_epochs=5)],
        verbose=0,
    )
    trainer.fit(
        X_train={"X_wide": X_wide, "X_tab": X_tab, "target": target},
        X_val={"X_wide": X_wide_val, "X_tab": X_tab_val, "target": target_val},
        n_epochs=5,
        batch_size=16,
    )
    out = []
    out.append(len(trainer.history["train_loss"]) == len_loss_output)

    try:
        lr_list = list(chain.from_iterable(trainer.lr_history["lr_deeptabular_0"]))
    except Exception:
        try:
            lr_list = trainer.lr_history["lr_deeptabular_0"]
        except Exception:
            lr_list = trainer.lr_history["lr_0"]

    out.append(len(lr_list) == len_lr_output)
    if init_lr is not None and schedulers_type == "step":
        out.append(lr_list[-1] == init_lr / 10)
    elif init_lr is not None and schedulers_type == "cyclic":
        out.append(lr_list[-1] == init_lr)
    assert all(out)


###############################################################################
# Test that EarlyStopping stops as expected
###############################################################################
def test_early_stop():
    wide = Wide(np.unique(X_wide).shape[0], 1)
    deeptabular = TabMlp(
        column_idx=column_idx,
        cat_embed_input=embed_input,
        continuous_cols=colnames[-5:],
        mlp_hidden_dims=[32, 16],
        mlp_dropout=[0.5, 0.5],
    )
    model = WideDeep(wide=wide, deeptabular=deeptabular)
    trainer = Trainer(
        model=model,
        objective="binary",
        callbacks=[
            EarlyStopping(
                min_delta=5.0, patience=3, restore_best_weights=True, verbose=1
            )
        ],
        verbose=0,
    )
    trainer.fit(X_wide=X_wide, X_tab=X_tab, target=target, val_split=0.2, n_epochs=5)
    # length of history = patience+1
    assert len(trainer.history["train_loss"]) == 3 + 1


###############################################################################
# Test that ModelCheckpoint behaves as expected
###############################################################################
@pytest.mark.parametrize(
    "fpath, save_best_only, max_save, n_files",
    [
        ("tests/test_model_functioning/weights/test_weights", True, 2, 2),
        ("tests/test_model_functioning/weights/test_weights", False, 2, 2),
        ("tests/test_model_functioning/weights/test_weights", False, 0, 5),
        (None, False, 0, 0),
    ],
)
def test_model_checkpoint(fpath, save_best_only, max_save, n_files):
    wide = Wide(np.unique(X_wide).shape[0], 1)
    deeptabular = TabMlp(
        mlp_hidden_dims=[32, 16],
        mlp_dropout=[0.5, 0.5],
        column_idx=column_idx,
        cat_embed_input=embed_input,
        continuous_cols=colnames[-5:],
    )
    model = WideDeep(wide=wide, deeptabular=deeptabular)
    trainer = Trainer(
        model=model,
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
    trainer.fit(X_wide=X_wide, X_tab=X_tab, target=target, n_epochs=5, val_split=0.2)
    if fpath:
        n_saved = len(os.listdir("tests/test_model_functioning/weights/"))
        shutil.rmtree("tests/test_model_functioning/weights/")
    else:
        n_saved = 0
    assert n_saved <= n_files


def test_filepath_error():
    wide = Wide(np.unique(X_wide).shape[0], 1)
    deeptabular = TabMlp(
        mlp_hidden_dims=[16, 4],
        column_idx=column_idx,
        cat_embed_input=embed_input,
        continuous_cols=colnames[-5:],
    )
    model = WideDeep(wide=wide, deeptabular=deeptabular)
    with pytest.raises(ValueError):
        trainer = Trainer(  # noqa: F841
            model=model,
            objective="binary",
            callbacks=[ModelCheckpoint(filepath="wrong_file_path")],
            verbose=0,
        )


###############################################################################
# Repeat 1st set of tests for TabTransormer
###############################################################################

# Wide array
X_wide = np.random.choice(50, (32, 10))

# Tab Array
colnames = list(string.ascii_lowercase)[:10]
embed_cols = [np.random.choice(np.arange(5), 32) for _ in range(5)]
embeds_input = [(i, j) for i, j in zip(colnames[:5], [5] * 5)]  # type: ignore[misc]
cont_cols = [np.random.rand(32) for _ in range(5)]
column_idx = {k: v for v, k in enumerate(colnames)}
X_tab = np.vstack(embed_cols + cont_cols).transpose()

# target
target = np.random.choice(2, 32)

wide = Wide(np.unique(X_wide).shape[0], 1)
tab_transformer = TabTransformer(
    column_idx={k: v for v, k in enumerate(colnames)},
    cat_embed_input=embeds_input,
    continuous_cols=colnames[5:],
)
model_tt = WideDeep(wide=wide, deeptabular=tab_transformer)

# 1. Single optimizers_1, single scheduler, not cyclic and both passed directly
optimizers_1 = torch.optim.Adam(model_tt.parameters())
lr_schedulers_1 = StepLR(optimizers_1, step_size=4)

# 2. Multiple optimizers, single scheduler, cyclic and pass via a 1 item
# dictionary
wide_opt_2 = torch.optim.Adam(model_tt.wide.parameters())
deep_opt_2 = torch.optim.AdamW(model_tt.deeptabular.parameters())
deep_sch_2 = CyclicLR(
    deep_opt_2, base_lr=0.001, max_lr=0.01, step_size_up=5, cycle_momentum=False
)
optimizers_2 = {"wide": wide_opt_2, "deeptabular": deep_opt_2}
lr_schedulers_2 = {"deeptabular": deep_sch_2}

# 3. Multiple schedulers no cyclic
wide_opt_3 = torch.optim.Adam(model_tt.wide.parameters())
deep_opt_3 = torch.optim.AdamW(model_tt.deeptabular.parameters())
wide_sch_3 = StepLR(wide_opt_3, step_size=4)
deep_sch_3 = StepLR(deep_opt_3, step_size=4)
optimizers_3 = {"wide": wide_opt_3, "deeptabular": deep_opt_3}
lr_schedulers_3 = {"wide": wide_sch_3, "deeptabular": deep_sch_3}

# 4. Multiple schedulers with cyclic
wide_opt_4 = torch.optim.Adam(model_tt.wide.parameters())
deep_opt_4 = torch.optim.AdamW(model_tt.deeptabular.parameters())
wide_sch_4 = StepLR(wide_opt_4, step_size=4)
deep_sch_4 = CyclicLR(
    deep_opt_4, base_lr=0.001, max_lr=0.01, step_size_up=5, cycle_momentum=False
)
optimizers_4 = {"wide": wide_opt_4, "deeptabular": deep_opt_4}
lr_schedulers_4 = {"wide": wide_sch_4, "deeptabular": deep_sch_4}


@pytest.mark.parametrize(
    "optimizers, schedulers, len_loss_output, len_lr_output, init_lr, schedulers_type",
    [
        (optimizers_1, lr_schedulers_1, 5, 5, 0.001, "step"),
        (optimizers_2, lr_schedulers_2, 5, 11, 0.001, "cyclic"),
        (optimizers_3, lr_schedulers_3, 5, 5, None, None),
        (optimizers_4, lr_schedulers_4, 5, 11, None, None),
    ],
)
def test_history_callback_w_tabtransformer(
    optimizers, schedulers, len_loss_output, len_lr_output, init_lr, schedulers_type
):
    trainer_tt = Trainer(
        model_tt,
        objective="binary",
        optimizers=optimizers,
        lr_schedulers=schedulers,
        callbacks=[LRHistory(n_epochs=5)],
        verbose=0,
    )
    trainer_tt.fit(
        X_wide=X_wide,
        X_tab=X_tab,
        target=target,
        n_epochs=5,
        batch_size=16,
    )
    out = []
    out.append(len(trainer_tt.history["train_loss"]) == len_loss_output)
    try:
        lr_list = list(chain.from_iterable(trainer_tt.lr_history["lr_deeptabular_0"]))
    except TypeError:
        lr_list = trainer_tt.lr_history["lr_deeptabular_0"]
    except Exception:
        lr_list = trainer_tt.lr_history["lr_0"]
    out.append(len(lr_list) == len_lr_output)
    if init_lr is not None and schedulers_type == "step":
        out.append(lr_list[-1] == init_lr / 10)
    elif init_lr is not None and schedulers_type == "cyclic":
        out.append(lr_list[-1] == init_lr)
    assert all(out)


def test_modelcheckpoint_mode_warning():

    fpath = "tests/test_model_functioning/modelcheckpoint/weights_out"

    with pytest.warns(RuntimeWarning):
        model_checkpoint = ModelCheckpoint(  # noqa: F841
            filepath=fpath, monitor="val_loss", mode="unknown"
        )

    shutil.rmtree("tests/test_model_functioning/modelcheckpoint/")


def test_modelcheckpoint_mode_options():

    fpath = "tests/test_model_functioning/modelcheckpoint/weights_out"

    model_checkpoint_1 = ModelCheckpoint(filepath=fpath, monitor="val_loss", mode="min")
    model_checkpoint_2 = ModelCheckpoint(filepath=fpath, monitor="val_loss")
    model_checkpoint_3 = ModelCheckpoint(filepath=fpath, monitor="acc", mode="max")
    model_checkpoint_4 = ModelCheckpoint(filepath=fpath, monitor="acc")
    model_checkpoint_5 = ModelCheckpoint(filepath=None, monitor="acc")

    is_min = model_checkpoint_1.monitor_op is np.less
    best_inf = model_checkpoint_1.best is np.Inf
    auto_is_min = model_checkpoint_2.monitor_op is np.less
    auto_best_inf = model_checkpoint_2.best is np.Inf
    is_max = model_checkpoint_3.monitor_op is np.greater
    best_minus_inf = -model_checkpoint_3.best == np.Inf
    auto_is_max = model_checkpoint_4.monitor_op is np.greater
    auto_best_minus_inf = -model_checkpoint_4.best == np.Inf
    auto_is_max = model_checkpoint_5.monitor_op is np.greater
    auto_best_minus_inf = -model_checkpoint_5.best == np.Inf

    shutil.rmtree("tests/test_model_functioning/modelcheckpoint/")

    assert all(
        [
            is_min,
            best_inf,
            is_max,
            best_minus_inf,
            auto_is_min,
            auto_best_inf,
            auto_is_max,
            auto_best_minus_inf,
        ]
    )


def test_modelcheckpoint_get_state():

    fpath = "tests/test_model_functioning/modelcheckpoint/"

    model_checkpoint = ModelCheckpoint(
        filepath="/".join([fpath, "weights_out"]), monitor="val_loss"
    )

    trainer = Trainer(
        model,
        objective="binary",
        callbacks=[model_checkpoint],
        verbose=0,
    )
    trainer.fit(
        X_wide=X_wide,
        X_tab=X_tab,
        target=target,
        n_epochs=1,
        batch_size=16,
    )

    with open("/".join([fpath, "checkpoint.p"]), "wb") as f:
        pickle.dump(model_checkpoint, f)

    with open("/".join([fpath, "checkpoint.p"]), "rb") as f:
        model_checkpoint = pickle.load(f)

    self_dict_keys = model_checkpoint.__dict__.keys()

    no_trainer = "trainer" not in self_dict_keys
    no_model = "model" not in self_dict_keys

    shutil.rmtree("tests/test_model_functioning/modelcheckpoint/")

    assert no_trainer and no_model


def test_early_stop_mode_warning():

    with pytest.warns(RuntimeWarning):
        model_checkpoint = EarlyStopping(  # noqa: F841
            monitor="val_loss", mode="unknown"
        )


def test_early_stop_mode_options():

    early_stopping_1 = EarlyStopping(monitor="val_loss", mode="min")
    early_stopping_2 = EarlyStopping(monitor="val_loss")
    early_stopping_3 = EarlyStopping(monitor="acc", mode="max")
    early_stopping_4 = EarlyStopping(monitor="acc")

    is_min = early_stopping_1.monitor_op is np.less
    auto_is_min = early_stopping_2.monitor_op is np.less
    is_max = early_stopping_3.monitor_op is np.greater
    auto_is_max = early_stopping_4.monitor_op is np.greater

    assert all(
        [
            is_min,
            is_max,
            auto_is_min,
            auto_is_max,
        ]
    )


def test_early_stopping_get_state():

    early_stopping_path = Path("tests/test_model_functioning/early_stopping")
    early_stopping_path.mkdir()

    early_stopping = EarlyStopping()

    trainer_tt = Trainer(
        model,
        objective="binary",
        callbacks=[early_stopping],
        verbose=0,
    )
    trainer_tt.fit(
        X_wide=X_wide,
        X_tab=X_tab,
        target=target,
        n_epochs=1,
        batch_size=16,
    )

    with open(early_stopping_path / "early_stopping.p", "wb") as f:
        pickle.dump(early_stopping, f)

    with open(early_stopping_path / "early_stopping.p", "rb") as f:
        early_stopping = pickle.load(f)

    self_dict_keys = early_stopping.__dict__.keys()

    no_trainer = "trainer" not in self_dict_keys
    no_model = "model" not in self_dict_keys

    shutil.rmtree("tests/test_model_functioning/early_stopping/")

    assert no_trainer and no_model


###############################################################################
# Test RayTuneReporter
###############################################################################


def test_ray_tune_reporter():

    rt_wide = Wide(np.unique(X_wide).shape[0], 1)
    rt_deeptabular = TabMlp(
        mlp_hidden_dims=[32, 16],
        mlp_dropout=[0.5, 0.5],
        column_idx=column_idx,
        cat_embed_input=embed_input,
        continuous_cols=colnames[-5:],
    )
    rt_model = WideDeep(wide=rt_wide, deeptabular=rt_deeptabular)

    config = {
        "batch_size": tune.grid_search([8, 16]),
    }

    def training_function(config):
        batch_size = config["batch_size"]

        trainer = Trainer(
            rt_model,
            objective="binary",
            callbacks=[RayTuneReporter],
            verbose=0,
        )

        trainer.fit(
            X_wide=X_wide,
            X_tab=X_tab,
            target=target,
            n_epochs=1,
            batch_size=batch_size,
        )

    analysis = tune.run(
        tune.with_parameters(training_function),
        config=config,
        resources_per_trial={"cpu": 1, "gpu": 0}
        if not torch.cuda.is_available()
        else {"cpu": 0, "gpu": 1},
        verbose=0,
    )

    assert any(["train_loss" in el for el in analysis.results_df.keys()])
