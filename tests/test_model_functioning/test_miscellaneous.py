import os
import shutil
import string
from copy import deepcopy

import numpy as np
import torch
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from pytorch_widedeep.models import (
    Wide,
    TabMlp,
    TabNet,
    DeepText,
    WideDeep,
    DeepImage,
    TabResnet,
    TabTransformer,
)
from pytorch_widedeep.metrics import Accuracy, Precision
from pytorch_widedeep.training import Trainer
from pytorch_widedeep.callbacks import EarlyStopping
from pytorch_widedeep.preprocessing import TabPreprocessor

# Wide array
X_wide = np.random.choice(50, (32, 10))

# Deep Array
colnames = list(string.ascii_lowercase)[:10]
embed_cols = [np.random.choice(np.arange(5), 32) for _ in range(5)]
embed_input = [(u, i, j) for u, i, j in zip(colnames[:5], [5] * 5, [16] * 5)]
embed_input_tt = [(u, i) for u, i in zip(colnames[:5], [5] * 5)]
cont_cols = [np.random.rand(32) for _ in range(5)]
X_tab = np.vstack(embed_cols + cont_cols).transpose()

# Text Array
padded_sequences = np.random.choice(np.arange(1, 100), (32, 48))
X_text = np.hstack((np.repeat(np.array([[0, 0]]), 32, axis=0), padded_sequences))
vocab_size = 100

# Image Array
X_img = np.random.choice(256, (32, 224, 224, 3))
X_img_norm = X_img / 255.0

# Target
target = np.random.choice(2, 32)
target_multi = np.random.choice(3, 32)

# train/validation split
(
    X_wide_tr,
    X_wide_val,
    X_tab_tr,
    X_tab_val,
    X_text_tr,
    X_text_val,
    X_img_tr,
    X_img_val,
    y_train,
    y_val,
) = train_test_split(X_wide, X_tab, X_text, X_img, target)

# build model components
wide = Wide(np.unique(X_wide).shape[0], 1)
tabmlp = TabMlp(
    mlp_hidden_dims=[32, 16],
    mlp_dropout=[0.5, 0.5],
    column_idx={k: v for v, k in enumerate(colnames)},
    embed_input=embed_input,
    continuous_cols=colnames[-5:],
)
tabresnet = TabResnet(
    blocks_dims=[32, 16],
    column_idx={k: v for v, k in enumerate(colnames)},
    embed_input=embed_input,
    continuous_cols=colnames[-5:],
)
tabtransformer = TabTransformer(
    column_idx={k: v for v, k in enumerate(colnames)},
    embed_input=embed_input_tt,
    continuous_cols=colnames[5:],
)
tabnet = TabNet(
    column_idx={k: v for v, k in enumerate(colnames)},
    embed_input=embed_input,
    continuous_cols=colnames[5:],
    ghost_bn=False,
)
deeptext = DeepText(vocab_size=vocab_size, embed_dim=32, padding_idx=0)
deepimage = DeepImage(pretrained=True)

###############################################################################
# test consistecy between optimizers and lr_schedulers format
###############################################################################


def test_optimizer_scheduler_format():
    model = WideDeep(deeptabular=tabmlp)
    optimizers = {
        "deeptabular": torch.optim.Adam(model.deeptabular.parameters(), lr=0.01)
    }
    schedulers = torch.optim.lr_scheduler.StepLR(optimizers["deeptabular"], step_size=3)
    with pytest.raises(ValueError):
        trainer = Trainer(  # noqa: F841
            model,
            objective="binary",
            optimizers=optimizers,
            lr_schedulers=schedulers,
        )


###############################################################################
# test that callbacks are properly initialised internally
###############################################################################


def test_non_instantiated_callbacks():
    model = WideDeep(wide=wide, deeptabular=tabmlp)
    callbacks = [EarlyStopping]
    trainer = Trainer(model, objective="binary", callbacks=callbacks)
    assert trainer.callbacks[2].__class__.__name__ == "EarlyStopping"


###############################################################################
# test that multiple metrics are properly constructed internally
###############################################################################


def test_multiple_metrics():
    model = WideDeep(wide=wide, deeptabular=tabmlp)
    metrics = [Accuracy, Precision]
    trainer = Trainer(model, objective="binary", metrics=metrics)
    assert (
        trainer.metric._metrics[0].__class__.__name__ == "Accuracy"
        and trainer.metric._metrics[1].__class__.__name__ == "Precision"
    )


###############################################################################
# test the train step with metrics runs well for a binary prediction
###############################################################################


@pytest.mark.parametrize(
    "wide, deeptabular",
    [
        (wide, tabmlp),
        (wide, tabresnet),
        (wide, tabtransformer),
    ],
)
def test_basic_run_with_metrics_binary(wide, deeptabular):
    model = WideDeep(wide=wide, deeptabular=deeptabular)
    trainer = Trainer(model, objective="binary", metrics=[Accuracy], verbose=False)
    trainer.fit(
        X_wide=X_wide,
        X_tab=X_tab,
        target=target,
        n_epochs=1,
        batch_size=16,
        val_split=0.2,
    )
    assert (
        "train_loss" in trainer.history.keys() and "train_acc" in trainer.history.keys()
    )


###############################################################################
# test the train step with metrics runs well for a muticlass prediction
###############################################################################


def test_basic_run_with_metrics_multiclass():
    wide = Wide(np.unique(X_wide).shape[0], 3)
    deeptabular = TabMlp(
        mlp_hidden_dims=[32, 16],
        mlp_dropout=[0.5, 0.5],
        column_idx={k: v for v, k in enumerate(colnames)},
        embed_input=embed_input,
        continuous_cols=colnames[-5:],
    )
    model = WideDeep(wide=wide, deeptabular=deeptabular, pred_dim=3)
    trainer = Trainer(model, objective="multiclass", metrics=[Accuracy], verbose=False)
    trainer.fit(
        X_wide=X_wide,
        X_tab=X_tab,
        target=target_multi,
        n_epochs=1,
        batch_size=16,
        val_split=0.2,
    )
    assert (
        "train_loss" in trainer.history.keys() and "train_acc" in trainer.history.keys()
    )


###############################################################################
# test predict method for individual components
###############################################################################


@pytest.mark.parametrize(
    "wide, deeptabular, deeptext, deepimage, X_wide, X_tab, X_text, X_img, target",
    [
        (wide, None, None, None, X_wide, None, None, None, target),
        (None, tabmlp, None, None, None, X_tab, None, None, target),
        (None, tabresnet, None, None, None, X_tab, None, None, target),
        (None, tabtransformer, None, None, None, X_tab, None, None, target),
        (None, None, deeptext, None, None, None, X_text, None, target),
        (None, None, None, deepimage, None, None, None, X_img, target),
    ],
)
def test_predict_with_individual_component(
    wide, deeptabular, deeptext, deepimage, X_wide, X_tab, X_text, X_img, target
):

    model = WideDeep(
        wide=wide, deeptabular=deeptabular, deeptext=deeptext, deepimage=deepimage
    )
    trainer = Trainer(model, objective="binary", verbose=0)
    trainer.fit(
        X_wide=X_wide,
        X_tab=X_tab,
        X_text=X_text,
        X_img=X_img,
        target=target,
        batch_size=16,
    )
    # simply checking that runs and produces outputs
    preds = trainer.predict(X_wide=X_wide, X_tab=X_tab, X_text=X_text, X_img=X_img)

    assert preds.shape[0] == 32 and "train_loss" in trainer.history


###############################################################################
# test save
###############################################################################


def test_save_and_load():
    model = WideDeep(wide=wide, deeptabular=tabmlp)
    trainer = Trainer(model, objective="binary", verbose=0)
    trainer.fit(X_wide=X_wide, X_tab=X_tab, target=target, batch_size=16)
    wide_weights = model.wide.wide_linear.weight.data
    trainer.save("tests/test_model_functioning/model_dir/")
    n_model = torch.load("tests/test_model_functioning/model_dir/wd_model.pt")
    n_wide_weights = n_model.wide.wide_linear.weight.data
    assert torch.allclose(wide_weights, n_wide_weights)


def test_save_and_load_dict():
    wide = Wide(np.unique(X_wide).shape[0], 1)
    tabmlp = TabMlp(
        mlp_hidden_dims=[32, 16],
        column_idx={k: v for v, k in enumerate(colnames)},
        embed_input=embed_input,
        continuous_cols=colnames[-5:],
    )
    model1 = WideDeep(wide=deepcopy(wide), deeptabular=deepcopy(tabmlp))
    trainer1 = Trainer(model1, objective="binary", verbose=0)
    trainer1.fit(
        X_wide=X_wide,
        X_tab=X_tab,
        X_text=X_text,
        X_img=X_img,
        target=target,
        batch_size=16,
    )
    wide_weights = model1.wide.wide_linear.weight.data
    trainer1.save(path="tests/test_model_functioning/model_dir/", save_state_dict=True)
    model2 = WideDeep(wide=wide, deeptabular=tabmlp)
    trainer2 = Trainer(model2, objective="binary", verbose=0)
    trainer2.model.load_state_dict(
        torch.load("tests/test_model_functioning/model_dir/wd_model.pt")
    )
    n_wide_weights = trainer2.model.wide.wide_linear.weight.data
    same_weights = torch.allclose(wide_weights, n_wide_weights)
    if os.path.isfile(
        "tests/test_model_functioning/model_dir/history/train_eval_history.json"
    ):
        history_saved = True
    else:
        history_saved = False
    shutil.rmtree("tests/test_model_functioning/model_dir/")
    assert same_weights and history_saved


###############################################################################
# test explain matrices and feature importance for TabNet
###############################################################################


def test_explain_mtx_and_feat_imp():
    model = WideDeep(deeptabular=tabnet)
    trainer = Trainer(model, objective="binary", verbose=0)
    trainer.fit(
        X_tab=X_tab,
        target=target,
        batch_size=16,
    )

    checks = []
    checks.append(len(trainer.feature_importance) == len(tabnet.column_idx))

    expl_mtx, step_masks = trainer.explain(X_tab[:6], save_step_masks=True)
    checks.append(expl_mtx.shape[0] == 6)
    checks.append(expl_mtx.shape[1] == 10)

    for i in range(tabnet.n_steps):
        checks.append(step_masks[i].shape[0] == 6)
        checks.append(step_masks[i].shape[1] == 10)

    assert all(checks)


###############################################################################
# test save load and predict
###############################################################################


def test_save_load_and_predict():

    fpath = "tests/test_model_functioning/test_wd_model"
    if not os.path.exists(fpath):
        os.makedirs(fpath)

    model = WideDeep(deeptabular=tabmlp)
    trainer = Trainer(model, objective="binary", verbose=0)
    trainer.fit(
        X_tab=X_tab,
        target=target,
        batch_size=16,
    )

    trainer.save(path=fpath, save_state_dict=True)

    model_new = WideDeep(deeptabular=tabmlp)
    model_new.load_state_dict(torch.load("/".join([fpath, "wd_model.pt"])))
    trainer_new = Trainer(model, objective="binary", verbose=0)
    preds = trainer_new.predict(X_tab=X_tab, batch_size=16)

    shutil.rmtree(fpath)

    assert preds.shape[0] == X_tab.shape[0]


###############################################################################
# test get_embeddings DeprecationWarning
###############################################################################


def create_test_dataset(input_type, input_type_2=None):
    df = pd.DataFrame()
    col1 = list(np.random.choice(input_type, 32))
    if input_type_2 is not None:
        col2 = list(np.random.choice(input_type_2, 32))
    else:
        col2 = list(np.random.choice(input_type, 32))
    df["col1"], df["col2"] = col1, col2
    return df


some_letters = ["a", "b", "c", "d", "e"]

df = create_test_dataset(some_letters)
df["col3"] = np.round(np.random.rand(32), 3)
df["col4"] = np.round(np.random.rand(32), 3)
df["target"] = np.random.choice(2, 32)


def test_get_embeddings_deprecation_warning():

    embed_cols = [("col1", 5), ("col2", 5)]
    continuous_cols = ["col3", "col4"]

    tab_preprocessor = TabPreprocessor(
        embed_cols=embed_cols, continuous_cols=continuous_cols
    )
    X_tab = tab_preprocessor.fit_transform(df)
    target = df.target.values

    tabmlp = TabMlp(
        mlp_hidden_dims=[32, 16],
        mlp_dropout=[0.5, 0.5],
        column_idx={k: v for v, k in enumerate(df.columns)},
        embed_input=tab_preprocessor.embeddings_input,
        continuous_cols=tab_preprocessor.continuous_cols,
    )

    model = WideDeep(deeptabular=tabmlp)
    trainer = Trainer(model, objective="binary", verbose=0)
    trainer.fit(
        X_tab=X_tab,
        target=target,
        batch_size=16,
    )

    with pytest.warns(DeprecationWarning):
        trainer.get_embeddings(
            col_name="col1",
            cat_encoding_dict=tab_preprocessor.label_encoder.encoding_dict,
        )
