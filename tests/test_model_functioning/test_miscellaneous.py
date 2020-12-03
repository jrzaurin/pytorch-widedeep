import string

import numpy as np
import torch
import pytest
from sklearn.model_selection import train_test_split

from pytorch_widedeep.models import (
    Wide,
    DeepText,
    WideDeep,
    DeepDense,
    DeepImage,
)
from pytorch_widedeep.metrics import Accuracy, Precision
from pytorch_widedeep.callbacks import EarlyStopping

# Wide array
X_wide = np.random.choice(50, (32, 10))

# Deep Array
colnames = list(string.ascii_lowercase)[:10]
embed_cols = [np.random.choice(np.arange(5), 32) for _ in range(5)]
embed_input = [(u, i, j) for u, i, j in zip(colnames[:5], [5] * 5, [16] * 5)]
cont_cols = [np.random.rand(32) for _ in range(5)]
X_deep = np.vstack(embed_cols + cont_cols).transpose()

#  Text Array
padded_sequences = np.random.choice(np.arange(1, 100), (32, 48))
X_text = np.hstack((np.repeat(np.array([[0, 0]]), 32, axis=0), padded_sequences))
vocab_size = 100

#  Image Array
X_img = np.random.choice(256, (32, 224, 224, 3))
X_img_norm = X_img / 255.0

# Target
target = np.random.choice(2, 32)
target_multi = np.random.choice(3, 32)

# train/validation split
(
    X_wide_tr,
    X_wide_val,
    X_deep_tr,
    X_deep_val,
    X_text_tr,
    X_text_val,
    X_img_tr,
    X_img_val,
    y_train,
    y_val,
) = train_test_split(X_wide, X_deep, X_text, X_img, target)

# build model components
wide = Wide(np.unique(X_wide).shape[0], 1)
deepdense = DeepDense(
    hidden_layers=[32, 16],
    dropout=[0.5, 0.5],
    deep_column_idx={k: v for v, k in enumerate(colnames)},
    embed_input=embed_input,
    continuous_cols=colnames[-5:],
)
deeptext = DeepText(vocab_size=vocab_size, embed_dim=32, padding_idx=0)
deepimage = DeepImage(pretrained=True)

###############################################################################
#  test consistecy between optimizers and lr_schedulers format
###############################################################################


def test_optimizer_scheduler_format():
    model = WideDeep(deepdense=deepdense)
    optimizers = {"deepdense": torch.optim.Adam(model.deepdense.parameters(), lr=0.01)}
    schedulers = torch.optim.lr_scheduler.StepLR(optimizers["deepdense"], step_size=3)
    with pytest.raises(ValueError):
        model.compile(
            method="binary",
            optimizers=optimizers,
            lr_schedulers=schedulers,
        )


###############################################################################
#  test that callbacks are properly initialised internally
###############################################################################


def test_non_instantiated_callbacks():
    model = WideDeep(wide=wide, deepdense=deepdense)
    callbacks = [EarlyStopping]
    model.compile(method="binary", callbacks=callbacks)
    assert model.callbacks[1].__class__.__name__ == "EarlyStopping"


###############################################################################
#  test that multiple metrics are properly constructed internally
###############################################################################


def test_multiple_metrics():
    model = WideDeep(wide=wide, deepdense=deepdense)
    metrics = [Accuracy, Precision]
    model.compile(method="binary", metrics=metrics)
    assert (
        model.metric._metrics[0].__class__.__name__ == "Accuracy"
        and model.metric._metrics[1].__class__.__name__ == "Precision"
    )


###############################################################################
#  test the train step with metrics runs well for a binary prediction
###############################################################################


def test_basic_run_with_metrics_binary():
    model = WideDeep(wide=wide, deepdense=deepdense)
    model.compile(method="binary", metrics=[Accuracy], verbose=False)
    model.fit(
        X_wide=X_wide,
        X_deep=X_deep,
        target=target,
        n_epochs=1,
        batch_size=16,
        val_split=0.2,
    )
    assert (
        "train_loss" in model.history._history.keys()
        and "train_acc" in model.history._history.keys()
    )


###############################################################################
#  test the train step with metrics runs well for a muticlass prediction
###############################################################################


def test_basic_run_with_metrics_multiclass():
    wide = Wide(np.unique(X_wide).shape[0], 3)
    deepdense = DeepDense(
        hidden_layers=[32, 16],
        dropout=[0.5, 0.5],
        deep_column_idx={k: v for v, k in enumerate(colnames)},
        embed_input=embed_input,
        continuous_cols=colnames[-5:],
    )
    model = WideDeep(wide=wide, deepdense=deepdense, pred_dim=3)
    model.compile(method="multiclass", metrics=[Accuracy], verbose=False)
    model.fit(
        X_wide=X_wide,
        X_deep=X_deep,
        target=target_multi,
        n_epochs=1,
        batch_size=16,
        val_split=0.2,
    )
    assert (
        "train_loss" in model.history._history.keys()
        and "train_acc" in model.history._history.keys()
    )


###############################################################################
#  test predict method for individual components
###############################################################################


@pytest.mark.parametrize(
    "wide, deepdense, deeptext, deepimage, X_wide, X_deep, X_text, X_img, target",
    [
        (wide, None, None, None, X_wide, None, None, None, target),
        (None, deepdense, None, None, None, X_deep, None, None, target),
        (None, None, deeptext, None, None, None, X_text, None, target),
        (None, None, None, deepimage, None, None, None, X_img, target),
    ],
)
def test_predict_with_individual_component(
    wide, deepdense, deeptext, deepimage, X_wide, X_deep, X_text, X_img, target
):

    model = WideDeep(
        wide=wide, deepdense=deepdense, deeptext=deeptext, deepimage=deepimage
    )
    model.compile(method="binary", verbose=0)
    model.fit(
        X_wide=X_wide,
        X_deep=X_deep,
        X_text=X_text,
        X_img=X_img,
        target=target,
        batch_size=16,
    )
    # simply checking that runs and produces outputs
    preds = model.predict(X_wide=X_wide, X_deep=X_deep, X_text=X_text, X_img=X_img)

    assert preds.shape[0] == 32 and "train_loss" in model.history._history
