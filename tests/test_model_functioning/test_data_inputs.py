import string

import numpy as np
import pytest
from torch import nn
from torchvision.transforms import ToTensor, Normalize
from sklearn.model_selection import train_test_split

from pytorch_widedeep.models import Wide, TabMlp, DeepText, WideDeep, DeepImage
from pytorch_widedeep.training import Trainer

np.random.seed(1)

# Wide array
X_wide = np.random.choice(50, (32, 10))

# Deep Array
colnames = list(string.ascii_lowercase)[:10]
embed_cols = [np.random.choice(np.arange(5), 32) for _ in range(5)]
embed_input = [(u, i, j) for u, i, j in zip(colnames[:5], [5] * 5, [16] * 5)]
cont_cols = [np.random.rand(32) for _ in range(5)]
X_tab = np.vstack(embed_cols + cont_cols).transpose()

# Text Array
padded_sequences = np.random.choice(np.arange(1, 100), (32, 48))
X_text = np.hstack((np.repeat(np.array([[0, 0]]), 32, axis=0), padded_sequences))
vocab_size = 110

# Image Array
X_img = np.random.choice(256, (32, 224, 224, 3))
X_img_norm = X_img / 255.0

# Target
target = np.random.choice(2, 32)

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
deeptabular = TabMlp(
    mlp_hidden_dims=[32, 16],
    mlp_dropout=[0.5, 0.5],
    column_idx={k: v for v, k in enumerate(colnames)},
    embed_input=embed_input,
    continuous_cols=colnames[-5:],
)
deeptext = DeepText(vocab_size=vocab_size, embed_dim=32, padding_idx=0)
deepimage = DeepImage(pretrained=True)

# transforms
mean = [0.406, 0.456, 0.485]  # BGR
std = [0.225, 0.224, 0.229]  # BGR
transforms1 = [ToTensor, Normalize(mean=mean, std=std)]
transforms2 = [Normalize(mean=mean, std=std)]

deephead_ds = nn.Sequential(nn.Linear(16, 8), nn.Linear(8, 4))
deephead_dt = nn.Sequential(nn.Linear(64, 8), nn.Linear(8, 4))
deephead_di = nn.Sequential(nn.Linear(512, 8), nn.Linear(8, 4))

# #############################################################################
# Test that runs many possible scenarios of data inputs I can think off.
# Surely users will input something unexpected
# #############################################################################


@pytest.mark.parametrize(
    "X_wide, X_tab, X_text, X_img, X_train, X_val, target, val_split, transforms",
    [
        (X_wide, X_tab, X_text, X_img, None, None, target, None, transforms1),
        (X_wide, X_tab, X_text, X_img, None, None, target, None, transforms2),
        (X_wide, X_tab, X_text, X_img, None, None, target, None, None),
        (
            X_wide,
            X_tab,
            X_text,
            X_img_norm,
            None,
            None,
            target,
            None,
            transforms2,
        ),
        (
            X_wide,
            X_tab,
            X_text,
            X_img_norm,
            None,
            None,
            target,
            None,
            transforms1,
        ),
        (X_wide, X_tab, X_text, X_img_norm, None, None, target, None, None),
        (X_wide, X_tab, X_text, X_img, None, None, target, 0.2, None),
        (
            None,
            None,
            None,
            None,
            {
                "X_wide": X_wide,
                "X_tab": X_tab,
                "X_text": X_text,
                "X_img": X_img,
                "target": target,
            },
            None,
            None,
            None,
            None,
        ),
        (
            None,
            None,
            None,
            None,
            {
                "X_wide": X_wide,
                "X_tab": X_tab,
                "X_text": X_text,
                "X_img": X_img,
                "target": target,
            },
            None,
            None,
            None,
            transforms1,
        ),
        (
            None,
            None,
            None,
            None,
            {
                "X_wide": X_wide,
                "X_tab": X_tab,
                "X_text": X_text,
                "X_img": X_img,
                "target": target,
            },
            None,
            None,
            0.2,
            None,
        ),
        (
            None,
            None,
            None,
            None,
            {
                "X_wide": X_wide,
                "X_tab": X_tab,
                "X_text": X_text,
                "X_img": X_img,
                "target": target,
            },
            None,
            None,
            0.2,
            transforms2,
        ),
        (
            None,
            None,
            None,
            None,
            {
                "X_wide": X_wide_tr,
                "X_tab": X_tab_tr,
                "X_text": X_text_tr,
                "X_img": X_img_tr,
                "target": y_train,
            },
            {
                "X_wide": X_wide_val,
                "X_tab": X_tab_val,
                "X_text": X_text_val,
                "X_img": X_img_val,
                "target": y_val,
            },
            None,
            None,
            None,
        ),
        (
            None,
            None,
            None,
            None,
            {
                "X_wide": X_wide_tr,
                "X_tab": X_tab_tr,
                "X_text": X_text_tr,
                "X_img": X_img_tr,
                "target": y_train,
            },
            {
                "X_wide": X_wide_val,
                "X_tab": X_tab_val,
                "X_text": X_text_val,
                "X_img": X_img_val,
                "target": y_val,
            },
            None,
            None,
            transforms1,
        ),
    ],
)
def test_widedeep_inputs(
    X_wide,
    X_tab,
    X_text,
    X_img,
    X_train,
    X_val,
    target,
    val_split,
    transforms,
):
    model = WideDeep(
        wide=wide, deeptabular=deeptabular, deeptext=deeptext, deepimage=deepimage
    )
    trainer = Trainer(model, objective="binary", transforms=transforms, verbose=0)
    trainer.fit(
        X_wide=X_wide,
        X_tab=X_tab,
        X_text=X_text,
        X_img=X_img,
        X_train=X_train,
        X_val=X_val,
        target=target,
        val_split=val_split,
        batch_size=16,
    )
    assert trainer.history["train_loss"] is not None


@pytest.mark.parametrize(
    "X_wide, X_tab, X_text, X_img, X_train, X_val, target",
    [
        (
            X_wide,
            X_tab,
            X_text,
            X_img,
            None,
            {
                "X_wide": X_wide_val,
                "X_tab": X_tab_val,
                "X_text": X_text_val,
                "X_img": X_img_val,
                "target": y_val,
            },
            target,
        ),
    ],
)
def test_xtrain_xval_assertion(
    X_wide,
    X_tab,
    X_text,
    X_img,
    X_train,
    X_val,
    target,
):
    model = WideDeep(
        wide=wide, deeptabular=deeptabular, deeptext=deeptext, deepimage=deepimage
    )
    trainer = Trainer(model, objective="binary", verbose=0)
    with pytest.raises(AssertionError):
        trainer.fit(
            X_wide=X_wide,
            X_tab=X_tab,
            X_text=X_text,
            X_img=X_img,
            X_train=X_train,
            X_val=X_val,
            target=target,
            batch_size=16,
        )


@pytest.mark.parametrize(
    "wide, deeptabular, deeptext, deepimage, X_wide, X_tab, X_text, X_img, target",
    [
        (wide, None, None, None, X_wide, None, None, None, target),
        (None, deeptabular, None, None, None, X_tab, None, None, target),
        (None, None, deeptext, None, None, None, X_text, None, target),
        (None, None, None, deepimage, None, None, None, X_img, target),
    ],
)
def test_individual_inputs(
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
    # check it has run succesfully
    assert len(trainer.history) == 1


###############################################################################
# test deephead is not None and individual components
###############################################################################


@pytest.mark.parametrize(
    "deeptabular, deeptext, deepimage, X_tab, X_text, X_img, deephead, target",
    [
        (deeptabular, None, None, X_tab, None, None, deephead_ds, target),
        (None, deeptext, None, None, X_text, None, deephead_dt, target),
        (None, None, deepimage, None, None, X_img, deephead_di, target),
    ],
)
def test_deephead_individual_components(
    deeptabular, deeptext, deepimage, X_tab, X_text, X_img, deephead, target
):
    model = WideDeep(
        deeptabular=deeptabular,
        deeptext=deeptext,
        deepimage=deepimage,
        deephead=deephead,
    )  # noqa: F841
    trainer = Trainer(model, objective="binary", verbose=0)
    trainer.fit(
        X_wide=X_wide,
        X_tab=X_tab,
        X_text=X_text,
        X_img=X_img,
        target=target,
        batch_size=16,
    )
    # check it has run succesfully
    assert len(trainer.history) == 1


###############################################################################
# test deephead is None and head_layers is not None and individual components
###############################################################################


@pytest.mark.parametrize(
    "deeptabular, deeptext, deepimage, X_tab, X_text, X_img, target",
    [
        (deeptabular, None, None, X_tab, None, None, target),
        (None, deeptext, None, None, X_text, None, target),
        (None, None, deepimage, None, None, X_img, target),
    ],
)
def test_head_layers_individual_components(
    deeptabular, deeptext, deepimage, X_tab, X_text, X_img, target
):
    model = WideDeep(
        deeptabular=deeptabular,
        deeptext=deeptext,
        deepimage=deepimage,
        head_hidden_dims=[8, 4],
    )  # noqa: F841
    trainer = Trainer(model, objective="binary", verbose=0)
    trainer.fit(
        X_wide=X_wide,
        X_tab=X_tab,
        X_text=X_text,
        X_img=X_img,
        target=target,
        batch_size=16,
    )
    # check it has run succesfully
    assert len(trainer.history) == 1
