import string

import numpy as np
import pytest
from torchvision.transforms import ToTensor, Normalize
from sklearn.model_selection import train_test_split

from pytorch_widedeep.models import (
    Wide,
    DeepText,
    WideDeep,
    DeepDense,
    DeepImage,
)

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

# transforms
mean = [0.406, 0.456, 0.485]  # BGR
std = [0.225, 0.224, 0.229]  # BGR
transforms1 = [ToTensor, Normalize(mean=mean, std=std)]
transforms2 = [Normalize(mean=mean, std=std)]


##############################################################################
# Test many possible scenarios of data inputs I can think off. Surely users
# will input something unexpected
##############################################################################
@pytest.mark.parametrize(
    "X_wide, X_deep, X_text, X_img, X_train, X_val, target, val_split, transforms, nepoch, null",
    [
        (X_wide, X_deep, X_text, X_img, None, None, target, None, transforms1, 0, None),
        (X_wide, X_deep, X_text, X_img, None, None, target, None, transforms2, 0, None),
        (X_wide, X_deep, X_text, X_img, None, None, target, None, None, 0, None),
        (
            X_wide,
            X_deep,
            X_text,
            X_img_norm,
            None,
            None,
            target,
            None,
            transforms2,
            0,
            None,
        ),
        (
            X_wide,
            X_deep,
            X_text,
            X_img_norm,
            None,
            None,
            target,
            None,
            transforms1,
            0,
            None,
        ),
        (X_wide, X_deep, X_text, X_img_norm, None, None, target, None, None, 0, None),
        (X_wide, X_deep, X_text, X_img, None, None, target, 0.2, None, 0, None),
        (
            None,
            None,
            None,
            None,
            {
                "X_wide": X_wide,
                "X_deep": X_deep,
                "X_text": X_text,
                "X_img": X_img,
                "target": target,
            },
            None,
            None,
            None,
            None,
            0,
            None,
        ),
        (
            None,
            None,
            None,
            None,
            {
                "X_wide": X_wide,
                "X_deep": X_deep,
                "X_text": X_text,
                "X_img": X_img,
                "target": target,
            },
            None,
            None,
            None,
            transforms1,
            0,
            None,
        ),
        (
            None,
            None,
            None,
            None,
            {
                "X_wide": X_wide,
                "X_deep": X_deep,
                "X_text": X_text,
                "X_img": X_img,
                "target": target,
            },
            None,
            None,
            0.2,
            None,
            0,
            None,
        ),
        (
            None,
            None,
            None,
            None,
            {
                "X_wide": X_wide,
                "X_deep": X_deep,
                "X_text": X_text,
                "X_img": X_img,
                "target": target,
            },
            None,
            None,
            0.2,
            transforms2,
            0,
            None,
        ),
        (
            None,
            None,
            None,
            None,
            {
                "X_wide": X_wide_tr,
                "X_deep": X_deep_tr,
                "X_text": X_text_tr,
                "X_img": X_img_tr,
                "target": y_train,
            },
            {
                "X_wide": X_wide_val,
                "X_deep": X_deep_val,
                "X_text": X_text_val,
                "X_img": X_img_val,
                "target": y_val,
            },
            None,
            None,
            None,
            0,
            None,
        ),
        (
            None,
            None,
            None,
            None,
            {
                "X_wide": X_wide_tr,
                "X_deep": X_deep_tr,
                "X_text": X_text_tr,
                "X_img": X_img_tr,
                "target": y_train,
            },
            {
                "X_wide": X_wide_val,
                "X_deep": X_deep_val,
                "X_text": X_text_val,
                "X_img": X_img_val,
                "target": y_val,
            },
            None,
            None,
            transforms1,
            0,
            None,
        ),
    ],
)
def test_widedeep_inputs(
    X_wide,
    X_deep,
    X_text,
    X_img,
    X_train,
    X_val,
    target,
    val_split,
    transforms,
    nepoch,
    null,
):
    model = WideDeep(
        wide=wide, deepdense=deepdense, deeptext=deeptext, deepimage=deepimage
    )
    model.compile(method="binary", transforms=transforms, verbose=0)
    model.fit(
        X_wide=X_wide,
        X_deep=X_deep,
        X_text=X_text,
        X_img=X_img,
        X_train=X_train,
        X_val=X_val,
        target=target,
        val_split=val_split,
        batch_size=16,
    )
    assert (
        model.history.epoch[0] == nepoch
        and model.history._history["train_loss"] is not null
    )
