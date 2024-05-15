import os

import cv2
import numpy as np
import torch
import pandas as pd
import pytest
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import TabMlp, Vision, BasicRNN, WideDeep
from pytorch_widedeep.metrics import F1Score, Accuracy
from pytorch_widedeep.initializers import XavierNormal, KaimingNormal
from pytorch_widedeep.preprocessing import TabPreprocessor, TextPreprocessor

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = "/".join([current_dir, "load_from_folder_test_data"])

if not os.path.exists(data_dir):
    raise FileNotFoundError("The data directory does not exist")

train_df = pd.read_csv(data_dir + "/train.csv")
valid_df = pd.read_csv(data_dir + "/val.csv")
test_df = pd.read_csv(data_dir + "/test.csv")

text_cols = ["text_col1", "text_col2"]
img_cols = ["image_col1", "image_col2"]

text_preprocessor_1 = TextPreprocessor(
    text_col=text_cols[0], max_vocab=100, min_freq=2, maxlen=10, n_cpus=1, verbose=0
)
X_text_tr_1 = text_preprocessor_1.fit_transform(train_df)
X_text_val_1 = text_preprocessor_1.transform(valid_df)
# in the real world, one would merge train and valid and refit the preprocessor
X_text_te_1 = text_preprocessor_1.transform(test_df)

text_preprocessor_2 = TextPreprocessor(
    text_col=text_cols[1], max_vocab=100, min_freq=2, maxlen=10, n_cpus=1, verbose=0
)
X_text_tr_2 = text_preprocessor_2.fit_transform(train_df)
X_text_val_2 = text_preprocessor_2.transform(valid_df)
X_text_te_2 = text_preprocessor_2.transform(test_df)

# use the training, validation and test sets to load the corresponding images in the image cols
X_img_tr_1 = np.asarray(
    [cv2.imread(data_dir + "/images/" + img) for img in train_df[img_cols[0]].values]
)
X_img_val_1 = np.asarray(
    [cv2.imread(data_dir + "/images/" + img) for img in valid_df[img_cols[0]].values]
)
X_img_te_1 = np.asarray(
    [cv2.imread(data_dir + "/images/" + img) for img in test_df[img_cols[0]].values]
)

X_img_tr_2 = np.asarray(
    [cv2.imread(data_dir + "/images/" + img) for img in train_df[img_cols[1]].values]
)
X_img_val_2 = np.asarray(
    [cv2.imread(data_dir + "/images/" + img) for img in valid_df[img_cols[1]].values]
)
X_img_te_2 = np.asarray(
    [cv2.imread(data_dir + "/images/" + img) for img in test_df[img_cols[1]].values]
)

tab_preprocessor = TabPreprocessor(
    embed_cols=["cat_col"], continuous_cols=["num_col"], default_embed_dim=4
)
X_tab_tr = tab_preprocessor.fit_transform(train_df)
X_tab_val = tab_preprocessor.transform(valid_df)
X_tab_te = tab_preprocessor.transform(test_df)

tab_mlp = TabMlp(
    column_idx=tab_preprocessor.column_idx,
    cat_embed_input=tab_preprocessor.cat_embed_input,
    continuous_cols=tab_preprocessor.continuous_cols,
    mlp_hidden_dims=[16, 4],
)

vision_1 = Vision(
    channel_sizes=[16, 32],
    kernel_sizes=[3, 3],
    strides=[1, 1],
    head_hidden_dims=[16, 8],
)

vision_2 = Vision(
    channel_sizes=[16, 32],
    kernel_sizes=[3, 3],
    strides=[1, 1],
    head_hidden_dims=[16, 4],  # just to make the head_hidden_dims different
)

rnn_1 = BasicRNN(
    vocab_size=len(text_preprocessor_1.vocab.itos),
    embed_dim=16,
    hidden_dim=16,
    n_layers=1,
    bidirectional=False,
    head_hidden_dims=[16, 8],
)

rnn_2 = BasicRNN(
    vocab_size=len(text_preprocessor_2.vocab.itos),
    embed_dim=16,
    hidden_dim=16,
    n_layers=1,
    bidirectional=False,
    head_hidden_dims=[16, 4],  # just to make the head_hidden_dims different
)

model = WideDeep(
    deeptabular=tab_mlp,
    deeptext=[rnn_1, rnn_2],
    deepimage=[vision_1, vision_2],
    pred_dim=1,
)


@pytest.mark.parametrize(
    "X_tab, X_text, X_img, X_train, X_val, val_split, target",
    [
        (
            X_tab_tr,
            [X_text_tr_1, X_text_tr_2],
            [X_img_tr_1, X_img_tr_2],
            None,
            None,
            None,
            train_df["target"].values,
        ),
        (
            X_tab_tr,
            [X_text_tr_1, X_text_tr_2],
            [X_img_tr_1, X_img_tr_2],
            None,
            None,
            0.2,
            train_df["target"].values,
        ),
        (
            None,
            None,
            None,
            {
                "X_tab": X_tab_tr,
                "X_text": [X_text_tr_1, X_text_tr_2],
                "X_img": [X_img_tr_1, X_img_tr_2],
                "target": train_df["target"].values,
            },
            {
                "X_tab": X_tab_val,
                "X_text": [X_text_val_1, X_text_val_2],
                "X_img": [X_img_val_1, X_img_val_2],
                "target": valid_df["target"].values,
            },
            None,
            None,
        ),
        (
            None,
            None,
            None,
            {
                "X_tab": X_tab_tr,
                "X_text": [X_text_tr_1, X_text_tr_2],
                "X_img": [X_img_tr_1, X_img_tr_2],
                "target": train_df["target"].values,
            },
            None,
            0.2,
            None,
        ),
    ],
)
def test_multi_text_or_image_cols(
    X_tab, X_text, X_img, X_train, X_val, val_split, target
):

    trainer = Trainer(
        model,
        objective="binary",
    )

    trainer.fit(
        X_tab=X_tab,
        X_text=X_text,
        X_img=X_img,
        X_train=X_train,
        X_val=X_val,
        val_split=val_split,
        target=target,
        n_epochs=1,
        batch_size=4,
    )

    return True
