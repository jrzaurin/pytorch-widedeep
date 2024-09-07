# NOTE FOR ME: Most of the tests here could go to
# the 'test_multi_text_or_image_cols.py' script. However the functionality to
# add multiple tabular model components came later and I prefer to write
# separate tests for it. There is going to be a lot of code repetition.
import os

import numpy as np
import torch
import pandas as pd
import pytest

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import TabMlp, TabNet, BasicRNN, WideDeep, ModelFuser
from pytorch_widedeep.metrics import F1Score, Accuracy
from pytorch_widedeep.callbacks import LRHistory
from pytorch_widedeep.initializers import XavierNormal, KaimingNormal
from pytorch_widedeep.preprocessing import TabPreprocessor, TextPreprocessor
from pytorch_widedeep.models._base_wd_model_component import BaseWDModelComponent


class CustomHead(BaseWDModelComponent):

    def __init__(self, input_units: int, output_units: int):
        super(CustomHead, self).__init__()
        self.fc = torch.nn.Linear(input_units, output_units)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.fc(X)

    @property
    def output_dim(self) -> int:
        return self.fc.out_features


current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = "/".join([current_dir, "data_for_muti_tabular_components"])

if not os.path.exists(data_dir):
    raise FileNotFoundError("The data directory does not exist")

train_df = pd.read_csv(data_dir + "/train.csv")
valid_df = pd.read_csv(data_dir + "/val.csv")
test_df = pd.read_csv(data_dir + "/test.csv")

user_cols = ["age", "gender", "location"]
item_cols = ["price", "color", "category"]

tab_preprocessor_user = TabPreprocessor(
    cat_embed_cols=["gender", "location"],
    continuous_cols=["age"],
    cols_to_scale=["age"],
)
X_tab_user_tr = tab_preprocessor_user.fit_transform(train_df)
X_tab_user_val = tab_preprocessor_user.transform(valid_df)
# in the real world, one would merge train and valid and refit the preprocessor
X_tab_user_te = tab_preprocessor_user.transform(test_df)

tab_preprocessor_item = TabPreprocessor(
    cat_embed_cols=["color", "category"],
    continuous_cols=["price"],
    cols_to_scale=["price"],
)
X_tab_item_tr = tab_preprocessor_item.fit_transform(train_df)
X_tab_item_val = tab_preprocessor_item.transform(valid_df)
X_tab_item_te = tab_preprocessor_item.transform(test_df)

text_cols = ["review", "description"]

text_preprocessor_reviews = TextPreprocessor(
    text_col="review", max_vocab=100, min_freq=2, maxlen=10, n_cpus=1, verbose=0
)
X_text_review_tr = text_preprocessor_reviews.fit_transform(train_df)
X_text_review_val = text_preprocessor_reviews.transform(valid_df)
X_text_review_te = text_preprocessor_reviews.transform(test_df)

text_preprocessor_descriptions = TextPreprocessor(
    text_col="description", max_vocab=100, min_freq=2, maxlen=10, n_cpus=1, verbose=0
)
X_text_description_tr = text_preprocessor_descriptions.fit_transform(train_df)
X_text_description_val = text_preprocessor_descriptions.transform(valid_df)
X_text_description_te = text_preprocessor_descriptions.transform(test_df)

tab_mlp_user = TabMlp(
    column_idx=tab_preprocessor_user.column_idx,
    cat_embed_input=tab_preprocessor_user.cat_embed_input,
    continuous_cols=tab_preprocessor_user.continuous_cols,
    mlp_hidden_dims=[16, 4],
)

tab_mlp_item = TabMlp(
    column_idx=tab_preprocessor_item.column_idx,
    cat_embed_input=tab_preprocessor_item.cat_embed_input,
    continuous_cols=tab_preprocessor_item.continuous_cols,
    mlp_hidden_dims=[16, 4],
)


rnn_reviews = BasicRNN(
    vocab_size=len(text_preprocessor_reviews.vocab.itos),
    embed_dim=16,
    hidden_dim=16,
    n_layers=1,
    bidirectional=False,
    head_hidden_dims=[16, 8],
)

rnn_descriptions = BasicRNN(
    vocab_size=len(text_preprocessor_descriptions.vocab.itos),
    embed_dim=16,
    hidden_dim=16,
    n_layers=1,
    bidirectional=False,
    head_hidden_dims=[16, 4],  # just to make the head_hidden_dims different
)

global_model = WideDeep(
    deeptabular=[tab_mlp_user, tab_mlp_item],
    deeptext=[rnn_reviews, rnn_descriptions],
    pred_dim=1,
)


@pytest.mark.parametrize(
    "X_tab, X_text, X_train, X_val, val_split, target",
    [
        (
            [X_tab_user_tr, X_tab_item_tr],
            [X_text_review_tr, X_text_description_tr],
            None,
            None,
            None,
            train_df["purchased"].values,
        ),
        (
            [X_tab_user_tr, X_tab_item_tr],
            [X_text_review_tr, X_text_description_tr],
            None,
            None,
            0.2,
            train_df["purchased"].values,
        ),
        (
            None,
            None,
            {
                "X_tab": [X_tab_user_tr, X_tab_item_tr],
                "X_text": [X_text_review_tr, X_text_description_tr],
                "target": train_df["purchased"].values,
            },
            {
                "X_tab": [X_tab_user_val, X_tab_item_val],
                "X_text": [X_text_review_val, X_text_description_val],
                "target": valid_df["purchased"].values,
            },
            None,
            None,
        ),
        (
            None,
            None,
            {
                "X_tab": [X_tab_user_tr, X_tab_item_tr],
                "X_text": [X_text_review_tr, X_text_description_tr],
                "target": train_df["purchased"].values,
            },
            None,
            0.2,
            None,
        ),
    ],
)
def test_multi_text_and_tab_col_input_options(
    X_tab, X_text, X_train, X_val, val_split, target
):

    trainer = Trainer(
        global_model,
        objective="binary",
    )

    trainer.fit(
        X_tab=X_tab,
        X_text=X_text,
        X_train=X_train,
        X_val=X_val,
        val_split=val_split,
        target=target,
        n_epochs=1,
        batch_size=32,
    )

    assert trainer.history["train_loss"] is not None


def test_multiple_setups_for_multi_text_or_tab_cols():

    model = WideDeep(
        deeptabular=[tab_mlp_user, tab_mlp_item],
        deeptext=[rnn_reviews, rnn_descriptions],
        pred_dim=1,
    )

    tab_opt_user = torch.optim.Adam(model.deeptabular[0].parameters(), lr=0.01)
    tab_opt_item = torch.optim.Adam(model.deeptabular[1].parameters(), lr=0.01)

    text_opt_reviews = torch.optim.Adam(model.deeptext[0].parameters(), lr=0.01)
    text_opt2_descriptions = torch.optim.Adam(model.deeptext[1].parameters(), lr=0.01)

    tab_scheduler_user = torch.optim.lr_scheduler.StepLR(tab_opt_user, step_size=2)
    tab_scheduler_item = torch.optim.lr_scheduler.StepLR(tab_opt_item, step_size=3)

    text_scheduler_reviews = torch.optim.lr_scheduler.StepLR(
        text_opt_reviews, step_size=2
    )
    text_scheduler_descriptions = torch.optim.lr_scheduler.StepLR(
        text_opt2_descriptions, step_size=3
    )

    optimizers = {
        "deeptabular": [tab_opt_user, tab_opt_item],
        "deeptext": [text_opt_reviews, text_opt2_descriptions],
    }
    schedulers = {
        "deeptabular": [tab_scheduler_user, tab_scheduler_item],
        "deeptext": [text_scheduler_reviews, text_scheduler_descriptions],
    }
    initializers = {
        "deeptabular": [XavierNormal, KaimingNormal],
        "deepimage": [XavierNormal, KaimingNormal],
    }

    n_epochs = 6
    trainer = Trainer(
        model,
        objective="binary",
        optimizers=optimizers,
        lr_schedulers=schedulers,
        initializers=initializers,
        metrics=[Accuracy(), F1Score(average=True)],
        callbacks=[LRHistory(n_epochs=n_epochs)],
    )

    X_train = {
        "X_tab": [X_tab_user_tr, X_tab_item_tr],
        "X_text": [X_text_review_tr, X_text_description_tr],
        "target": train_df["purchased"].values,
    }
    X_val = {
        "X_tab": [X_tab_user_val, X_tab_item_val],
        "X_text": [X_text_review_val, X_text_description_val],
        "target": valid_df["purchased"].values,
    }
    trainer.fit(
        X_train=X_train,
        X_val=X_val,
        n_epochs=n_epochs,
        batch_size=4,
        verbose=0,
    )

    assert len(trainer.history["train_loss"]) == n_epochs

    deeptabular_keys = sorted(
        [k for k in trainer.lr_history.keys() if "deeptabular" in k]
    )
    deeptext_keys = sorted([k for k in trainer.lr_history.keys() if "deeptext" in k])

    for k, sz in zip(
        deeptabular_keys, [tab_scheduler_user.step_size, tab_scheduler_item.step_size]
    ):
        n_lr_decreases = n_epochs // sz - 1 if n_epochs % sz == 0 else n_epochs // sz
        lr_decrease_factor = 10**n_lr_decreases
        assert len(trainer.lr_history[k]) == n_epochs
        assert np.allclose(
            trainer.lr_history[k][0] / trainer.lr_history[k][-1], lr_decrease_factor
        )

    for k, sz in zip(
        deeptext_keys, [tab_scheduler_user.step_size, tab_scheduler_item.step_size]
    ):
        n_lr_decreases = n_epochs // sz - 1 if n_epochs % sz == 0 else n_epochs // sz
        lr_decrease_factor = 10**n_lr_decreases
        assert len(trainer.lr_history[k]) == n_epochs
        assert np.allclose(
            trainer.lr_history[k][0] / trainer.lr_history[k][-1], lr_decrease_factor
        )


def test_finetune_all_for_multi_text_or_tab_cols():

    model = WideDeep(
        deeptabular=[tab_mlp_user, tab_mlp_item],
        deeptext=[rnn_reviews, rnn_descriptions],
        pred_dim=1,
    )

    n_epochs = 5
    trainer = Trainer(
        model,
        objective="binary",
    )

    X_train = {
        "X_tab": [X_tab_user_tr, X_tab_item_tr],
        "X_text": [X_text_review_tr, X_text_description_tr],
        "target": train_df["purchased"].values,
    }
    X_val = {
        "X_tab": [X_tab_user_val, X_tab_item_val],
        "X_text": [X_text_review_val, X_text_description_val],
        "target": valid_df["purchased"].values,
    }
    trainer.fit(
        X_train=X_train,
        X_val=X_val,
        n_epochs=n_epochs,
        batch_size=4,
        finetune=True,
        finetune_epochs=2,
        verbose=0,
    )

    # weak assertion, but anyway...
    assert len(trainer.history["train_loss"]) == n_epochs


@pytest.mark.parametrize("routine", ["felbo", "howard"])
def test_finetune_gradual_for_multi_text_or_tab_cols(routine):

    model = WideDeep(
        deeptabular=[tab_mlp_user, tab_mlp_item],
        deeptext=[rnn_reviews, rnn_descriptions],
        pred_dim=1,
    )

    deeptabular_users_layers = [
        model.deeptabular[0][0].encoder.mlp[1],
        model.deeptabular[0][0].encoder.mlp[0],
    ]

    deeptabular_items_layers = [
        model.deeptabular[1][0].encoder.mlp[1],
        model.deeptabular[1][0].encoder.mlp[0],
    ]

    deeptext_reviews_layers = [
        model.deeptext[0][0].rnn_mlp.mlp[1],
        model.deeptext[0][0].rnn_mlp.mlp[0],
    ]
    deeptext_descriptions_layers = [
        model.deeptext[1][0].rnn_mlp.mlp[1],
        model.deeptext[1][0].rnn_mlp.mlp[0],
    ]

    n_epochs = 5
    trainer = Trainer(
        model,
        objective="binary",
    )

    X_train = {
        "X_tab": [X_tab_user_tr, X_tab_item_tr],
        "X_text": [X_text_review_tr, X_text_description_tr],
        "target": train_df["purchased"].values,
    }
    X_val = {
        "X_tab": [X_tab_user_val, X_tab_item_val],
        "X_text": [X_text_review_val, X_text_description_val],
        "target": valid_df["purchased"].values,
    }
    trainer.fit(
        X_train=X_train,
        X_val=X_val,
        n_epochs=n_epochs,
        batch_size=4,
        finetune=True,
        finetune_epochs=2,
        routine=routine,  # add alias as finetune_routine
        deeptabular_gradual=True,
        deeptabular_layers=[deeptabular_users_layers, deeptabular_items_layers],
        deeptabular_max_lr=0.01,
        deeptext_gradual=True,
        deeptext_layers=[deeptext_reviews_layers, deeptext_descriptions_layers],
        deepteext_max_lr=0.01,
        deepimage_gradual=True,
        deepimage_max_lr=0.01,
        verbose=0,
    )

    # weak assertion, but anyway...
    assert len(trainer.history["train_loss"]) == n_epochs


@pytest.mark.parametrize(
    "fusion_method",
    [
        "concatenate",
        "mean",
        "max",
        "sum",
        "mult",
        "dot",
        "head",
        ["max", "dot"],
        ["concatenate", "mean"],
        ["concatenate", "max", "mean"],
        ["concatenate", "max", "mean", "mult"],
    ],
)
def test_tab_model_fusion_methods(fusion_method):

    tab_mlp_user = TabMlp(
        column_idx=tab_preprocessor_user.column_idx,
        cat_embed_input=tab_preprocessor_user.cat_embed_input,
        continuous_cols=tab_preprocessor_user.continuous_cols,
        mlp_hidden_dims=[16, 4],
    )

    tab_mlp_item = TabMlp(
        column_idx=tab_preprocessor_item.column_idx,
        cat_embed_input=tab_preprocessor_item.cat_embed_input,
        continuous_cols=tab_preprocessor_item.continuous_cols,
        mlp_hidden_dims=[16, 4],
    )

    if isinstance(fusion_method, list) and "dot" in fusion_method:
        with pytest.raises(ValueError):
            models_fuser = ModelFuser(
                models=[tab_mlp_user, tab_mlp_item],
                fusion_method=fusion_method,
            )
        return
    else:
        models_fuser = ModelFuser(
            models=[tab_mlp_user, tab_mlp_item],
            fusion_method=fusion_method,
            projection_method="max",
            head_hidden_dims=[32, 8] if "head" in fusion_method else None,
        )

    X_tab_user_tr_tnsr = torch.from_numpy(X_tab_user_tr)[:16]
    X_tab_item_tr_tnsr = torch.from_numpy(X_tab_item_tr)[:16]

    out = models_fuser([X_tab_user_tr_tnsr, X_tab_item_tr_tnsr])

    if fusion_method == "dot":
        assert out.shape[1] == 1 == models_fuser.output_dim
    elif fusion_method == "concatenate":
        assert (
            out.shape[1]
            == tab_mlp_user.output_dim + tab_mlp_item.output_dim
            == models_fuser.output_dim
        )
    elif any(
        [
            fusion_method == "mean",
            fusion_method == "max",
            fusion_method == "sum",
            fusion_method == "mult",
        ]
    ):
        assert (
            out.shape[1]
            == max(tab_mlp_user.output_dim, tab_mlp_item.output_dim)
            == models_fuser.output_dim
        )
    elif fusion_method == "head":
        assert (
            out.shape[1] == models_fuser.head_hidden_dims[-1] == models_fuser.output_dim
        )
    elif fusion_method == ["concatenate", "mean"]:
        assert (
            out.shape[1]
            == tab_mlp_user.output_dim
            + tab_mlp_item.output_dim
            + max(tab_mlp_user.output_dim, tab_mlp_item.output_dim)
            == models_fuser.output_dim
        )
    elif fusion_method == ["concatenate", "max", "mean"]:
        assert (
            out.shape[1]
            == tab_mlp_user.output_dim
            + tab_mlp_item.output_dim
            + max(tab_mlp_user.output_dim, tab_mlp_item.output_dim) * 2
            == models_fuser.output_dim
        )
    else:
        # ["concatenate", "max", "mean", "mult"]
        assert (
            out.shape[1]
            == tab_mlp_user.output_dim
            + tab_mlp_item.output_dim
            + max(tab_mlp_user.output_dim, tab_mlp_item.output_dim) * 3
            == models_fuser.output_dim
        )


def test_model_fusion_custom_head():

    tab_mlp_user = TabMlp(
        column_idx=tab_preprocessor_user.column_idx,
        cat_embed_input=tab_preprocessor_user.cat_embed_input,
        continuous_cols=tab_preprocessor_user.continuous_cols,
        mlp_hidden_dims=[16, 4],
    )

    tab_mlp_item = TabMlp(
        column_idx=tab_preprocessor_item.column_idx,
        cat_embed_input=tab_preprocessor_item.cat_embed_input,
        continuous_cols=tab_preprocessor_item.continuous_cols,
        mlp_hidden_dims=[16, 4],
    )

    custom_head = CustomHead(tab_mlp_user.output_dim + tab_mlp_item.output_dim, 8)

    models_fuser = ModelFuser(
        models=[tab_mlp_user, tab_mlp_item],
        fusion_method="head",
        custom_head=custom_head,
        projection_method="max",
    )

    X_tab_user_tr_tnsr = torch.from_numpy(X_tab_user_tr)[:16]
    X_tab_item_tr_tnsr = torch.from_numpy(X_tab_item_tr)[:16]
    out = models_fuser([X_tab_user_tr_tnsr, X_tab_item_tr_tnsr])

    assert out.shape[1] == custom_head.output_dim == models_fuser.output_dim


@pytest.mark.parametrize(
    "projection_method",
    ["min", "max", "mean"],
)
def test_model_fusion_projection_methods(projection_method):

    tab_mlp_user = TabMlp(
        column_idx=tab_preprocessor_user.column_idx,
        cat_embed_input=tab_preprocessor_user.cat_embed_input,
        continuous_cols=tab_preprocessor_user.continuous_cols,
        mlp_hidden_dims=[16, 4],
    )

    tab_mlp_item = TabMlp(
        column_idx=tab_preprocessor_item.column_idx,
        cat_embed_input=tab_preprocessor_item.cat_embed_input,
        continuous_cols=tab_preprocessor_item.continuous_cols,
        mlp_hidden_dims=[16, 4],
    )

    models_fuser = ModelFuser(
        models=[tab_mlp_user, tab_mlp_item],
        fusion_method="mean",
        projection_method=projection_method,
    )

    X_tab_user_tr_tnsr = torch.from_numpy(X_tab_user_tr)[:16]
    X_tab_item_tr_tnsr = torch.from_numpy(X_tab_item_tr)[:16]
    out = models_fuser([X_tab_user_tr_tnsr, X_tab_item_tr_tnsr])

    if projection_method == "min":
        proj_dim = min(tab_mlp_user.output_dim, tab_mlp_item.output_dim)
    elif projection_method == "max":
        proj_dim = max(tab_mlp_user.output_dim, tab_mlp_item.output_dim)
    else:
        proj_dim = int((tab_mlp_user.output_dim + tab_mlp_item.output_dim) / 2)

    assert out.shape[1] == proj_dim == models_fuser.output_dim


@pytest.mark.parametrize("head_type", [None, "via_params", "custom"])
def test_full_process_with_fusion(head_type):

    fused_tab_model = ModelFuser(
        models=[tab_mlp_user, tab_mlp_item],
        fusion_method="mean",
        projection_method="max",
    )

    fused_text_model = ModelFuser(
        models=[rnn_reviews, rnn_descriptions],
        fusion_method="mean",
        projection_method="min",
    )

    if head_type == "via_params":
        head_hidden_dims = [fused_tab_model.output_dim + fused_text_model.output_dim, 8]
        custom_head = None
    elif head_type == "custom":
        head_hidden_dims = None
        custom_head = CustomHead(
            fused_tab_model.output_dim + fused_text_model.output_dim, 8
        )
    else:
        head_hidden_dims = None
        custom_head = None

    model = WideDeep(
        deeptabular=fused_tab_model,
        deeptext=fused_text_model,
        pred_dim=1,
        head_hidden_dims=head_hidden_dims,
        deephead=custom_head,
    )

    n_epochs = 2
    trainer = Trainer(
        model,
        objective="binary",
        verbose=0,
    )

    X_train = {
        "X_tab": [X_tab_user_tr, X_tab_item_tr],
        "X_text": [X_text_review_tr, X_text_description_tr],
        "target": train_df["purchased"].values,
    }
    X_val = {
        "X_tab": [X_tab_user_val, X_tab_item_val],
        "X_text": [X_text_review_val, X_text_description_val],
        "target": valid_df["purchased"].values,
    }
    trainer.fit(
        X_train=X_train,
        X_val=X_val,
        n_epochs=n_epochs,
        batch_size=4,
    )

    # weak assertion, but anyway...
    assert len(trainer.history["train_loss"]) == n_epochs


@pytest.mark.parametrize("head_type", [None, "via_params", "custom"])
def test_full_process_without_fusion(head_type):

    # the 4 models to be combined are tab_mlp_user, tab_mlp_item, rnn_reviews,
    # rnn_descriptions
    if head_type == "via_params":
        head_hidden_dims = [
            tab_mlp_user.output_dim
            + tab_mlp_item.output_dim
            + rnn_reviews.output_dim
            + rnn_descriptions.output_dim,
            8,
        ]
        custom_head = None
    elif head_type == "custom":
        head_hidden_dims = None
        custom_head = CustomHead(
            tab_mlp_user.output_dim
            + tab_mlp_item.output_dim
            + rnn_reviews.output_dim
            + rnn_descriptions.output_dim,
            8,
        )
    else:
        head_hidden_dims = None
        custom_head = None

    model = WideDeep(
        deeptabular=[tab_mlp_user, tab_mlp_item],
        deeptext=[rnn_reviews, rnn_descriptions],
        pred_dim=1,
        head_hidden_dims=head_hidden_dims,
        deephead=custom_head,
    )

    n_epochs = 2
    trainer = Trainer(
        model,
        objective="binary",
        verbose=0,
    )

    X_train = {
        "X_tab": [X_tab_user_tr, X_tab_item_tr],
        "X_text": [X_text_review_tr, X_text_description_tr],
        "target": train_df["purchased"].values,
    }
    X_val = {
        "X_tab": [X_tab_user_val, X_tab_item_val],
        "X_text": [X_text_review_val, X_text_description_val],
        "target": valid_df["purchased"].values,
    }
    trainer.fit(
        X_train=X_train,
        X_val=X_val,
        n_epochs=n_epochs,
        batch_size=4,
    )

    # weak assertion, but anyway...
    assert len(trainer.history["train_loss"]) == n_epochs


@pytest.mark.parametrize("fuse_models", [True, False])
def test_catch_tabnet_error(fuse_models):

    tabnet_user = TabNet(
        column_idx=tab_preprocessor_user.column_idx,
        cat_embed_input=tab_preprocessor_user.cat_embed_input,
        continuous_cols=tab_preprocessor_user.continuous_cols,
    )

    tab_mlp_item = TabMlp(
        column_idx=tab_preprocessor_item.column_idx,
        cat_embed_input=tab_preprocessor_item.cat_embed_input,
        continuous_cols=tab_preprocessor_item.continuous_cols,
    )

    if fuse_models:
        with pytest.raises(ValueError):
            fused_model = ModelFuser(  # noqa: F841
                models=[tabnet_user, tab_mlp_item],
                fusion_method="mean",
                projection_method="max",
            )
    else:
        with pytest.raises(ValueError):
            model = WideDeep(deeptabular=[tabnet_user, tab_mlp_item])  # noqa: F841
