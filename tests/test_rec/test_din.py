import numpy as np
import torch
import pytest

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import WideDeep
from pytorch_widedeep.models.rec import DeepInterestNetwork

from .utils_test_rec import prepare_data_for_din

data, config, column_idx = prepare_data_for_din()

item_seq_config = config["item_seq_config"]
item_purchased_seq_config = config["item_purchased_seq_config"]
item_category_seq_config = config["item_category_seq_config"]
cat_embed_input = config["cat_embed_input"]
continuous_cols = config["continuous_cols"]


def test_din_initialization():

    din = DeepInterestNetwork(
        column_idx=column_idx,
        target_item_col="target_item",
        user_behavior_confiq=item_seq_config,
        action_seq_config=item_purchased_seq_config,
        other_seq_cols_confiq=item_category_seq_config,
        cat_embed_input=cat_embed_input,
        continuous_cols=continuous_cols,
    )

    # Other cols assertion:
    other_col_embed = din.other_col_embed.cat_embed.embed_layers
    for col, n_unique, embed_dim in cat_embed_input:
        assert other_col_embed[f"emb_layer_{col}"].num_embeddings == n_unique + 1
        assert other_col_embed[f"emb_layer_{col}"].embedding_dim == embed_dim

    # we have 5 users, so user_behavior_embed must have 6 num_embeddings (5 + padding)
    assert din.user_behavior_embed.cat_embed.embed.num_embeddings == 6

    # we have 2 actions per user, so action_embed must have 3 num_embeddings (2 + padding)
    assert din.action_embed.cat_embed.embed.num_embeddings == 3

    # we have 5 categories, so other_cols_embed must have 6 num_embeddings (5 + padding)
    assert din.other_seq_cols_embed["seq_0"].cat_embed.embed.num_embeddings


def test_din_forward():

    X_tab = torch.tensor(data["train"][0])
    din = DeepInterestNetwork(
        column_idx=column_idx,
        target_item_col="target_item",
        user_behavior_confiq=item_seq_config,
        action_seq_config=item_purchased_seq_config,
        other_seq_cols_confiq=item_category_seq_config,
        cat_embed_input=cat_embed_input,
        continuous_cols=continuous_cols,
    )

    res = din(X_tab)
    assert res.shape[0] == X_tab.shape[0]


@pytest.mark.parametrize("use_cat_bias", [True, False])
@pytest.mark.parametrize("cat_embed_activation", ["relu", None])
@pytest.mark.parametrize("cont_norm_layer", ["layernorm", "batchnorm"])
@pytest.mark.parametrize("cont_embed_activation", ["relu", None])
@pytest.mark.parametrize("mlp_hidden_dims", [None, [16, 8]])
def test_din_with_params(
    use_cat_bias,
    cat_embed_activation,
    cont_norm_layer,
    cont_embed_activation,
    mlp_hidden_dims,
):

    X_tab = torch.tensor(data["train"][0])

    din = DeepInterestNetwork(
        column_idx=column_idx,
        user_behavior_confiq=item_seq_config,
        target_item_col="target_item",
        action_seq_config=item_purchased_seq_config,
        other_seq_cols_confiq=item_category_seq_config,
        cat_embed_input=cat_embed_input,
        cat_embed_dropout=0.2,
        use_cat_bias=use_cat_bias,
        cat_embed_activation=cat_embed_activation,
        continuous_cols=continuous_cols,
        cont_norm_layer=cont_norm_layer,
        embed_continuous=True,
        embed_continuous_method="standard",
        cont_embed_dim=4,
        cont_embed_dropout=0.2,
        cont_embed_activation=cont_embed_activation,
        mlp_hidden_dims=mlp_hidden_dims,
    )

    res = din(X_tab)

    assert res.shape[0] == X_tab.shape[0]


def test_din_with_piecewise():
    X_tab = torch.tensor(data["train"][0])

    quantization_setup = {}
    for col, idx in column_idx.items():
        if col in continuous_cols:
            quantization_setup[col] = list(
                np.quantile(X_tab[:, idx], [0.0, 0.25, 0.5, 1.0])
            )

    din = DeepInterestNetwork(
        column_idx=column_idx,
        target_item_col="target_item",
        user_behavior_confiq=item_seq_config,
        action_seq_config=item_purchased_seq_config,
        other_seq_cols_confiq=item_category_seq_config,
        cat_embed_input=cat_embed_input,
        continuous_cols=continuous_cols,
        cont_embed_dim=4,
        embed_continuous_method="piecewise",
        quantization_setup=quantization_setup,
        mlp_hidden_dims=[16, 8],
    )

    res = din(X_tab)
    assert res.shape[0] == X_tab.shape[0]


def test_din_with_periodic():
    X_tab = torch.tensor(data["train"][0])

    din = DeepInterestNetwork(
        column_idx=column_idx,
        target_item_col="target_item",
        user_behavior_confiq=item_seq_config,
        action_seq_config=item_purchased_seq_config,
        other_seq_cols_confiq=item_category_seq_config,
        cat_embed_input=cat_embed_input,
        continuous_cols=continuous_cols,
        cont_embed_dim=4,
        embed_continuous_method="periodic",
        cont_embed_activation="relu",
        n_frequencies=4,
        sigma=0.1,
        share_last_layer=True,
        mlp_hidden_dims=[16, 8],
    )

    res = din(X_tab)
    assert res.shape[0] == X_tab.shape[0]


@pytest.mark.parametrize("mlp_hidden_dims", [None, [16, 8]])
def test_din_full_process(mlp_hidden_dims):

    X_tab_tr, y_tr = data["train"]
    X_tab_val, y_val = data["val"]
    X_tab_te, _ = data["test"]

    quantization_setup = {}
    for col, idx in column_idx.items():
        if col in continuous_cols:
            quantization_setup[col] = list(
                np.quantile(X_tab_tr[:, idx], [0.0, 0.25, 0.5, 1.0])
            )

    din = DeepInterestNetwork(
        column_idx=column_idx,
        target_item_col="target_item",
        user_behavior_confiq=item_seq_config,
        action_seq_config=item_purchased_seq_config,
        other_seq_cols_confiq=item_category_seq_config,
        cat_embed_input=cat_embed_input,
        continuous_cols=continuous_cols,
        cont_embed_dim=4,
        embed_continuous_method="piecewise",
        quantization_setup=quantization_setup,
        mlp_hidden_dims=mlp_hidden_dims,
    )

    model = WideDeep(deeptabular=din)

    trainer = Trainer(model, objective="binary", verbose=0)

    X_train = {"X_tab": X_tab_tr, "target": y_tr}
    X_val = {"X_tab": X_tab_val, "target": y_val}
    trainer.fit(X_train=X_train, X_val=X_val, n_epochs=1)

    preds = trainer.predict_proba(X_tab=X_tab_te)

    assert preds.shape[0] == X_tab_te.shape[0]
    assert (
        trainer.history is not None
        and "train_loss" in trainer.history
        and "val_loss" in trainer.history
    )
