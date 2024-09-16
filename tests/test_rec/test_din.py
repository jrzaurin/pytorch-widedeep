import torch
import pytest

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import Wide, WideDeep
from pytorch_widedeep.models.rec import DeepInterestNetwork
from pytorch_widedeep.preprocessing import TabPreprocessor, WidePreprocessor

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
