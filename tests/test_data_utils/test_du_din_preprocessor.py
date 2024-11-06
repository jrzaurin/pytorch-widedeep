import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import WideDeep
from pytorch_widedeep.models.rec import DeepInterestNetwork
from pytorch_widedeep.preprocessing import DINPreprocessor

full_path = os.path.realpath(__file__)
path = os.path.split(full_path)[0]

save_dir = Path(path) / "interactions_df_for_rec_preprocessor"

df_interactions = pd.read_csv(save_dir / "interactions_data.csv")

df_interactions = df_interactions.sort_values(by=["user_id", "timestamp"]).reset_index(
    drop=True
)

cat_embed_cols = ["user_id", "age", "gender"]
continuous_cols = ["height", "weight"]


def user_behaviour_well_encoded(
    input_df: pd.DataFrame,
    din_preprocessor: DINPreprocessor,
    X: np.ndarray,
    max_seq_length: int,
):

    user_id: int = 5

    encoded_user_id = (  # type: ignore
        din_preprocessor.tab_preprocessor.label_encoder.encoding_dict["user_id"][
            user_id
        ]
    )

    user_items = (
        input_df[input_df["user_id"] == user_id]
        .groupby("user_id")["item_id"]
        .agg(list)
        .values[0]
    )[:max_seq_length]

    encoded_items = [
        din_preprocessor.item_le.encoding_dict["item_id"][item] for item in user_items
    ]

    rows = np.where(
        X[:, din_preprocessor.din_columns_idx["user_id"]] == encoded_user_id
    )[0]
    X_user_id = X[rows][:1]

    item_seq_cols = din_preprocessor.user_behaviour_config[0]
    item_seq_cols_idx = [din_preprocessor.din_columns_idx[col] for col in item_seq_cols]

    X_user_items = list(X_user_id[:, item_seq_cols_idx].astype(int)[0])

    return X_user_items == encoded_items


def test_din_preprocessor():

    din_preprocessor = DINPreprocessor(
        user_id_col="user_id",
        item_embed_col="item_id",
        target_col="interaction",
        action_col="interaction",
        other_seq_embed_cols=["category", "price"],
        cat_embed_cols=cat_embed_cols,
        continuous_cols=continuous_cols,
        cols_to_scale=continuous_cols,
        max_seq_length=5,
    )

    X, y = din_preprocessor.fit_transform(df_interactions)
    # 5 items + 5 actions + 2 * 5 other_seq_embed_cols + (1 target item + 1
    # target category + 1 target price) + 5 continuous_cols and
    # cat_embed_cols
    expected_n_cols = 5 + 5 + (2 * 5) + 3 + 5
    assert X.shape[1] == expected_n_cols
    assert din_preprocessor.is_fitted
    assert user_behaviour_well_encoded(df_interactions, din_preprocessor, X, 5)


@pytest.mark.parametrize(
    "with_cat_embed_cols, with_continuous_cols, with_action_col, with_other_seq_embed_cols",
    [
        (True, True, True, True),
        (True, True, True, False),
        (True, True, False, True),
        (True, True, False, False),
        (True, False, True, True),
        (True, False, True, False),
        (True, False, False, True),
        (True, False, False, False),
        (False, True, True, True),
        (False, True, True, False),
        (False, True, False, True),
        (False, True, False, False),
        (False, False, True, True),
        (False, False, True, False),
        (False, False, False, True),
        (False, False, False, False),
    ],
)
def test_din_preprocessor_diff_input_params(
    with_cat_embed_cols,
    with_continuous_cols,
    with_action_col,
    with_other_seq_embed_cols,
):

    max_seq_length = 5

    din_preprocessor = DINPreprocessor(
        user_id_col="user_id",
        item_embed_col="item_id",
        target_col="interaction",
        action_col="interaction" if with_action_col else None,
        other_seq_embed_cols=(
            ["category", "price"] if with_other_seq_embed_cols else None
        ),
        cat_embed_cols=cat_embed_cols if with_cat_embed_cols else None,
        continuous_cols=continuous_cols if with_continuous_cols else None,
        cols_to_scale=continuous_cols if with_continuous_cols else None,
        max_seq_length=max_seq_length,
    )

    X, y = din_preprocessor.fit_transform(df_interactions)

    expected_n_cols = max_seq_length + 1

    assert din_preprocessor.user_behaviour_config[0] == [
        f"item_{i+1}" for i in range(max_seq_length)
    ]

    if with_action_col:
        expected_n_cols += 5
        assert din_preprocessor.action_seq_config[0] == [
            f"action_{i+1}" for i in range(max_seq_length)
        ]

    if with_other_seq_embed_cols:
        for i, col in enumerate(["category", "price"]):
            expected_n_cols += 5 + 1
            assert din_preprocessor.other_seq_config[i][0] == [
                f"{col}_{i+1}" for i in range(max_seq_length)
            ]

    if with_cat_embed_cols:
        expected_n_cols += len(cat_embed_cols)

    if with_continuous_cols:
        expected_n_cols += len(continuous_cols)

    assert din_preprocessor.is_fitted
    assert X.shape[1] == expected_n_cols
    if with_cat_embed_cols:
        assert user_behaviour_well_encoded(df_interactions, din_preprocessor, X, 5)


def test_din_full_process_w_processor():

    din_preprocessor = DINPreprocessor(
        user_id_col="user_id",
        item_embed_col="item_id",
        target_col="interaction",
        action_col="interaction",
        other_seq_embed_cols=["category", "price"],
        cat_embed_cols=cat_embed_cols,
        continuous_cols=continuous_cols,
        cols_to_scale=continuous_cols,
        max_seq_length=5,
    )

    X, y = din_preprocessor.fit_transform(df_interactions)

    din = DeepInterestNetwork(
        column_idx=din_preprocessor.din_columns_idx,
        user_behavior_confiq=din_preprocessor.user_behaviour_config,
        target_item_col="target_item",
        action_seq_config=din_preprocessor.action_seq_config,
        other_seq_cols_confiq=din_preprocessor.other_seq_config,
        cat_embed_input=din_preprocessor.tab_preprocessor.cat_embed_input,  # type: ignore
        continuous_cols=din_preprocessor.tab_preprocessor.continuous_cols,
        mlp_hidden_dims=[16, 8],
    )

    model = WideDeep(deeptabular=din)

    trainer = Trainer(model, objective="binary", verbose=0)

    trainer.fit(X_tab=X, target=y, n_epochs=2)

    preds = trainer.predict(X_tab=X)

    assert preds.shape[0] == X.shape[0]
    assert trainer.history is not None and "train_loss" in trainer.history
