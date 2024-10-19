# silence DeprecationWarning
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from pytorch_widedeep.models import TabMlp, WideDeep
from pytorch_widedeep.metrics import (
    F1Score,
    Accuracy,
    MAP_at_k,
    NDCG_at_k,
    Recall_at_k,
    HitRatio_at_k,
    Precision_at_k,
    BinaryNDCG_at_k,
)
from pytorch_widedeep.training import Trainer
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.training._wd_dataset import WideDeepDataset


def generate_user_item_interactions(
    n_users: int = 10,
    n_interactions_per_user: int = 10,
    n_items: int = 5,
    random_seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(random_seed)

    user_ids = np.repeat(range(1, n_users + 1), n_interactions_per_user)
    item_ids = np.random.randint(1, n_items + 1, size=n_users * n_interactions_per_user)
    user_categories = np.random.choice(
        ["A", "B", "C"], size=n_users * n_interactions_per_user
    )
    item_categories = np.random.choice(
        ["X", "Y", "Z"], size=n_users * n_interactions_per_user
    )
    binary_target = np.random.randint(0, 2, size=n_users * n_interactions_per_user)
    categorical_target = np.random.randint(0, 3, size=n_users * n_interactions_per_user)

    df = pd.DataFrame(
        {
            "user_id": user_ids,
            "item_id": item_ids,
            "user_category": user_categories,
            "item_category": item_categories,
            "liked": binary_target,
            "rating": categorical_target,
        }
    )

    return df


def split_train_validation(
    df: pd.DataFrame, validation_interactions_per_user: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    grouped = (
        df.groupby("user_id")
        .apply(lambda x: x.sample(frac=1, random_state=42))
        .reset_index(drop=True)
    )

    train_df = (
        grouped.groupby("user_id")
        .apply(lambda x: x.iloc[:-validation_interactions_per_user])
        .reset_index(drop=True)
    )
    val_df = (
        grouped.groupby("user_id")
        .apply(lambda x: x.iloc[-validation_interactions_per_user:])
        .reset_index(drop=True)
    )

    return train_df, val_df


@pytest.fixture
def user_item_data(request):
    validation_interactions_per_user = request.param.get(
        "validation_interactions_per_user", 5
    )
    df = generate_user_item_interactions()
    train_df, val_df = split_train_validation(df, validation_interactions_per_user)
    return train_df, val_df


# strategy 1:
# 	* ranking metrics for both training and validation sets.
# 	* Same number of items per user in both sets


@pytest.mark.parametrize(
    "user_item_data", [{"validation_interactions_per_user": 5}], indirect=True
)
@pytest.mark.parametrize(
    "metric",
    [
        MAP_at_k(n_cols=5, k=3),
        NDCG_at_k(n_cols=5, k=3),
        Recall_at_k(n_cols=5, k=3),
        HitRatio_at_k(n_cols=5, k=3),
        Precision_at_k(n_cols=5, k=3),
        BinaryNDCG_at_k(n_cols=5, k=3),
    ],
)
def test_binary_classification_strategy_1(user_item_data, metric):

    train_df, val_df = user_item_data

    categorical_cols = ["user_id", "item_id", "user_category", "item_category"]

    target_col = "liked"

    tab_preprocessor = TabPreprocessor(
        embed_cols=categorical_cols,
        for_transformer=False,
    )

    tab_preprocessor.fit(train_df)

    X_train = tab_preprocessor.transform(train_df)
    X_val = tab_preprocessor.transform(val_df)

    tab_mlp = TabMlp(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,  # type: ignore[arg-type]
    )
    model = WideDeep(deeptabular=tab_mlp)

    trainer = Trainer(model, objective="binary", metrics=[metric])

    trainer.fit(
        X_train={"X_tab": X_train, "target": train_df[target_col].values},
        X_val={"X_tab": X_val, "target": val_df[target_col].values},
        n_epochs=1,
        batch_size=2 * 5,
    )

    # predict on validation, this is just a test...
    preds = trainer.predict(X_tab=X_val)

    assert preds.shape[0] == X_val.shape[0]
    assert (
        trainer.history is not None
        and "train_loss" in trainer.history
        and "val_loss" in trainer.history
        and f"train_{metric._name}" in trainer.history
        and f"val_{metric._name}" in trainer.history
    )
