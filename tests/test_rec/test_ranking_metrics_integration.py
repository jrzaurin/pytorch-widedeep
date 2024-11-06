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
from pytorch_widedeep.dataloaders import CustomDataLoader
from pytorch_widedeep.preprocessing import TabPreprocessor


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
#   * Same ranking metrics for both training and validation sets.
#   * Same number of items per user in both sets


@pytest.mark.parametrize(
    "user_item_data", [{"validation_interactions_per_user": 5}], indirect=True
)
@pytest.mark.parametrize(
    "metric",
    [
        MAP_at_k(n_cols=5, k=3),
        Recall_at_k(n_cols=5, k=3),
        HitRatio_at_k(n_cols=5, k=3),
        Precision_at_k(n_cols=5, k=3),
        BinaryNDCG_at_k(n_cols=5, k=3),
        [Accuracy(), MAP_at_k(n_cols=5, k=3)],
        [F1Score(), MAP_at_k(n_cols=5, k=3)],
        [Accuracy(), BinaryNDCG_at_k(n_cols=5, k=3)],
        [F1Score(), BinaryNDCG_at_k(n_cols=5, k=3)],
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

    if isinstance(metric, list):
        trainer = Trainer(model, objective="binary", metrics=metric)
    else:
        trainer = Trainer(model, objective="binary", metrics=[metric])

    trainer.fit(
        X_train={"X_tab": X_train, "target": train_df[target_col].values},
        X_val={"X_tab": X_val, "target": val_df[target_col].values},
        n_epochs=1,
        batch_size=2 * 5,
        verbose=0,
    )

    # predict on validation, this is just a test...
    preds = trainer.predict(X_tab=X_val)

    assert preds.shape[0] == X_val.shape[0]
    assert (
        trainer.history is not None
        and "train_loss" in trainer.history
        and "val_loss" in trainer.history
    )
    if not isinstance(metric, list):
        metric = [metric]
    for m in metric:
        assert f"train_{m._name}" in trainer.history
        assert f"val_{m._name}" in trainer.history


@pytest.mark.parametrize(
    "user_item_data", [{"validation_interactions_per_user": 5}], indirect=True
)
def test_multiclass_classification_strategy_1(user_item_data):

    train_df, val_df = user_item_data

    categorical_cols = ["user_id", "item_id", "user_category", "item_category"]

    target_col = "rating"

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
    model = WideDeep(deeptabular=tab_mlp, pred_dim=3)

    trainer = Trainer(
        model,
        objective="multiclass",
        metrics=[Accuracy(), NDCG_at_k(n_cols=5, k=3)],
        verbose=0,
    )

    trainer.fit(
        X_train={"X_tab": X_train, "target": train_df[target_col].values},
        X_val={"X_tab": X_val, "target": val_df[target_col].values},
        n_epochs=1,
        batch_size=2 * 5,  # 2 * n_items
    )

    # predict on validation, this is just a test...
    preds = trainer.predict(X_tab=X_val)

    assert preds.shape[0] == X_val.shape[0]
    assert (
        trainer.history is not None
        and "train_loss" in trainer.history
        and "val_loss" in trainer.history
    )


# strategy 2:
#   * Diff metrics for training and validation sets.
#   * Same number of items per user in both sets


@pytest.mark.parametrize(
    "user_item_data", [{"validation_interactions_per_user": 5}], indirect=True
)
@pytest.mark.parametrize("target_col", ["liked", "rating"])
def test_strategy_2(user_item_data, target_col):

    train_df, val_df = user_item_data

    categorical_cols = ["user_id", "item_id", "user_category", "item_category"]

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
    model = WideDeep(deeptabular=tab_mlp, pred_dim=1 if target_col == "liked" else 3)

    eval_metrics = (
        [BinaryNDCG_at_k(n_cols=5, k=3)]
        if target_col == "liked"
        else [NDCG_at_k(n_cols=5, k=3)]
    )
    trainer = Trainer(
        model,
        objective="binary" if target_col == "liked" else "multiclass",
        metrics=[Accuracy()],
        eval_metrics=eval_metrics,
        verbose=0,
    )

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
        and "train_acc" in trainer.history
        and "val_binary_ndcg@3" in trainer.history
        if target_col == "liked"
        else "val_ndcg@3" in trainer.history
    )


# strategy 3:
#   * Diff number of items per user in both sets which implies CustomDataLoaders
#   * everything else


@pytest.mark.parametrize(
    "user_item_data", [{"validation_interactions_per_user": 6}], indirect=True
)
@pytest.mark.parametrize(
    "metrics",
    [
        [[BinaryNDCG_at_k(n_cols=4, k=3)], [BinaryNDCG_at_k(n_cols=6, k=3)]],
        [[Accuracy(), F1Score()], [F1Score(), BinaryNDCG_at_k(n_cols=6, k=3)]],
        [[Accuracy(), F1Score()], [F1Score(), MAP_at_k(n_cols=6, k=3)]],
        [[Accuracy(), F1Score()], [F1Score(), Precision_at_k(n_cols=6, k=3)]],
        [[Accuracy(), F1Score()], [F1Score(), Recall_at_k(n_cols=6, k=3)]],
        [[Accuracy(), F1Score()], [F1Score(), HitRatio_at_k(n_cols=6, k=3)]],
    ],
)
@pytest.mark.parametrize("with_train_data_loader", [True, False])
def test_binary_classification_strategy_3(
    user_item_data, metrics, with_train_data_loader
):

    train_metrics, val_metrics = metrics[0], metrics[1]

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

    trainer = Trainer(
        model,
        objective="binary",
        metrics=train_metrics,
        eval_metrics=val_metrics,
    )

    if with_train_data_loader:
        train_dl = CustomDataLoader(
            batch_size=2 * 4, shuffle=False  # in case there is a ranking metric
        )
    else:
        train_dl = None

    valid_dl = CustomDataLoader(
        batch_size=2 * 6,
        shuffle=False,
    )

    trainer.fit(
        X_train={"X_tab": X_train, "target": train_df[target_col].values},
        X_val={"X_tab": X_val, "target": val_df[target_col].values},
        n_epochs=1,
        batch_size=2
        * 4,  # it only applies to the training set. Will be ignored if train_dl is not None
        train_dataloader=train_dl,
        eval_dataloader=valid_dl,
    )

    train_metric_names = [m._name for m in train_metrics]
    val_metric_names = [m._name for m in val_metrics]

    # predict on validation, this is just a test...
    preds = trainer.predict(X_tab=X_val)

    assert valid_dl.dataset.X_tab.shape[0] == X_val.shape[0] == 60

    if with_train_data_loader:
        assert train_dl.dataset.X_tab.shape[0] == X_train.shape[0] == 40

    assert preds.shape[0] == X_val.shape[0]
    assert (
        trainer.history is not None
        and "train_loss" in trainer.history
        and "val_loss" in trainer.history
    )

    for train_metric_name in train_metric_names:
        assert f"train_{train_metric_name}" in trainer.history

    for val_metric_name in val_metric_names:
        assert f"val_{val_metric_name}" in trainer.history


@pytest.mark.parametrize(
    "user_item_data", [{"validation_interactions_per_user": 6}], indirect=True
)
@pytest.mark.parametrize(
    "metrics",
    [
        [[NDCG_at_k(n_cols=4, k=3)], [NDCG_at_k(n_cols=6, k=3)]],
        [[Accuracy(), F1Score()], [F1Score(), NDCG_at_k(n_cols=6, k=3)]],
    ],
)
@pytest.mark.parametrize("with_train_data_loader", [True, False])
def test_multiclass_classification_strategy_3(
    user_item_data, metrics, with_train_data_loader
):

    train_metrics, val_metrics = metrics[0], metrics[1]

    train_df, val_df = user_item_data

    categorical_cols = ["user_id", "item_id", "user_category", "item_category"]

    target_col = "rating"

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
    model = WideDeep(deeptabular=tab_mlp, pred_dim=3)

    trainer = Trainer(
        model,
        objective="multiclass",
        metrics=train_metrics,
        eval_metrics=val_metrics,
    )

    if with_train_data_loader:
        train_dl = CustomDataLoader(
            batch_size=2 * 4, shuffle=False  # in case there is a ranking metric
        )
    else:
        train_dl = None

    valid_dl = CustomDataLoader(
        batch_size=2 * 6,
        shuffle=False,
    )

    trainer.fit(
        X_train={"X_tab": X_train, "target": train_df[target_col].values},
        X_val={"X_tab": X_val, "target": val_df[target_col].values},
        n_epochs=1,
        batch_size=2
        * 4,  # it only applies to the training set. Will be ignored if train_dl is not None
        train_dataloader=train_dl,
        eval_dataloader=valid_dl,
    )

    train_metric_names = [m._name for m in train_metrics]
    val_metric_names = [m._name for m in val_metrics]

    # predict on validation, this is just a test...
    preds = trainer.predict(X_tab=X_val)

    assert valid_dl.dataset.X_tab.shape[0] == X_val.shape[0] == 60

    if with_train_data_loader:
        assert train_dl.dataset.X_tab.shape[0] == X_train.shape[0] == 40

    assert preds.shape[0] == X_val.shape[0]
    assert (
        trainer.history is not None
        and "train_loss" in trainer.history
        and "val_loss" in trainer.history
    )

    for train_metric_name in train_metric_names:
        assert f"train_{train_metric_name}" in trainer.history

    for val_metric_name in val_metric_names:
        assert f"val_{val_metric_name}" in trainer.history
