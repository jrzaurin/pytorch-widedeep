import numpy as np
import torch
import pandas as pd
import pytest
import torch.nn.functional as F
from sklearn.datasets import make_regression, make_classification

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import TabMlp, WideDeep
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.losses_multitarget import (
    MultiTargetRegressionLoss,
    MultiTargetClassificationLoss,
    MutilTargetRegressionAndClassificationLoss,
)


def create_multitarget_data() -> pd.DataFrame:
    # Generate binary classification target
    X_classification_binary, y_classification_binary = make_classification(
        n_samples=64,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        n_classes=2,
        random_state=42,
    )

    # Generate multi-class classification target
    X_classification_multiclass, y_classification_multiclass = make_classification(
        n_samples=64,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        n_classes=3,
        random_state=42,
    )

    # Generate regression targets
    X_regression1, y_regression1 = make_regression(
        n_samples=64, n_features=2, noise=0.1, random_state=42
    )
    X_regression2, y_regression2 = make_regression(
        n_samples=64, n_features=2, noise=0.1, random_state=42
    )

    # Create a pandas DataFrame
    df = pd.DataFrame(
        np.hstack(
            [
                X_regression1,
                X_regression2,
                X_classification_binary,
                X_classification_multiclass,
            ]
        ),
        columns=["col1", "col2", "col3", "col4", "col5", "col6", "col7", "col8"],
    )
    df["target1_regression"] = y_regression1.astype(np.float32)
    df["target2_regression"] = y_regression2.astype(np.float32)
    df["target3_binary"] = y_classification_binary.astype(np.float32)
    df["target4_multiclass"] = y_classification_multiclass.astype(np.float32)

    return df


df = create_multitarget_data()


@pytest.mark.parametrize("weights", [None, [1, 5]])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("with_alias", [False, True])
def test_multi_target_regression_loss(weights, reduction, with_alias):

    tab_preprocessor = TabPreprocessor(continuous_cols=["col1", "col2", "col3", "col4"])
    X_tab = tab_preprocessor.fit_transform(df)
    X_tab_tnsr = torch.tensor(X_tab, dtype=torch.float32)

    tab_ml = TabMlp(
        column_idx=tab_preprocessor.column_idx,
        continuous_cols=tab_preprocessor.continuous_cols,
        mlp_hidden_dims=[16, 8],
    )

    if with_alias:
        regression_loss = MultiTargetRegressionLoss(
            target_weights=weights, target_reduction=reduction
        )
    else:
        regression_loss = MultiTargetRegressionLoss(
            weights=weights, reduction=reduction
        )

    model = WideDeep(deeptabular=tab_ml, pred_dim=2)

    y_true = torch.tensor(
        df[["target1_regression", "target2_regression"]].values, dtype=torch.float32
    )
    y_pred = model({"deeptabular": X_tab_tnsr})

    multi_target_loss = regression_loss(y_pred, y_true)

    if reduction == "mean":
        if weights is not None:
            manual_loss = 0.5 * weights[0] * F.mse_loss(
                y_pred[:, 0], y_true[:, 0]
            ) + 0.5 * weights[1] * F.mse_loss(y_pred[:, 1], y_true[:, 1])

        else:
            manual_loss = 0.5 * F.mse_loss(
                y_pred[:, 0], y_true[:, 0]
            ) + 0.5 * F.mse_loss(y_pred[:, 1], y_true[:, 1])
    else:
        if weights is not None:
            manual_loss = weights[0] * F.mse_loss(
                y_pred[:, 0], y_true[:, 0], reduction="sum"
            ) + weights[1] * F.mse_loss(y_pred[:, 1], y_true[:, 1], reduction="sum")
        else:
            manual_loss = F.mse_loss(
                y_pred[:, 0], y_true[:, 0], reduction="sum"
            ) + F.mse_loss(y_pred[:, 1], y_true[:, 1], reduction="sum")

    assert torch.allclose(multi_target_loss, manual_loss)


@pytest.mark.parametrize("binary_trick", [False, True])
def test_multi_target_classification_loss(binary_trick):

    tab_preprocessor = TabPreprocessor(continuous_cols=["col1", "col2", "col3", "col4"])
    X_tab = tab_preprocessor.fit_transform(df)
    X_tab_tnsr = torch.tensor(X_tab, dtype=torch.float32)

    tab_ml = TabMlp(
        column_idx=tab_preprocessor.column_idx,
        continuous_cols=tab_preprocessor.continuous_cols,
        mlp_hidden_dims=[16, 8],
    )

    classification_loss = MultiTargetClassificationLoss(
        binary_config=[0], multiclass_config=[(1, 3)], binary_trick=binary_trick
    )

    model = WideDeep(deeptabular=tab_ml, pred_dim=2 + 3)

    y_true = torch.tensor(
        df[["target3_binary", "target4_multiclass"]].values, dtype=torch.float32
    )
    y_pred = model({"deeptabular": X_tab_tnsr})

    multi_target_loss = classification_loss(y_pred, y_true)

    # just assert it has run
    assert multi_target_loss.item() > 0


def test_multi_target_classification_loss_with_weights():

    tab_preprocessor = TabPreprocessor(continuous_cols=["col1", "col2", "col3", "col4"])
    X_tab = tab_preprocessor.fit_transform(df)
    X_tab_tnsr = torch.tensor(X_tab, dtype=torch.float32)

    tab_ml = TabMlp(
        column_idx=tab_preprocessor.column_idx,
        continuous_cols=tab_preprocessor.continuous_cols,
        mlp_hidden_dims=[16, 8],
    )

    classification_loss = MultiTargetClassificationLoss(
        binary_config=[(0, 0.2)],
        multiclass_config=[(1, 3, [1.0, 2.0, 3.0])],
        target_weights=[1.0, 5.0],
    )

    model = WideDeep(deeptabular=tab_ml, pred_dim=1 + 3)

    y_true = torch.tensor(
        df[["target3_binary", "target4_multiclass"]].values, dtype=torch.float32
    )
    y_pred = model({"deeptabular": X_tab_tnsr})

    multi_target_loss = classification_loss(y_pred, y_true)

    # just assert it has run
    assert multi_target_loss.item() > 0


@pytest.mark.parametrize("binary_trick", [False, True])
def test_multi_target_regression_and_classification_loss(binary_trick):

    tab_preprocessor = TabPreprocessor(
        continuous_cols=["col1", "col2", "col3", "col4", "col5", "col6", "col7", "col8"]
    )
    X_tab = tab_preprocessor.fit_transform(df)
    X_tab_tnsr = torch.tensor(X_tab, dtype=torch.float32)

    tab_ml = TabMlp(
        column_idx=tab_preprocessor.column_idx,
        continuous_cols=tab_preprocessor.continuous_cols,
        mlp_hidden_dims=[16, 8],
    )

    loss = MutilTargetRegressionAndClassificationLoss(
        regression_config=[0, 1],
        binary_config=[2] if binary_trick else [(2, 0.2)],
        multiclass_config=[(3, 3)] if binary_trick else [(3, 3, [1.0, 2.0, 3.0])],
        binary_trick=binary_trick,
        weights=None if binary_trick else [1.0, 2.0, 3.0, 4.0],
    )

    regres_dim = 1
    bin_dim = 2 if binary_trick else 1
    multiclass_dim = 3
    model = WideDeep(
        deeptabular=tab_ml, pred_dim=regres_dim + regres_dim + bin_dim + multiclass_dim
    )

    y_true = torch.tensor(
        df[
            [
                "target1_regression",
                "target2_regression",
                "target3_binary",
                "target4_multiclass",
            ]
        ].values,
        dtype=torch.float32,
    )
    y_pred = model({"deeptabular": X_tab_tnsr})

    multi_target_loss = loss(y_pred, y_true)

    # just assert it has run
    assert multi_target_loss.item() > 0


def test_multi_target_regression_loss_errors():

    tab_preprocessor = TabPreprocessor(
        continuous_cols=["col1", "col2", "col3", "col4", "col5", "col6", "col7", "col8"]
    )
    X_tab = tab_preprocessor.fit_transform(df)
    X_tab_tnsr = torch.tensor(X_tab, dtype=torch.float32)

    tab_ml = TabMlp(
        column_idx=tab_preprocessor.column_idx,
        continuous_cols=tab_preprocessor.continuous_cols,
        mlp_hidden_dims=[16, 8],
    )

    multi_target_regression_loss = MultiTargetRegressionLoss(
        weights=[1, 2, 3], reduction="sum"
    )

    model = WideDeep(deeptabular=tab_ml, pred_dim=2)

    y_true = torch.tensor(
        df[["target1_regression", "target2_regression"]].values, dtype=torch.float32
    )
    y_pred = model({"deeptabular": X_tab_tnsr})

    with pytest.raises(AssertionError):
        # weights must have the same length as the number of targets
        multi_target_regression_loss(y_pred, y_true)

    with pytest.raises(ValueError):
        # reduction must be either 'mean' or 'sum'
        MultiTargetRegressionLoss(reduction="wrong")


def test_multi_target_classification_loss_errors():

    with pytest.raises(ValueError):
        # reduction must be either 'mean' or 'sum'
        MultiTargetClassificationLoss(
            binary_config=[0], multiclass_config=[(1, 3)], reduction="wrong"
        )

    with pytest.raises(ValueError):
        # weights must have the same length as the number of targets
        MultiTargetClassificationLoss(
            binary_config=[0], multiclass_config=[(1, 3)], weights=[1.0, 2.0, 3.0]
        )

    with pytest.raises(ValueError):
        # If binary_trick is True, binary_config must be a list of integers
        MultiTargetClassificationLoss(
            binary_config=[(0, 0.2)],
            multiclass_config=[(1, 3)],
            binary_trick=True,
        )

    with pytest.raises(ValueError):
        # If binary_trick is True, multiclass_config must be a list of tuples
        MultiTargetClassificationLoss(
            binary_config=[0],
            multiclass_config=[(1, 3, [1.0, 2.0, 3.0])],
            binary_trick=True,
        )

    with pytest.raises(ValueError):
        # if binary_trick is True, the binary targets must be the first targets
        MultiTargetClassificationLoss(
            binary_config=[1], multiclass_config=[(0, 3)], binary_trick=True
        )


def test_multi_target_regression_and_classification_loss_errors():

    with pytest.raises(AssertionError):
        # binary_config and multiclass_config cannot be both None
        MutilTargetRegressionAndClassificationLoss(
            regression_config=[0, 1],
        )

    with pytest.raises(ValueError):
        # if binary_trick is True, the target order should be regression,
        # binary, multiclass
        MutilTargetRegressionAndClassificationLoss(
            regression_config=[1],
            binary_config=[0],
            binary_trick=True,
        )

    with pytest.raises(ValueError):
        # if binary_trick is True, the target order should be regression,
        # binary, multiclass
        MutilTargetRegressionAndClassificationLoss(
            regression_config=[0],
            binary_config=[2],
            multiclass_config=[(1, 3)],
            binary_trick=True,
        )

    with pytest.raises(ValueError):
        # If weigths is not None, it must have the same length as the number of targets
        MutilTargetRegressionAndClassificationLoss(
            regression_config=[0, 1],
            binary_config=[(2, 0.2)],
            multiclass_config=[(3, 3, [1.0, 2.0, 3.0])],
            weights=[1.0, 2.0],
        )


@pytest.mark.parametrize(
    "problem", ["regression", "classification", "regression_and_classification"]
)
def test_multi_target_losses_integration(problem):

    tab_preprocessor = TabPreprocessor(
        continuous_cols=["col1", "col2", "col3", "col4", "col5", "col6", "col7", "col8"]
    )
    X_tab = tab_preprocessor.fit_transform(df)

    tab_ml = TabMlp(
        column_idx=tab_preprocessor.column_idx,
        continuous_cols=tab_preprocessor.continuous_cols,
        mlp_hidden_dims=[16, 8],
    )

    if problem == "regression":
        loss = MultiTargetRegressionLoss()
        pred_dim = 2
        target = df[["target1_regression", "target2_regression"]].values
    elif problem == "classification":
        loss = MultiTargetClassificationLoss(
            binary_config=[0],
            multiclass_config=[(1, 3)],
        )
        pred_dim = 1 + 3
        target = df[["target3_binary", "target4_multiclass"]].values
    else:
        loss = MutilTargetRegressionAndClassificationLoss(
            regression_config=[0, 1],
            binary_config=[2],
            multiclass_config=[(3, 3)],
        )
        pred_dim = 2 + 1 + 3
        target = df[
            [
                "target1_regression",
                "target2_regression",
                "target3_binary",
                "target4_multiclass",
            ]
        ].values

    model = WideDeep(deeptabular=tab_ml, pred_dim=pred_dim)

    trainer = Trainer(
        model, objective="multitarget", custom_loss_function=loss, verbose=0
    )

    trainer.fit(X_tab=X_tab, target=target, n_epochs=1)

    if problem == "regression":
        preds = trainer.predict(X_tab=X_tab)
    else:
        preds = trainer.predict_proba(X_tab=X_tab)

    assert trainer.history["train_loss"][0] != 0
    assert preds.shape[0] == df.shape[0] and preds.shape[1] == pred_dim


@pytest.mark.parametrize("problem", ["classification", "regression_and_classification"])
def test_predict_error_for_classification_problems(problem):

    tab_preprocessor = TabPreprocessor(
        continuous_cols=["col1", "col2", "col3", "col4", "col5", "col6", "col7", "col8"]
    )
    X_tab = tab_preprocessor.fit_transform(df)

    tab_ml = TabMlp(
        column_idx=tab_preprocessor.column_idx,
        continuous_cols=tab_preprocessor.continuous_cols,
        mlp_hidden_dims=[16, 8],
    )

    if problem == "classification":
        loss = MultiTargetClassificationLoss(
            binary_config=[0],
            multiclass_config=[(1, 3)],
        )
        pred_dim = 1 + 3
        target = df[["target3_binary", "target4_multiclass"]].values
    else:
        loss = MutilTargetRegressionAndClassificationLoss(
            regression_config=[0, 1],
            binary_config=[2],
            multiclass_config=[(3, 3)],
        )
        pred_dim = 2 + 1 + 3
        target = df[
            [
                "target1_regression",
                "target2_regression",
                "target3_binary",
                "target4_multiclass",
            ]
        ].values

    model = WideDeep(deeptabular=tab_ml, pred_dim=pred_dim)

    trainer = Trainer(
        model, objective="multitarget", custom_loss_function=loss, verbose=0
    )

    trainer.fit(X_tab=X_tab, target=target, n_epochs=1)

    with pytest.raises(ValueError):
        trainer.predict(X_tab=X_tab)
