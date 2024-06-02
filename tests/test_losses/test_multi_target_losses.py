import numpy as np
import torch
import pandas as pd
import pytest
import torch.nn.functional as F
from sklearn.datasets import make_regression, make_classification

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
    df["target1_regression"] = y_regression1
    df["target2_regression"] = y_regression2
    df["target3_binary"] = y_classification_binary
    df["target4_multiclass"] = y_classification_multiclass

    return df


df = create_multitarget_data()


@pytest.mark.parametrize("weights", [None, [1, 5]])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_multi_target_regression_loss(weights, reduction):

    tab_preprocessor = TabPreprocessor(continuous_cols=["col1", "col2", "col3", "col4"])
    X_tab = tab_preprocessor.fit_transform(df)
    X_tab_tnsr = torch.tensor(X_tab, dtype=torch.float32)

    tab_ml = TabMlp(
        column_idx=tab_preprocessor.column_idx,
        continuous_cols=tab_preprocessor.continuous_cols,
        mlp_hidden_dims=[16, 8],
    )

    regression_loss = MultiTargetRegressionLoss(weights=weights, reduction=reduction)

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
        weights=[1.0, 5.0],
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


# TO DO: check assertion and ValueErrors
