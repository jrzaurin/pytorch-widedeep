from copy import deepcopy

import numpy as np
import torch
import pytest
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    recall_score,
    accuracy_score,
    precision_score,
)

from pytorch_widedeep.metrics import (
    Recall,
    F1Score,
    Accuracy,
    Precision,
    FBetaScore,
)


def f2_score_bin(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2)


y_true_bin_np = np.array([1, 0, 0, 0, 1, 1, 0])
y_pred_bin_np = np.array([0.6, 0.3, 0.2, 0.8, 0.4, 0.9, 0.6])

y_true_bin_pt = torch.from_numpy(y_true_bin_np)
y_pred_bin_pt = torch.from_numpy(y_pred_bin_np).view(-1, 1)


###############################################################################
# Test binary metrics
###############################################################################
@pytest.mark.parametrize(
    "sklearn_metric, widedeep_metric",
    [
        (accuracy_score, Accuracy()),
        (precision_score, Precision()),
        (recall_score, Recall()),
        (f1_score, F1Score()),
        (f2_score_bin, FBetaScore(beta=2)),
    ],
)
def test_binary_metrics(sklearn_metric, widedeep_metric):
    assert np.isclose(
        sklearn_metric(y_true_bin_np, y_pred_bin_np.round()),
        widedeep_metric(y_pred_bin_pt, y_true_bin_pt),
    )


###############################################################################
# Test top_k for Accuracy
###############################################################################
@pytest.mark.parametrize("top_k, expected_acc", [(1, 0.33), (2, 0.66)])
def test_categorical_accuracy_topk(top_k, expected_acc):
    y_true = torch.from_numpy(np.random.choice(3, 100))
    y_pred = torch.from_numpy(np.random.rand(100, 3))
    metric = Accuracy(top_k=top_k)
    acc = metric(y_pred, y_true)
    assert np.isclose(acc, expected_acc, atol=0.3)


###############################################################################
# Test multiclass metrics
###############################################################################
y_true_multi_np = np.array([1, 0, 2, 1, 1, 2, 2, 0, 0, 0])
y_pred_muli_np = np.array(
    [
        [0.2, 0.6, 0.2],
        [0.4, 0.5, 0.1],
        [0.1, 0.1, 0.8],
        [0.1, 0.6, 0.3],
        [0.1, 0.8, 0.1],
        [0.1, 0.6, 0.6],
        [0.2, 0.6, 0.8],
        [0.6, 0.1, 0.3],
        [0.7, 0.2, 0.1],
        [0.1, 0.7, 0.2],
    ]
)

y_true_multi_pt = torch.from_numpy(y_true_multi_np)
y_pred_multi_pt = torch.from_numpy(y_pred_muli_np)


def f2_score_multi(y_true, y_pred, average):
    return fbeta_score(y_true, y_pred, average=average, beta=2)


@pytest.mark.parametrize(
    "sklearn_metric, widedeep_metric",
    [
        (accuracy_score, Accuracy()),
        (precision_score, Precision()),
        (recall_score, Recall()),
        (f1_score, F1Score()),
        (f2_score_multi, FBetaScore(beta=2)),
    ],
)
def test_muticlass_metrics(sklearn_metric, widedeep_metric):
    if sklearn_metric.__name__ == "accuracy_score":
        assert np.isclose(
            sklearn_metric(y_true_multi_np, y_pred_muli_np.argmax(axis=1)),
            widedeep_metric(y_pred_multi_pt, y_true_multi_pt),
        )
    else:
        assert np.isclose(
            sklearn_metric(
                y_true_multi_np, y_pred_muli_np.argmax(axis=1), average="macro"
            ),
            widedeep_metric(y_pred_multi_pt, y_true_multi_pt),
        )
