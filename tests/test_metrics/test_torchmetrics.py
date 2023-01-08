import numpy as np
import torch
import pytest
from torchmetrics import Recall, F1Score, Accuracy, Precision, FBetaScore
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    recall_score,
    accuracy_score,
    precision_score,
)

from pytorch_widedeep.metrics import MultipleMetrics


def f2_score_bin(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2)


y_true_bin_np = np.array([1, 0, 0, 0, 1, 1, 0]).reshape((-1, 1))
y_pred_bin_np = np.array([0.6, 0.3, 0.2, 0.8, 0.4, 0.9, 0.6]).reshape((-1, 1))

y_true_bin_pt = torch.from_numpy(y_true_bin_np)
y_pred_bin_pt = torch.from_numpy(y_pred_bin_np)


###############################################################################
# Test binary metrics
###############################################################################
@pytest.mark.parametrize(
    "metric_name, sklearn_metric, torch_metric",
    [
        ("Accuracy", accuracy_score, Accuracy(task="binary")),
        ("Precision", precision_score, Precision(task="binary")),
        ("Recall", recall_score, Recall(task="binary")),
        ("F1Score", f1_score, F1Score(task="binary")),
        ("FBetaScore", f2_score_bin, FBetaScore(task="binary", beta=2)),
    ],
)
def test_binary_metrics(metric_name, sklearn_metric, torch_metric):
    sk_res = sklearn_metric(y_true_bin_np, y_pred_bin_np.round())
    wd_metric = MultipleMetrics(metrics=[torch_metric])
    wd_logs = wd_metric(y_pred_bin_pt, y_true_bin_pt)
    wd_res = wd_logs[metric_name]
    if wd_res.size != 1:
        wd_res = wd_res[1]
    assert np.isclose(sk_res, wd_res)


# ###############################################################################
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
        [0.1, 0.3, 0.6],
        [0.1, 0.1, 0.8],
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
    "metric_name, sklearn_metric, torch_metric",
    [
        (
            "Accuracy",
            accuracy_score,
            Accuracy(task="multiclass", num_classes=3, average="micro"),
        ),
        (
            "Precision",
            precision_score,
            Precision(task="multiclass", num_classes=3, average="macro"),
        ),
        (
            "Recall",
            recall_score,
            Recall(task="multiclass", num_classes=3, average="macro"),
        ),
        (
            "F1Score",
            f1_score,
            F1Score(task="multiclass", num_classes=3, average="macro"),
        ),
        (
            "MulticlassAccuracy",
            accuracy_score,
            Accuracy(task="multiclass", num_classes=3, average="micro"),
        ),
        (
            "MulticlassPrecision",
            precision_score,
            Precision(task="multiclass", num_classes=3, average="macro"),
        ),
        (
            "MulticlassRecall",
            recall_score,
            Recall(task="multiclass", num_classes=3, average="macro"),
        ),
        (
            "MulticlassF1Score",
            f1_score,
            F1Score(task="multiclass", num_classes=3, average="macro"),
        ),
        (
            "MulticlassFBetaScore",
            f2_score_multi,
            FBetaScore(task="multiclass", beta=3, num_classes=3, average="macro"),
        ),
    ],
)
def test_muticlass_metrics(metric_name, sklearn_metric, torch_metric):
    if metric_name == "MulticlassAccuracy":
        sk_res = sklearn_metric(y_true_multi_np, y_pred_muli_np.argmax(axis=1))
    else:
        sk_res = sklearn_metric(
            y_true_multi_np, y_pred_muli_np.argmax(axis=1), average="macro"
        )

    wd_metric = MultipleMetrics(metrics=[torch_metric])
    wd_logs = wd_metric(y_pred_multi_pt, y_true_multi_pt)
    wd_res = wd_logs[metric_name]

    assert np.isclose(sk_res, wd_res, atol=0.01)
