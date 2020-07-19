import numpy as np
import torch

from .wdtypes import *
from .callbacks import Callback


class Metric(object):
    def __init__(self):
        self._name = ""

    def reset(self):
        raise NotImplementedError("Custom Metrics must implement this function")

    def __call__(self, y_pred: Tensor, y_true: Tensor):
        raise NotImplementedError("Custom Metrics must implement this function")


class MultipleMetrics(object):
    def __init__(self, metrics: List[Metric], prefix: str = ""):

        instantiated_metrics = []
        for metric in metrics:
            if isinstance(metric, type):
                instantiated_metrics.append(metric())
            else:
                instantiated_metrics.append(metric)
        self._metrics = instantiated_metrics
        self.prefix = prefix

    def reset(self):
        for metric in self._metrics:
            metric.reset()

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Dict:
        logs = {}
        for metric in self._metrics:
            logs[self.prefix + metric._name] = metric(y_pred, y_true)
        return logs


class MetricCallback(Callback):
    def __init__(self, container: MultipleMetrics):
        self.container = container

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        self.container.reset()


class Accuracy(Metric):
    r"""Class to calculate the accuracy for both binary and categorical problems

    Parameters
    ----------
    top_k: int, default = 1
        Accuracy will be computed using the top k most likely classes in
        multiclass problems
    """

    def __init__(self, top_k: int = 1):
        self.top_k = top_k
        self.correct_count = 0
        self.total_count = 0

        self._name = "acc"

    def reset(self):
        """
        resets counters to 0
        """
        self.correct_count = 0
        self.total_count = 0

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> np.ndarray:
        num_classes = y_pred.size(1)

        if num_classes == 1:
            y_pred = y_pred.round()
            y_true = y_true.view(-1, 1)
        elif num_classes > 1:
            y_pred = y_pred.topk(self.top_k, 1)[1]
            y_true = y_true.view(-1, 1).expand_as(y_pred)  # type: ignore

        self.correct_count += y_pred.eq(y_true).sum().item()  # type: ignore
        self.total_count += len(y_pred)  # type: ignore
        accuracy = float(self.correct_count) / float(self.total_count)
        return accuracy


class Precision(Metric):
    r"""Class to calculate the precision for both binary and categorical problems

    Parameters
    ----------
    average: bool, default = True
        This applies only to multiclass problems. if `True` calculate
        precision for each label, and find their unweighted mean.
    """

    def __init__(self, average: bool = True):
        self.average = average
        self.true_positives = 0
        self.all_positives = 0
        self.eps = 1e-20

        self._name = "prec"

    def reset(self) -> None:
        """
        resets counters to 0
        """
        self.true_positives = 0
        self.all_positives = 0

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> np.ndarray:
        num_class = y_pred.size(1)

        if num_class == 1:
            y_pred = y_pred.round()
            y_true = y_true.view(-1, 1)
        elif num_class > 1:
            y_true = torch.eye(num_class)[y_true.long()]
            y_pred = y_pred.topk(1, 1)[1].view(-1)
            y_pred = torch.eye(num_class)[y_pred.long()]

        self.true_positives += (y_true * y_pred).sum(dim=0)  # type:ignore
        self.all_positives += y_pred.sum(dim=0)  # type:ignore

        precision = self.true_positives / (self.all_positives + self.eps)

        if self.average:
            return precision.mean().item()  # type:ignore
        else:
            return precision


class Recall(Metric):
    r"""Class to calculate the recall for both binary and categorical problems

    Parameters
    ----------
    average: bool, default = True
        This applies only to multiclass problems. if `True` calculate recall
        for each label, and find their unweighted mean.
    """

    def __init__(self, average: bool = True):
        self.average = average
        self.true_positives = 0
        self.actual_positives = 0
        self.eps = 1e-20

        self._name = "rec"

    def reset(self) -> None:
        """
        resets counters to 0
        """
        self.true_positives = 0
        self.actual_positives = 0

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> np.ndarray:
        num_class = y_pred.size(1)

        if num_class == 1:
            y_pred = y_pred.round()
            y_true = y_true.view(-1, 1)
        elif num_class > 1:
            y_true = torch.eye(num_class)[y_true.long()]
            y_pred = y_pred.topk(1, 1)[1].view(-1)
            y_pred = torch.eye(num_class)[y_pred.long()]

        self.true_positives += (y_true * y_pred).sum(dim=0)  # type: ignore
        self.actual_positives += y_true.sum(dim=0)  # type: ignore

        recall = self.true_positives / (self.actual_positives + self.eps)

        if self.average:
            return recall.mean().item()  # type:ignore
        else:
            return recall


class FBetaScore(Metric):
    r"""Class to calculate the fbeta score for both binary and categorical problems

    ``FBeta = ((1 + beta^2) * Precision * Recall) / (beta^2 * Precision + Recall)``

    Parameters
    ----------
    beta: int
        Coefficient to control the balance between precision and recall
    average: bool, default = True
        This applies only to multiclass problems. if `True` calculate fbeta
        for each label, and find their unweighted mean.
    """

    def __init__(self, beta: int, average: bool = True):
        self.average = average

        self.precision = Precision(average=False)
        self.recall = Recall(average=False)

        self.beta = beta

        self._name = "".join(["f", str(beta)])

    def reset(self) -> None:
        """
        resets precision and recall
        """
        self.precision.reset()
        self.recall.reset()

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> np.ndarray:

        prec = self.precision(y_pred, y_true)
        rec = self.recall(y_pred, y_true)
        beta2 = self.beta ** 2

        fbeta = ((1 + beta2) * prec * rec) / (beta2 * prec + rec)

        if self.average:
            return fbeta.mean().item()
        else:
            return fbeta


class F1Score(Metric):
    r"""Class to calculate the f1 score for both binary and categorical problems

    Parameters
    ----------
    average: bool, default = True
        This applies only to multiclass problems. if `True` calculate f1 for
        each label, and find their unweighted mean.
    """

    def __init__(self, average: bool = True):
        self.f1 = FBetaScore(beta=1, average=average)
        self._name = self.f1._name

    def reset(self) -> None:
        """
        resets counters to 0
        """
        self.f1.reset()

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> np.ndarray:
        return self.f1(y_pred, y_true)
