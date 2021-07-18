import numpy as np
import torch
from torchmetrics import Metric as TorchMetric

from .wdtypes import *  # noqa: F403


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
            if isinstance(metric, Metric):
                logs[self.prefix + metric._name] = metric(y_pred, y_true)
            if isinstance(metric, TorchMetric):
                if metric.num_classes == 2:
                    metric.update(torch.round(y_pred).int(), y_true.int())
                if metric.num_classes > 2:  # type: ignore[operator]
                    metric.update(torch.max(y_pred, dim=1).indices, y_true.int())  # type: ignore[attr-defined]
                logs[self.prefix + type(metric).__name__] = (
                    metric.compute().detach().cpu().numpy()
                )
        return logs


class Accuracy(Metric):
    r"""Class to calculate the accuracy for both binary and categorical problems

    Parameters
    ----------
    top_k: int, default = 1
        Accuracy will be computed using the top k most likely classes in
        multiclass problems

    Examples
    --------
    >>> import torch
    >>>
    >>> from pytorch_widedeep.metrics import Accuracy
    >>>
    >>> acc = Accuracy()
    >>> y_true = torch.tensor([0, 1, 0, 1]).view(-1, 1)
    >>> y_pred = torch.tensor([[0.3, 0.2, 0.6, 0.7]]).view(-1, 1)
    >>> acc(y_pred, y_true)
    array(0.5)
    >>>
    >>> acc = Accuracy(top_k=2)
    >>> y_true = torch.tensor([0, 1, 2])
    >>> y_pred = torch.tensor([[0.3, 0.5, 0.2], [0.1, 0.1, 0.8], [0.1, 0.5, 0.4]])
    >>> acc(y_pred, y_true)
    array(0.66666667)
    """

    def __init__(self, top_k: int = 1):
        super(Accuracy, self).__init__()

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
            y_true = y_true
        elif num_classes > 1:
            y_pred = y_pred.topk(self.top_k, 1)[1]
            y_true = y_true.view(-1, 1).expand_as(y_pred)

        self.correct_count += y_pred.eq(y_true).sum().item()  # type: ignore[assignment]
        self.total_count += len(y_pred)
        accuracy = float(self.correct_count) / float(self.total_count)
        return np.array(accuracy)


class Precision(Metric):
    r"""Class to calculate the precision for both binary and categorical problems

    Parameters
    ----------
    average: bool, default = True
        This applies only to multiclass problems. if ``True`` calculate
        precision for each label, and finds their unweighted mean.

    Examples
    --------
    >>> import torch
    >>>
    >>> from pytorch_widedeep.metrics import Precision
    >>>
    >>> prec = Precision()
    >>> y_true = torch.tensor([0, 1, 0, 1]).view(-1, 1)
    >>> y_pred = torch.tensor([[0.3, 0.2, 0.6, 0.7]]).view(-1, 1)
    >>> prec(y_pred, y_true)
    array(0.5)
    >>>
    >>> prec = Precision(average=True)
    >>> y_true = torch.tensor([0, 1, 2])
    >>> y_pred = torch.tensor([[0.7, 0.1, 0.2], [0.1, 0.1, 0.8], [0.1, 0.5, 0.4]])
    >>> prec(y_pred, y_true)
    array(0.33333334)
    """

    def __init__(self, average: bool = True):
        super(Precision, self).__init__()

        self.average = average
        self.true_positives = 0
        self.all_positives = 0
        self.eps = 1e-20
        self._name = "prec"

    def reset(self):
        """
        resets counters to 0
        """
        self.true_positives = 0
        self.all_positives = 0

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> np.ndarray:
        num_class = y_pred.size(1)

        if num_class == 1:
            y_pred = y_pred.round()
            y_true = y_true
        elif num_class > 1:
            y_true = torch.eye(num_class)[y_true.squeeze().long()]
            y_pred = y_pred.topk(1, 1)[1].view(-1)
            y_pred = torch.eye(num_class)[y_pred.long()]

        self.true_positives += (y_true * y_pred).sum(dim=0)  # type:ignore
        self.all_positives += y_pred.sum(dim=0)  # type:ignore

        precision = self.true_positives / (self.all_positives + self.eps)

        if self.average:
            return np.array(precision.mean().item())  # type:ignore
        else:
            return precision.detach().cpu().numpy()  # type: ignore[attr-defined]


class Recall(Metric):
    r"""Class to calculate the recall for both binary and categorical problems

    Parameters
    ----------
    average: bool, default = True
        This applies only to multiclass problems. if ``True`` calculate recall
        for each label, and finds their unweighted mean.

    Examples
    --------
    >>> import torch
    >>>
    >>> from pytorch_widedeep.metrics import Recall
    >>>
    >>> rec = Recall()
    >>> y_true = torch.tensor([0, 1, 0, 1]).view(-1, 1)
    >>> y_pred = torch.tensor([[0.3, 0.2, 0.6, 0.7]]).view(-1, 1)
    >>> rec(y_pred, y_true)
    array(0.5)
    >>>
    >>> rec = Recall(average=True)
    >>> y_true = torch.tensor([0, 1, 2])
    >>> y_pred = torch.tensor([[0.7, 0.1, 0.2], [0.1, 0.1, 0.8], [0.1, 0.5, 0.4]])
    >>> rec(y_pred, y_true)
    array(0.33333334)
    """

    def __init__(self, average: bool = True):
        super(Recall, self).__init__()

        self.average = average
        self.true_positives = 0
        self.actual_positives = 0
        self.eps = 1e-20
        self._name = "rec"

    def reset(self):
        """
        resets counters to 0
        """
        self.true_positives = 0
        self.actual_positives = 0

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> np.ndarray:
        num_class = y_pred.size(1)

        if num_class == 1:
            y_pred = y_pred.round()
            y_true = y_true
        elif num_class > 1:
            y_true = torch.eye(num_class)[y_true.squeeze().long()]
            y_pred = y_pred.topk(1, 1)[1].view(-1)
            y_pred = torch.eye(num_class)[y_pred.long()]

        self.true_positives += (y_true * y_pred).sum(dim=0)  # type: ignore
        self.actual_positives += y_true.sum(dim=0)  # type: ignore

        recall = self.true_positives / (self.actual_positives + self.eps)

        if self.average:
            return np.array(recall.mean().item())  # type:ignore
        else:
            return recall.detach().cpu().numpy()  # type: ignore[attr-defined]


class FBetaScore(Metric):
    r"""Class to calculate the fbeta score for both binary and categorical problems

    :math:`F_{\beta} = ((1 + {\beta}^2) * \frac{(precision * recall)}{({\beta}^2 * precision + recall)}`

    Parameters
    ----------
    beta: int
        Coefficient to control the balance between precision and recall
    average: bool, default = True
        This applies only to multiclass problems. if ``True`` calculate fbeta
        for each label, and find their unweighted mean.

    Examples
    --------
    >>> import torch
    >>>
    >>> from pytorch_widedeep.metrics import FBetaScore
    >>>
    >>> fbeta = FBetaScore(beta=2)
    >>> y_true = torch.tensor([0, 1, 0, 1]).view(-1, 1)
    >>> y_pred = torch.tensor([[0.3, 0.2, 0.6, 0.7]]).view(-1, 1)
    >>> fbeta(y_pred, y_true)
    array(0.5)
    >>>
    >>> fbeta = FBetaScore(beta=2)
    >>> y_true = torch.tensor([0, 1, 2])
    >>> y_pred = torch.tensor([[0.7, 0.1, 0.2], [0.1, 0.1, 0.8], [0.1, 0.5, 0.4]])
    >>> fbeta(y_pred, y_true)
    array(0.33333334)
    """

    def __init__(self, beta: int, average: bool = True):
        super(FBetaScore, self).__init__()

        self.beta = beta
        self.average = average
        self.precision = Precision(average=False)
        self.recall = Recall(average=False)
        self.eps = 1e-20
        self._name = "".join(["f", str(self.beta)])

    def reset(self):
        """
        resets precision and recall
        """
        self.precision.reset()
        self.recall.reset()

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> np.ndarray:

        prec = self.precision(y_pred, y_true)
        rec = self.recall(y_pred, y_true)
        beta2 = self.beta ** 2

        fbeta = ((1 + beta2) * prec * rec) / (beta2 * prec + rec + self.eps)

        if self.average:
            return np.array(fbeta.mean().item())  # type: ignore[attr-defined]
        else:
            return fbeta


class F1Score(Metric):
    r"""Class to calculate the f1 score for both binary and categorical problems

    Parameters
    ----------
    average: bool, default = True
        This applies only to multiclass problems. if ``True`` calculate f1 for
        each label, and find their unweighted mean.

    Examples
    --------
    >>> import torch
    >>>
    >>> from pytorch_widedeep.metrics import F1Score
    >>>
    >>> f1 = F1Score()
    >>> y_true = torch.tensor([0, 1, 0, 1]).view(-1, 1)
    >>> y_pred = torch.tensor([[0.3, 0.2, 0.6, 0.7]]).view(-1, 1)
    >>> f1(y_pred, y_true)
    array(0.5)
    >>>
    >>> f1 = F1Score()
    >>> y_true = torch.tensor([0, 1, 2])
    >>> y_pred = torch.tensor([[0.7, 0.1, 0.2], [0.1, 0.1, 0.8], [0.1, 0.5, 0.4]])
    >>> f1(y_pred, y_true)
    array(0.33333334)
    """

    def __init__(self, average: bool = True):
        super(F1Score, self).__init__()

        self.average = average
        self.f1 = FBetaScore(beta=1, average=self.average)
        self._name = self.f1._name

    def reset(self):
        """
        resets counters to 0
        """
        self.f1.reset()

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> np.ndarray:
        return self.f1(y_pred, y_true)


class R2Score(Metric):
    r"""
    Calculates the R-Squared, the
    `coefficient of determination <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_:

    :math:`R^2 = 1 - \frac{\sum_{j=1}^n(y_j - \hat{y_j})^2}{\sum_{j=1}^n(y_j - \bar{y})^2}`,

    where :math:`\hat{y_j}` is the ground truth, :math:`y_j` is the predicted value and
    :math:`\bar{y}` is the mean of the ground truth.

    Examples
    --------
    >>> import torch
    >>>
    >>> from pytorch_widedeep.metrics import R2Score
    >>>
    >>> r2 = R2Score()
    >>> y_true = torch.tensor([3, -0.5, 2, 7]).view(-1, 1)
    >>> y_pred = torch.tensor([2.5, 0.0, 2, 8]).view(-1, 1)
    >>> r2(y_pred, y_true)
    array(0.94860814)
    """

    def __init__(self):
        self.numerator = 0
        self.denominator = 0
        self.num_examples = 0
        self.y_true_sum = 0

        self._name = "r2"

    def reset(self):
        """
        resets counters to 0
        """
        self.numerator = 0
        self.denominator = 0
        self.num_examples = 0
        self.y_true_sum = 0

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> np.ndarray:

        self.numerator += ((y_pred - y_true) ** 2).sum().item()

        self.num_examples += y_true.shape[0]
        self.y_true_sum += y_true.sum().item()
        y_true_avg = self.y_true_sum / self.num_examples
        self.denominator += ((y_true - y_true_avg) ** 2).sum().item()
        return np.array((1 - (self.numerator / self.denominator)))
