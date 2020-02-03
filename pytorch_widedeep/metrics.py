import numpy as np

from .callbacks import Callback
from .wdtypes import *


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


class CategoricalAccuracy(Metric):
    def __init__(self, top_k=1):
        self.top_k = top_k
        self.correct_count = 0
        self.total_count = 0

        self._name = "acc"

    def reset(self):
        self.correct_count = 0
        self.total_count = 0

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> np.ndarray:
        top_k = y_pred.topk(self.top_k, 1)[1]
        true_k = y_true.view(len(y_true), 1).expand_as(top_k)  # type: ignore
        self.correct_count += top_k.eq(true_k).float().sum().item()
        self.total_count += len(y_pred)  # type: ignore
        accuracy = float(self.correct_count) / float(self.total_count)
        return np.round(accuracy, 4)


class BinaryAccuracy(Metric):
    def __init__(self):
        self.correct_count = 0
        self.total_count = 0

        self._name = "acc"

    def reset(self):
        self.correct_count = 0
        self.total_count = 0

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> np.ndarray:
        y_pred_round = y_pred.round()
        self.correct_count += y_pred_round.eq(y_true.view(-1, 1)).float().sum().item()
        self.total_count += len(y_pred)  # type: ignore
        accuracy = float(self.correct_count) / float(self.total_count)
        return np.round(accuracy, 4)
