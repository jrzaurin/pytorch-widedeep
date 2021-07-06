import torch

from pytorch_widedeep.callbacks import Callback
from torchmetrics import Metric
import pdb
from torchmetrics import Accuracy, Precision, Recall, F1

from .wdtypes import *  # noqa: F403


class MultipleMetrics(object):
    def __init__(self, metrics: List[Metric], prefix: str = ""):

        instantiated_metrics = []
        for metric in metrics:
            if isinstance(metric, Metric):
                instantiated_metrics.append(metric)
            else:
                NotImplementedError("Custom Metrics must be implemented "
                                    "using torchmetrics Metric class")
        self._metrics = instantiated_metrics
        self.prefix = prefix

    def reset(self):
        for metric in self._metrics:
            metric.reset()

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Dict:
        logs = {}
        for metric in self._metrics:
            metric.update(y_pred, y_true)
            logs[self.prefix + type(metric).__name__] = metric.compute().detach().cpu().numpy()
        return logs


class MetricCallback(Callback):
    def __init__(self, container: MultipleMetrics):
        self.container = container

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        self.container.reset()

    def on_eval_begin(self, logs: Optional[Dict] = None):
        self.container.reset()
