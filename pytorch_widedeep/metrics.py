import numpy as np
import torch
from torchmetrics import Metric as TorchMetric

from pytorch_widedeep.wdtypes import Dict, List, Union, Tensor, Optional
from pytorch_widedeep.utils.general_utils import alias


class Metric(object):
    def __init__(self):
        self._name = ""

    def reset(self):
        raise NotImplementedError("Custom Metrics must implement this function")

    def __call__(self, y_pred: Tensor, y_true: Tensor):
        raise NotImplementedError("Custom Metrics must implement this function")


class MultipleMetrics(object):
    def __init__(self, metrics: List[Union[Metric, object]], prefix: str = ""):
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
            elif isinstance(metric, TorchMetric):
                metric.update(y_pred, y_true.int())  # type: ignore[attr-defined]
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
            y_true = torch.eye(num_class)[y_true.squeeze().cpu().long()]
            y_pred = y_pred.topk(1, 1)[1].view(-1)
            y_pred = torch.eye(num_class)[y_pred.cpu().long()]

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
            y_true = torch.eye(num_class)[y_true.squeeze().cpu().long()]
            y_pred = y_pred.topk(1, 1)[1].view(-1)
            y_pred = torch.eye(num_class)[y_pred.cpu().long()]

        self.true_positives += (y_true * y_pred).sum(dim=0)  # type: ignore
        self.actual_positives += y_true.sum(dim=0)  # type: ignore

        recall = self.true_positives / (self.actual_positives + self.eps)

        if self.average:
            return np.array(recall.mean().item())  # type:ignore
        else:
            return recall.detach().cpu().numpy()  # type: ignore[attr-defined]


class FBetaScore(Metric):
    r"""Class to calculate the fbeta score for both binary and categorical problems

    $$
    F_{\beta} = ((1 + {\beta}^2) * \frac{(precision * recall)}{({\beta}^2 * precision + recall)}
    $$

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
        beta2 = self.beta**2

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
    Calculates R-Squared, the
    [coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination>):

    $$
    R^2 = 1 - \frac{\sum_{j=1}^n(y_j - \hat{y_j})^2}{\sum_{j=1}^n(y_j - \bar{y})^2}
    $$

    where $\hat{y_j}$ is the ground truth, $y_j$ is the predicted value and
    $\bar{y}$ is the mean of the ground truth.

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


def reshape_to_2d(tensor: Tensor, n_columns: int) -> Tensor:
    if tensor.dim() == 1:
        if tensor.size(0) % n_columns != 0:
            raise ValueError(
                f"Tensor length ({tensor.size(0)}) must be divisible by n_columns ({n_columns})"
            )
        n_rows = tensor.size(0) // n_columns
        return tensor.reshape(n_rows, n_columns)
    elif tensor.dim() == 2 and tensor.size(1) == 1:
        if tensor.size(0) % n_columns != 0:
            raise ValueError(
                f"Tensor length ({tensor.size(0)}) must be divisible by n_columns ({n_columns})"
            )
        n_rows = tensor.size(0) // n_columns
        return tensor.reshape(n_rows, n_columns)
    else:
        raise ValueError(
            "Input tensor must be 1-dimensional or 2-dimensional with one column"
        )


class NDCG_at_k(Metric):
    r"""
    Normalized Discounted Cumulative Gain (NDCG) at k.

    Parameters
    ----------
    n_cols: int, default = 10
        Number of columns in the input tensors. This parameter is neccessary
        because the input tensors are reshaped to 2D tensors. n_cols is the
        number of columns in the reshaped tensor. Alias for this parameter
        are: 'n_items', 'n_items_per_query',
        'n_items_per_id', 'n_items_per_user'
    k: int, Optional, default = None
        Number of top items to consider. It must be less than or equal to n_cols.
        If is None, k will be equal to n_cols.
    eps: float, default = 1e-8
        Small value to avoid division by zero.

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.metrics import NDCG_at_k
    >>>
    >>> ndcg = NDCG_at_k(k=10)
    >>> y_pred = torch.rand(100, 5)
    >>> y_true = torch.randint(2, (100,))
    >>> score = ndcg(y_pred, y_true)
    """

    @alias(
        "n_cols", ["n_items", "n_items_per_query", "n_items_per_id", "n_items_per_user"]
    )
    def __init__(self, n_cols: int = 10, k: Optional[int] = None, eps: float = 1e-8):
        super(NDCG_at_k, self).__init__()

        if k is not None and k > n_cols:
            raise ValueError(
                f"k must be less than or equal to n_cols. Got k: {k}, n_cols: {n_cols}"
            )

        self.n_cols = n_cols
        self.k = k if k is not None else n_cols
        self.eps = eps
        self._name = f"ndcg@{k}"
        self.reset()

    def reset(self):
        self.sum_ndcg = 0.0
        self.count = 0

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> np.ndarray:
        # NDGC@k is supposed to be used when the output reflects interest
        # scores, i.e, could be used in a regression or a multiclass problem.
        # If regression y_pred will be a float tensor, if multiclass, y_pred
        # will be a float tensor with the output of a softmax activation
        # function and we need to turn it into a 1D tensor with the class.
        # Finally, for binary problems, please use BinaryNDCG_at_k
        device = y_pred.device

        if y_pred.ndim > 1 and y_pred.size(1) > 1:
            # multiclass
            y_pred = y_pred.topk(1, 1)[1]

        y_pred_2d = reshape_to_2d(y_pred, self.n_cols)
        y_true_2d = reshape_to_2d(y_true, self.n_cols)

        batch_size = y_true_2d.shape[0]

        _, top_k_indices = torch.topk(y_pred_2d, self.k, dim=1)
        top_k_relevance = y_true_2d.gather(1, top_k_indices)
        discounts = 1.0 / torch.log2(
            torch.arange(2, top_k_relevance.shape[1] + 2, device=device)
        )

        dcg = (torch.pow(2, top_k_relevance) - 1) * discounts.unsqueeze(0)
        dcg = dcg.sum(dim=1)

        sorted_relevance, _ = torch.sort(y_true_2d, dim=1, descending=True)
        ideal_relevance = sorted_relevance[:, : self.k]

        idcg = (torch.pow(2, ideal_relevance) - 1) * discounts[
            : ideal_relevance.shape[1]
        ].unsqueeze(0)

        idcg = idcg.sum(dim=1)
        ndcg = dcg / (idcg + self.eps)

        self.sum_ndcg += ndcg.sum().item()
        self.count += batch_size

        return np.array(self.sum_ndcg / max(self.count, 1))


class BinaryNDCG_at_k(Metric):
    r"""
    Normalized Discounted Cumulative Gain (NDCG) at k for binary relevance.

    Parameters
    ----------
    n_cols: int, default = 10
        Number of columns in the input tensors. This parameter is neccessary
        because the input tensors are reshaped to 2D tensors. n_cols is the
        number of columns in the reshaped tensor. Alias for this parameter
        are: 'n_items', 'n_items_per_query',
        'n_items_per_id', 'n_items_per_user'
    k: int, Optional, default = None
        Number of top items to consider. It must be less than or equal to n_cols.
        If is None, k will be equal to n_cols.
    eps: float, default = 1e-8
        Small value to avoid division by zero.

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.metrics import BinaryNDCG_at_k
    >>>
    >>> ndcg = BinaryNDCG_at_k(k=10)
    >>> y_pred = torch.randint(2, (100,))
    >>> y_true = torch.randint(2, (100,))
    >>> score = ndcg(y_pred, y_true)
    """

    @alias(
        "n_cols", ["n_items", "n_items_per_query", "n_items_per_id", "n_items_per_user"]
    )
    def __init__(self, n_cols: int = 10, k: Optional[int] = None, eps: float = 1e-8):
        super(BinaryNDCG_at_k, self).__init__()

        if k is not None and k > n_cols:
            raise ValueError(
                f"k must be less than or equal to n_cols. Got k: {k}, n_cols: {n_cols}"
            )

        self.n_cols = n_cols
        self.k = k if k is not None else n_cols
        self.eps = eps
        self._name = f"binary_ndcg@{k}"
        self.reset()

    def reset(self):
        self.sum_ndcg = 0.0
        self.count = 0

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> np.ndarray:
        device = y_pred.device

        y_pred_2d = reshape_to_2d(y_pred, self.n_cols)
        y_true_2d = reshape_to_2d(y_true, self.n_cols)

        batch_size = y_pred_2d.shape[0]

        _, top_k_indices = torch.topk(y_pred_2d, self.k, dim=1)
        top_k_mask = torch.zeros_like(y_pred_2d, dtype=torch.bool).scatter_(
            1, top_k_indices, 1
        )

        _discounts = 1.0 / torch.log2(
            torch.arange(2, self.k + 2, device=device).float()
        )
        expanded_discounts = _discounts.repeat(1, batch_size)
        discounts = torch.zeros_like(top_k_mask, dtype=torch.float)
        discounts[top_k_mask] = expanded_discounts

        dcg = (y_true_2d * top_k_mask * discounts).sum(dim=1)
        n_relevant = torch.minimum(
            y_true_2d.sum(dim=1), torch.tensor(self.k, device=device)
        ).int()
        ideal_discounts = 1.0 / torch.log2(
            torch.arange(2, y_true_2d.shape[1] + 2, device=device).float()
        )

        idcg = torch.zeros(batch_size, device=device)
        for i in range(batch_size):
            idcg[i] = ideal_discounts[: n_relevant[i]].sum()

        ndcg = dcg / (idcg + self.eps)

        self.sum_ndcg += ndcg.sum().item()
        self.count += batch_size

        return np.array(self.sum_ndcg / max(self.count, 1))


class MAP_at_k(Metric):
    r"""
    Mean Average Precision (MAP) at k.

    Parameters
    ----------
    n_cols: int, default = 10
        Number of columns in the input tensors. This parameter is neccessary
        because the input tensors are reshaped to 2D tensors. n_cols is the
        number of columns in the reshaped tensor. Alias for this parameter
        are: 'n_items', 'n_items_per_query', 'n_items_per_id', 'n_items_per_user'
    k: int, Optional, default = None
        Number of top items to consider. It must be less than or equal to n_cols.
        If is None, k will be equal to n_cols.

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.metrics import MAP_at_k
    >>>
    >>> map_at_k = MAP_at_k(k=10)
    >>> y_pred = torch.randint(2, (100,))
    >>> y_true = torch.randint(2, (100,))
    >>> score = map_at_k(y_pred, y_true)
    """

    @alias(
        "n_cols", ["n_items", "n_items_per_query", "n_items_per_id", "n_items_per_user"]
    )
    def __init__(self, n_cols: int = 10, k: Optional[int] = None):
        super(MAP_at_k, self).__init__()

        if k is not None and k > n_cols:
            raise ValueError(
                f"k must be less than or equal to n_cols. Got k: {k}, n_cols: {n_cols}"
            )

        self.n_cols = n_cols
        self.k = k if k is not None else n_cols
        self._name = f"map@{k}"
        self.reset()

    def reset(self):
        self.sum_avg_precision = 0.0
        self.count = 0

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> np.ndarray:

        y_pred_2d = reshape_to_2d(y_pred, self.n_cols)
        y_true_2d = reshape_to_2d(y_true, self.n_cols)

        batch_size = y_pred_2d.shape[0]
        _, top_k_indices = torch.topk(y_pred_2d, self.k, dim=1)
        batch_relevance = y_true_2d.gather(1, top_k_indices)
        cumsum_relevance = torch.cumsum(batch_relevance, dim=1)
        precision_at_i = cumsum_relevance / torch.arange(
            1, self.k + 1, device=y_pred_2d.device
        ).float().unsqueeze(0)
        avg_precision = (precision_at_i * batch_relevance).sum(dim=1) / torch.clamp(
            y_true_2d.sum(dim=1), min=1
        )
        self.sum_avg_precision += avg_precision.sum().item()
        self.count += batch_size
        return np.array(self.sum_avg_precision / max(self.count, 1))


class HitRatio_at_k(Metric):
    r"""
    Hit Ratio (HR) at k.

    Parameters
    ----------
    n_cols: int, default = 10
        Number of columns in the input tensors. This parameter is neccessary
        because the input tensors are reshaped to 2D tensors. n_cols is the
        number of columns in the reshaped tensor. Alias for this parameter
        are: 'n_items', 'n_items_per_query', 'n_items_per_id', 'n_items_per_user'
    k: int, Optional, default = None
        Number of top items to consider. It must be less than or equal to n_cols.
        If is None, k will be equal to n_cols.

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.metrics import HitRatio_at_k
    >>>
    >>> hr_at_k = HitRatio_at_k(k=10)
    >>> y_pred = torch.randint(2, (100,))
    >>> y_true = torch.randint(2, (100,))
    >>> score = hr_at_k(y_pred, y_true)
    """

    @alias(
        "n_cols", ["n_items", "n_items_per_query", "n_items_per_id", "n_items_per_user"]
    )
    def __init__(self, n_cols: int = 10, k: Optional[int] = None):
        super(HitRatio_at_k, self).__init__()

        if k is not None and k > n_cols:
            raise ValueError(
                f"k must be less than or equal to n_cols. Got k: {k}, n_cols: {n_cols}"
            )

        self.n_cols = n_cols
        self.k = k if k is not None else n_cols
        self._name = f"hr@{k}"
        self.reset()

    def reset(self):
        self.sum_hr = 0.0
        self.count = 0

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> np.ndarray:
        y_pred_2d = reshape_to_2d(y_pred, self.n_cols)
        y_true_2d = reshape_to_2d(y_true, self.n_cols)
        batch_size = y_pred_2d.shape[0]
        _, top_k_indices = torch.topk(y_pred_2d, self.k, dim=1)
        batch_relevance = y_true_2d.gather(1, top_k_indices)
        hit = (batch_relevance.sum(dim=1) > 0).float()
        self.sum_hr += hit.sum().item()
        self.count += batch_size
        return np.array(self.sum_hr / max(self.count, 1))


class Precision_at_k(Metric):
    r"""
    Precision at k.

    Parameters
    ----------
    n_cols: int, default = 10
        Number of columns in the input tensors. This parameter is neccessary
        because the input tensors are reshaped to 2D tensors. n_cols is the
        number of columns in the reshaped tensor. Alias for this parameter
        are: 'n_items', 'n_items_per_query',
        'n_items_per_id', 'n_items_per
    k: int, Optional, default = None
        Number of top items to consider. It must be less than or equal to n_cols.
        If is None, k will be equal to n_cols.

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.metrics import Precision_at_k
    >>>
    >>> prec_at_k = Precision_at_k(k=10)
    >>> y_pred = torch.randint(2, (100,))
    >>> y_true = torch.randint(2, (100,))
    >>> score = prec_at_k(y_pred, y_true)
    """

    @alias(
        "n_cols", ["n_items", "n_items_per_query", "n_items_per_id", "n_items_per_user"]
    )
    def __init__(self, n_cols: int = 10, k: Optional[int] = None):
        super(Precision_at_k, self).__init__()

        if k is not None and k > n_cols:
            raise ValueError(
                f"k must be less than or equal to n_cols. Got k: {k}, n_cols: {n_cols}"
            )

        self.n_cols = n_cols
        self.k = k if k is not None else n_cols
        self._name = f"precision@{k}"
        self.reset()

    def reset(self):
        self.sum_precision = 0.0
        self.count = 0

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> np.ndarray:
        y_pred_2d = reshape_to_2d(y_pred, self.n_cols)
        y_true_2d = reshape_to_2d(y_true, self.n_cols)
        batch_size = y_pred_2d.shape[0]
        _, top_k_indices = torch.topk(y_pred_2d, self.k, dim=1)
        batch_relevance = y_true_2d.gather(1, top_k_indices)
        precision = batch_relevance.sum(dim=1) / self.k
        self.sum_precision += precision.sum().item()
        self.count += batch_size
        return np.array(self.sum_precision / max(self.count, 1))


class Recall_at_k(Metric):
    r"""
    Recall at k.

    Parameters
    ----------
    n_cols: int, default = 10
        Number of columns in the input tensors. This parameter is neccessary
        because the input tensors are reshaped to 2D tensors. n_cols is the
        number of columns in the reshaped tensor. Alias for this parameter
        are: 'n_items', 'n_items_per_query',
        'n_items_per_id', 'n_items_per_user'
    k: int, default = 10
        Number of top items to consider.

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.metrics import Recall_at_k
    >>>
    >>> rec_at_k = Recall_at_k(k=10)
    >>> y_pred = torch.randint(2, (100,))
    >>> y_true = torch.randint(2, (100,))
    >>> score = rec_at_k(y_pred, y_true)
    """

    @alias(
        "n_cols", ["n_items", "n_items_per_query", "n_items_per_id", "n_items_per_user"]
    )
    def __init__(self, n_cols: int = 10, k: Optional[int] = None):
        super(Recall_at_k, self).__init__()

        if k is not None and k > n_cols:
            raise ValueError(
                f"k must be less than or equal to n_cols. Got k: {k}, n_cols: {n_cols}"
            )

        self.n_cols = n_cols
        self.k = k if k is not None else n_cols
        self._name = f"recall@{k}"
        self.reset()

    def reset(self):
        self.sum_recall = 0.0
        self.count = 0

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> np.ndarray:
        y_pred_2d = reshape_to_2d(y_pred, self.n_cols)
        y_true_2d = reshape_to_2d(y_true, self.n_cols)
        batch_size = y_pred_2d.shape[0]
        _, top_k_indices = torch.topk(y_pred_2d, self.k, dim=1)
        batch_relevance = y_true_2d.gather(1, top_k_indices)
        recall = batch_relevance.sum(dim=1) / torch.clamp(y_true_2d.sum(dim=1), min=1)
        self.sum_recall += recall.sum().item()
        self.count += batch_size
        return np.array(self.sum_recall / max(self.count, 1))


RankingMetrics = Union[
    BinaryNDCG_at_k, MAP_at_k, HitRatio_at_k, Precision_at_k, Recall_at_k
]


# tnis is just because we want to support python 3.9. In 3.10 we could define
# a type hint like Union[BinaryNDCG_at_k, MAP_at_k, HitRatio_at_k,
# Precision_at_k, Recall_at_k]
def is_ranking_metric(metric) -> bool:
    return isinstance(
        metric,
        (
            BinaryNDCG_at_k,
            MAP_at_k,
            HitRatio_at_k,
            Precision_at_k,
            Recall_at_k,
        ),
    )
