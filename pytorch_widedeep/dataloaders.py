import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.training._wd_dataset import WideDeepDataset


def get_class_weights(dataset: WideDeepDataset) -> Tuple[np.ndarray, int, int]:
    """Helper function to get weights of classes in the imbalanced dataset.

    Parameters
    ----------
    dataset: ``WideDeepDataset``
        dataset containing target classes in dataset.Y

    Returns
    ----------
    weights: array
        numpy array with weights
    minor_class_count: int
        count of samples in the smallest class for undersampling
    num_classes: int
        number of classes
    """
    weights = 1 / np.unique(dataset.Y, return_counts=True)[1]
    minor_class_count = min(np.unique(dataset.Y, return_counts=True)[1])
    num_classes = len(np.unique(dataset.Y))
    return weights, minor_class_count, num_classes


class DataLoaderDefault(DataLoader):
    def __init__(self, dataset, batch_size, num_workers, **kwargs):
        super().__init__(dataset, batch_size, num_workers)


class DataLoaderImbalanced(DataLoader):
    r"""Class to load and shuffle batches with adjusted weights for imbalanced
    datasets. If the classes do not begin from 0 remapping is necessary. See
    `here <https://towardsdatascience.com/pytorch-tabular-multiclass-classification-9f8211a123ab>`_

    Parameters
    ----------
    dataset: ``WideDeepDataset``
        see ``pytorch_widedeep.training._wd_dataset``
    batch_size: int
        size of batch
    num_workers: int
        number of workers
    """

    def __init__(
        self, dataset: WideDeepDataset, batch_size: int, num_workers: int, **kwargs
    ):
        if "oversample_mul" in kwargs:
            oversample_mul = kwargs["oversample_mul"]
        else:
            oversample_mul = 1
        weights, minor_cls_cnt, num_clss = get_class_weights(dataset)
        num_samples = int(minor_cls_cnt * num_clss * oversample_mul)
        samples_weight = list(np.array([weights[i] for i in dataset.Y]))
        sampler = WeightedRandomSampler(samples_weight, num_samples, replacement=True)
        super().__init__(dataset, batch_size, num_workers=num_workers, sampler=sampler)
