from typing import Tuple

import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

from pytorch_widedeep.training._wd_dataset import WideDeepDataset


def get_class_weights(dataset: WideDeepDataset) -> Tuple[np.ndarray, int, int]:
    """Helper function to get weights of classes in the imbalanced dataset.

    Parameters
    ----------
    dataset: `WideDeepDataset`
        dataset containing target classes in dataset.Y

    Returns
    ----------
    weights: array
        numpy array with weights
    minor_class_count: int
        count of samples in the smallest class for undersampling
    num_classes: int
        number of classes

    Other Parameters
    ----------------
    **kwargs: Dict
        This can include any parameter that can be passed to the _'standard'_
        pytorch[DataLoader]
        (https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
        and that is not already explicitely passed to the class.
    """
    weights = 1 / np.unique(dataset.Y, return_counts=True)[1]
    minor_class_count = min(np.unique(dataset.Y, return_counts=True)[1])
    num_classes = len(np.unique(dataset.Y))
    return weights, minor_class_count, num_classes


class DataLoaderDefault(DataLoader):
    def __init__(
        self, dataset: WideDeepDataset, batch_size: int, num_workers: int, **kwargs
    ):
        self.with_lds = dataset.with_lds
        super().__init__(
            dataset=dataset, batch_size=batch_size, num_workers=num_workers, **kwargs
        )


class DataLoaderImbalanced(DataLoader):
    r"""Class to load and shuffle batches with adjusted weights for imbalanced
    datasets. If the classes do not begin from 0 remapping is necessary. See
    [here](https://towardsdatascience.com/pytorch-tabular-multiclass-classification-9f8211a123ab).

    Parameters
    ----------
    dataset: `WideDeepDataset`
        see `pytorch_widedeep.training._wd_dataset`
    batch_size: int
        size of batch
    num_workers: int
        number of workers

    Other Parameters
    ----------------
    **kwargs: Dict
        This can include any parameter that can be passed to the _'standard'_
        pytorch
        [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
        and that is not already explicitely passed to the class. In addition,
        the dictionary can also include the extra parameter `oversample_mul` which
        will multiply the number of samples of the minority class to be sampled by
        the [`WeightedRandomSampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler).

        In other words, the `num_samples` param in `WeightedRandomSampler` will be defined as:

        $$
        minority \space class \space count \times number \space of \space classes \times oversample\_mul
        $$
    """

    def __init__(
        self, dataset: WideDeepDataset, batch_size: int, num_workers: int, **kwargs
    ):
        assert dataset.Y is not None, (
            "The 'dataset' instance of WideDeepDataset must contain a "
            "target array 'Y'"
        )

        self.with_lds = dataset.with_lds
        if "oversample_mul" in kwargs:
            oversample_mul = kwargs["oversample_mul"]
            del kwargs["oversample_mul"]
        else:
            oversample_mul = 1
        weights, minor_cls_cnt, num_clss = get_class_weights(dataset)
        num_samples = int(minor_cls_cnt * num_clss * oversample_mul)
        samples_weight = list(np.array([weights[i] for i in dataset.Y]))
        sampler = WeightedRandomSampler(samples_weight, num_samples, replacement=True)
        super().__init__(
            dataset, batch_size, num_workers=num_workers, sampler=sampler, **kwargs
        )
