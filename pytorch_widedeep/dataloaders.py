from typing import Tuple, Optional

import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

from pytorch_widedeep.training._wd_dataset import WideDeepDataset


class DatasetAlreadySetError(Exception):
    """Exception raised when attempting to set a dataset that has already been set."""

    def __init__(self, message="Dataset has already been set and cannot be changed."):
        self.message = message
        super().__init__(self.message)


class CustomDataLoader(DataLoader):
    r"""
    Wrapper around the `torch.utils.data.DataLoader` class that allows
    to set the dataset after the class has been instantiated.
    """

    def __init__(self, dataset: Optional[WideDeepDataset] = None, *args, **kwargs):
        self.dataset_set = dataset is not None
        if self.dataset_set:
            super().__init__(dataset, *args, **kwargs)
        else:
            self.args = args
            self.kwargs = kwargs

    def set_dataset(self, dataset: WideDeepDataset):
        if self.dataset_set:
            raise DatasetAlreadySetError()

        self.dataset_set = True
        super().__init__(dataset, *self.args, **self.kwargs)

    def __iter__(self):
        if not self.dataset_set:
            raise ValueError(
                "Dataset has not been set. Use set_dataset method to set a dataset."
            )
        return super().__iter__()


# From here on is legacy code and there are better ways to do it. It will be
# removed in the next version.
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


class DataLoaderImbalanced(CustomDataLoader):
    r"""Class to load and shuffle batches with adjusted weights for imbalanced
    datasets. If the classes do not begin from 0 remapping is necessary. See
    [here](https://towardsdatascience.com/pytorch-tabular-multiclass-classification-9f8211a123ab).

    Parameters
    ----------
    dataset: `WideDeepDataset`
        see `pytorch_widedeep.training._wd_dataset`

    Other Parameters
    ----------------
    *args: Any
        Positional arguments to be passed to the parent CustomDataLoader.
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

    def __init__(self, dataset: Optional[WideDeepDataset] = None, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

        if dataset is not None:
            self._setup_sampler(dataset)
            super().__init__(dataset, *args, sampler=self.sampler, **kwargs)
        else:
            super().__init__()

    def set_dataset(self, dataset: WideDeepDataset):
        sampler = self._setup_sampler(dataset)
        # update the kwargs with the new sampler
        self.kwargs["sampler"] = sampler
        super().set_dataset(dataset)

    def _setup_sampler(self, dataset: WideDeepDataset) -> WeightedRandomSampler:
        assert dataset.Y is not None, (
            "The 'dataset' instance of WideDeepDataset must contain a "
            "target array 'Y'"
        )

        oversample_mul = self.kwargs.pop("oversample_mul", 1)
        weights, minor_cls_cnt, num_clss = get_class_weights(dataset)
        num_samples = int(minor_cls_cnt * num_clss * oversample_mul)
        samples_weight = list(np.array([weights[i] for i in dataset.Y]))
        sampler = WeightedRandomSampler(samples_weight, num_samples, replacement=True)
        return sampler
