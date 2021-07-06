from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np


def get_class_weights(dataset):
    """Helper function to get weights of classes in the imbalanced dataset.
    Args:
        dataset (WideDeepDataset): dataset containing target classes in dataset.Y 
    Returns:
        weights (np.array): numpy array with weights
        minor_class_count (int): count of samples in the smallest class for undersampling
        num_classes (int): number of classes
    """
    weights = 1/np.unique(dataset.Y, return_counts=True)[1]
    minor_class_count = min(np.unique(dataset.Y, return_counts=True)[1])
    num_classes = len(np.unique(dataset.Y))
    return weights, minor_class_count, num_classes


class DataLoader_default(DataLoader):

    def __init__(self, dataset, batch_size, num_workers, **kwargs):
        super().__init__(dataset, batch_size, num_workers)


class DataLoader_imbalanced(DataLoader):
    """Helper function to load and shuffle tensors into models in
    batches with adjusted weights to "fight" against imbalance of the classes.
    If the classes do not begin from 0 remapping is necessary, see:
    https://towardsdatascience.com/pytorch-tabular-multiclass-classification-9f8211a123ab
    Args:
        dataset (WideDeepDataset): dataset containing target classes in dataset.Y 
        batch_size (int): size of batch
        num_workers (int): number of workers
        oversample_mul (float): multiplicator for random oversampling of minority class
    Returns:
        DataLoader: PyTorch DataLoader object
    """
    def __init__(self, dataset, batch_size, num_workers, **kwargs):
        if 'oversample_mul' in kwargs: 
            oversample_mul = kwargs['oversample_mul']
        else:
            oversample_mul = 1
        weights, minor_cls_cnt, num_clss = get_class_weights(dataset)
        num_samples = int(minor_cls_cnt * num_clss * oversample_mul)
        # weight for each sample
        samples_weight = np.array([weights[i] for i in dataset.Y])
        # draw len(dataset) samples with given weights
        sampler = WeightedRandomSampler(
            samples_weight, num_samples, replacement=True)
        # sampler option is mutually exclusive with shuffle, can't set shuffle to
        # false/true
        # setting num_worker>0 with sampler causes error "DataLoader worker (pid 1362) is killed by signal: Segmentation fault"
        # I could not find a workaround, seems its related to sampling/multiprocessing
        super().__init__(dataset, batch_size, num_workers=0, sampler=sampler)