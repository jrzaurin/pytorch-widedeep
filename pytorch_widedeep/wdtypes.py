import sys

from torch.nn import Module
from torch import Tensor
from torchvision.transforms import (
    CenterCrop,
    ColorJitter,
    Compose,
    FiveCrop,
    Grayscale,
    Lambda,
    LinearTransformation,
    Normalize,
    Pad,
    RandomAffine,
    RandomApply,
    RandomChoice,
    RandomCrop,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomOrder,
    RandomResizedCrop,
    RandomRotation,
    RandomSizedCrop,
    RandomVerticalFlip,
    Resize,
    Scale,
    TenCrop,
    ToPILImage,
    ToTensor,
)
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from pathlib import PosixPath
from typing import (
    List,
    Any,
    Union,
    Dict,
    Callable,
    Optional,
    Tuple,
    Generator,
    Collection,
    Iterable,
    Match,
    Iterator,
)
from scipy.sparse.csr import csr_matrix as sparse_matrix
from types import SimpleNamespace


ListRules = Collection[Callable[[str], str]]
Tokens = Collection[Collection[str]]
Transforms = Union[
    CenterCrop,
    ColorJitter,
    Compose,
    FiveCrop,
    Grayscale,
    Lambda,
    LinearTransformation,
    Normalize,
    Pad,
    RandomAffine,
    RandomApply,
    RandomChoice,
    RandomCrop,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomOrder,
    RandomResizedCrop,
    RandomRotation,
    RandomSizedCrop,
    RandomVerticalFlip,
    Resize,
    Scale,
    TenCrop,
    ToPILImage,
    ToTensor,
]
LRScheduler = _LRScheduler
ModelParams = Generator[Tensor, Tensor, Tensor]
