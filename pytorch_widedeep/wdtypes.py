import sys
from types import SimpleNamespace
from typing import (
    Any,
    Dict,
    List,
    Match,
    Tuple,
    Union,
    Callable,
    Iterable,
    Iterator,
    Optional,
    Generator,
    Collection,
)
from pathlib import PosixPath

from torch import Tensor
from torch.nn import Module
from scipy.sparse.csr import csr_matrix as sparse_matrix
from torch.optim.optimizer import Optimizer
from torchvision.transforms import (
    Pad,
    Scale,
    Lambda,
    Resize,
    Compose,
    TenCrop,
    FiveCrop,
    ToTensor,
    Grayscale,
    Normalize,
    CenterCrop,
    RandomCrop,
    ToPILImage,
    ColorJitter,
    RandomApply,
    RandomOrder,
    RandomAffine,
    RandomChoice,
    RandomRotation,
    RandomGrayscale,
    RandomSizedCrop,
    RandomResizedCrop,
    RandomVerticalFlip,
    LinearTransformation,
    RandomHorizontalFlip,
)
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data.dataloader import DataLoader

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
