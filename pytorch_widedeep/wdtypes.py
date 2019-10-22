import sys
import scipy

from torch.nn import Module
from torch import Tensor
from torchvision.transforms import *
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from pathlib import PosixPath
from typing import (List, Any, Union, Dict, Callable, Optional, Tuple,
	Generator, Collection, Iterable)


sparse_matrix = Union[scipy.sparse.csr.csr_matrix]

SimpleNamespace = type(sys.implementation)
ListRules = Collection[Callable[[str],str]]
Tokens = Collection[Collection[str]]

Transforms= Union[CenterCrop, ColorJitter, Compose, FiveCrop, Grayscale,
	Lambda, LinearTransformation, Normalize, Pad, RandomAffine,
	RandomApply, RandomChoice, RandomCrop, RandomGrayscale,
	RandomHorizontalFlip, RandomOrder, RandomResizedCrop, RandomRotation,
	RandomSizedCrop, RandomVerticalFlip, Resize, Scale, TenCrop,
	ToPILImage, ToTensor]

LRScheduler = _LRScheduler
ModelParams = Generator[Tensor,Tensor,Tensor]
