from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from pathlib import PosixPath
from typing import List, Any, Union, Dict, Optional, Tuple, Generator, Collection, Iterable

LRScheduler = _LRScheduler
ModelParams = Generator[Tensor,Tensor,Tensor]
