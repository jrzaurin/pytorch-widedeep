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

# isort: off
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    try:
        from typing_extensions import Literal
    except ModuleNotFoundError:
        pass
# isort: on

from pathlib import PosixPath

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torchvision.transforms import (
    Pad,
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
    PILToTensor,
    RandomApply,
    RandomOrder,
    GaussianBlur,
    RandomAffine,
    RandomChoice,
    RandomInvert,
    RandomErasing,
    RandomEqualize,
    RandomRotation,
    RandomSolarize,
    RandomGrayscale,
    RandomPosterize,
    ConvertImageDtype,
    InterpolationMode,
    RandomPerspective,
    RandomResizedCrop,
    RandomAutocontrast,
    RandomVerticalFlip,
    LinearTransformation,
    RandomHorizontalFlip,
    RandomAdjustSharpness,
)
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data.dataloader import DataLoader

from pytorch_widedeep.models import WideDeep
from pytorch_widedeep.models.tabular.tabnet.sparsemax import (
    Entmax15,
    Sparsemax,
)
from pytorch_widedeep.bayesian_models._base_bayesian_model import (
    BaseBayesianModel,
)

ListRules = Collection[Callable[[str], str]]
Tokens = Collection[Collection[str]]
Transforms = Union[
    Pad,
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
    PILToTensor,
    RandomApply,
    RandomOrder,
    GaussianBlur,
    RandomAffine,
    RandomChoice,
    RandomInvert,
    RandomErasing,
    RandomEqualize,
    RandomRotation,
    RandomSolarize,
    RandomGrayscale,
    RandomPosterize,
    ConvertImageDtype,
    InterpolationMode,
    RandomPerspective,
    RandomResizedCrop,
    RandomAutocontrast,
    RandomVerticalFlip,
    LinearTransformation,
    RandomHorizontalFlip,
    RandomAdjustSharpness,
]
LRScheduler = _LRScheduler
ModelParams = Generator[Tensor, Tensor, Tensor]
NormLayers = Union[torch.nn.Identity, torch.nn.LayerNorm, torch.nn.BatchNorm1d]
