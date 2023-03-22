import sys
from types import SimpleNamespace
from typing import (
    Any,
    Dict,
    List,
    Match,
    Tuple,
    Union,
    Literal,
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
from torchvision.models._api import WeightsEnum
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data.dataloader import DataLoader

from pytorch_widedeep.models import (
    SAINT,
    TabMlp,
    TabNet,
    WideDeep,
    TabResnet,
    TabPerceiver,
    FTTransformer,
    TabFastFormer,
    TabMlpDecoder,
    TabNetDecoder,
    TabTransformer,
    SelfAttentionMLP,
    TabResnetDecoder,
    ContextAttentionMLP,
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
ModelParams = Generator[Tensor, Tensor, Tensor]

ModelWithoutAttention = Union[
    TabMlp,
    TabNet,
    TabResnet,
]

ModelWithAttention = Union[
    TabTransformer,
    SAINT,
    FTTransformer,
    TabFastFormer,
    TabPerceiver,
    ContextAttentionMLP,
    SelfAttentionMLP,
]

DecoderWithoutAttention = Union[
    TabMlpDecoder,
    TabNetDecoder,
    TabResnetDecoder,
]
