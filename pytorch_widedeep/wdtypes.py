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

# isort: off
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    try:
        from typing_extensions import Literal
    except ModuleNotFoundError:
        pass
# isort: on

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
from torch.optim.lr_scheduler import _LRScheduler
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
LRScheduler = _LRScheduler
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
