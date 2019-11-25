from typing import Union

from .wide import Wide
from .deep_dense import DeepDense
from .deep_text import DeepText
from .deep_image import DeepImage

WDModel = Union[Wide, DeepDense, DeepText, DeepImage]
