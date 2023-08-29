import os
from typing import Callable, Optional

# from pytorch_widedeep.wdtypes import Tensor
import numpy as np

from pytorch_widedeep.preprocessing.text_preprocessor import (
    ChunkTextPreprocessor,
)


class TextFolder:
    def __init__(
        self,
        directory: str,
        preprocessor: ChunkTextPreprocessor,
        loader: Optional[Callable] = None,
    ):
        assert (
            preprocessor.is_fitted
        ), "The preprocessor must be fitted before using this class"

        self.directory = directory
        self.preprocessor = preprocessor

    def get_item(
        self, fname: Optional[str] = None, text: Optional[str] = None
    ) -> np.ndarray:
        if fname is not None:
            path = os.path.join(self.directory, fname)

            with open(path, "r") as f:
                text = f.read().replace("\n", "")

        sample = self.preprocessor.transform(text)

        return sample
