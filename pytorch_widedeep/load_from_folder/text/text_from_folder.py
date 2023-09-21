import os
from typing import Union, Optional

import numpy as np

from pytorch_widedeep.preprocessing.text_preprocessor import (
    TextPreprocessor,
    ChunkTextPreprocessor,
)


class TextFromFolder:
    def __init__(
        self,
        preprocessor: Union[TextPreprocessor, ChunkTextPreprocessor],
    ):
        assert (
            preprocessor.is_fitted
        ), "The preprocessor must be fitted before using this class"

        self.preprocessor = preprocessor

    def get_item(self, text: Optional[str] = None) -> np.ndarray:
        if (
            isinstance(self.preprocessor, ChunkTextPreprocessor)
            and self.preprocessor.root_dir is not None
        ):
            path = os.path.join(self.preprocessor.root_dir, text)

            with open(path, "r") as f:
                sample = f.read().replace("\n", "")
        else:
            sample = text

        processed_sample = self.preprocessor.transform_sample(sample)

        return processed_sample
