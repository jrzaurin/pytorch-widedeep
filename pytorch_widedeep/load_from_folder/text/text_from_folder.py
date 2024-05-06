import os
from typing import Union

import numpy as np

from pytorch_widedeep.preprocessing.hf_preprocessor import (
    HFPreprocessor,
    ChunkHFPreprocessor,
)
from pytorch_widedeep.preprocessing.text_preprocessor import (
    TextPreprocessor,
    ChunkTextPreprocessor,
)


class TextFromFolder:
    """
    This class is used to load the text dataset (i.e. the text files) from a
    folder, or to retrieve the text given a texts column specified within the
    preprocessor object.

    For examples, please, see the examples folder in the repo.

    Parameters
    ----------
    preprocessor: Union[TextPreprocessor, ChunkTextPreprocessor, HFPreprocessor, ChunkHFPreprocessor]
        The preprocessor used to process the text. It must be fitted before using
        this class
    """

    def __init__(
        self,
        preprocessor: Union[
            TextPreprocessor, ChunkTextPreprocessor, HFPreprocessor, ChunkHFPreprocessor
        ],
    ):
        assert (
            preprocessor.is_fitted
        ), "The preprocessor must be fitted before using this class"

        self.preprocessor = preprocessor

    def get_item(self, text: str) -> np.ndarray:
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

    def __repr__(self):
        return f"{self.__class__.__name__}({self.preprocessor.__class__.__name__})"
