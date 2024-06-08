import os
from typing import List, Union

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
            TextPreprocessor,
            ChunkTextPreprocessor,
            HFPreprocessor,
            ChunkHFPreprocessor,
            List[TextPreprocessor],
            List[ChunkTextPreprocessor],
            List[HFPreprocessor],
            List[ChunkHFPreprocessor],
        ],
    ):
        if isinstance(preprocessor, list):
            for p in preprocessor:
                assert (
                    p.is_fitted
                ), "All preprocessors must be fitted before using this class"
        else:
            assert (
                preprocessor.is_fitted
            ), "The preprocessor must be fitted before using this class"

        self.preprocessor = preprocessor

    def get_item(
        self, text: Union[str, List[str]]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        if isinstance(self.preprocessor, list):
            assert isinstance(text, list)
            processed_sample: Union[np.ndarray, List[np.ndarray]] = [
                self._preprocess_one_sample(t, self.preprocessor[i])
                for i, t in enumerate(text)
            ]
        else:
            assert isinstance(text, str)
            processed_sample = self._preprocess_one_sample(text, self.preprocessor)

        return processed_sample

    def _preprocess_one_sample(
        self,
        text: str,
        preprocessor: Union[
            TextPreprocessor,
            ChunkTextPreprocessor,
            HFPreprocessor,
            ChunkHFPreprocessor,
        ],
    ) -> np.ndarray:
        if (
            isinstance(preprocessor, ChunkTextPreprocessor)
            and preprocessor.root_dir is not None
        ):
            path = os.path.join(preprocessor.root_dir, text)

            with open(path, "r") as f:
                sample = f.read().replace("\n", "")
        else:
            sample = text

        processed_sample = preprocessor.transform_sample(sample)

        return processed_sample

    def __repr__(self):
        if isinstance(self.preprocessor, list):
            return f"{self.__class__.__name__}({[p.__class__.__name__ for p in self.preprocessor]})"
        else:
            return f"{self.__class__.__name__}({self.preprocessor.__class__.__name__})"
