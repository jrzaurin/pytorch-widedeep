import os
from typing import Optional

from pytorch_widedeep.wdtypes import Tensor


class BatchTextPreprocessor:
    def fit(self) -> "BatchTextPreprocessor":
        pass

    def transform(self) -> Tensor:
        pass


class TextFolder:
    def __init__(
        self, directory: str, text_preprocessor: BatchTextPreprocessor
    ) -> None:
        self.directory = directory
        self.text_preprocessor = text_preprocessor

    def get_item(self, fname: Optional[str] = None, text: Optional[str] = None):
        if fname is not None:
            path = os.path.join(self.directory, fname)

            with open(path, "r") as f:
                sample = f.read().replace("\n", "")
        elif text is not None:
            sample = self.text_preprocessor.transform(text)

        return sample
