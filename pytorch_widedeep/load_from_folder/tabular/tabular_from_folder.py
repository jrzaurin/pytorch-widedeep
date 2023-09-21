import os
from typing import Tuple, Union, Optional

import numpy as np
import pandas as pd

from pytorch_widedeep.preprocessing import (
    TabPreprocessor,
    WidePreprocessor,
    ChunkTabPreprocessor,
    ChunkWidePreprocessor,
)

TabularPreprocessor = Union[
    TabPreprocessor, WidePreprocessor, ChunkTabPreprocessor, ChunkWidePreprocessor
]


class TabFromFolder:
    def __init__(
        self,
        directory: str,
        fname: str,
        target_col: str,
        preprocessor: TabularPreprocessor,
        text_col: Optional[str] = None,
        img_col: Optional[str] = None,
    ):
        assert (
            preprocessor.is_fitted
        ), "The preprocessor must be fitted before using this class"

        self.directory = directory
        self.fname = fname
        self.target_col = target_col
        self.preprocessor = preprocessor
        self.text_col = text_col
        self.img_col = img_col

    def get_item(self, idx: int) -> Tuple[np.ndarray, str, str, Union[int, float]]:
        path = os.path.join(self.directory, self.fname)

        try:
            if not hasattr(self, "colnames"):
                self.colnames = pd.read_csv(path, nrows=0).columns.tolist()

            # TO DO: we need to look into this as the treatment is different whether
            # the csv contains headers or not

            _sample = pd.read_csv(
                path, skiprows=lambda x: x != idx + 1, header=None
            ).values
            sample = pd.DataFrame(_sample, columns=self.colnames)
        except Exception:
            raise ValueError("Currently only csv format is supported.")

        text_fname_or_text: str = (
            sample[self.text_col].to_list()[0] if self.text_col is not None else None
        )
        img_fname: str = (
            sample[self.img_col].to_list()[0] if self.img_col is not None else None
        )

        processed_sample = self.preprocessor.transform_sample(sample)
        target = sample[self.target_col].to_list()[0]

        return processed_sample, text_fname_or_text, img_fname, target
