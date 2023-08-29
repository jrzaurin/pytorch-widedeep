import os
from typing import List, Tuple, Union, Optional

import numpy as np
import pandas as pd

from pytorch_widedeep.preprocessing import (
    ChunkTabPreprocessor,
    ChunkWidePreprocessor,
)


class TabFolder:
    def __init__(
        self,
        directory: str,
        target_col: str,
        preprocessor: Union[ChunkTabPreprocessor, ChunkWidePreprocessor],
        colnames: List[str],
        text_col: Optional[str] = None,
        img_col: Optional[str] = None,
    ):
        assert (
            preprocessor.is_fitted
        ), "The preprocessor must be fitted before using this class"

        self.directory = directory
        self.target_col = target_col
        self.preprocessor = preprocessor
        self.colnames = colnames
        self.text_col = text_col
        self.img_col = img_col

    def get_item(
        self, fname: str, idx: str
    ) -> Tuple[np.ndarray, str, str, Union[int, float]]:
        path = os.path.join(self.directory, fname)

        try:
            _sample = pd.read_csv(path, skiprows=lambda x: x != idx, header=None).values
            sample = pd.DataFrame(_sample, columns=self.colnames + [self.target_col])
        except Exception:
            raise ValueError("Currently only csv format is supported.")

        text_fname_or_text: str = (
            sample[self.text_col].to_list()[0] if self.text_col is not None else None
        )
        img_fname: str = (
            sample[self.img_col].to_list()[0] if self.img_col is not None else None
        )

        processed_sample = self.preprocessor.transform(sample)
        target = sample[self.target_col].to_list()[0]

        return processed_sample, text_fname_or_text, img_fname, target
