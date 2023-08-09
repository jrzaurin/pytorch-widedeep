import os
from typing import Optional

import pandas as pd

from pytorch_widedeep.wdtypes import List, Literal
from pytorch_widedeep.preprocessing.tab_preprocessor import (
    ChunkTabPreprocessor,
)

TAB_EXTENSIONS = (
    ".parquet",
    ".csv",
)


class TabFolder:
    def __init__(
        self,
        directory: str,
        tab_preprocessor: ChunkTabPreprocessor,
        data_format: Literal[
            "csv",
            "parquet",
        ],
        colnames: List[str],
        text_col: Optional[str],
        img_col: Optional[str],
    ):
        self.directory = directory
        self.tab_preprocessor = tab_preprocessor
        self.data_format = data_format
        self.colnames = colnames
        self.text_col = text_col
        self.img_col = img_col

    def get_item(self, fname: str, idx: str):
        assert fname.lower().endswith(TAB_EXTENSIONS)

        path = os.path.join(self.directory, fname)

        if self.data_format == "csv":
            _sample = pd.read_csv(path, skiprows=lambda x: x != idx)
            sample = pd.DataFrame([list(_sample)], columns=self.colnames)

        if self.data_format == "parquet":
            pass

        text_fname_or_text = sample[self.text_col].to_list()[0]
        img_fname = sample[self.img_col].to_list()[0]

        sample = self.tab_preprocessor.transform(sample)

        return sample, text_fname_or_text, img_fname
