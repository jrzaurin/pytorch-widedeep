import os
from typing import Any, List, Tuple, Union, Optional

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
    """
    This class is used to load tabular data from disk. The current constrains are:

    1. The only file format supported right now is csv
    2. The csv file must contain headers

    For examples, please, see the examples folder in the repo.

    Parameters
    ----------
    fname: str
        the name of the csv file
    directory: str, Optional, default = None
        the path to the directory where the csv file is located. If None,
        a `TabFromFolder` reference object must be provided
    target_col: str, Optional, default = None
        the name of the target column. If None, a `TabFromFolder` reference
        object must be provided
    preprocessor: `TabularPreprocessor`, Optional, default = None
        a fitted `TabularPreprocessor` object. If None, a `TabFromFolder`
        reference object must be provided
    text_col: str, Optional, default = None
        the name of the column with the texts themselves or the names of the
        files that contain the text dataset. If None, either there is no text
        column or a `TabFromFolder` reference object must be provided
    img_col: str, Optional, default = None
        the name of the column with the the names of the images. If None,
        either there is no image column or a `TabFromFolder` reference object
        must be provided
    ignore_target: bool, default = False
        whether to ignore the target column. This is normally set to True when
        this class is used for a test dataset.
    reference: `TabFromFolder`, Optional, default = None
        a reference `TabFromFolder` object. If provided, the `TabFromFolder`
        object will be created using the attributes of the reference object.
        This is useful to instantiate a `TabFromFolder` object for evaluation
        or test purposes
    verbose: int, default = 1
        verbosity. If 0, no output will be printed during the process.
    """

    def __init__(
        self,
        fname: str,
        directory: Optional[str] = None,
        target_col: Optional[str] = None,
        preprocessor: Optional[TabularPreprocessor] = None,
        text_col: Optional[Union[str, List[str]]] = None,
        img_col: Optional[Union[str, List[str]]] = None,
        ignore_target: bool = False,
        reference: Optional[Any] = None,  # is Type["TabFromFolder"],
        verbose: Optional[int] = 1,
    ):
        self.fname = fname
        self.ignore_target = ignore_target
        self.verbose = verbose

        if reference is not None:
            (
                self.directory,
                self.target_col,
                self.preprocessor,
                self.text_col,
                self.img_col,
            ) = self._set_from_reference(reference, preprocessor)
        else:
            assert (
                directory is not None
                and (target_col is not None and not ignore_target)
                and preprocessor is not None
            ), (
                "if no reference is provided, 'directory', 'target_col' and 'preprocessor' "
                "must be provided"
            )

            self.directory = directory
            self.target_col = target_col
            self.preprocessor = preprocessor
            self.text_col = text_col
            self.img_col = img_col

        assert (
            self.preprocessor.is_fitted
        ), "The preprocessor must be fitted before passing it to this class"

    def get_item(self, idx: int) -> Tuple[  # noqa: C901
        np.ndarray,
        Optional[Union[str, List[str]]],
        Optional[Union[str, List[str]]],
        Optional[Union[int, float]],
    ]:
        path = os.path.join(self.directory, self.fname)

        try:
            if not hasattr(self, "colnames"):
                self.colnames = pd.read_csv(path, nrows=0).columns.tolist()

            # TO DO: we need to look into this as the treatment is different
            # whether the csv contains headers or not. For the time being we
            # will require that the csv file has headers

            _sample = pd.read_csv(
                path, skiprows=lambda x: x != idx + 1, header=None
            ).values
            sample = pd.DataFrame(_sample, columns=self.colnames)
        except Exception:
            raise ValueError("Currently only csv format is supported.")

        text_fnames_or_text: Optional[Union[str, List[str]]] = None
        if self.text_col is not None:
            if isinstance(self.text_col, list):
                text_fnames_or_text = [
                    sample[col].to_list()[0] for col in self.text_col
                ]
            else:
                text_fnames_or_text = sample[self.text_col].to_list()[0]

        img_fname: Optional[Union[str, List[str]]] = None
        if self.img_col is not None:
            if isinstance(self.img_col, list):
                img_fname = [sample[col].to_list()[0] for col in self.img_col]
            else:
                img_fname = sample[self.img_col].to_list()[0]

        processed_sample = self.preprocessor.transform_sample(sample)

        if not self.ignore_target:
            target = sample[self.target_col].to_list()[0]
        else:
            target = None

        return processed_sample, text_fnames_or_text, img_fname, target

    def _set_from_reference(
        self,
        reference: Any,  # is Type["TabFromFolder"],
        preprocessor: Optional[TabularPreprocessor],
    ) -> Tuple[
        str,
        str,
        TabularPreprocessor,
        Optional[Union[str, List[str]]],
        Optional[Union[str, List[str]]],
    ]:
        (
            directory,
            target_col,
            _preprocessor,
            text_col,
            img_col,
        ) = self._get_from_reference(reference)

        if preprocessor is not None:
            preprocessor = preprocessor
            if self.verbose:
                UserWarning(
                    "The preprocessor from the reference object is overwritten "
                    "by the provided preprocessor"
                )
        else:
            preprocessor = _preprocessor

        return directory, target_col, preprocessor, text_col, img_col

    @staticmethod
    def _get_from_reference(
        reference: Any,  # is Type["TabFromFolder"],
    ) -> Tuple[str, str, TabularPreprocessor, Optional[str], Optional[str]]:
        return (
            reference.directory,
            reference.target_col,
            reference.preprocessor,
            reference.text_col,
            reference.img_col,
        )

    def __repr__(self) -> str:  # noqa: C901
        list_of_params: List[str] = []
        if self.fname is not None:
            list_of_params.append("fname={fname}")
        if self.directory is not None:
            list_of_params.append("directory={directory}")
        if self.target_col is not None:
            list_of_params.append("target_col={target_col}")
        if self.preprocessor is not None:
            list_of_params.append(
                f"preprocessor={self.preprocessor.__class__.__name__}"
            )
        if self.text_col is not None:
            if isinstance(self.text_col, list):
                list_of_params.append(
                    f"text_col={[text_col for text_col in self.text_col]}"
                )
            else:
                list_of_params.append("text_col={text_col}")
        if self.img_col is not None:
            if isinstance(self.img_col, list):
                list_of_params.append(
                    f"img_col={[img_col for img_col in self.img_col]}"
                )
            else:
                list_of_params.append("img_col={img_col}")
        if self.ignore_target is not None:
            list_of_params.append("ignore_target={ignore_target}")
        if self.verbose is not None:
            list_of_params.append("verbose={verbose}")
        all_params = ", ".join(list_of_params)
        return f"{self.__class__.__name__}({all_params.format(**self.__dict__)})"


class WideFromFolder(TabFromFolder):
    """
    This class is mostly identical to `TabFromFolder` but exists because we
    want to separate the treatment of the wide and the deep tabular
    components

    Parameters
    ----------
    fname: str
        the name of the csv file
    directory: str, Optional, default = None
        the path to the directory where the csv file is located. If None,
        a `WideFromFolder` reference object must be provided
    target_col: str, Optional, default = None
        the name of the target column. If None, a `WideFromFolder` reference
        object must be provided
    preprocessor: `TabularPreprocessor`, Optional, default = None
        a fitted `TabularPreprocessor` object. If None, a `WideFromFolder`
        reference object must be provided
    text_col: str, Optional, default = None
        the name of the column with the texts themselves or the names of the
        files that contain the text dataset. If None, either there is no text
        column or a `WideFromFolder` reference object must be provided=
    img_col: str, Optional, default = None
        the name of the column with the the names of the images. If None,
        either there is no image column or a `WideFromFolder` reference object
        must be provided
    ignore_target: bool, default = False
        whether to ignore the target column. This is normally used when this
        class is used for a test dataset.
    reference: `WideFromFolder`, Optional, default = None
        a reference `WideFromFolder` object. If provided, the `WideFromFolder`
        object will be created using the attributes of the reference object.
        This is useful to instantiate a `WideFromFolder` object for evaluation
        or test purposes
    verbose: int, default = 1
        verbosity. If 0, no output will be printed during the process.
    """

    def __init__(
        self,
        fname: str,
        directory: Optional[str] = None,
        target_col: Optional[str] = None,
        preprocessor: Optional[TabularPreprocessor] = None,
        text_col: Optional[str] = None,
        img_col: Optional[str] = None,
        ignore_target: bool = False,
        reference: Optional[Any] = None,  # is Type["WideFromFolder"],
        verbose: int = 1,
    ):
        super(WideFromFolder, self).__init__(
            fname=fname,
            directory=directory,
            target_col=target_col,
            preprocessor=preprocessor,
            text_col=text_col,
            img_col=img_col,
            reference=reference,
            ignore_target=ignore_target,
            verbose=verbose,
        )
