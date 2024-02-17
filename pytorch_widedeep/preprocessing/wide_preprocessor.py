import sys
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from pytorch_widedeep.preprocessing.base_preprocessor import (
    BasePreprocessor,
    check_is_fitted,
)


class WidePreprocessor(BasePreprocessor):
    r"""Preprocessor to prepare the wide input dataset

    This Preprocessor prepares the data for the wide, linear component.
    This linear model is implemented via an Embedding layer that is
    connected to the output neuron. `WidePreprocessor` numerically
    encodes all the unique values of all categorical columns `wide_cols +
    crossed_cols`. See the Example below.

    Parameters
    ----------
    wide_cols: List
        List of strings with the name of the columns that will label
        encoded and passed through the `wide` component
    crossed_cols: List, default = None
        List of Tuples with the name of the columns that will be `'crossed'`
        and then label encoded. e.g. _[('education', 'occupation'), ...]_. For
        binary features, a cross-product transformation is 1 if and only if
        the constituent features are all 1, and 0 otherwise.

    Attributes
    ----------
    wide_crossed_cols: List
        List with the names of all columns that will be label encoded
    encoding_dict: Dict
        Dictionary where the keys are the result of pasting `colname + '_' +
        column value` and the values are the corresponding mapped integer.
    inverse_encoding_dict: Dict
        the inverse encoding dictionary
    wide_dim: int
        Dimension of the wide model (i.e. dim of the linear layer)

    Examples
    --------
    >>> import pandas as pd
    >>> from pytorch_widedeep.preprocessing import WidePreprocessor
    >>> df = pd.DataFrame({'color': ['r', 'b', 'g'], 'size': ['s', 'n', 'l']})
    >>> wide_cols = ['color']
    >>> crossed_cols = [('color', 'size')]
    >>> wide_preprocessor = WidePreprocessor(wide_cols=wide_cols, crossed_cols=crossed_cols)
    >>> X_wide = wide_preprocessor.fit_transform(df)
    >>> X_wide
    array([[1, 4],
           [2, 5],
           [3, 6]])
    >>> wide_preprocessor.encoding_dict
    {'color_r': 1, 'color_b': 2, 'color_g': 3, 'color_size_r-s': 4, 'color_size_b-n': 5, 'color_size_g-l': 6}
    >>> wide_preprocessor.inverse_transform(X_wide)
      color color_size
    0     r        r-s
    1     b        b-n
    2     g        g-l
    """

    def __init__(
        self, wide_cols: List[str], crossed_cols: Optional[List[Tuple[str, str]]] = None
    ):
        super(WidePreprocessor, self).__init__()

        self.wide_cols = wide_cols
        self.crossed_cols = crossed_cols

        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "WidePreprocessor":
        r"""Fits the Preprocessor and creates required attributes

        Parameters
        ----------
        df: pd.DataFrame
            Input pandas dataframe

        Returns
        -------
        WidePreprocessor
            `WidePreprocessor` fitted object
        """
        df_wide = self._prepare_wide(df)
        self.wide_crossed_cols = df_wide.columns.tolist()
        glob_feature_list = self._make_global_feature_list(
            df_wide[self.wide_crossed_cols]
        )
        # leave 0 for padding/"unseen" categories
        self.encoding_dict = {v: i + 1 for i, v in enumerate(glob_feature_list)}
        self.wide_dim = len(self.encoding_dict)
        self.inverse_encoding_dict = {k: v for v, k in self.encoding_dict.items()}
        self.inverse_encoding_dict[0] = "unseen"

        self.is_fitted = True

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        r"""
        Parameters
        ----------
        df: pd.DataFrame
            Input pandas dataframe

        Returns
        -------
        np.ndarray
            transformed input dataframe
        """
        check_is_fitted(self, attributes=["encoding_dict"])
        df_wide = self._prepare_wide(df)
        encoded = np.zeros([len(df_wide), len(self.wide_crossed_cols)])
        for col_i, col in enumerate(self.wide_crossed_cols):
            encoded[:, col_i] = df_wide[col].apply(
                lambda x: (
                    self.encoding_dict[col + "_" + str(x)]
                    if col + "_" + str(x) in self.encoding_dict
                    else 0
                )
            )
        return encoded.astype("int64")

    def transform_sample(self, df: pd.DataFrame) -> np.ndarray:
        return self.transform(df)[0]

    def inverse_transform(self, encoded: np.ndarray) -> pd.DataFrame:
        r"""Takes as input the output from the `transform` method and it will
        return the original values.

        Parameters
        ----------
        encoded: np.ndarray
            numpy array with the encoded values that are the output from the
            `transform` method

        Returns
        -------
        pd.DataFrame
            Pandas dataframe with the original values
        """
        decoded = pd.DataFrame(encoded, columns=self.wide_crossed_cols)

        if pd.__version__ >= "2.1.0":
            decoded = decoded.map(lambda x: self.inverse_encoding_dict[x])
        else:
            decoded = decoded.applymap(lambda x: self.inverse_encoding_dict[x])

        for col in decoded.columns:
            rm_str = "".join([col, "_"])
            decoded[col] = decoded[col].apply(lambda x: x.replace(rm_str, ""))
        return decoded

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Combines `fit` and `transform`

        Parameters
        ----------
        df: pd.DataFrame
            Input pandas dataframe

        Returns
        -------
        np.ndarray
            transformed input dataframe
        """
        return self.fit(df).transform(df)

    def _make_global_feature_list(self, df: pd.DataFrame) -> List:
        glob_feature_list = []
        for column in df.columns:
            glob_feature_list += self._make_column_feature_list(df[column])
        return glob_feature_list

    def _make_column_feature_list(self, s: pd.Series) -> List:
        return [s.name + "_" + str(x) for x in s.unique()]

    def _cross_cols(self, df: pd.DataFrame):
        df_cc = df.copy()
        crossed_colnames = []
        for cols in self.crossed_cols:
            for c in cols:
                df_cc[c] = df_cc[c].astype("str")
            colname = "_".join(cols)
            df_cc[colname] = df_cc[list(cols)].apply(lambda x: "-".join(x), axis=1)
            crossed_colnames.append(colname)
        return df_cc[crossed_colnames]

    def _prepare_wide(self, df: pd.DataFrame):
        if self.crossed_cols is not None:
            df_cc = self._cross_cols(df)
            return pd.concat([df[self.wide_cols], df_cc], axis=1)
        else:
            return df.copy()[self.wide_cols]

    def __repr__(self) -> str:
        list_of_params: List[str] = ["wide_cols={wide_cols}"]
        if self.crossed_cols is not None:
            list_of_params.append("crossed_cols={crossed_cols}")
        all_params = ", ".join(list_of_params)
        return f"WidePreprocessor({all_params.format(**self.__dict__)})"


class ChunkWidePreprocessor(WidePreprocessor):
    r"""Preprocessor to prepare the wide input dataset

    This Preprocessor prepares the data for the wide, linear component.
    This linear model is implemented via an Embedding layer that is
    connected to the output neuron. `ChunkWidePreprocessor` numerically
    encodes all the unique values of all categorical columns `wide_cols +
    crossed_cols`. See the Example below.

    Parameters
    ----------
    wide_cols: List
        List of strings with the name of the columns that will label
        encoded and passed through the `wide` component
    crossed_cols: List, default = None
        List of Tuples with the name of the columns that will be `'crossed'`
        and then label encoded. e.g. _[('education', 'occupation'), ...]_. For
        binary features, a cross-product transformation is 1 if and only if
        the constituent features are all 1, and 0 otherwise.

    Attributes
    ----------
    wide_crossed_cols: List
        List with the names of all columns that will be label encoded
    encoding_dict: Dict
        Dictionary where the keys are the result of pasting `colname + '_' +
        column value` and the values are the corresponding mapped integer.
    inverse_encoding_dict: Dict
        the inverse encoding dictionary
    wide_dim: int
        Dimension of the wide model (i.e. dim of the linear layer)

    Examples
    --------
    >>> import pandas as pd
    >>> from pytorch_widedeep.preprocessing import ChunkWidePreprocessor
    >>> chunk = pd.DataFrame({'color': ['r', 'b', 'g'], 'size': ['s', 'n', 'l']})
    >>> wide_cols = ['color']
    >>> crossed_cols = [('color', 'size')]
    >>> chunk_wide_preprocessor = ChunkWidePreprocessor(wide_cols=wide_cols, crossed_cols=crossed_cols,
    ... n_chunks=1)
    >>> X_wide = chunk_wide_preprocessor.fit_transform(chunk)
    """

    def __init__(
        self,
        wide_cols: List[str],
        n_chunks: int,
        crossed_cols: Optional[List[Tuple[str, str]]] = None,
    ):
        super(ChunkWidePreprocessor, self).__init__(wide_cols, crossed_cols)

        self.n_chunks = n_chunks

        self.chunk_counter = 0

        self.is_fitted = False

    def partial_fit(self, chunk: pd.DataFrame) -> "ChunkWidePreprocessor":
        r"""Fits the Preprocessor and creates required attributes

        Parameters
        ----------
        chunk: pd.DataFrame
            Input pandas dataframe

        Returns
        -------
        ChunkWidePreprocessor
            `ChunkWidePreprocessor` fitted object
        """
        df_wide = self._prepare_wide(chunk)
        self.wide_crossed_cols = df_wide.columns.tolist()

        if self.chunk_counter == 0:
            self.glob_feature_set = set(
                self._make_global_feature_list(df_wide[self.wide_crossed_cols])
            )
        else:
            self.glob_feature_set.update(
                self._make_global_feature_list(df_wide[self.wide_crossed_cols])
            )

        self.chunk_counter += 1

        if self.chunk_counter == self.n_chunks:
            self.encoding_dict = {v: i + 1 for i, v in enumerate(self.glob_feature_set)}
            self.wide_dim = len(self.encoding_dict)
            self.inverse_encoding_dict = {k: v for v, k in self.encoding_dict.items()}
            self.inverse_encoding_dict[0] = "unseen"

            self.is_fitted = True

        return self

    def fit(self, df: pd.DataFrame) -> "ChunkWidePreprocessor":
        """
        Runs `partial_fit`. This is just to override the fit method in the base
        class. This class is not designed or thought to run fit
        """
        return self.partial_fit(df)

    def __repr__(self) -> str:
        list_of_params: List[str] = ["wide_cols={wide_cols}"]
        list_of_params.append("n_chunks={n_chunks}")
        if self.crossed_cols is not None:
            list_of_params.append("crossed_cols={crossed_cols}")
        all_params = ", ".join(list_of_params)
        return f"WidePreprocessor({all_params.format(**self.__dict__)})"
