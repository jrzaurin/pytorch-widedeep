import numpy as np
import pandas as pd

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.preprocessing.base_preprocessor import (
    BasePreprocessor,
    check_is_fitted,
)


class WidePreprocessor(BasePreprocessor):
    r"""Preprocessor to prepare the wide input dataset

    This Preprocessor prepares the data for the wide, linear component.
    This linear model is implemented via an Embedding layer that is
    connected to the output neuron. ``WidePreprocessor`` numerically
    encodes all the unique values of all categorical columns ``wide_cols +
    crossed_cols``. See the Example below.

    Parameters
    ----------
    wide_cols: List
        List of strings with the name of the columns that will label
        encoded and passed through the ``wide`` component
    crossed_cols: List, default = None
        List of Tuples with the name of the columns that will be `'crossed'`
        and then label encoded. e.g. [('education', 'occupation'), ...]. For
        binary features, a cross-product transformation is 1 if and only if
        the constituent features are all 1, and 0 otherwise".

    Attributes
    ----------
    wide_crossed_cols: List
        List with the names of all columns that will be label encoded
    encoding_dict: Dict
        Dictionary where the keys are the result of pasting `colname + '_' +
        column value` and the values are the corresponding mapped integer.
    wide_dim: int
        Dimension of the wide model (i.e. dim of the linear layer)

    Example
    -------
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
        self, wide_cols: List[str], crossed_cols: List[Tuple[str, str]] = None
    ):
        super(WidePreprocessor, self).__init__()

        self.wide_cols = wide_cols
        self.crossed_cols = crossed_cols

    def fit(self, df: pd.DataFrame) -> BasePreprocessor:
        r"""Fits the Preprocessor and creates required attributes"""
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
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        r"""Returns the processed dataframe"""
        check_is_fitted(self, attributes=["encoding_dict"])
        df_wide = self._prepare_wide(df)
        encoded = np.zeros([len(df_wide), len(self.wide_crossed_cols)])
        for col_i, col in enumerate(self.wide_crossed_cols):
            encoded[:, col_i] = df_wide[col].apply(
                lambda x: self.encoding_dict[col + "_" + str(x)]
                if col + "_" + str(x) in self.encoding_dict
                else 0
            )
        return encoded.astype("int64")

    def inverse_transform(self, encoded: np.ndarray) -> pd.DataFrame:
        r"""Takes as input the output from the ``transform`` method and it will
        return the original values.
        """
        decoded = pd.DataFrame(encoded, columns=self.wide_crossed_cols)
        decoded = decoded.applymap(lambda x: self.inverse_encoding_dict[x])
        for col in decoded.columns:
            rm_str = "".join([col, "_"])
            decoded[col] = decoded[col].apply(lambda x: x.replace(rm_str, ""))
        return decoded

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Combines ``fit`` and ``transform``"""
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
