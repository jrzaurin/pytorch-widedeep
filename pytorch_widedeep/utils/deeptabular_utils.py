import warnings

import numpy as np
import torch
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.exceptions import NotFittedError
from scipy.signal.windows import triang

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.utils.general_utils import Alias

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

__all__ = ["LabelEncoder", "find_bin", "get_kernel_window"]


class LabelEncoder:
    r"""Label Encode categorical values for multiple columns at once

    .. note:: LabelEncoder reserves 0 for `unseen` new categories. This is convenient
        when defining the embedding layers, since we can just set padding idx to 0.

    Parameters
    ----------
    columns_to_encode: list, Optional, default = None
        List of strings containing the names of the columns to encode. If
        ``None`` all columns of type ``object`` in the dataframe will be label
        encoded.
    with_attention: bool, default = False
        Boolean indicating whether the preprocessed data will be passed to an
        attention-based model.
        Aliased as ``for_transformer``.
    shared_embed: bool, default = False
        Boolean indicating if the embeddings will be "shared" when using
        attention-based models. The idea behind ``shared_embed`` is described
        in the Appendix A in the `TabTransformer paper
        <https://arxiv.org/abs/2012.06678>`_: `'The goal of having column
        embedding is to enable the model to distinguish the classes in one
        column from those in the other columns'`. In other words, the idea is
        to let the model learn which column is embedded at the time. See:
        :obj:`pytorch_widedeep.models.transformers._layers.SharedEmbeddings`.

    Attributes
    -----------
    encoding_dict: Dict
        Dictionary containing the encoding mappings in the format, e.g.

        `{'colname1': {'cat1': 1, 'cat2': 2, ...}, 'colname2': {'cat1': 1, 'cat2': 2, ...}, ...}`

    inverse_encoding_dict: Dict
        Dictionary containing the insverse encoding mappings in the format, e.g.

        `{'colname1': {1: 'cat1', 2: 'cat2', ...}, 'colname2': {1: 'cat1', 2: 'cat2', ...}, ...}`
    """

    @Alias("with_attention", "for_transformer")
    def __init__(
        self,
        columns_to_encode: Optional[List[str]] = None,
        with_attention: bool = False,
        shared_embed: bool = False,
    ):
        self.columns_to_encode = columns_to_encode

        self.shared_embed = shared_embed
        self.with_attention = with_attention

        self.reset_embed_idx = not self.with_attention or self.shared_embed

    def fit(self, df: pd.DataFrame) -> "LabelEncoder":
        """Creates encoding attributes"""

        df_inp = df.copy()

        if self.columns_to_encode is None:
            self.columns_to_encode = list(
                df_inp.select_dtypes(include=["object"]).columns
            )
        else:
            # sanity check to make sure all categorical columns are in an adequate
            # format
            for col in self.columns_to_encode:
                df_inp[col] = df_inp[col].astype("O")

        unique_column_vals = dict()
        for c in self.columns_to_encode:
            unique_column_vals[c] = df_inp[c].unique()

        self.encoding_dict = dict()
        if "cls_token" in unique_column_vals and self.shared_embed:
            self.encoding_dict["cls_token"] = {"[CLS]": 0}
            del unique_column_vals["cls_token"]
        # leave 0 for padding/"unseen" categories
        idx = 1
        for k, v in unique_column_vals.items():
            self.encoding_dict[k] = {
                o: i + idx for i, o in enumerate(unique_column_vals[k])
            }
            idx = 1 if self.reset_embed_idx else idx + len(unique_column_vals[k])

        self.inverse_encoding_dict = dict()
        for c in self.encoding_dict:
            self.inverse_encoding_dict[c] = {
                v: k for k, v in self.encoding_dict[c].items()
            }
            self.inverse_encoding_dict[c][0] = "unseen"

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label Encoded the categories in ``columns_to_encode``"""
        try:
            self.encoding_dict
        except AttributeError:
            raise NotFittedError(
                "This LabelEncoder instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this LabelEncoder."
            )

        df_inp = df.copy()
        # sanity check to make sure all categorical columns are in an adequate
        # format
        for col in self.columns_to_encode:  # type: ignore
            df_inp[col] = df_inp[col].astype("O")

        for k, v in self.encoding_dict.items():
            df_inp[k] = df_inp[k].apply(lambda x: v[x] if x in v.keys() else 0)

        return df_inp

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combines ``fit`` and ``transform``

        Examples
        --------

        >>> import pandas as pd
        >>> from pytorch_widedeep.utils import LabelEncoder
        >>> df = pd.DataFrame({'col1': [1,2,3], 'col2': ['me', 'you', 'him']})
        >>> columns_to_encode = ['col2']
        >>> encoder = LabelEncoder(columns_to_encode)
        >>> encoder.fit_transform(df)
           col1  col2
        0     1     1
        1     2     2
        2     3     3
        >>> encoder.encoding_dict
        {'col2': {'me': 1, 'you': 2, 'him': 3}}
        """
        return self.fit(df).transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns the original categories

        Examples
        --------

        >>> import pandas as pd
        >>> from pytorch_widedeep.utils import LabelEncoder
        >>> df = pd.DataFrame({'col1': [1,2,3], 'col2': ['me', 'you', 'him']})
        >>> columns_to_encode = ['col2']
        >>> encoder = LabelEncoder(columns_to_encode)
        >>> df_enc = encoder.fit_transform(df)
        >>> encoder.inverse_transform(df_enc)
           col1 col2
        0     1   me
        1     2  you
        2     3  him
        """
        for k, v in self.inverse_encoding_dict.items():
            df[k] = df[k].apply(lambda x: v[x])
        return df


def find_bin(
    bin_edges: Union[np.ndarray, Tensor],
    values: Union[np.ndarray, Tensor],
    ret_value: bool = True,
) -> Union[np.ndarray, Tensor]:
    """Returns histograms left bin edge value or array indices from monotonically
    increasing array of bin edges for each value in values.
    If ret_value

    Parameters
    ----------
    bin_edges: Union[np.ndarray, Tensor]
        monotonically increasing array of bin edges
    values: Union[np.ndarray, Tensor]
        values for which we want corresponding bins
    ret_value: bool
        if True, return bin values else indices

    Returns
    -------
    left_bin_edges: Union[np.ndarray, Tensor]
        left bin edges
    """
    if type(bin_edges) == np.ndarray and type(values) == np.ndarray:
        indices: Union[np.ndarray, Tensor] = np.searchsorted(
            bin_edges, values, side="left"
        )
        indices = np.where(
            (indices == 0) | (indices == len(bin_edges)), indices, indices - 1
        )
        indices = np.where(indices != len(bin_edges), indices, indices - 2)
    elif type(bin_edges) == Tensor and type(values) == Tensor:
        bin_edges = bin_edges.to(values.device)
        indices = torch.searchsorted(bin_edges, values, right=False)
        indices = torch.where(
            (indices == 0) | (indices == len(bin_edges)), indices, indices - 1
        )
        indices = torch.where(indices != len(bin_edges), indices, indices - 2)
    else:
        raise TypeError(
            "Both input arrays must be of teh same type, either np.ndarray of Tensor"
        )
    return indices if not ret_value else bin_edges[indices]  # type: ignore[index]


def _laplace(x):
    return np.exp(-abs(x) / sigma) / (2.0 * sigma)


def get_kernel_window(
    kernel: Literal["gaussian", "triang", "laplace"] = "gaussian",
    ks: int = 5,
    sigma: Union[int, float] = 2,
) -> List[float]:

    """Procedure to prepare window of values from symetrical kernel function for smoothing of the distribution in
    Label and Feature Distribution Smoothing (LDS & FDS).

    Parameters
    ----------
    kernel: Literal['gaussian', 'triang', 'laplace'] = 'gaussian'
        choice of kernel for label distribution smoothing
    ks: int = 5
        kernel size, i.e. count of samples in symmetric window
    sigma: Union[int,float] = 2
        standard deviation of ['gaussian','laplace'] kernel

    Returns
    -------
    kernel_window: list
        list with values from the chosen kernel function
    """
    half_ks = (ks - 1) // 2
    if kernel == "gaussian":
        base_kernel = [0.0] * half_ks + [1.0] + [0.0] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma)
    elif kernel == "triang":
        kernel_window = triang(ks) / sum(triang(ks))
    elif kernel == "laplace":
        kernel_window = list(map(_laplace, np.arange(-half_ks, half_ks + 1)))
    else:
        raise ValueError("Kernel can be only ['gaussian', 'triang', 'laplace'].")

    return kernel_window
