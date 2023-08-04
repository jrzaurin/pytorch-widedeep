import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from pytorch_widedeep.wdtypes import (
    Dict,
    List,
    Tuple,
    Union,
    Literal,
    Optional,
)
from pytorch_widedeep.utils.general_utils import Alias
from pytorch_widedeep.utils.deeptabular_utils import LabelEncoder
from pytorch_widedeep.preprocessing.base_preprocessor import (
    BasePreprocessor,
    check_is_fitted,
)


def embed_sz_rule(
    n_cat: int,
    embedding_rule: Literal["google", "fastai_old", "fastai_new"] = "fastai_new",
) -> int:
    r"""Rule of thumb to pick embedding size corresponding to ``n_cat``. Default rule is taken
    from recent fastai's Tabular API. The function also includes previously used rule by fastai
    and rule included in the Google's Tensorflow documentation

    Parameters
    ----------
    n_cat: int
        number of unique categorical values in a feature
    embedding_rule: str, default = fastai_old
        rule of thumb to be used for embedding vector size
    """
    if embedding_rule == "google":
        return int(round(n_cat**0.25))
    elif embedding_rule == "fastai_old":
        return int(min(50, (n_cat // 2) + 1))
    else:
        return int(min(600, round(1.6 * n_cat**0.56)))


class Quantizer:
    """Helper class to perform the quantization of continuous columns. It is
    included in this docs for completion, since depending on the value of the
    parameter `'quantization_setup'` of the `TabPreprocessor` class, that
    class might have an attribute of type `Quantizer`. However, this class is
    designed to always run internally within the `TabPreprocessor` class.

    Parameters
    ----------
    quantization_setup: Dict, default = None
        Dictionary where the keys are the column names to quantize and the
        values are the either integers indicating the number of bins or a
        list of scalars indicating the bin edges.
    """

    def __init__(
        self,
        quantization_setup: Dict[str, Union[int, List[float]]],
        **kwargs,
    ):
        self.quantization_setup = quantization_setup
        self.quant_args = kwargs

        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "Quantizer":
        self.bins: Dict[str, List[float]] = {}
        for col, bins in self.quantization_setup.items():
            _, self.bins[col] = pd.cut(
                df[col], bins, retbins=True, labels=False, **self.quant_args
            )

        self.inversed_bins: Dict[str, Dict[int, float]] = {}
        for col, bins in self.bins.items():
            self.inversed_bins[col] = {
                k: v
                for k, v in list(
                    zip(
                        range(len(bins)),
                        [(a + b) / 2.0 for a, b in zip(bins, bins[1:])],
                    )
                )
            }

        self.is_fitted = True

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, condition=self.is_fitted)

        dfc = df.copy()
        for col, bins in self.bins.items():
            dfc[col] = pd.cut(dfc[col], bins, labels=False, **self.quant_args)

        return dfc

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


class TabPreprocessor(BasePreprocessor):
    r"""Preprocessor to prepare the `deeptabular` component input dataset

    Parameters
    ----------
    cat_embed_cols: List, default = None
        List containing the name of the categorical columns that will be
        represented by embeddings (e.g. _['education', 'relationship', ...]_) or
        a Tuple with the name and the embedding dimension (e.g.: _[
        ('education',32), ('relationship',16), ...]_)
    continuous_cols: List, default = None
        List with the name of the continuous cols
    quantization_setup: int or Dict, default = None
        Continuous columns can be turned into categorical via `pd.cut`. If
        `quantization_setup` is an `int`, all continuous columns will be
        quantized using this value as the number of bins. Alternatively, a
        dictionary where the keys are the column names to quantize and the
        values are the either integers indicating the number of bins or a
        list of scalars indicating the bin edges.
    cols_to_scale: List, default = None,
        List with the names of the columns that will be standarised via
        sklearn's `StandardScaler`
    scale: bool, default = False
        :information_source: **note**: this arg will be removed in the next
         release. Please use `cols_to_scale` instead. <br/>
        Bool indicating whether or not to scale/standarise continuous cols. It
        is important to emphasize that all the DL models for tabular data in
        the library also include the possibility of normalising the input
        continuous features via a `BatchNorm` or a `LayerNorm`. <br/>
        Param alias: `scale_cont_cols`.
    already_standard: List, default = None
        :information_source: **note**: this arg will be removed in the next
         release. Please use `cols_to_scale` instead. <br/>
        List with the name of the continuous cols that do not need to be
        scaled/standarised.
    auto_embed_dim: bool, default = True
        Boolean indicating whether the embedding dimensions will be
        automatically defined via rule of thumb. See `embedding_rule`
        below.
    embedding_rule: str, default = 'fastai_new'
        If `auto_embed_dim=True`, this is the choice of embedding rule of
        thumb. Choices are:

        - _fastai_new_: $min(600, round(1.6 \times n_{cat}^{0.56}))$

        - _fastai_old_: $min(50, (n_{cat}//{2})+1)$

        - _google_: $min(600, round(n_{cat}^{0.24}))$
    default_embed_dim: int, default=16
        Dimension for the embeddings if the embed_dim is not provided in the
        `cat_embed_cols` parameter and `auto_embed_dim` is set to
        `False`.
    with_attention: bool, default = False
        Boolean indicating whether the preprocessed data will be passed to an
        attention-based model (more precisely a model where all embeddings
        must have the same dimensions). If `True`, the param `cat_embed_cols`
        must just be a list containing just the categorical column names:
        e.g.
        _['education', 'relationship', ...]_. This is because they will all be
         encoded using embeddings of the same dim, which will be specified
         later when the model is defined. <br/> Param alias:
         `for_transformer`
    with_cls_token: bool, default = False
        Boolean indicating if a `'[CLS]'` token will be added to the dataset
        when using attention-based models. The final hidden state
        corresponding to this token is used as the aggregated representation
        for classification and regression tasks. If not, the categorical
        (and continuous embeddings if present) will be concatenated before
        being passed to the final MLP (if present).
    shared_embed: bool, default = False
        Boolean indicating if the embeddings will be "shared" when using
        attention-based models. The idea behind `shared_embed` is
        described in the Appendix A in the [TabTransformer paper](https://arxiv.org/abs/2012.06678):
        _'The goal of having column embedding is to enable the model to
        distinguish the classes in one column from those in the other
        columns'_. In other words, the idea is to let the model learn which
        column is embedded at the time. See: `pytorch_widedeep.models.transformers._layers.SharedEmbeddings`.
    verbose: int, default = 1

    Other Parameters
    ----------------
    **kwargs: dict
        `pd.cut` and `StandardScaler` related args

    Attributes
    ----------
    embed_dim: Dict
        Dictionary where keys are the embed cols and values are the embedding
        dimensions. If `with_attention` is set to `True` this attribute
        is not generated during the `fit` process
    label_encoder: LabelEncoder
        see `pytorch_widedeep.utils.dense_utils.LabelEncder`
    cat_embed_input: List
        List of Tuples with the column name, number of individual values for
        that column and, If `with_attention` is set to `False`, the
        corresponding embeddings dim, e.g. _[('education', 16, 10),
        ('relationship', 6, 8), ...]_.
    standardize_cols: List
        List of the columns that will be standarized
    scaler: StandardScaler
        an instance of `sklearn.preprocessing.StandardScaler`
    column_idx: Dict
        Dictionary where keys are column names and values are column indexes.
        This is neccesary to slice tensors
    quantizer: Quantizer
        an instance of `Quantizer`

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pytorch_widedeep.preprocessing import TabPreprocessor
    >>> df = pd.DataFrame({'color': ['r', 'b', 'g'], 'size': ['s', 'n', 'l'], 'age': [25, 40, 55]})
    >>> cat_embed_cols = [('color',5), ('size',5)]
    >>> cont_cols = ['age']
    >>> deep_preprocessor = TabPreprocessor(cat_embed_cols=cat_embed_cols, continuous_cols=cont_cols)
    >>> X_tab = deep_preprocessor.fit_transform(df)
    >>> deep_preprocessor.embed_dim
    {'color': 5, 'size': 5}
    >>> deep_preprocessor.column_idx
    {'color': 0, 'size': 1, 'age': 2}
    >>> cont_df = pd.DataFrame({"col1": np.random.rand(10), "col2": np.random.rand(10) + 1})
    >>> cont_cols = ["col1", "col2"]
    >>> tab_preprocessor = TabPreprocessor(continuous_cols=cont_cols, quantization_setup=3)
    >>> ft_cont_df = tab_preprocessor.fit_transform(cont_df)
    >>> # or...
    >>> quantization_setup = {'col1': [0., 0.4, 1.], 'col2': [1., 1.4, 2.]}
    >>> tab_preprocessor2 = TabPreprocessor(continuous_cols=cont_cols, quantization_setup=quantization_setup)
    >>> ft_cont_df2 = tab_preprocessor2.fit_transform(cont_df)
    """

    @Alias("with_attention", "for_transformer")
    @Alias("cat_embed_cols", "embed_cols")
    @Alias("scale", "scale_cont_cols")
    def __init__(
        self,
        cat_embed_cols: Optional[Union[List[str], List[Tuple[str, int]]]] = None,
        continuous_cols: Optional[List[str]] = None,
        quantization_setup: Optional[
            Union[int, Dict[str, Union[int, List[float]]]]
        ] = None,
        cols_to_scale: Optional[List[str]] = None,
        auto_embed_dim: bool = True,
        embedding_rule: Literal["google", "fastai_old", "fastai_new"] = "fastai_new",
        default_embed_dim: int = 16,
        with_attention: bool = False,
        with_cls_token: bool = False,
        shared_embed: bool = False,
        verbose: int = 1,
        *,
        scale: bool = False,
        already_standard: List[str] = None,
        **kwargs,
    ):
        super(TabPreprocessor, self).__init__()

        self.continuous_cols = continuous_cols
        self.quantization_setup = quantization_setup
        self.cols_to_scale = cols_to_scale
        self.scale = scale
        self.already_standard = already_standard
        self.auto_embed_dim = auto_embed_dim
        self.embedding_rule = embedding_rule
        self.default_embed_dim = default_embed_dim
        self.with_attention = with_attention
        self.with_cls_token = with_cls_token
        self.shared_embed = shared_embed
        self.verbose = verbose

        self.quant_args = {
            k: v for k, v in kwargs.items() if k in pd.cut.__code__.co_varnames
        }
        self.scale_args = {
            k: v for k, v in kwargs.items() if k in StandardScaler().get_params()
        }

        self._check_inputs(cat_embed_cols)

        if with_cls_token:
            self.cat_embed_cols = (
                ["cls_token"] + cat_embed_cols  # type: ignore[operator]
                if cat_embed_cols is not None
                else ["cls_token"]
            )
        else:
            self.cat_embed_cols = cat_embed_cols  # type: ignore[assignment]

        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> BasePreprocessor:
        """Fits the Preprocessor and creates required attributes

        Parameters
        ----------
        df: pd.DataFrame
            Input pandas dataframe

        Returns
        -------
        TabPreprocessor
            `TabPreprocessor` fitted object
        """

        df_adj = self._insert_cls_token(df) if self.with_cls_token else df.copy()

        if self.cat_embed_cols is not None:
            df_emb = self._prepare_embed(df_adj)
            self.label_encoder = LabelEncoder(
                columns_to_encode=df_emb.columns.tolist(),
                shared_embed=self.shared_embed,
                with_attention=self.with_attention,
            )
            self.label_encoder.fit(df_emb)
            self.cat_embed_input: List = []
            for k, v in self.label_encoder.encoding_dict.items():
                if self.with_attention:
                    self.cat_embed_input.append((k, len(v)))
                else:
                    self.cat_embed_input.append((k, len(v), self.embed_dim[k]))
        if self.continuous_cols is not None:
            df_cont = self._prepare_continuous(df_adj)

            if self.standardize_cols is not None:
                self.scaler = StandardScaler(**self.scale_args).fit(
                    df_cont[self.standardize_cols].values
                )
            elif self.verbose:
                warnings.warn("Continuous columns will not be normalised")

            if self.cols_and_bins is not None:
                # we do not run 'Quantizer.fit' here since in the wild case
                # someone wants standardization and quantization for the same
                # columns, the Quantizer will run on the scaled data
                self.quantizer = Quantizer(self.cols_and_bins, **self.quant_args)

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Returns the processed `dataframe` as a np.ndarray

        Parameters
        ----------
        df: pd.DataFrame
            Input pandas dataframe

        Returns
        -------
        np.ndarray
            transformed input dataframe
        """
        check_is_fitted(self, condition=self.is_fitted)

        df_adj = self._insert_cls_token(df) if self.with_cls_token else df.copy()

        if self.cat_embed_cols is not None:
            df_emb = self._prepare_embed(df_adj)
            df_emb = self.label_encoder.transform(df_emb)
        if self.continuous_cols is not None:
            df_cont = self._prepare_continuous(df_adj)
            if self.standardize_cols:
                df_cont[self.standardize_cols] = self.scaler.transform(
                    df_cont[self.standardize_cols].values
                )
            if self.cols_and_bins is not None:
                df_cont = self.quantizer.fit_transform(df_cont)
        try:
            df_deep = pd.concat([df_emb, df_cont], axis=1)
        except NameError:
            try:
                df_deep = df_emb.copy()
            except NameError:
                df_deep = df_cont.copy()
        self.column_idx = {k: v for v, k in enumerate(df_deep.columns)}
        return df_deep.values

    def inverse_transform(self, encoded: np.ndarray) -> pd.DataFrame:
        r"""Takes as input the output from the `transform` method and it will
        return the original values.

        Parameters
        ----------
        encoded: np.ndarray
            array with the output of the `transform` method

        Returns
        -------
        pd.DataFrame
            Pandas dataframe with the original values
        """
        decoded = pd.DataFrame(encoded, columns=self.column_idx.keys())
        # embeddings back to original category
        if self.cat_embed_cols is not None:
            if isinstance(self.cat_embed_cols[0], tuple):
                emb_c: List = [c[0] for c in self.cat_embed_cols]
            else:
                emb_c = self.cat_embed_cols.copy()
            for c in emb_c:
                decoded[c] = decoded[c].map(self.label_encoder.inverse_encoding_dict[c])
        # quantized cols to the mid point
        if self.quantization_setup is not None:
            if self.verbose:
                print(
                    "Note that quantized cols will not be turned into the mid point of "
                    "the corresponding bin"
                )
            for k, v in self.quantizer.inversed_bins.items():
                decoded[k] = decoded[k].map(v)
        # continuous_cols back to non-standarised
        try:
            decoded[self.continuous_cols] = self.scaler.inverse_transform(
                decoded[self.continuous_cols]
            )
        except AttributeError:
            pass

        if "cls_token" in decoded.columns:
            decoded.drop("cls_token", axis=1, inplace=True)

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

    def _insert_cls_token(self, df: pd.DataFrame) -> pd.DataFrame:
        df_cls = df.copy()
        df_cls.insert(loc=0, column="cls_token", value="[CLS]")
        return df_cls

    def _prepare_embed(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.with_attention:
            return df[self.cat_embed_cols]
        else:
            if isinstance(self.cat_embed_cols[0], tuple):
                self.embed_dim: Dict = dict(self.cat_embed_cols)  # type: ignore
                embed_colname = [emb[0] for emb in self.cat_embed_cols]
            elif self.auto_embed_dim:
                n_cats = {col: df[col].nunique() for col in self.cat_embed_cols}
                self.embed_dim = {
                    # type: ignore[misc]
                    col: embed_sz_rule(n_cat, self.embedding_rule)
                    for col, n_cat in n_cats.items()
                }
                embed_colname = self.cat_embed_cols  # type: ignore
            else:
                self.embed_dim = {
                    e: self.default_embed_dim for e in self.cat_embed_cols
                }  # type: ignore
                embed_colname = self.cat_embed_cols  # type: ignore
            return df[embed_colname]

    def _prepare_continuous(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.is_fitted:
            return df[self.continuous_cols]
        else:
            if self.cols_to_scale is not None:
                self.standardize_cols = (
                    self.cols_to_scale
                    if self.cols_to_scale != "all"
                    else self.continuous_cols
                )
            elif self.scale:
                if self.already_standard is not None:
                    self.standardize_cols = [
                        c
                        for c in self.continuous_cols
                        if c not in self.already_standard
                    ]
                else:
                    self.standardize_cols = self.continuous_cols
            else:
                self.standardize_cols = None

            if self.quantization_setup is not None:
                if isinstance(self.quantization_setup, int):
                    self.cols_and_bins: Dict[str, Union[int, List[float]]] = {}
                    for col in self.continuous_cols:
                        self.cols_and_bins[col] = self.quantization_setup
                else:
                    self.cols_and_bins = self.quantization_setup.copy()
            else:
                self.cols_and_bins = None

            return df[self.continuous_cols]

    def _check_inputs(self, cat_embed_cols):  # noqa: C901
        if self.scale or self.already_standard is not None:
            warnings.warn(
                "'scale' and 'already_standard' will be deprecated in the next release. "
                "Please use 'cols_to_scale' instead",
                DeprecationWarning,
                stacklevel=2,
            )

        if self.scale:
            if self.already_standard is not None:
                standardize_cols = [
                    c for c in self.continuous_cols if c not in self.already_standard
                ]
            else:
                standardize_cols = self.continuous_cols
        elif self.cols_to_scale is not None:
            standardize_cols = self.cols_to_scale
        else:
            standardize_cols = None

        if standardize_cols is not None:
            if isinstance(self.quantization_setup, int):
                cols_to_quantize_and_standardize = [
                    c for c in standardize_cols if c in self.continuous_cols
                ]
            elif isinstance(self.quantization_setup, dict):
                cols_to_quantize_and_standardize = [
                    c for c in standardize_cols if c in self.quantization_setup
                ]
            else:
                cols_to_quantize_and_standardize = None
            if cols_to_quantize_and_standardize is not None:
                warnings.warn(
                    f"the following columns: {cols_to_quantize_and_standardize} will be first scaled"
                    " using a StandardScaler and then quantized. Make sure this is what you really want"
                )

        if self.with_cls_token and not self.with_attention:
            warnings.warn(
                "If 'with_cls_token' is set to 'True', 'with_attention' will be automatically ",
                "to 'True' if is 'False'",
            )
            self.with_attention = True

        if (cat_embed_cols is None) and (self.continuous_cols is None):
            raise ValueError(
                "'cat_embed_cols' and 'continuous_cols' are 'None'. Please, define at least one of the two."
            )

        if (
            cat_embed_cols is not None
            and self.continuous_cols is not None
            and len(np.intersect1d(cat_embed_cols, self.continuous_cols)) > 0
        ):
            overlapping_cols = list(
                np.intersect1d(cat_embed_cols, self.continuous_cols)
            )
            raise ValueError(
                "Currently passing columns as both categorical and continuum is not supported."
                " Please, choose one or the other for the following columns: {}".format(
                    ", ".join(overlapping_cols)
                )
            )
        transformer_error_message = (
            "If with_attention is 'True' cat_embed_cols must be a list "
            " of strings with the columns to be encoded as embeddings."
        )
        if (
            self.with_attention
            and cat_embed_cols is not None
            and isinstance(cat_embed_cols[0], tuple)
        ):
            raise ValueError(transformer_error_message)
