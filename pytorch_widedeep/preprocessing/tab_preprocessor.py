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
from pytorch_widedeep.utils.general_utils import alias
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
                        range(1, len(bins) + 1),
                        [(a + b) / 2.0 for a, b in zip(bins, bins[1:])],
                    )
                )
            }
            # 0
            self.inversed_bins[col][0] = np.nan

        self.is_fitted = True

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, condition=self.is_fitted)

        dfc = df.copy()
        for col, bins in self.bins.items():
            dfc[col] = pd.cut(dfc[col], bins, labels=False, **self.quant_args)
            # 0 will be left for numbers outside the bins, i.e. smaller than
            # the smaller boundary or larger than the largest boundary
            dfc[col] = dfc[col] + 1
            dfc[col] = dfc[col].fillna(0)
            dfc[col] = dfc[col].astype(int)

        return dfc

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def __repr__(self) -> str:
        return f"Quantizer(quantization_setup={self.quantization_setup})"


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
        list of scalars indicating the bin edges can also be used.
    cols_to_scale: List or str, default = None,
        List with the names of the columns that will be standarised via
        sklearn's `StandardScaler`. It can also be the string `'all'` in
        which case all the continuous cols will be scaled.
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
        Dimension for the embeddings if the embedding dimension is not
        provided in the `cat_embed_cols` parameter and `auto_embed_dim` is
        set to `False`.
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
        and/or continuous embeddings will be concatenated before being passed
        to the final MLP (if present).
    shared_embed: bool, default = False
        Boolean indicating if the embeddings will be "shared" when using
        attention-based models. The idea behind `shared_embed` is
        described in the Appendix A in the [TabTransformer paper](https://arxiv.org/abs/2012.06678):
        _'The goal of having column embedding is to enable the model to
        distinguish the classes in one column from those in the other
        columns'_. In other words, the idea is to let the model learn which
        column is embedded at the time. See: `pytorch_widedeep.models.transformers._layers.SharedEmbeddings`.
    verbose: int, default = 1
    scale: bool, default = False
        :information_source: **note**: this arg will be removed in upcoming
         releases. Please use `cols_to_scale` instead. <br/> Bool indicating
         whether or not to scale/standarise continuous cols. It is important
         to emphasize that all the DL models for tabular data in the library
         also include the possibility of normalising the input continuous
         features via a `BatchNorm` or a `LayerNorm`. <br/> Param alias:
         `scale_cont_cols`.
    already_standard: List, default = None
        :information_source: **note**: this arg will be removed in upcoming
         releases. Please use `cols_to_scale` instead. <br/> List with the
         name of the continuous cols that do not need to be
         scaled/standarised.

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
    >>> deep_preprocessor.cat_embed_cols
    [('color', 5), ('size', 5)]
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

    @alias("with_attention", ["for_transformer"])
    @alias("cat_embed_cols", ["embed_cols"])
    @alias("scale", ["scale_cont_cols"])
    @alias("quantization_setup", ["cols_and_bins"])
    def __init__(
        self,
        cat_embed_cols: Optional[Union[List[str], List[Tuple[str, int]]]] = None,
        continuous_cols: Optional[List[str]] = None,
        quantization_setup: Optional[
            Union[int, Dict[str, Union[int, List[float]]]]
        ] = None,
        cols_to_scale: Optional[Union[List[str], str]] = None,
        auto_embed_dim: bool = True,
        embedding_rule: Literal["google", "fastai_old", "fastai_new"] = "fastai_new",
        default_embed_dim: int = 16,
        with_attention: bool = False,
        with_cls_token: bool = False,
        shared_embed: bool = False,
        verbose: int = 1,
        *,
        scale: bool = False,
        already_standard: Optional[List[str]] = None,
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

    def fit(self, df: pd.DataFrame) -> BasePreprocessor:  # noqa: C901
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

        self.column_idx: Dict[str, int] = {}

        # Categorical embeddings logic
        if self.cat_embed_cols is not None or self.quantization_setup is not None:
            self.cat_embed_input: List[Union[Tuple[str, int], Tuple[str, int, int]]] = (
                []
            )

        if self.cat_embed_cols is not None:
            df_cat, cat_embed_dim = self._prepare_categorical(df_adj)

            self.label_encoder = LabelEncoder(
                columns_to_encode=df_cat.columns.tolist(),
                shared_embed=self.shared_embed,
                with_attention=self.with_attention,
            )
            self.label_encoder.fit(df_cat)

            for k, v in self.label_encoder.encoding_dict.items():
                if self.with_attention:
                    self.cat_embed_input.append((k, len(v)))
                else:
                    self.cat_embed_input.append((k, len(v), cat_embed_dim[k]))

            self.column_idx.update({k: v for v, k in enumerate(df_cat.columns)})

        # Continuous columns logic
        if self.continuous_cols is not None:
            df_cont, cont_embed_dim = self._prepare_continuous(df_adj)

            # Standardization logic
            if self.standardize_cols is not None:
                self.scaler = StandardScaler(**self.scale_args).fit(
                    df_cont[self.standardize_cols].values
                )
            elif self.verbose:
                warnings.warn("Continuous columns will not be normalised")

            # Quantization logic
            if self.cols_and_bins is not None:
                # we do not run 'Quantizer.fit' here since in the wild case
                # someone wants standardization and quantization for the same
                # columns, the Quantizer will run on the scaled data
                self.quantizer = Quantizer(self.cols_and_bins, **self.quant_args)

                if self.with_attention:
                    for col, n_cat, _ in cont_embed_dim:
                        self.cat_embed_input.append((col, n_cat))
                else:
                    self.cat_embed_input.extend(cont_embed_dim)

            self.column_idx.update(
                {k: v + len(self.column_idx) for v, k in enumerate(df_cont)}
            )

        self.is_fitted = True

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:  # noqa: C901
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
            df_cat = df_adj[self.cat_cols]
            df_cat = self.label_encoder.transform(df_cat)
        if self.continuous_cols is not None:
            df_cont = df_adj[self.continuous_cols]
            # Standardization logic
            if self.standardize_cols:
                df_cont[self.standardize_cols] = self.scaler.transform(
                    df_cont[self.standardize_cols].values
                )
            # Quantization logic
            if self.cols_and_bins is not None:
                # Adjustment so I don't have to override the method
                # in 'ChunkTabPreprocessor'
                if self.quantizer.is_fitted:
                    df_cont = self.quantizer.transform(df_cont)
                else:
                    df_cont = self.quantizer.fit_transform(df_cont)
        try:
            df_deep = pd.concat([df_cat, df_cont], axis=1)
        except NameError:
            try:
                df_deep = df_cat.copy()
            except NameError:
                df_deep = df_cont.copy()

        return df_deep.values

    def transform_sample(self, df: pd.DataFrame) -> np.ndarray:
        return self.transform(df).astype("float")[0]

    def inverse_transform(self, encoded: np.ndarray) -> pd.DataFrame:  # noqa: C901
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
        decoded = pd.DataFrame(encoded, columns=list(self.column_idx.keys()))
        # embeddings back to original category
        if self.cat_embed_cols is not None:
            decoded = self.label_encoder.inverse_transform(decoded)
        if self.continuous_cols is not None:
            # quantized cols to the mid point
            if self.cols_and_bins is not None:
                if self.verbose:
                    print(
                        "Note that quantized cols will be turned into the mid point of "
                        "the corresponding bin"
                    )
                for k, v in self.quantizer.inversed_bins.items():
                    decoded[k] = decoded[k].map(v)
            # continuous_cols back to non-standarised
            try:
                decoded[self.standardize_cols] = self.scaler.inverse_transform(
                    decoded[self.standardize_cols]
                )
            except Exception:  # KeyError:
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

    def _prepare_categorical(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        if isinstance(self.cat_embed_cols[0], tuple):
            self.cat_cols: List[str] = [emb[0] for emb in self.cat_embed_cols]
            cat_embed_dim: Dict[str, int] = dict(self.cat_embed_cols)  # type: ignore
        else:
            self.cat_cols = self.cat_embed_cols  # type: ignore[assignment]
            if self.auto_embed_dim:
                assert isinstance(self.cat_embed_cols[0], str), (
                    "If 'auto_embed_dim' is 'True' and 'with_attention' is 'False', "
                    "'cat_embed_cols' must be a list of strings with the columns to "
                    "be encoded as embeddings."
                )
                n_cats: Dict[str, int] = {col: df[col].nunique() for col in self.cat_embed_cols}  # type: ignore[misc]
                cat_embed_dim = {
                    col: embed_sz_rule(n_cat, self.embedding_rule)  # type: ignore[misc]
                    for col, n_cat in n_cats.items()
                }
            else:
                cat_embed_dim = {
                    e: self.default_embed_dim for e in self.cat_embed_cols  # type: ignore[misc]
                }  # type: ignore
        return df[self.cat_cols], cat_embed_dim

    def _prepare_continuous(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[List[Tuple[str, int, int]]]]:
        # Standardization logic
        if self.cols_to_scale is not None:
            self.standardize_cols = (
                self.cols_to_scale
                if self.cols_to_scale != "all"
                else self.continuous_cols
            )
        elif self.scale:
            if self.already_standard is not None:
                self.standardize_cols = [
                    c for c in self.continuous_cols if c not in self.already_standard
                ]
            else:
                self.standardize_cols = self.continuous_cols
        else:
            self.standardize_cols = None

        # Quantization logic
        if self.quantization_setup is not None:
            # the quantized columns are then treated as categorical
            quant_cont_embed_input: Optional[List[Tuple[str, int, int]]] = []
            if isinstance(self.quantization_setup, int):
                self.cols_and_bins: Optional[Dict[str, Union[int, List[float]]]] = {}
                for col in self.continuous_cols:
                    self.cols_and_bins[col] = self.quantization_setup
                    quant_cont_embed_input.append(
                        (
                            col,
                            self.quantization_setup,
                            (
                                embed_sz_rule(
                                    self.quantization_setup + 1, self.embedding_rule  # type: ignore[arg-type]
                                )
                                if self.auto_embed_dim
                                else self.default_embed_dim
                            ),
                        )
                    )
            else:
                for col, val in self.quantization_setup.items():
                    if isinstance(val, int):
                        quant_cont_embed_input.append(
                            (
                                col,
                                val,
                                (
                                    embed_sz_rule(
                                        val + 1, self.embedding_rule  # type: ignore[arg-type]
                                    )
                                    if self.auto_embed_dim
                                    else self.default_embed_dim
                                ),
                            )
                        )
                    else:
                        quant_cont_embed_input.append(
                            (
                                col,
                                len(val) - 1,
                                (
                                    embed_sz_rule(len(val), self.embedding_rule)  # type: ignore[arg-type]
                                    if self.auto_embed_dim
                                    else self.default_embed_dim
                                ),
                            )
                        )

                self.cols_and_bins = self.quantization_setup.copy()
        else:
            self.cols_and_bins = None
            quant_cont_embed_input = None

        return df[self.continuous_cols], quant_cont_embed_input

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
                "If 'with_cls_token' is set to 'True', 'with_attention' will be automatically "
                "to 'True' if is 'False'",
                UserWarning,
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

    def __repr__(self) -> str:  # noqa: C901
        list_of_params: List[str] = []
        if self.cat_embed_cols is not None:
            list_of_params.append("cat_embed_cols={cat_embed_cols}")
        if self.continuous_cols is not None:
            list_of_params.append("continuous_cols={continuous_cols}")
        if self.quantization_setup is not None:
            list_of_params.append("quantization_setup={quantization_setup}")
        if self.cols_to_scale is not None:
            list_of_params.append("cols_to_scale={cols_to_scale}")
        if not self.auto_embed_dim:
            list_of_params.append("auto_embed_dim={auto_embed_dim}")
        if self.embedding_rule != "fastai_new":
            list_of_params.append("embedding_rule='{embedding_rule}'")
        if self.default_embed_dim != 16:
            list_of_params.append("default_embed_dim={default_embed_dim}")
        if self.with_attention:
            list_of_params.append("with_attention={with_attention}")
        if self.with_cls_token:
            list_of_params.append("with_cls_token={with_cls_token}")
        if self.shared_embed:
            list_of_params.append("shared_embed={shared_embed}")
        if self.verbose != 1:
            list_of_params.append("verbose={verbose}")
        if self.scale:
            list_of_params.append("scale={scale}")
        if self.already_standard is not None:
            list_of_params.append("already_standard={already_standard}")
        if len(self.quant_args) > 0:
            list_of_params.append(
                ", ".join([f"{k}" + "=" + f"{v}" for k, v in self.quant_args.items()])
            )
        if len(self.scale_args) > 0:
            list_of_params.append(
                ", ".join([f"{k}" + "=" + f"{v}" for k, v in self.scale_args.items()])
            )
        all_params = ", ".join(list_of_params)
        return f"TabPreprocessor({all_params.format(**self.__dict__)})"


class ChunkTabPreprocessor(TabPreprocessor):
    r"""Preprocessor to prepare the `deeptabular` component input dataset

    Parameters
    ----------
    n_chunks: int
        Number of chunks that the tabular dataset is divided by.
    cat_embed_cols: List, default = None
        List containing the name of the categorical columns that will be
        represented by embeddings (e.g. _['education', 'relationship', ...]_) or
        a Tuple with the name and the embedding dimension (e.g.: _[
        ('education',32), ('relationship',16), ...]_)
    continuous_cols: List, default = None
        List with the name of the continuous cols
    cols_and_bins: Dict, default = None
        Continuous columns can be turned into categorical via
        `pd.cut`. 'cols_and_bins' is dictionary where the keys are the column
        names to quantize and the values are a list of scalars indicating the
        bin edges.
    cols_to_scale: List, default = None,
        List with the names of the columns that will be standarised via
        sklearn's `StandardScaler`
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
    scale: bool, default = False
        :information_source: **note**: this arg will be removed in upcoming
         releases. Please use `cols_to_scale` instead. <br/> Bool indicating
         whether or not to scale/standarise continuous cols. It is important
         to emphasize that all the DL models for tabular data in the library
         also include the possibility of normalising the input continuous
         features via a `BatchNorm` or a `LayerNorm`. <br/> Param alias:
         `scale_cont_cols`.
    already_standard: List, default = None
        :information_source: **note**: this arg will be removed in upcoming
         releases. Please use `cols_to_scale` instead. <br/> List with the
         name of the continuous cols that do not need to be
         scaled/standarised.

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
        if 'cols_to_scale' is not None or 'scale' is 'True'
    column_idx: Dict
        Dictionary where keys are column names and values are column indexes.
        This is neccesary to slice tensors
    quantizer: Quantizer
        an instance of `Quantizer`

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pytorch_widedeep.preprocessing import ChunkTabPreprocessor
    >>> np.random.seed(42)
    >>> chunk_df = pd.DataFrame({'cat_col': np.random.choice(['A', 'B', 'C'], size=8),
    ... 'cont_col': np.random.uniform(1, 100, size=8)})
    >>> cat_embed_cols = [('cat_col',4)]
    >>> cont_cols = ['cont_col']
    >>> tab_preprocessor = ChunkTabPreprocessor(
    ... n_chunks=1, cat_embed_cols=cat_embed_cols, continuous_cols=cont_cols
    ... )
    >>> X_tab = tab_preprocessor.fit_transform(chunk_df)
    >>> tab_preprocessor.cat_embed_cols
    [('cat_col', 4)]
    >>> tab_preprocessor.column_idx
    {'cat_col': 0, 'cont_col': 1}
    """

    @alias("with_attention", ["for_transformer"])
    @alias("cat_embed_cols", ["embed_cols"])
    @alias("scale", ["scale_cont_cols"])
    @alias("cols_and_bins", ["quantization_setup"])
    def __init__(
        self,
        n_chunks: int,
        cat_embed_cols: Optional[Union[List[str], List[Tuple[str, int]]]] = None,
        continuous_cols: Optional[List[str]] = None,
        cols_and_bins: Optional[Dict[str, List[float]]] = None,
        cols_to_scale: Optional[Union[List[str], str]] = None,
        default_embed_dim: int = 16,
        with_attention: bool = False,
        with_cls_token: bool = False,
        shared_embed: bool = False,
        verbose: int = 1,
        *,
        scale: bool = False,
        already_standard: Optional[List[str]] = None,
        **kwargs,
    ):
        super(ChunkTabPreprocessor, self).__init__(
            cat_embed_cols=cat_embed_cols,
            continuous_cols=continuous_cols,
            quantization_setup=None,
            cols_to_scale=cols_to_scale,
            auto_embed_dim=False,
            embedding_rule="google",  # does not matter, irrelevant
            default_embed_dim=default_embed_dim,
            with_attention=with_attention,
            with_cls_token=with_cls_token,
            shared_embed=shared_embed,
            verbose=verbose,
            scale=scale,
            already_standard=already_standard,
            **kwargs,
        )

        self.n_chunks = n_chunks
        self.chunk_counter = 0

        self.cols_and_bins = cols_and_bins  # type: ignore[assignment]
        if self.cols_and_bins is not None:
            self.quantizer = Quantizer(self.cols_and_bins, **self.quant_args)

        self.embed_prepared = False
        self.continuous_prepared = False

    def partial_fit(self, df: pd.DataFrame) -> "ChunkTabPreprocessor":  # noqa: C901
        # df here, and throughout the class, is a chunk of the original df
        self.chunk_counter += 1

        df_adj = self._insert_cls_token(df) if self.with_cls_token else df.copy()

        self.column_idx: Dict[str, int] = {}

        # Categorical embeddings logic
        if self.cat_embed_cols is not None:
            if not self.embed_prepared:
                df_cat, self.cat_embed_dim = self._prepare_categorical(df_adj)
                self.label_encoder = LabelEncoder(
                    columns_to_encode=df_cat.columns.tolist(),
                    shared_embed=self.shared_embed,
                    with_attention=self.with_attention,
                )
                self.label_encoder.partial_fit(df_cat)
            else:
                df_cat = df_adj[self.cat_cols]
                self.label_encoder.partial_fit(df_cat)

            self.column_idx.update({k: v for v, k in enumerate(df_cat.columns)})

        # Continuous columns logic
        if self.continuous_cols is not None:
            if not self.continuous_prepared:
                df_cont, self.cont_embed_dim = self._prepare_continuous(df_adj)
            else:
                df_cont = df[self.continuous_cols]

            self.column_idx.update(
                {k: v + len(self.column_idx) for v, k in enumerate(df_cont.columns)}
            )

            if self.standardize_cols is not None:
                self.scaler.partial_fit(df_cont[self.standardize_cols].values)

        if self.chunk_counter == self.n_chunks:
            if self.cat_embed_cols is not None or self.cols_and_bins is not None:
                self.cat_embed_input: List[
                    Union[Tuple[str, int], Tuple[str, int, int]]
                ] = []

            if self.cat_embed_cols is not None:
                for k, v in self.label_encoder.encoding_dict.items():
                    if self.with_attention:
                        self.cat_embed_input.append((k, len(v)))
                    else:
                        self.cat_embed_input.append((k, len(v), self.cat_embed_dim[k]))

            if self.cols_and_bins is not None:
                assert self.cont_embed_dim is not None  # just to make mypy happy
                if self.with_attention:
                    for col, n_cat, _ in self.cont_embed_dim:
                        self.cat_embed_input.append((col, n_cat))
                else:
                    self.cat_embed_input.extend(self.cont_embed_dim)

            self.is_fitted = True

        return self

    def fit(self, df: pd.DataFrame) -> "ChunkTabPreprocessor":
        # just to override the fit method in the base class. This class is not
        # designed or thought to run fit
        return self.partial_fit(df)

    def _prepare_categorical(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        # When dealing with chunks we will NOT support the option of
        # automatically define embeddings as this implies going through the
        # entire dataset
        if isinstance(self.cat_embed_cols[0], tuple):
            self.cat_cols: List[str] = [emb[0] for emb in self.cat_embed_cols]
            cat_embed_dim: Dict[str, int] = dict(self.cat_embed_cols)  # type: ignore
        else:
            self.cat_cols = self.cat_embed_cols  # type: ignore[assignment]
            cat_embed_dim = {
                e: self.default_embed_dim for e in self.cat_embed_cols  # type: ignore[misc]
            }  # type: ignore

        self.embed_prepared = True

        return df[self.cat_cols], cat_embed_dim

    def _prepare_continuous(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[List[Tuple[str, int, int]]]]:
        # Standardization logic
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
                    for c in self.continuous_cols  # type: ignore[misc]
                    if c not in self.already_standard
                ]
            else:
                self.standardize_cols = self.continuous_cols
        else:
            self.standardize_cols = None

        if self.standardize_cols is not None:
            self.scaler = StandardScaler(**self.scale_args)
        elif self.verbose:
            warnings.warn("Continuous columns will not be normalised")

        # Quantization logic
        if self.cols_and_bins is not None:
            # the quantized columns are then treated as categorical
            quant_cont_embed_input: Optional[List[Tuple[str, int, int]]] = []
            for col, val in self.cols_and_bins.items():
                if isinstance(val, int):
                    quant_cont_embed_input.append(
                        (
                            col,
                            val,
                            self.default_embed_dim,
                        )
                    )
                else:
                    quant_cont_embed_input.append(
                        (
                            col,
                            len(val) - 1,
                            self.default_embed_dim,
                        )
                    )
        else:
            quant_cont_embed_input = None

        self.continuous_prepared = True

        return df[self.continuous_cols], quant_cont_embed_input

    def __repr__(self) -> str:  # noqa: C901
        list_of_params: List[str] = []
        if self.n_chunks is not None:
            list_of_params.append("n_chunks={n_chunks}")
        if self.cat_embed_cols is not None:
            list_of_params.append("cat_embed_cols={cat_embed_cols}")
        if self.continuous_cols is not None:
            list_of_params.append("continuous_cols={continuous_cols}")
        if self.cols_and_bins is not None:
            list_of_params.append("cols_and_bins={cols_and_bins}")
        if self.cols_to_scale is not None:
            list_of_params.append("cols_to_scale={cols_to_scale}")
        if self.default_embed_dim != 16:
            list_of_params.append("default_embed_dim={default_embed_dim}")
        if self.with_attention:
            list_of_params.append("with_attention={with_attention}")
        if self.with_cls_token:
            list_of_params.append("with_cls_token={with_cls_token}")
        if self.shared_embed:
            list_of_params.append("shared_embed={shared_embed}")
        if self.verbose != 1:
            list_of_params.append("verbose={verbose}")
        if self.scale:
            list_of_params.append("scale={scale}")
        if self.already_standard is not None:
            list_of_params.append("already_standard={already_standard}")
        if len(self.quant_args) > 0:
            list_of_params.append(
                ", ".join([f"{k}" + "=" + f"{v}" for k, v in self.quant_args.items()])
            )
        if len(self.scale_args) > 0:
            list_of_params.append(
                ", ".join([f"{k}" + "=" + f"{v}" for k, v in self.scale_args.items()])
            )
        all_params = ", ".join(list_of_params)
        return f"ChunkTabPreprocessor({all_params.format(**self.__dict__)})"
