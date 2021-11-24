import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.utils.deeptabular_utils import LabelEncoder
from pytorch_widedeep.preprocessing.base_preprocessor import (
    BasePreprocessor,
    check_is_fitted,
)


def embed_sz_rule(n_cat: int, embedding_rule: str = "fastai_new") -> int:
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
        return int(round(n_cat ** 0.25))
    elif embedding_rule == "fastai_old":
        return int(min(50, (n_cat // 2) + 1))
    else:
        return int(min(600, round(1.6 * n_cat ** 0.56)))


class TabPreprocessor(BasePreprocessor):
    r"""Preprocessor to prepare the ``deeptabular`` component input dataset

    Parameters
    ----------
    embed_cols: List, default = None
        List containing the name of the columns that will be represented by
        embeddings or a Tuple with the name and the embedding dimension. e.g.:
        [('education',32), ('relationship',16), ...]
    continuous_cols: List, default = None
        List with the name of the so called continuous cols
    scale: bool, default = True
        Bool indicating whether or not to scale/standarise continuous cols.
        The user should bear in mind that all the ``deeptabular`` components
        available within ``pytorch-widedeep`` they also include the
        possibility of normalising the input continuous features via a
        ``BatchNorm`` or a ``LayerNorm`` layer. See
        :obj:`pytorch_widedeep.models.transformers._embedding_layers`
    auto_embed_dim: bool, default = True
        Boolean indicating whether the embedding dimensions will be
        automatically defined via rule of thumb
    embedding_rule: str, default = 'fastai_new'
        choice of embedding rule of thumb are:

        - 'fastai_new' -- :math:`min(600, round(1.6 \times n_{cat}^{0.56}))`

        - 'fastai_old' -- :math:`min(50, (n_{cat}//{2})+1)`

        - 'google' -- :math:`min(600, round(n_{cat}^{0.24}))`

    default_embed_dim: int, default=16
        Dimension for the embeddings used for the ``deeptabular``
        component if the embed_dim is not provided in the ``embed_cols``
        parameter
    already_standard: List, default = None
        List with the name of the continuous cols that do not need to be
        Standarised. For example, you might have Long and Lat in your
        dataset and might want to encode them somehow (e.g. see the
        ``LatLongScalarEnc`` available in the `autogluon
        <https://github.com/awslabs/autogluon/tree/master/tabular/src/autogluon/tabular>`_
        tabular library) and not standarize them any further
    for_transformer: bool, default = False
        Boolean indicating whether the preprocessed data will be passed to a
        transformer-based model
        (See :obj:`pytorch_widedeep.models.transformers`). If ``True``, the
        param ``embed_cols`` must just be a list containing the categorical
        columns: e.g.:['education', 'relationship', ...] This is because they
        will all be encoded using embeddings of the same dim.
    with_cls_token: bool, default = False
        Boolean indicating if a `'[CLS]'` token will be added to the dataset
        when using transformer-based models. The final hidden state
        corresponding to this token is used as the aggregated representation
        for classification and regression tasks. If not, the categorical
        (and continuous embeddings if present) will be concatenated before
        being passed to the final MLP.
    shared_embed: bool, default = False
        Boolean indicating if the embeddings will be "shared" when using
        transformer-based models. The idea behind ``shared_embed`` is
        described in the Appendix A in the `TabTransformer paper
        <https://arxiv.org/abs/2012.06678>`_: `'The goal of having column
        embedding is to enable the model to distinguish the classes in one
        column from those in the other columns'`. In other words, the idea is
        to let the model learn which column is embedded at the time. See:
        :obj:`pytorch_widedeep.models.transformers._layers.SharedEmbeddings`.
    verbose: int, default = 1

    Attributes
    ----------
    embed_dim: Dict
        Dictionary where keys are the embed cols and values are the embedding
        dimensions. If ``for_transformer`` is set to ``True`` the embedding
        dimensions are the same for all columns and this attributes is not
        generated during the ``fit`` process
    label_encoder: LabelEncoder
        see :class:`pytorch_widedeep.utils.dense_utils.LabelEncder`
    embeddings_input: List
        List of Tuples with the column name, number of individual values for
        that column and the corresponding embeddings dim, e.g. [
        ('education', 16, 10), ('relationship', 6, 8), ...]
    standardize_cols: List
        List of the columns that will be standarized
    scaler: StandardScaler
        an instance of :class:`sklearn.preprocessing.StandardScaler`
    column_idx: Dict
        Dictionary where keys are column names and values are column indexes.
        This is be neccesary to slice tensors

    Examples
    --------
    >>> import pandas as pd
    >>> from pytorch_widedeep.preprocessing import TabPreprocessor
    >>> df = pd.DataFrame({'color': ['r', 'b', 'g'], 'size': ['s', 'n', 'l'], 'age': [25, 40, 55]})
    >>> embed_cols = [('color',5), ('size',5)]
    >>> cont_cols = ['age']
    >>> deep_preprocessor = TabPreprocessor(embed_cols=embed_cols, continuous_cols=cont_cols)
    >>> X_tab = deep_preprocessor.fit_transform(df)
    >>> deep_preprocessor.embed_dim
    {'color': 5, 'size': 5}
    >>> deep_preprocessor.column_idx
    {'color': 0, 'size': 1, 'age': 2}
    """

    def __init__(
        self,
        embed_cols: Union[List[str], List[Tuple[str, int]]] = None,
        continuous_cols: List[str] = None,
        scale: bool = True,
        auto_embed_dim: bool = True,
        embedding_rule: str = "fastai_new",
        default_embed_dim: int = 16,
        already_standard: List[str] = None,
        for_transformer: bool = False,
        with_cls_token: bool = False,
        shared_embed: bool = False,
        verbose: int = 1,
    ):
        super(TabPreprocessor, self).__init__()

        self.embed_cols = embed_cols
        self.continuous_cols = continuous_cols
        self.scale = scale
        self.auto_embed_dim = auto_embed_dim
        self.embedding_rule = embedding_rule
        self.default_embed_dim = default_embed_dim
        self.already_standard = already_standard
        self.for_transformer = for_transformer
        self.with_cls_token = with_cls_token
        self.shared_embed = shared_embed
        self.verbose = verbose

        self.is_fitted = False

        if (self.embed_cols is None) and (self.continuous_cols is None):
            raise ValueError(
                "'embed_cols' and 'continuous_cols' are 'None'. Please, define at least one of the two."
            )

        transformer_error_message = (
            "If for_transformer is 'True' embed_cols must be a list "
            " of strings with the columns to be encoded as embeddings."
        )
        if self.for_transformer and self.embed_cols is None:
            raise ValueError(transformer_error_message)
        if self.for_transformer and isinstance(self.embed_cols[0], tuple):  # type: ignore[index]
            raise ValueError(transformer_error_message)

    def fit(self, df: pd.DataFrame) -> BasePreprocessor:
        """Fits the Preprocessor and creates required attributes"""
        if self.embed_cols is not None:
            df_emb = self._prepare_embed(df)
            self.label_encoder = LabelEncoder(
                columns_to_encode=df_emb.columns.tolist(),
                shared_embed=self.shared_embed,
                for_transformer=self.for_transformer,
            )
            self.label_encoder.fit(df_emb)
            self.embeddings_input: List = []
            for k, v in self.label_encoder.encoding_dict.items():
                if self.for_transformer:
                    self.embeddings_input.append((k, len(v)))
                else:
                    self.embeddings_input.append((k, len(v), self.embed_dim[k]))
        if self.continuous_cols is not None:
            df_cont = self._prepare_continuous(df)
            if self.scale:
                df_std = df_cont[self.standardize_cols]
                self.scaler = StandardScaler().fit(df_std.values)
            elif self.verbose:
                warnings.warn("Continuous columns will not be normalised")
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Returns the processed ``dataframe`` as a np.ndarray"""
        check_is_fitted(self, condition=self.is_fitted)
        if self.embed_cols is not None:
            df_emb = self._prepare_embed(df)
            df_emb = self.label_encoder.transform(df_emb)
        if self.continuous_cols is not None:
            df_cont = self._prepare_continuous(df)
            if self.scale:
                df_std = df_cont[self.standardize_cols]
                df_cont[self.standardize_cols] = self.scaler.transform(df_std.values)
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
        r"""Takes as input the output from the ``transform`` method and it will
        return the original values.

        Parameters
        ----------
        encoded: np.ndarray
            array with the output of the ``transform`` method
        """
        decoded = pd.DataFrame(encoded, columns=self.column_idx.keys())
        # embeddings back to original category
        if self.embed_cols is not None:
            if isinstance(self.embed_cols[0], tuple):
                emb_c: List = [c[0] for c in self.embed_cols]
            else:
                emb_c = self.embed_cols.copy()
            for c in emb_c:
                decoded[c] = decoded[c].map(self.label_encoder.inverse_encoding_dict[c])
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
        """Combines ``fit`` and ``transform``"""
        return self.fit(df).transform(df)

    def _prepare_embed(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.for_transformer:
            if self.with_cls_token:
                df_cls = df.copy()[self.embed_cols]
                df_cls.insert(loc=0, column="cls_token", value="[CLS]")
                return df_cls
            else:
                return df.copy()[self.embed_cols]
        else:
            if isinstance(self.embed_cols[0], tuple):
                self.embed_dim = dict(self.embed_cols)  # type: ignore
                embed_colname = [emb[0] for emb in self.embed_cols]
            elif self.auto_embed_dim:
                n_cats = {col: df[col].nunique() for col in self.embed_cols}
                self.embed_dim = {
                    col: embed_sz_rule(n_cat, self.embedding_rule)  # type: ignore[misc]
                    for col, n_cat in n_cats.items()
                }
                embed_colname = self.embed_cols  # type: ignore
            else:
                self.embed_dim = {e: self.default_embed_dim for e in self.embed_cols}  # type: ignore
                embed_colname = self.embed_cols  # type: ignore
            return df.copy()[embed_colname]

    def _prepare_continuous(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.scale:
            if self.already_standard is not None:
                self.standardize_cols = [
                    c for c in self.continuous_cols if c not in self.already_standard
                ]
            else:
                self.standardize_cols = self.continuous_cols
        return df.copy()[self.continuous_cols]
