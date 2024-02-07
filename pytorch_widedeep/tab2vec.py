import warnings

import numpy as np
import torch
import einops
import pandas as pd

from pytorch_widedeep.wdtypes import (
    List,
    Tuple,
    Union,
    Callable,
    Optional,
    WideDeep,
)
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.bayesian_models import BayesianWide, BayesianTabMlp
from pytorch_widedeep.bayesian_models._base_bayesian_model import (
    BaseBayesianModel,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Tab2Vec:
    r"""Class to transform an input dataframe into vectorized form.

    This class will take an input dataframe in the form of the dataframe used
    for training, and it will turn it into a vectorised form based on the
    processing applied by the model to the categorical and continuous
    columns.

    :information_source: **NOTE**: Currently this class is only implemented
     for the deeptabular component. Therefore, if the input dataframe has a
     text column or a column with the path to images, these will be ignored.
     We will be adding these functionalities in future versions

    Parameters
    ----------
    model: `WideDeep`, `BayesianWide` or `BayesianTabMlp`
        `WideDeep`, `BayesianWide` or `BayesianTabMlp` model. Must be trained.
    tab_preprocessor: `TabPreprocessor`
        `TabPreprocessor` object. Must be fitted.
    return_dataframe: bool
        Boolean indicating of the returned object(s) will be array(s) or
        pandas dataframe(s)

    Attributes
    ----------
    vectorizer: nn.Module
        Torch module with the categorical and continuous encoding process

    Examples
    --------
    >>> import string
    >>> from random import choices
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pytorch_widedeep import Tab2Vec
    >>> from pytorch_widedeep.models import TabMlp, WideDeep
    >>> from pytorch_widedeep.preprocessing import TabPreprocessor
    >>>
    >>> colnames = list(string.ascii_lowercase)[:4]
    >>> cat_col1_vals = ["a", "b", "c"]
    >>> cat_col2_vals = ["d", "e", "f"]
    >>>
    >>> # Create the toy input dataframe and a toy dataframe to be vectorised
    >>> cat_inp = [np.array(choices(c, k=5)) for c in [cat_col1_vals, cat_col2_vals]]
    >>> cont_inp = [np.round(np.random.rand(5), 2) for _ in range(2)]
    >>> df_inp = pd.DataFrame(np.vstack(cat_inp + cont_inp).transpose(), columns=colnames)
    >>> cat_t2v = [np.array(choices(c, k=5)) for c in [cat_col1_vals, cat_col2_vals]]
    >>> cont_t2v = [np.round(np.random.rand(5), 2) for _ in range(2)]
    >>> df_t2v = pd.DataFrame(np.vstack(cat_t2v + cont_t2v).transpose(), columns=colnames)
    >>>
    >>> # fit the TabPreprocessor
    >>> embed_cols = [("a", 2), ("b", 4)]
    >>> cont_cols = ["c", "d"]
    >>> tab_preprocessor = TabPreprocessor(cat_embed_cols=embed_cols, continuous_cols=cont_cols)
    >>> X_tab = tab_preprocessor.fit_transform(df_inp)
    >>>
    >>> # define the model (and let's assume we train it)
    >>> tabmlp = TabMlp(
    ... column_idx=tab_preprocessor.column_idx,
    ... cat_embed_input=tab_preprocessor.cat_embed_input,
    ... continuous_cols=tab_preprocessor.continuous_cols,
    ... mlp_hidden_dims=[8, 4])
    >>> model = WideDeep(deeptabular=tabmlp)
    >>> # ...train the model...
    >>>
    >>> # vectorise the dataframe
    >>> t2v = Tab2Vec(tab_preprocessor, model)
    >>> X_vec = t2v.transform(df_t2v)
    """

    def __init__(
        self,
        tab_preprocessor: TabPreprocessor,
        model: Union[WideDeep, BayesianWide, BayesianTabMlp],
        return_dataframe: bool = False,
        verbose: bool = False,
    ):
        super(Tab2Vec, self).__init__()

        self._check_inputs(tab_preprocessor, model, verbose)

        self.tab_preprocessor = tab_preprocessor
        self.return_dataframe = return_dataframe
        self.verbose = verbose

        self.vectorizer = self._set_vectorizer(model)

        self._set_dim_attributes(tab_preprocessor, model)

    def fit(self, df: pd.DataFrame, target_col: Optional[str] = None) -> "Tab2Vec":
        r"""This is an empty method i.e. Returns the unchanged object itself. Is
        only included for consistency in case `Tab2Vec` is used as part of a
        Pipeline

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame to be vectorised, i.e. the categorical and continuous
            columns will be encoded based on the processing applied within
            the model
        target_col: str, Optional
            Column name of the target_col variable. If `None` only the array of
            predictors will be returned

        Returns
        -------
        Tab2Vec
        """

        return self

    def transform(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
    ) -> Union[
        np.ndarray,
        Tuple[np.ndarray, np.ndarray],
        pd.DataFrame,
        Tuple[pd.DataFrame, pd.Series],
    ]:
        r"""Transforms the input dataframe into vectorized form. If a target
        column name is passed the target values will be returned separately
        in their corresponding type (np.ndarray or pd.DataFrame)

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame to be vectorised, i.e. the categorical and continuous
            columns will be encoded based on the processing applied within
            the model
        target_col: str, Optional
            Column name of the target_col variable. If `None` only the array of
            predictors will be returned

        Returns
        -------
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray], pd.DataFrame, Tuple[pd.DataFrame, pd.Series]
            Returns eiter a numpy array with the vectorised values, or a Tuple
            of numpy arrays with the vectorised values and the target. The
            same applies to dataframes in case we choose to set
            `return_dataframe = True`
        """

        X_tab = self.tab_preprocessor.transform(df)
        X = torch.from_numpy(X_tab.astype("float")).to(device)

        with torch.no_grad():
            if self.is_tab_transformer:
                x_vec, x_cont_not_embed = self.vectorizer(X)
            else:
                x_vec = self.vectorizer(X)
                x_cont_not_embed = None

        if self.tab_preprocessor.with_cls_token:
            x_vec = x_vec[:, 1:, :]

        if self.tab_preprocessor.with_attention:
            x_vec = einops.rearrange(x_vec, "s c e -> s (c e)")

        if x_cont_not_embed is not None:
            x_vec = torch.cat([x_vec, x_cont_not_embed], 1).detach().cpu().numpy()
        else:
            x_vec = x_vec.detach().cpu().numpy()

        if self.return_dataframe:
            new_colnames = self._new_colnames()
            if target_col:
                return pd.DataFrame(data=x_vec, columns=new_colnames), df[[target_col]]
            else:
                return pd.DataFrame(data=x_vec, columns=new_colnames)
        else:
            if target_col:
                return x_vec, df[target_col].values
            else:
                return x_vec

    def fit_transform(
        self, df: pd.DataFrame, target_col: Optional[str] = None
    ) -> Union[
        np.ndarray,
        Tuple[np.ndarray, np.ndarray],
        pd.DataFrame,
        Tuple[pd.DataFrame, pd.Series],
    ]:
        r"""Combines `fit` and `transform`"""
        return self.fit(df, target_col).transform(df, target_col)

    def _new_colnames(self) -> List[str]:
        if self.tab_preprocessor.with_attention:
            return self._new_colnames_with_attention()
        else:
            return self._new_colnames_without_attention()

    def _new_colnames_with_attention(
        self,
    ) -> List[str]:
        cat_cols: List[str] = (
            [
                ei[0]
                for ei in self.tab_preprocessor.cat_embed_input
                if ei[0] != "cls_token"
            ]
            if self.tab_preprocessor.cat_embed_input is not None
            else []
        )

        cont_cols: List[str] = (
            self.tab_preprocessor.continuous_cols
            if self.tab_preprocessor.continuous_cols is not None
            else []
        )

        if self.are_cont_embed:
            all_embed_cols = cat_cols + cont_cols
        else:
            all_embed_cols = cat_cols

        new_colnames = []
        for colname in all_embed_cols:
            assert self.input_dim is not None
            new_colnames.extend(
                [
                    "_".join([colname, "embed", str(i + 1)])
                    for i in range(self.input_dim)
                ]
            )

        if not self.are_cont_embed:
            new_colnames += cont_cols

        return new_colnames

    def _new_colnames_without_attention(
        self,
    ) -> List[str]:
        new_colnames: List[str] = []

        if self.tab_preprocessor.cat_embed_input is not None:
            for colname, _, embedding_dim in self.tab_preprocessor.cat_embed_input:  # type: ignore[misc]
                new_colnames.extend(
                    [
                        "_".join([colname, "embed", str(i + 1)])
                        for i in range(embedding_dim)
                    ]
                )

        if self.tab_preprocessor.continuous_cols is not None:
            if self.are_cont_embed:
                for colname in self.tab_preprocessor.continuous_cols:
                    assert self.cont_embed_dim is not None
                    new_colnames.extend(
                        [
                            "_".join([colname, "embed", str(i + 1)])
                            for i in range(self.cont_embed_dim)
                        ]
                    )
            else:
                new_colnames += self.tab_preprocessor.continuous_cols

        return new_colnames

    def _check_inputs(
        self,
        tab_preprocessor: TabPreprocessor,
        model: Union[WideDeep, BayesianWide, BayesianTabMlp],
        verbose: bool,
    ) -> None:
        if not isinstance(model, BaseBayesianModel):
            if verbose:
                if model.deepimage is not None or model.deeptext is not None:
                    warnings.warn(
                        "Currently 'Tab2Vec' is only implemented for the 'deeptabular' component."
                    )

            if model.deeptabular is None:
                raise RuntimeError(
                    "Currently 'Tab2Vec' is only implemented for the 'deeptabular' component."
                )
            if not tab_preprocessor.is_fitted:
                raise RuntimeError(
                    "The 'tab_preprocessor' must be fitted before is passed to 'Tab2Vec'"
                )

    def _set_vectorizer(
        self, model: Union[WideDeep, BayesianWide, BayesianTabMlp]
    ) -> Callable:
        if isinstance(model, BaseBayesianModel):
            vectorizer = model._get_embeddings
            self.is_tab_transformer = False
        elif model.deeptabular[0].__class__.__name__.lower() == "tabtransformer":  # type: ignore
            self.is_tab_transformer = True
            vectorizer = model.deeptabular[0]._get_embeddings_tt  # type: ignore
        else:
            self.is_tab_transformer = False
            vectorizer = model.deeptabular[0]._get_embeddings  # type: ignore
        return vectorizer

    def _set_dim_attributes(
        self,
        tab_preprocessor: TabPreprocessor,
        model: Union[WideDeep, BayesianWide, BayesianTabMlp],
    ) -> None:
        if isinstance(model, BaseBayesianModel):
            self.are_cont_embed: bool = (
                model.embed_continuous if model.embed_continuous else False
            )
            self.input_dim: Optional[int] = None
            self.cont_embed_dim: Optional[int] = model.cont_embed_dim
        else:
            self.are_cont_embed = (
                model.deeptabular[0].embed_continuous_method is not None  # type: ignore
            )
            if tab_preprocessor.with_attention:
                self.input_dim = model.deeptabular[0].input_dim  # type: ignore
                self.cont_embed_dim = None
            else:
                self.input_dim = None
                self.cont_embed_dim = model.deeptabular[0].cont_embed_dim  # type: ignore
