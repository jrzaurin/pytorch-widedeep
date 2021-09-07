import warnings
from copy import deepcopy

import numpy as np
import torch
import einops
import pandas as pd

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.preprocessing import TabPreprocessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Tab2Vec:
    r"""Class to transform an input dataframe into vectorized form.

    This class will take an input dataframe in the form of the dataframe used
    for training, and it will turn it into a vectorised form based on the
    processing applied to the categorical and continuous columns.

    .. note:: Currently this class is only implemented for the deeptabular component.
        Therefore, if the input dataframe has a text column or a column with
        the path to images, these will be ignored. We will be adding these
        functionalities in future versions

    Parameters
    ----------
    model: ``WideDeep``
        ``WideDeep`` model. Must be trained.
        See :obj:`pytorch-widedeep.models.wide_deep.WideDeep`
    tab_preprocessor: ``TabPreprocessor``
        ``TabPreprocessor`` object. Must be fitted.
        See :obj:`pytorch-widedeep.preprocessing.tab_preprocessor.TabPreprocessor`

    Attributes
    ----------
    vectorizer: ``nn.Module``
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
    >>> tab_preprocessor = TabPreprocessor(embed_cols=embed_cols, continuous_cols=cont_cols)
    >>> X_tab = tab_preprocessor.fit_transform(df_inp)
    >>>
    >>> # define the model (and let's assume we train it)
    >>> tabmlp = TabMlp(
    ... column_idx=tab_preprocessor.column_idx,
    ... embed_input=tab_preprocessor.embeddings_input,
    ... continuous_cols=tab_preprocessor.continuous_cols,
    ... mlp_hidden_dims=[8, 4])
    >>> model = WideDeep(deeptabular=tabmlp)
    >>> # ...train the model...
    >>>
    >>> # vectorise the dataframe
    >>> t2v = Tab2Vec(model, tab_preprocessor)
    >>> X_vec = t2v.transform(df_t2v)
    """

    def __init__(
        self, model: WideDeep, tab_preprocessor: TabPreprocessor, verbose: bool = False
    ):
        super(Tab2Vec, self).__init__()

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

        self.tab_preprocessor = tab_preprocessor

        transformer_family = [
            "tabtransformer",
            "saint",
            "fttransformer",
            "tabperceiver",
            "tabfastformer",
        ]
        self.is_transformer = (
            model.deeptabular[0].__class__.__name__.lower() in transformer_family  # type: ignore[index]
        )
        self.vectorizer = (
            deepcopy(model.deeptabular[0].cat_embed_and_cont)  # type: ignore[index]
            if not self.is_transformer
            else deepcopy(model.deeptabular[0].cat_and_cont_embed)  # type: ignore[index]
        )
        self.vectorizer.to(device)

    def fit(self, df: pd.DataFrame, target_col: Optional[str] = None) -> "Tab2Vec":
        r"""Empty method. Returns the object itself. Is only included for
        consistency in case ``Tab2Vec`` is used as part of a Pipeline
        """
        return self

    def transform(
        self, df: pd.DataFrame, target_col: Optional[str] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        r"""
        Parameters
        ----------
        df: pd.DataFrame
            DataFrame to be vectorised, i.e. the categorical and continuous
            columns will be encoded based on the processing applied within
            the model
        target_col: str, Optional
            Column name of the target_col variable. If 'None' only the array of
            predictors will be returned
        """

        X_tab = self.tab_preprocessor.transform(df)
        X = torch.from_numpy(X_tab).to(device)

        with torch.no_grad():
            x_cat, x_cont = self.vectorizer(X)

        if self.tab_preprocessor.with_cls_token:
            x_cat = x_cat[:, 1:, :]

        if self.is_transformer:
            x_cat = einops.rearrange(x_cat, "s c e -> s (c e)")
            if len(list(x_cont.shape)) == 3:
                x_cont = einops.rearrange(x_cont, "s c e -> s (c e)")

        X_vec = torch.cat([x_cat, x_cont], 1).cpu().numpy()
        if target_col:
            return X_vec, df[target_col].values
        else:
            return X_vec

    def fit_transform(
        self, df: pd.DataFrame, target_col: Optional[str] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        r"""Combines ``fit`` and ``transform``"""
        return self.fit(df, target_col).transform(df, target_col)
