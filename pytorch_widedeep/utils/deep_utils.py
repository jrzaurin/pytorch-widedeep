import numpy as np
import pandas as pd
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .base_util import DataProcessor
from ..wdtypes import *

pd.options.mode.chained_assignment = None


def label_encoder(df_inp:pd.DataFrame, cols:Optional[List[str]]=None,
    val_to_idx:Optional[Dict[str,Dict[str,int]]]=None):

    df = df_inp.copy()
    if cols == None:
        cols = list(df.select_dtypes(include=['object']).columns)

    if not val_to_idx:
        val_types = dict()
        for c in cols:
            val_types[c] = df[c].unique()
        val_to_idx = dict()
        for k, v in val_types.items():
            val_to_idx[k] = {o: i for i, o in enumerate(val_types[k])}

    for k, v in val_to_idx.items():
        df[k] = df[k].apply(lambda x: v[x])

    return df, val_to_idx


class DeepProcessor(DataProcessor):
    def __init__(self,
        embed_cols:List[Union[str,Tuple[str,int]]]=None,
        continuous_cols:List[str]=None,
        already_standard:Optional[List[str]]=None,
        scale:bool=True,
        default_embed_dim:int=8):
        super(DeepProcessor, self).__init__()

        self.embed_cols=embed_cols
        self.continuous_cols=continuous_cols
        self.already_standard=already_standard
        self.scale=scale
        self.default_embed_dim=default_embed_dim

        assert (self.embed_cols is not None) or (self.continuous_cols is not None), \
        'Either the embedding columns or continuous columns must not be passed'

    def _prepare_embed(self, df:pd.DataFrame)->pd.DataFrame:
        if isinstance(self.embed_cols[0], Tuple):
            self.embed_dim = dict(self.embed_cols)
            embed_colname = [emb[0] for emb in self.embed_cols]
        else:
            self.embed_dim = {e:self.default_embed_dim for e in self.embed_cols}
            embed_colname = self.embed_cols
        return df.copy()[embed_colname]

    def _prepare_continuous(self, df:pd.DataFrame)->pd.DataFrame:
        if self.scale:
            if self.already_standard is not None:
                self.standardize_cols = [c for c in self.continuous_cols if c not in self.already_standard]
            else: self.standardize_cols = self.continuous_cols
        return df.copy()[self.continuous_cols]

    def fit(self, df:pd.DataFrame)->DataProcessor:
        if self.embed_cols is not None:
            df_emb = self._prepare_embed(df)
            _, self.encoding_dict = label_encoder(df_emb, cols=df_emb.columns.tolist())
            self.embeddings_input = []
            for k,v in self.encoding_dict.items():
                self.embeddings_input.append((k, len(v), self.embed_dim[k]))
        if self.continuous_cols is not None:
            df_cont = self._prepare_continuous(df)
            if self.scale:
                df_std = df_cont[self.standardize_cols]
                self.scaler = StandardScaler().fit(df_std.values)
            else:
                warnings.warn('Continuous columns will not be normalised')
        return self

    def transform(self, df:pd.DataFrame)->np.ndarray:
        if self.embed_cols is not None:
            df_emb = self._prepare_embed(df)
            df_emb, _ = label_encoder(df_emb, cols=df_emb.columns.tolist(),
                val_to_idx=self.encoding_dict)
        if self.continuous_cols is not None:
            df_cont = self._prepare_continuous(df)
            if self.scale:
                check_is_fitted(self.scaler, 'mean_')
                df_std = df_cont[self.standardize_cols]
                df_cont[self.standardize_cols] = self.scaler.transform(df_std.values)
        try:
            df_deep = pd.concat([df_emb, df_cont], axis=1)
        except:
            try:
                df_deep = df_emb.copy()
            except:
                df_deep = df_cont.copy()
        return df_deep.values

    def fit_transform(self, df:pd.DataFrame)->np.ndarray:
        return self.fit(df).transform(df)