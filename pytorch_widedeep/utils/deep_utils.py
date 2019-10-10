import numpy as np
import pandas as pd
import warnings

from sklearn.preprocessing import StandardScaler

from ..wdtypes import *

pd.options.mode.chained_assignment = None


def label_encode(df_inp:pd.DataFrame, cols:Optional[List[str]]=None,
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


class PrepareDeep(object):
    def __init__(self,
        embed_cols:List[Union[str,Tuple[str,int]]]=None,
        continuous_cols:List[str]=None,
        already_standard:Optional[List[str]]=None,
        scale:bool=True,
        default_embed_dim:int=8):
        super(PrepareDeep, self).__init__()

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

    def fit(self, df:pd.DataFrame)->np.ndarray:
        if self.embed_cols is not None:
            df_deep = self._prepare_embed(df)
            df_deep, self.encoding_dict = label_encode(df_deep, cols=df_deep.columns.tolist())
            self.embeddings_input = []
            for k,v in self.encoding_dict.items():
                self.embeddings_input.append((k, len(v), self.embed_dim[k]))

        if self.continuous_cols is not None:
            df_cont = self._prepare_continuous(df)
            if self.scale:
                df_std = df_cont[self.standardize_cols]
                self.scaler = StandardScaler().fit(df_std.values)
                df_cont[self.standardize_cols] = self.scaler.transform(df_std.values)
            else:
                warnings.warn('Continuous columns will not be normalised')
        try:
            df_deep = pd.concat([df_deep, df_cont], axis=1)
        except:
            df_deep = df_cont.copy()

        return df_deep.values

    def transform(self, df:pd.DataFrame)->np.ndarray:

        if self.embed_cols is not None:
            df_deep = self._prepare_embed(df)
            df_deep, _ = label_encode(df_deep, cols=df_deep.columns.tolist(),
                val_to_idx=self.encoding_dict)

        if self.continuous_cols is not None:
            df_cont = self._prepare_continuous(df)
            if self.scale:
                df_std = df_cont[self.standardize_cols]
                df_cont[self.standardize_cols] = self.scaler.transform(df_std.values)
        try:
            df_deep = pd.concat([df_deep, df_cont], axis=1)
        except:
            df_deep = df_cont.copy()

        return df_deep.values

    def fit_transform(self, df:pd.DataFrame)->np.ndarray:
        return self.fit(df)


# def prepare_deep(df:pd.DataFrame,
#     embed_cols:List[Union[str, Tuple[str,int]]]=None,
#     cat_encodings:Optional[Dict[str,Dict[str,int]]]=None,
#     continuous_cols:List[str]=None,
#     already_standard:Optional[List[str]]=None, scale:bool=True,
#     default_embed_dim:int=8):

#     assert (embed_cols is not None) or (continuous_cols is not None), \
#     'Either the embedding columns or continuous columns must not be passed'

#     # set the categorical columns that will be represented by embeddings
#     if embed_cols is not None:
#         if isinstance(embed_cols[0], Tuple):
#             embed_dim = dict(embed_cols)
#             embed_coln = [emb[0] for emb in embed_cols]
#         else:
#             embed_dim = {e:default_embed_dim for e in embed_cols}
#             embed_coln = embed_cols
#         df_deep = df.copy()[embed_coln]
#         df_deep, encoding_dict = label_encode(df_deep, cols=embed_coln, val_to_idx=cat_encodings)
#         embeddings_input = []
#         for k,v in encoding_dict.items():
#             embeddings_input.append((k, len(v), embed_dim[k]))
#     else:
#         embeddings_input, encoding_dict = None, None

#     # set the continous columns
#     if continuous_cols is not None:
#         df_cont = df.copy()[continuous_cols]
#         if scale:
#             scaler = StandardScaler()
#             if already_standard is not None:
#                 standardize_cols = [c for c in continuous_cols if c not in already_standard]
#             else: standardize_cols = continuous_cols
#             for cc in standardize_cols:
#                 df_cont[cc]  = scaler.fit_transform(df_cont[cc].values.reshape(-1,1).astype(float))
#         else:
#             warnings.warn('Continuous columns will not be normalised')
#         try:
#             df_deep = pd.concat([df_deep, df_cont], axis=1)
#         except:
#             df_deep = df_cont.copy()

#     if not 'scaler' in locals(): scaler=None
#     deep_column_idx = {k:v for v,k in enumerate(df_deep.columns)}

#     return df_deep.values, embeddings_input, encoding_dict, deep_column_idx, scaler


# def label_encode(df_inp:pd.DataFrame, cols:Optional[List[str]]=None,
#     val_to_idx:Optional[Dict[str,Dict[str,int]]]=None):

#     df = df_inp.copy()
#     if cols == None:
#         cols = list(df.select_dtypes(include=['object']).columns)

#     if not val_to_idx:
#         val_types = dict()
#         for c in cols:
#             val_types[c] = df[c].unique()
#         val_to_idx = dict()
#         for k, v in val_types.items():
#             val_to_idx[k] = {o: i for i, o in enumerate(val_types[k])}

#     for k, v in val_to_idx.items():
#         df[k] = df[k].apply(lambda x: v[x])

#     return df, val_to_idx