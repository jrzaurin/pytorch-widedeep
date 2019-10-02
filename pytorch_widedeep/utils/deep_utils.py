import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from ..wdtypes import *

pd.options.mode.chained_assignment = None


def prepare_deep(df:pd.DataFrame, embed_cols:List[Union[str, Tuple[str,int]]],
    continuous_cols:List[str], already_standard:Optional[List[str]]=None, scale:bool=True,
    default_embed_dim:int=8):

    if isinstance(embed_cols[0], Tuple):
        embed_dim = dict(embed_cols)
        embed_coln = [emb[0] for emb in embed_cols]
    else:
        embed_dim = {e:default_embed_dim for e in embed_cols}
        embed_coln = embed_cols
    deep_cols = embed_coln + continuous_cols

    df_deep = df.copy()[deep_cols]
    df_deep, encoding_dict = label_encode(df_deep, cols=embed_coln)
    embeddings_input = []
    for k,v in encoding_dict.items():
        embeddings_input.append((k, len(v), embed_dim[k]))
    deep_column_idx = {k:v for v,k in enumerate(df_deep.columns)}

    if scale:
        scaler = StandardScaler()
        if already_standard is not None:
            standardize_cols = [c for c in continuous_cols if c not in already_standard]
        else: standardize_cols = continuous_cols
        for cc in standardize_cols:
            df_deep[cc]  = scaler.fit_transform(df_deep[cc].values.reshape(-1,1).astype(float))

    return df_deep.values, embeddings_input, encoding_dict, deep_column_idx


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