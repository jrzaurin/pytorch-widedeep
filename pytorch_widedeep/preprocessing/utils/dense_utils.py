import numpy as np
import pandas as pd

from ...wdtypes import *


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