import pandas as pd

from typing import List, Dict, Optional

def label_encode(df_inp:pd.DataFrame, cols:Optional[List[str]]=None,
    val_to_idx:Optional[Dict[str,Dict[str,int]]]=None):
    """
    Helper function to label-encode some features of a given dataset.

    Parameters:
    -----------
    df_inp: pd.Dataframe
        input dataframe
    cols: List
        optional - columns to be label-encoded
    val_to_idx: Dict
        optional - dictionary with the encodings

    Returns:
    --------
    df: pd.Dataframe
        df with Label-encoded features.
    val_to_idx: Dict
        Dictionary with the encoding information
    """
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
