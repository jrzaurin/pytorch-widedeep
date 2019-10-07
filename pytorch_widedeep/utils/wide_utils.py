import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from ..wdtypes import *


def prepare_wide(df:pd.DataFrame, wide_cols:List[str], crossed_cols,
    already_dummies:Optional[List[str]]=None) -> np.ndarray:

    df_wide = df.copy()[wide_cols]

    crossed_columns = []
    for cols in crossed_cols:
        cols = list(cols)
        colname = '_'.join(cols)
        df_wide[colname] = df_wide[cols].apply(lambda x: '-'.join(x), axis=1)
        crossed_columns.append(colname)

    if already_dummies:
        dummy_cols = [c for c in wide_cols+crossed_columns if c not in already_dummies]
    else:
        dummy_cols = wide_cols+crossed_columns
    df_wide = pd.get_dummies(df_wide, columns=dummy_cols)

    return df_wide.values