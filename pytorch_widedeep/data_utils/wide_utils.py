import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from ..wdtypes import *

def prepare_wide(df:pd.DataFrame, wide_cols:List[str],
    crossed_cols:List[Tuple[str,str]], already_dummies:Optional[List[str]]=None):
    """
    Function to prepare the features that will be passed through the "Wide" model.

    Parameters:
    ----------
    df: pd.Dataframe
    target: str
    wide_cols: List
        List with the name of the columns that will be one-hot encoded and
        pass through the Wide model
    crossed_cols: List
        List of Tuples with the name of the columns that will be "crossed"
        and then one-hot encoded. e.g. (['education', 'occupation'], ...)
    already_dummies: List
        List of columns that are already dummies/one-hot encoded

    Returns:
    df_wide.values: np.ndarray
        values that will be passed through the Wide Model
    """

    df_wide = df.copy()[wide_cols]

    crossed_columns = []
    for cols in crossed_cols:
        colname = '_'.join(cols)
        df_wide[colname] = df_wide[cols].apply(lambda x: '-'.join(x), axis=1)
        crossed_columns.append(colname)

    if already_dummies:
        dummy_cols = [c for c in wide_cols+crossed_columns if c not in already_dummies]
    else:
        dummy_cols = wide_cols+crossed_columns
    df_wide = pd.get_dummies(df_wide, columns=dummy_cols)

    return df_wide.values
