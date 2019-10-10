import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

from ..wdtypes import *


class PrepareWide(object):
    def __init__(self, wide_cols:List[str], crossed_cols=None,
        already_dummies:Optional[List[str]]=None):
        super(PrepareWide, self).__init__()
        self.wide_cols = wide_cols
        self.crossed_cols = crossed_cols
        self.already_dummies = already_dummies
        self.one_hot_enc = OneHotEncoder(sparse=False)

    def cross_cols(self, df:pd.DataFrame):
        crossed_colnames = []
        for cols in self.crossed_cols:
            cols = list(cols)
            for c in cols: df[c] = df[c].astype('str')
            colname = '_'.join(cols)
            df[colname] = df[cols].apply(lambda x: '-'.join(x), axis=1)
            crossed_colnames.append(colname)
        return df, crossed_colnames

    def fit(self, df:pd.DataFrame)->np.ndarray:
        df_wide = df.copy()[self.wide_cols]
        if self.crossed_cols is not None:
            df_wide, crossed_colnames = self.cross_cols(df_wide)
            self.wide_crossed_cols = self.wide_cols + crossed_colnames
        else:
            self.wide_crossed_cols = self.wide_cols

        if self.already_dummies:
            X_oh_1 = self.df_wide[self.already_dummies].values
            dummy_cols = [c for c in self.wide_crossed_cols if c not in self.already_dummies]
            X_oh_2=self.one_hot_enc.fit_transform(df_wide[dummy_cols])
            return np.hstack((X_oh_1, X_oh_2))
        else:
            return self.one_hot_enc.fit_transform(df_wide[self.wide_crossed_cols])

    def transform(self, df:pd.DataFrame)->np.ndarray:
        df_wide = df.copy()[self.wide_cols]
        if self.crossed_cols is not None:
            df_wide, _ = self.cross_cols(df_wide)

        if self.already_dummies:
            X_oh_1 = self.df_wide[self.already_dummies].values
            dummy_cols = [c for c in self.wide_crossed_cols if c not in self.already_dummies]
            X_oh_2=self.one_hot_enc.transform(df_wide[dummy_cols])
            return np.hstack((X_oh_1, X_oh_2))
        else:
            return self.one_hot_enc.transform(df_wide[self.wide_crossed_cols])

    def fit_transform(self, df:pd.DataFrame)->np.ndarray:
        return self.fit(df)