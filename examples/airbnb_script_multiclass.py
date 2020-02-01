import numpy as np
import pandas as pd
import torch

from pytorch_widedeep.preprocessing import WidePreprocessor, DeepPreprocessor
from pytorch_widedeep.models import Wide, DeepDense, WideDeep
from pytorch_widedeep.optim import RAdam
from pytorch_widedeep.initializers import *
from pytorch_widedeep.callbacks import *
from pytorch_widedeep.metrics import *

import pdb

use_cuda = torch.cuda.is_available()

if __name__ == '__main__':

    df = pd.read_csv('data/airbnb/airbnb_sample.csv')

    crossed_cols = (['property_type', 'room_type'],)
    already_dummies = [c for c in df.columns if 'amenity' in c] + ['has_house_rules']
    wide_cols = ['is_location_exact', 'property_type', 'room_type', 'host_gender',
    'instant_bookable'] + already_dummies
    cat_embed_cols = [(c, 16) for c in df.columns if 'catg' in c] + \
        [('neighbourhood_cleansed', 64), ('cancellation_policy', 16)]
    continuous_cols = ['latitude', 'longitude', 'security_deposit', 'extra_people']
    already_standard = ['latitude', 'longitude']
    df['yield_cat'] = pd.cut(df['yield'], bins=[0.2, 65, 163, 600], labels=[0,1,2])
    df.drop('yield', axis=1, inplace=True)
    target = 'yield_cat'

    target = np.array(df[target].values)
    prepare_wide = WidePreprocessor(wide_cols=wide_cols, crossed_cols=crossed_cols)
    X_wide = prepare_wide.fit_transform(df)
    prepare_deep = DeepPreprocessor(embed_cols=cat_embed_cols, continuous_cols=continuous_cols)
    X_deep = prepare_deep.fit_transform(df)
    wide = Wide(
        wide_dim=X_wide.shape[1],
        output_dim=3)
    deepdense = DeepDense(
        hidden_layers=[64,32],
        dropout=[0.2,0.2],
        deep_column_idx=prepare_deep.deep_column_idx,
        embed_input=prepare_deep.embeddings_input,
        continuous_cols=continuous_cols)
    model = WideDeep(wide=wide, deepdense=deepdense, output_dim=3)
    model.compile(method='multiclass', metrics=[CategoricalAccuracy])
    model.fit(
        X_wide=X_wide,
        X_deep=X_deep,
        target=target,
        n_epochs=1,
        batch_size=32,
        val_split=0.2)