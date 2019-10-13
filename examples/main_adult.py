import numpy as np
import pandas as pd
import torch
from pathlib import Path

from pytorch_widedeep.utils.wide_utils import WideProcessor
from pytorch_widedeep.utils.deep_utils import DeepProcessor

from pytorch_widedeep.models.wide import Wide
from pytorch_widedeep.models.deep_dense import DeepDense

from pytorch_widedeep.models.wide_deep import WideDeep

from pytorch_widedeep.initializers import *
from pytorch_widedeep.optimizers import *
from pytorch_widedeep.lr_schedulers import *
from pytorch_widedeep.callbacks import *
from pytorch_widedeep.metrics import *

# use_cuda = torch.cuda.is_available()

import pdb

if __name__ == '__main__':

    DATA_PATH = Path('../data')
    df = pd.read_csv(DATA_PATH/'adult/adult.csv.zip')
    df.columns = [c.replace("-", "_") for c in df.columns]
    df['age_buckets'] = pd.cut(df.age, bins=[16, 25, 30, 35, 40, 45, 50, 55, 60, 91], labels=np.arange(9))
    df['income_label'] = (df["income"].apply(lambda x: ">50K" in x)).astype(int)
    df.drop('income', axis=1, inplace=True)
    df.head()

    wide_cols = ['age_buckets', 'education', 'relationship','workclass','occupation',
        'native_country','gender']
    crossed_cols = [('education', 'occupation'), ('native_country', 'occupation')]
    cat_embed_cols = [('education',10), ('relationship',8), ('workclass',10),
        ('occupation',10),('native_country',10)]
    continuous_cols = ["age","hours_per_week"]
    target = 'income_label'

    target = df[target].values
    prepare_wide = WideProcessor(wide_cols=wide_cols, crossed_cols=crossed_cols)
    X_wide = prepare_wide.fit_transform(df)
    prepare_deep = DeepProcessor(embed_cols=cat_embed_cols, continuous_cols=continuous_cols)
    X_deep = prepare_deep.fit_transform(df)

    wide = Wide(
        wide_dim=X_wide.shape[1],
        output_dim=1)
    deepdense = DeepDense(
        hidden_layers=[32,16],
        dropout=[0.5],
        deep_column_idx=prepare_deep.deep_column_idx,
        embed_input=prepare_deep.embeddings_input,
        continuous_cols=continuous_cols,
        output_dim=1)
    model = WideDeep(wide=wide, deepdense=deepdense)

    initializers = {'wide': Normal, 'deepdense':Normal}
    optimizers = {'wide': Adam, 'deepdense':RAdam(lr=0.001)}
    schedulers = {'wide': StepLR(step_size=5), 'deepdense':StepLR(step_size=5)}

    callbacks = [EarlyStopping, ModelCheckpoint(filepath='../model_weights/wd_out')]
    metrics = [BinaryAccuracy]

    model.compile(
        method='logistic',
        initializers=initializers,
        optimizers=optimizers,
        lr_schedulers=schedulers,
        callbacks=callbacks,
        metrics=metrics)

    model.fit(
        X_wide=X_wide,
        X_deep=X_deep,
        target=target,
        n_epochs=10,
        batch_size=256,
        val_split=0.2)
    pdb.set_trace()
