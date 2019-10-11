import numpy as np
import pandas as pd
import torch
from pathlib import Path

from pytorch_widedeep.utils.wide_utils import WideProcessor
from pytorch_widedeep.utils.deep_utils import DeepProcessor

from pytorch_widedeep.models.wide import Wide
from pytorch_widedeep.models.deep_dense import DeepDense

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

    prepare_wide = WideProcessor(wide_cols=wide_cols, crossed_cols=crossed_cols)
    X_wide = prepare_wide.fit_transform(df)

    prepare_deep = DeepProcessor(embed_cols=cat_embed_cols, continuous_cols=continuous_cols)
    X_deep = prepare_deep.fit_transform(df)

    wide = Wide(X_wide.shape[1], 1)
    pred_wide = wide(torch.tensor(X_wide[:10]))

    deep = DeepDense(
        hidden_layers=[32,16],
        dropout=[0.5],
        deep_column_idx=prepare_deep.deep_column_idx,
        embed_input=prepare_deep.embeddings_input,
        continuous_cols=continuous_cols,
        batchnorm=True,
        output_dim=1)
    pred_deep = deep(torch.tensor(X_deep[:10]))
    pdb.set_trace()


    # wd_dataset = prepare_data(df,
    #     target=target,
    #     wide_cols=wide_cols,
    #     crossed_cols=crossed_cols,
    #     cat_embed_cols=cat_embed_cols,
    #     continuous_cols=continuous_cols)

    # model = WideDeep(
    #     output_dim=1,
    #     wide_dim=wd_dataset.wide.shape[1],
    #     cat_embed_input = wd_dataset.cat_embed_input,
    #     continuous_cols=wd_dataset.continuous_cols,
    #     deep_column_idx=wd_dataset.deep_column_idx)

    # initializers = {'wide': Normal, 'deepdense':Normal}
    # optimizers = {'wide': Adam, 'deepdense':RAdam(lr=0.001)}
    # schedulers = {'wide': StepLR(step_size=5), 'deepdense':StepLR(step_size=5)}

    # callbacks = [EarlyStopping, ModelCheckpoint(filepath='../model_weights/wd_out.pt')]
    # metrics = [BinaryAccuracy]

    # model.compile(
    #     method='logistic',
    #     initializers=initializers,
    #     optimizers=optimizers,
    #     lr_schedulers=schedulers,
    #     callbacks=callbacks,
    #     metrics=metrics)

    # model.fit(
    #     X_wide=wd_dataset.wide,
    #     X_deep=wd_dataset.deepdense,
    #     target=wd_dataset.target,
    #     n_epochs=5,
    #     batch_size=256,
    #     val_split=0.2)