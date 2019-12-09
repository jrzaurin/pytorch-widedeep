## Binary Classification with different optimizers, schedulers, etc.

In this notebook we will use the Adult Census dataset. Download the data from [here](https://www.kaggle.com/wenruliu/adult-income-dataset/downloads/adult.csv/2).


```python
import numpy as np
import pandas as pd
import torch

from pytorch_widedeep.preprocessing import WidePreprocessor, DeepPreprocessor
from pytorch_widedeep.models import Wide, DeepDense, WideDeep
from pytorch_widedeep.metrics import BinaryAccuracy
```


```python
df = pd.read_csv('data/adult/adult.csv.zip')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>educational-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>gender</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>25</td>
      <td>Private</td>
      <td>226802</td>
      <td>11th</td>
      <td>7</td>
      <td>Never-married</td>
      <td>Machine-op-inspct</td>
      <td>Own-child</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <td>1</td>
      <td>38</td>
      <td>Private</td>
      <td>89814</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Farming-fishing</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <td>2</td>
      <td>28</td>
      <td>Local-gov</td>
      <td>336951</td>
      <td>Assoc-acdm</td>
      <td>12</td>
      <td>Married-civ-spouse</td>
      <td>Protective-serv</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <td>3</td>
      <td>44</td>
      <td>Private</td>
      <td>160323</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Married-civ-spouse</td>
      <td>Machine-op-inspct</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>7688</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <td>4</td>
      <td>18</td>
      <td>?</td>
      <td>103497</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Never-married</td>
      <td>?</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>




```python
# For convenience, we'll replace '-' with '_'
df.columns = [c.replace("-", "_") for c in df.columns]
# binary target
df['income_label'] = (df["income"].apply(lambda x: ">50K" in x)).astype(int)
df.drop('income', axis=1, inplace=True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>educational_num</th>
      <th>marital_status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>gender</th>
      <th>capital_gain</th>
      <th>capital_loss</th>
      <th>hours_per_week</th>
      <th>native_country</th>
      <th>income_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>25</td>
      <td>Private</td>
      <td>226802</td>
      <td>11th</td>
      <td>7</td>
      <td>Never-married</td>
      <td>Machine-op-inspct</td>
      <td>Own-child</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>38</td>
      <td>Private</td>
      <td>89814</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Farming-fishing</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
      <td>United-States</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>28</td>
      <td>Local-gov</td>
      <td>336951</td>
      <td>Assoc-acdm</td>
      <td>12</td>
      <td>Married-civ-spouse</td>
      <td>Protective-serv</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>44</td>
      <td>Private</td>
      <td>160323</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Married-civ-spouse</td>
      <td>Machine-op-inspct</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>7688</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>18</td>
      <td>?</td>
      <td>103497</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Never-married</td>
      <td>?</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>United-States</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Preparing the data

Have a look to notebooks one and two if you want to get a good understanding of the next few lines of code (although there is no need to use the package)


```python
wide_cols = ['education', 'relationship','workclass','occupation','native_country','gender']
crossed_cols = [('education', 'occupation'), ('native_country', 'occupation')]
cat_embed_cols = [('education',16), ('relationship',8), ('workclass',16),
    ('occupation',16),('native_country',16)]
continuous_cols = ["age","hours_per_week"]
target_col = 'income_label'
```


```python
# TARGET
target = df[target_col].values

# WIDE
preprocess_wide = WidePreprocessor(wide_cols=wide_cols, crossed_cols=crossed_cols)
X_wide = preprocess_wide.fit_transform(df)

# DEEP
preprocess_deep = DeepPreprocessor(embed_cols=cat_embed_cols, continuous_cols=continuous_cols)
X_deep = preprocess_deep.fit_transform(df)
```


```python
print(X_wide)
print(X_wide.shape)
```

    [[0. 1. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     ...
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]]
    (48842, 796)



```python
print(X_deep)
print(X_deep.shape)
```

    [[ 0.          0.          0.         ...  0.         -0.99512893
      -0.03408696]
     [ 1.          1.          0.         ...  0.         -0.04694151
       0.77292975]
     [ 2.          1.          1.         ...  0.         -0.77631645
      -0.03408696]
     ...
     [ 1.          3.          0.         ...  0.          1.41180837
      -0.03408696]
     [ 1.          0.          0.         ...  0.         -1.21394141
      -1.64812038]
     [ 1.          4.          6.         ...  0.          0.97418341
      -0.03408696]]
    (48842, 7)


As you can see, you can run a wide and deep model in just a few lines of code

Let's now see how to use `WideDeep` with varying parameters

###  2.1 Dropout and Batchnorm


```python
wide = Wide(wide_dim=X_wide.shape[1], output_dim=1)
# We can add dropout and batchnorm to the dense layers
deepdense = DeepDense(hidden_layers=[64,32], dropout=[0.5, 0.5], batchnorm=True,
                      deep_column_idx=preprocess_deep.deep_column_idx,
                      embed_input=preprocess_deep.embeddings_input,
                      continuous_cols=continuous_cols)
model = WideDeep(wide=wide, deepdense=deepdense)
```


```python
model
```




    WideDeep(
      (wide): Wide(
        (wide_linear): Linear(in_features=796, out_features=1, bias=True)
      )
      (deepdense): Sequential(
        (0): DeepDense(
          (embed_layers): ModuleDict(
            (emb_layer_education): Embedding(16, 16)
            (emb_layer_native_country): Embedding(42, 16)
            (emb_layer_occupation): Embedding(15, 16)
            (emb_layer_relationship): Embedding(6, 8)
            (emb_layer_workclass): Embedding(9, 16)
          )
          (embed_dropout): Dropout(p=0.0, inplace=False)
          (dense): Sequential(
            (dense_layer_0): Sequential(
              (0): Linear(in_features=74, out_features=64, bias=True)
              (1): LeakyReLU(negative_slope=0.01, inplace=True)
              (2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (3): Dropout(p=0.5, inplace=False)
            )
            (dense_layer_1): Sequential(
              (0): Linear(in_features=64, out_features=32, bias=True)
              (1): LeakyReLU(negative_slope=0.01, inplace=True)
              (2): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (3): Dropout(p=0.5, inplace=False)
            )
          )
        )
        (1): Linear(in_features=32, out_features=1, bias=True)
      )
    )



We can use different initializers, optimizers and learning rate schedulers for each `branch` of the model

###  Optimizers, LR schedulers, Initializers and Callbacks


```python
from pytorch_widedeep.initializers import KaimingNormal, XavierNormal
from pytorch_widedeep.callbacks import ModelCheckpoint, LRHistory, EarlyStopping
from pytorch_widedeep.optim import RAdam
```


```python
# Optimizers
wide_opt = torch.optim.Adam(model.wide.parameters())
deep_opt = RAdam(model.deepdense.parameters())
# LR Schedulers
wide_sch = torch.optim.lr_scheduler.StepLR(wide_opt, step_size=3)
deep_sch = torch.optim.lr_scheduler.StepLR(deep_opt, step_size=5)
```

the component-dependent settings must be passed as dictionaries, while general settings are simply lists


```python
# Component-dependent settings as Dict
optimizers = {'wide': wide_opt, 'deepdense':deep_opt}
schedulers = {'wide': wide_sch, 'deepdense':deep_sch}
initializers = {'wide': KaimingNormal, 'deepdense':XavierNormal}
# General settings as List
callbacks = [LRHistory(n_epochs=10), EarlyStopping, ModelCheckpoint(filepath='model_weights/wd_out')]
metrics = [BinaryAccuracy]
```


```python
model.compile(method='binary', optimizers=optimizers, lr_schedulers=schedulers, 
              initializers=initializers,
              callbacks=callbacks,
              metrics=metrics)
```


```python
model.fit(X_wide=X_wide, X_deep=X_deep, target=target, n_epochs=10, batch_size=256, val_split=0.2)
```

      0%|          | 0/153 [00:00<?, ?it/s]

    Training


    epoch 1: 100%|██████████| 153/153 [00:03<00:00, 44.53it/s, loss=0.783, metrics={'acc': 0.6151}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 116.60it/s, loss=0.469, metrics={'acc': 0.6493}]
    epoch 2: 100%|██████████| 153/153 [00:03<00:00, 47.32it/s, loss=0.529, metrics={'acc': 0.7565}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 95.90it/s, loss=0.396, metrics={'acc': 0.7685}]
    epoch 3: 100%|██████████| 153/153 [00:03<00:00, 46.55it/s, loss=0.457, metrics={'acc': 0.7907}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 116.63it/s, loss=0.372, metrics={'acc': 0.798}]
    epoch 4: 100%|██████████| 153/153 [00:03<00:00, 49.69it/s, loss=0.421, metrics={'acc': 0.8038}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 128.04it/s, loss=0.366, metrics={'acc': 0.8091}]
    epoch 5: 100%|██████████| 153/153 [00:03<00:00, 50.27it/s, loss=0.398, metrics={'acc': 0.815}] 
    valid: 100%|██████████| 39/39 [00:00<00:00, 131.81it/s, loss=0.36, metrics={'acc': 0.8188}]
    epoch 6: 100%|██████████| 153/153 [00:03<00:00, 50.16it/s, loss=0.388, metrics={'acc': 0.817}] 
    valid: 100%|██████████| 39/39 [00:00<00:00, 130.33it/s, loss=0.36, metrics={'acc': 0.8204}]
    epoch 7: 100%|██████████| 153/153 [00:03<00:00, 50.06it/s, loss=0.386, metrics={'acc': 0.8175}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 133.34it/s, loss=0.359, metrics={'acc': 0.8208}]
    epoch 8: 100%|██████████| 153/153 [00:03<00:00, 50.43it/s, loss=0.387, metrics={'acc': 0.8189}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 124.92it/s, loss=0.359, metrics={'acc': 0.8221}]
    epoch 9: 100%|██████████| 153/153 [00:03<00:00, 50.34it/s, loss=0.385, metrics={'acc': 0.8185}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 130.44it/s, loss=0.358, metrics={'acc': 0.8219}]
    epoch 10: 100%|██████████| 153/153 [00:03<00:00, 50.29it/s, loss=0.384, metrics={'acc': 0.8191}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 129.58it/s, loss=0.358, metrics={'acc': 0.8225}]



```python
dir(model)
```




    ['__call__',
     '__class__',
     '__delattr__',
     '__dict__',
     '__dir__',
     '__doc__',
     '__eq__',
     '__format__',
     '__ge__',
     '__getattr__',
     '__getattribute__',
     '__gt__',
     '__hash__',
     '__init__',
     '__init_subclass__',
     '__le__',
     '__lt__',
     '__module__',
     '__ne__',
     '__new__',
     '__reduce__',
     '__reduce_ex__',
     '__repr__',
     '__setattr__',
     '__setstate__',
     '__sizeof__',
     '__str__',
     '__subclasshook__',
     '__weakref__',
     '_activation_fn',
     '_apply',
     '_backward_hooks',
     '_buffers',
     '_forward_hooks',
     '_forward_pre_hooks',
     '_get_name',
     '_load_from_state_dict',
     '_load_state_dict_pre_hooks',
     '_loss_fn',
     '_lr_scheduler_step',
     '_modules',
     '_named_members',
     '_parameters',
     '_predict',
     '_register_load_state_dict_pre_hook',
     '_register_state_dict_hook',
     '_save_to_state_dict',
     '_slow_forward',
     '_state_dict_hooks',
     '_tracing_name',
     '_train_val_split',
     '_training_step',
     '_validation_step',
     '_version',
     '_warm_up',
     'add_module',
     'apply',
     'batch_size',
     'buffers',
     'callback_container',
     'callbacks',
     'children',
     'class_weight',
     'compile',
     'cpu',
     'cuda',
     'cyclic',
     'deepdense',
     'deephead',
     'deepimage',
     'deeptext',
     'double',
     'dump_patches',
     'early_stop',
     'eval',
     'extra_repr',
     'fit',
     'float',
     'forward',
     'get_embeddings',
     'half',
     'history',
     'initializer',
     'load_state_dict',
     'lr_history',
     'lr_scheduler',
     'method',
     'metric',
     'modules',
     'named_buffers',
     'named_children',
     'named_modules',
     'named_parameters',
     'optimizer',
     'parameters',
     'predict',
     'predict_proba',
     'register_backward_hook',
     'register_buffer',
     'register_forward_hook',
     'register_forward_pre_hook',
     'register_parameter',
     'requires_grad_',
     'seed',
     'share_memory',
     'state_dict',
     'to',
     'train',
     'train_running_loss',
     'training',
     'transforms',
     'type',
     'valid_running_loss',
     'verbose',
     'wide',
     'with_focal_loss',
     'zero_grad']



You see that, among many methods and attributes we have the `history` and `lr_history` attributes


```python
model.history.epoch
```




    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]




```python
print(model.history._history)
```

    {'train_loss': [0.7826161832591287, 0.5294494130253012, 0.45743006565212424, 0.4206276263286865, 0.3982163554702709, 0.3881325295158461, 0.3862898593244989, 0.38681577603801404, 0.38500378529230755, 0.38388273743243], 'train_acc': [0.6151, 0.7565, 0.7907, 0.8038, 0.815, 0.817, 0.8175, 0.8189, 0.8185, 0.8191], 'val_loss': [0.4694176025879689, 0.3960292133001181, 0.37219820802028364, 0.3658289725963886, 0.3600605313594525, 0.35951805343994725, 0.35915129765486103, 0.3585702692851042, 0.3578468553530864, 0.3576407875770178], 'val_acc': [0.6493, 0.7685, 0.798, 0.8091, 0.8188, 0.8204, 0.8208, 0.8221, 0.8219, 0.8225]}



```python
print(model.lr_history)
```

    {'lr_wide_0': [0.001, 0.001, 0.001, 0.0001, 0.0001, 0.0001, 1.0000000000000003e-05, 1.0000000000000003e-05, 1.0000000000000003e-05, 1.0000000000000002e-06], 'lr_deepdense_0': [0.001, 0.001, 0.001, 0.001, 0.001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]}


We can see that the learning rate effectively decreases by a factor of 0.1 (the default) after the corresponding `step_size`. Note that the keys of the dictionary have a suffix `_0`. This is because if you pass different parameter groups to the torch optimizers, these will also be recorded. We'll see this in the `Regression` notebook. 

And I guess one has a good idea of how to use the package. Before we leave this notebook just mentioning that the `WideDeep` class comes with a useful method to "rescue" the learned embeddings. For example, let's say I want to use the embeddings learned for the different levels of the categorical feature `education`


```python
model.get_embeddings(col_name='education', cat_encoding_dict=preprocess_deep.encoding_dict)
```




    {'11th': array([-1.08425401e-01,  5.09871461e-04,  1.25755548e-01, -1.20801523e-01,
            -2.56043434e-01, -3.55644524e-02, -8.66190940e-02, -1.39202878e-01,
             1.11087626e-04,  4.54997361e-01, -2.31609955e-01, -1.36443637e-02,
             8.78131837e-02, -3.07353675e-01, -1.10240346e-02,  6.45920560e-02],
           dtype=float32),
     'HS-grad': array([ 0.19832617,  0.12040217, -0.5314197 ,  0.35005897, -0.15391229,
            -0.22196807,  0.09345723,  0.06745315,  0.25015768,  0.08744714,
             0.24480642, -0.08957793,  0.27947524, -0.26326123, -0.19119193,
            -0.10995993], dtype=float32),
     'Assoc-acdm': array([ 0.06525454, -0.2618052 , -0.09840333,  0.10541438,  0.33471954,
            -0.04292247,  0.10712572,  0.34287837, -0.18687049, -0.13836485,
            -0.1715912 ,  0.15273218, -0.03476759, -0.07450581,  0.56081617,
             0.29201028], dtype=float32),
     'Some-college': array([-0.45491776, -0.17205039,  0.21580465, -0.2539856 ,  0.02358766,
            -0.05496917, -0.01120283,  0.09221312, -0.12831998,  0.17159238,
             0.196605  , -0.2090644 , -0.11193639, -0.18394227, -0.16056207,
             0.02444198], dtype=float32),
     '10th': array([-0.5581912 , -0.20644131,  0.1300292 , -0.10135209,  0.4538276 ,
            -0.27146348,  0.12652951,  0.5233289 ,  0.01145706, -0.05667543,
             0.43509725, -0.74307233,  0.00139265,  0.07225899,  0.0781986 ,
            -0.2610258 ], dtype=float32),
     'Prof-school': array([-6.5744489e-02,  1.3956554e-01,  5.7986474e-01,  2.7874210e-01,
            -2.4446699e-01,  7.9873689e-02, -3.8569799e-01,  2.2757685e-01,
            -3.8109139e-02,  3.3144853e-01, -3.8229354e-02,  2.9802489e-01,
            -1.5467829e-01,  5.4805580e-04, -2.1627106e-01, -2.6592135e-02],
           dtype=float32),
     '7th-8th': array([ 0.10858492,  0.42190084,  0.07536066, -0.11707054,  0.05351719,
             0.32636967,  0.14053936,  0.45679298, -0.2558197 , -0.47910702,
             0.4725715 , -0.0981419 ,  0.3462793 ,  0.07776859, -0.45930195,
             0.12625834], dtype=float32),
     'Bachelors': array([ 0.01805384, -0.10573057,  0.25564098, -0.27709666, -0.16297452,
            -0.1851758 , -0.5702467 , -0.23569717,  0.067039  , -0.28916818,
            -0.22313781, -0.23893505,  0.37708414,  0.17465928, -0.47459307,
             0.04889947], dtype=float32),
     'Masters': array([ 0.11953138,  0.11543513, -0.3954705 ,  0.32583147,  0.23851769,
            -0.6448425 ,  0.00705628,  0.10673986, -0.08305098, -0.10872949,
            -0.46080047, -0.05367367, -0.18693425,  0.14182107, -0.39178014,
            -0.23969549], dtype=float32),
     'Doctorate': array([ 0.04873321, -0.19027464, -0.10777274, -0.17476888,  0.47248197,
            -0.2873778 , -0.29792303, -0.06811561,  0.16541322, -0.17425427,
            -0.09404507,  0.06525683,  0.06408301,  0.38656166,  0.13369907,
             0.10825544], dtype=float32),
     '5th-6th': array([ 0.08566641,  0.03589746,  0.17174615,  0.08747724,  0.2698885 ,
             0.08344392, -0.23652045,  0.31357667,  0.3546634 , -0.29814255,
             0.10943606,  0.45218074, -0.0614133 , -0.31987205,  0.34947518,
             0.07603104], dtype=float32),
     'Assoc-voc': array([-0.07830544,  0.0278313 ,  0.34295908, -0.27213913, -0.20097388,
             0.10972344,  0.14000823, -0.24098383, -0.16614872,  0.19084413,
            -0.02334382,  0.5209352 ,  0.24089335, -0.1350642 , -0.23480216,
            -0.32963687], dtype=float32),
     '9th': array([ 0.12994888,  0.02475524, -0.12875263,  0.0097373 ,  0.38253692,
            -0.2718543 ,  0.13766348,  0.27845392, -0.2036348 , -0.20567507,
            -0.11305337, -0.47028974,  0.07009655, -0.29621345, -0.17303236,
             0.15854478], dtype=float32),
     '12th': array([-0.15079321, -0.26879913, -0.5159767 ,  0.30044943,  0.0295292 ,
            -0.32494095,  0.20975012,  0.35193697, -0.5034315 , -0.14420179,
             0.06113023,  0.22398257,  0.0087006 ,  0.09041765, -0.09754901,
            -0.21647781], dtype=float32),
     '1st-4th': array([-0.3199786 ,  0.10094872, -0.10035568,  0.10014401, -0.09340642,
            -0.00761677,  0.50759906,  0.288856  , -0.18745485,  0.05442255,
             0.6481828 ,  0.18515776,  0.21597311, -0.21534163,  0.01798662,
            -0.22816893], dtype=float32),
     'Preschool': array([ 0.10035816, -0.24015287,  0.00935481,  0.05356123, -0.18744251,
            -0.39735606,  0.03849271, -0.2864288 , -0.10379744,  0.20251973,
             0.14565234, -0.24607188, -0.14268415,  0.1209868 ,  0.04662501,
             0.41015574], dtype=float32)}


