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
cat_embed_cols = [('education',16), ('relationship',8), ('workclass',16), ('occupation',16),('native_country',16)]
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


    epoch 1: 100%|██████████| 153/153 [00:03<00:00, 47.06it/s, loss=0.731, metrics={'acc': 0.6468}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 118.07it/s, loss=0.418, metrics={'acc': 0.6785}]
    epoch 2: 100%|██████████| 153/153 [00:03<00:00, 49.72it/s, loss=0.51, metrics={'acc': 0.7637}] 
    valid: 100%|██████████| 39/39 [00:00<00:00, 114.38it/s, loss=0.376, metrics={'acc': 0.7765}]
    epoch 3: 100%|██████████| 153/153 [00:03<00:00, 49.32it/s, loss=0.448, metrics={'acc': 0.7927}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 113.23it/s, loss=0.361, metrics={'acc': 0.8007}]
    epoch 4: 100%|██████████| 153/153 [00:03<00:00, 48.23it/s, loss=0.413, metrics={'acc': 0.8079}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 118.17it/s, loss=0.355, metrics={'acc': 0.8132}]
    epoch 5: 100%|██████████| 153/153 [00:03<00:00, 48.27it/s, loss=0.395, metrics={'acc': 0.8149}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 114.71it/s, loss=0.352, metrics={'acc': 0.8191}]
    epoch 6: 100%|██████████| 153/153 [00:03<00:00, 48.97it/s, loss=0.387, metrics={'acc': 0.8157}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 116.88it/s, loss=0.352, metrics={'acc': 0.8199}]
    epoch 7: 100%|██████████| 153/153 [00:03<00:00, 48.56it/s, loss=0.388, metrics={'acc': 0.8153}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 116.33it/s, loss=0.352, metrics={'acc': 0.8195}]
    epoch 8: 100%|██████████| 153/153 [00:03<00:00, 48.54it/s, loss=0.383, metrics={'acc': 0.8184}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 93.05it/s, loss=0.351, metrics={'acc': 0.822}] 
    epoch 9: 100%|██████████| 153/153 [00:03<00:00, 45.71it/s, loss=0.385, metrics={'acc': 0.8196}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 110.48it/s, loss=0.351, metrics={'acc': 0.8229}]
    epoch 10: 100%|██████████| 153/153 [00:03<00:00, 48.44it/s, loss=0.382, metrics={'acc': 0.8194}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 114.40it/s, loss=0.35, metrics={'acc': 0.8228}]



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

    {'train_loss': [0.7313813343157176, 0.5101876866583731, 0.44813506724008545, 0.41332343941420513, 0.3945406624694276, 0.3871746306715448, 0.3884129401515512, 0.38312816230300206, 0.3847907395923839, 0.3817657043341718], 'train_acc': [0.6468, 0.7637, 0.7927, 0.8079, 0.8149, 0.8157, 0.8153, 0.8184, 0.8196, 0.8194], 'val_loss': [0.41844800649545133, 0.3759944920356457, 0.36132928041311413, 0.3554159953044011, 0.3523857922126085, 0.3518377657120044, 0.35156664175864977, 0.35120767278549, 0.35089820012068135, 0.35047405576094603], 'val_acc': [0.6785, 0.7765, 0.8007, 0.8132, 0.8191, 0.8199, 0.8195, 0.822, 0.8229, 0.8228]}



```python
print(model.lr_history)
```

    {'lr_wide_0': [0.001, 0.001, 0.001, 0.0001, 0.0001, 0.0001, 1.0000000000000003e-05, 1.0000000000000003e-05, 1.0000000000000003e-05, 1.0000000000000002e-06], 'lr_deepdense_0': [0.001, 0.001, 0.001, 0.001, 0.001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]}


We can see that the learning rate effectively decreases by a factor of 0.1 (the default) after the corresponding `step_size`. Note that the keys of the dictionary have a suffix `_0`. This is because if you pass different parameter groups to the torch optimizers, these will also be recorded. We'll see this in the `Regression` notebook. 

And I guess one has a good idea of how to use the package. Before we leave this notebook just mentioning that the `WideDeep` class comes with a useful method to "rescue" the learned embeddings. For example, let's say I want to use the embeddings learned for the different levels of the categorical feature `education`


```python
model.get_embeddings(col_name='education', cat_encoding_dict=preprocess_deep.encoding_dict)
```




    {'11th': array([-0.04807916,  0.21404432,  0.12517522, -0.154123  ,  0.06864536,
             0.00092955, -0.38516527, -0.18440197,  0.15861034,  0.12012056,
             0.55413646, -0.16920644,  0.1356924 , -0.37921003,  0.53833497,
             0.08743049], dtype=float32),
     'HS-grad': array([ 0.37504154,  0.34191516,  0.27299362,  0.22921972,  0.07420117,
             0.34922913,  0.19239122, -0.42343035, -0.845824  , -0.07287297,
             0.27455565,  0.19505064,  0.07062761, -0.5201107 ,  0.37823108,
             0.46134958], dtype=float32),
     'Assoc-acdm': array([ 0.22331461,  0.15005238,  0.13472553, -0.16886246, -0.12053325,
            -0.04233408, -0.08905135, -0.54481906,  0.24300168, -0.21069968,
            -0.00685616, -0.38423738, -0.00281451,  0.10599079, -0.05224385,
             0.2891064 ], dtype=float32),
     'Some-college': array([-0.09498356, -0.16801773, -0.09181987,  0.05381393, -0.03607363,
            -0.05759075,  0.09382061,  0.33274302, -0.11906563,  0.14481838,
            -0.1765725 ,  0.20070277,  0.2960993 , -0.02055654,  0.26645136,
             0.4075843 ], dtype=float32),
     '10th': array([-0.12961714, -0.27546212,  0.24345328, -0.24318363,  0.31552687,
             0.16653115, -0.05234893,  0.06825106,  0.2388588 ,  0.10887478,
            -0.12004007, -0.00373614, -0.0223387 ,  0.133562  ,  0.29672143,
             0.03046475], dtype=float32),
     'Prof-school': array([-0.1589678 , -0.07629952,  0.00763621,  0.13788143,  0.4114019 ,
             0.07717889, -0.17072953,  0.29419565, -0.18929462, -0.09182461,
            -0.08409152,  0.01395322, -0.20351669,  0.18333136, -0.03983613,
            -0.31888708], dtype=float32),
     '7th-8th': array([ 0.39654806,  0.26095334, -0.3147828 , -0.41267306, -0.23983437,
            -0.08034727,  0.4807234 ,  0.3054779 , -0.3085564 , -0.07860225,
            -0.1279486 , -0.2846014 ,  0.1358583 ,  0.24006395, -0.18911272,
            -0.2299538 ], dtype=float32),
     'Bachelors': array([ 0.35242578, -0.03246311,  0.15835243, -0.06434399,  0.03403192,
             0.0088449 ,  0.00627425, -0.31485453, -0.30984947, -0.23008366,
            -0.09467663,  0.17246258, -0.09432375,  0.07691337,  0.70925283,
             0.18795769], dtype=float32),
     'Masters': array([-0.14503758,  0.0048258 ,  0.58242404,  0.28511924, -0.13773848,
             0.35109136,  0.05824559,  0.3609631 ,  0.4700086 ,  0.4251728 ,
            -0.2538366 , -0.00297809,  0.1424264 , -0.12481072, -0.09403807,
             0.00634856], dtype=float32),
     'Doctorate': array([-0.12487873, -0.1699961 ,  0.2220065 , -0.04808738,  0.09443628,
            -0.21019349, -0.23745097,  0.28523713,  0.05516997, -0.04004707,
             0.3316393 ,  0.18710822,  0.4153885 , -0.12905155,  0.03055826,
             0.0664137 ], dtype=float32),
     '5th-6th': array([ 0.21891987, -0.13600409, -0.03123563,  0.16288632, -0.03479639,
            -0.4221951 ,  0.4688111 ,  0.08145971, -0.29254073,  0.18396533,
            -0.20204993, -0.03327556, -0.2558647 ,  0.56448   , -0.30299884,
             0.07629355], dtype=float32),
     'Assoc-voc': array([-0.01987046, -0.06434393,  0.00226   ,  0.08150155, -0.33775425,
            -0.13507745,  0.12741297,  0.0542295 ,  0.09895965,  0.067229  ,
            -0.1718493 ,  0.01054914,  0.10441845, -0.18814586, -0.01663602,
             0.03088147], dtype=float32),
     '9th': array([-0.24095939,  0.2750888 ,  0.01418325, -0.36754113,  0.5431856 ,
            -0.19582956,  0.03485603,  0.22838333, -0.05723334,  0.10631263,
             0.06331363, -0.09572615,  0.21977316, -0.02579625, -0.13822857,
             0.28736743], dtype=float32),
     '12th': array([-0.20278502, -0.19245535, -0.04846343,  0.14459866,  0.25858438,
             0.15333128,  0.5074635 , -0.15141617, -0.19331448, -0.2630267 ,
            -0.1378872 , -0.16868882,  0.4048257 , -0.34108582, -0.23098588,
             0.2859633 ], dtype=float32),
     '1st-4th': array([-0.53678703,  0.19669479, -0.18026853,  0.33791658,  0.14260627,
             0.20269199,  0.00518189,  0.01120056,  0.01568659,  0.28752655,
             0.3359768 ,  0.01758064,  0.11630564, -0.35470524, -0.05704446,
             0.41216984], dtype=float32),
     'Preschool': array([ 0.10326536, -0.02895411,  0.11348445,  0.03685748,  0.55893034,
            -0.2522173 , -0.07186767, -0.30955225, -0.17825711,  0.02907414,
            -0.61121726,  0.40596214,  0.63471395,  0.3304132 ,  0.05272925,
            -0.4266447 ], dtype=float32)}


