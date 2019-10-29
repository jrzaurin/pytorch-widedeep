## <font color='MediumSeaGreen '>1. Simple Binary Classification with defaults.</font>

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



### <font color='MediumSeaGreen '>1.1 Preparing the data</font>

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


### <font color='MediumSeaGreen '>1.2. Defining the model</font>


```python
wide = Wide(wide_dim=X_wide.shape[1], output_dim=1)
deepdense = DeepDense(hidden_layers=[64,32], 
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
          (dense): Sequential(
            (dense_layer_0): Sequential(
              (0): Linear(in_features=74, out_features=64, bias=True)
              (1): LeakyReLU(negative_slope=0.01, inplace=True)
              (2): Dropout(p=0.0, inplace=False)
            )
            (dense_layer_1): Sequential(
              (0): Linear(in_features=64, out_features=32, bias=True)
              (1): LeakyReLU(negative_slope=0.01, inplace=True)
              (2): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (1): Linear(in_features=32, out_features=1, bias=True)
      )
    )



As you can see, the model is not particularly complex. In mathematical terms (Eq 3 in the [original paper](https://arxiv.org/pdf/1606.07792.pdf)): 

$$
pred = \sigma(W^{T}_{wide}[x, \phi(x)] + W^{T}_{deep}a_{deep}^{(l_f)} +  b) 
$$ 


The architecture above will output the 1st and the second term in the parenthesis. `WideDeep` will then add them and apply an activation function (`sigmoid` in this case). For more details, please refer to the paper.

### <font color='MediumSeaGreen '>1.3 Compiling and Running/Fitting</font>
Once the model is built, we just need to compile it and run it


```python
model.compile(method='binary', metrics=[BinaryAccuracy])
```


```python
model.fit(X_wide=X_wide, X_deep=X_deep, target=target, n_epochs=5, batch_size=256, val_split=0.2)
```

    epoch 1: 100%|██████████| 153/153 [00:01<00:00, 87.08it/s, loss=0.421, metrics={'acc': 0.8067}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 136.50it/s, loss=0.364, metrics={'acc': 0.8115}]
    epoch 2: 100%|██████████| 153/153 [00:01<00:00, 96.64it/s, loss=0.352, metrics={'acc': 0.8354}] 
    valid: 100%|██████████| 39/39 [00:00<00:00, 146.08it/s, loss=0.354, metrics={'acc': 0.8356}]
    epoch 3: 100%|██████████| 153/153 [00:01<00:00, 97.69it/s, loss=0.344, metrics={'acc': 0.8388}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 141.73it/s, loss=0.35, metrics={'acc': 0.8388}]
    epoch 4: 100%|██████████| 153/153 [00:01<00:00, 99.13it/s, loss=0.34, metrics={'acc': 0.841}]   
    valid: 100%|██████████| 39/39 [00:00<00:00, 139.63it/s, loss=0.348, metrics={'acc': 0.8407}]
    epoch 5: 100%|██████████| 153/153 [00:01<00:00, 94.87it/s, loss=0.337, metrics={'acc': 0.8422}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 134.58it/s, loss=0.348, metrics={'acc': 0.8417}]


As you can see, you can run a wide and deep model in just a few lines of code

Let's now see how to use `WideDeep` with varying parameters

## <font color='MediumSeaGreen '>2. Binary Classification with varying parameters.</font>

###  <font color='MediumSeaGreen '>2.1 Dropout and Batchnorm</font>


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
          (dense): Sequential(
            (dense_layer_0): Sequential(
              (0): Linear(in_features=74, out_features=64, bias=True)
              (1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): LeakyReLU(negative_slope=0.01, inplace=True)
              (3): Dropout(p=0.5, inplace=False)
            )
            (dense_layer_1): Sequential(
              (0): Linear(in_features=64, out_features=32, bias=True)
              (1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): LeakyReLU(negative_slope=0.01, inplace=True)
              (3): Dropout(p=0.5, inplace=False)
            )
          )
        )
        (1): Linear(in_features=32, out_features=1, bias=True)
      )
    )



We can use different initializers, optimizers and learning rate schedulers for each `branch` of the model

###  <font color='MediumSeaGreen '>2.1 Optimizers, LR schedulers, Initializers and Callbacks</font>


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

    epoch 1: 100%|██████████| 153/153 [00:02<00:00, 76.36it/s, loss=0.72, metrics={'acc': 0.6214}] 
    valid: 100%|██████████| 39/39 [00:00<00:00, 133.66it/s, loss=0.433, metrics={'acc': 0.6602}]
    epoch 2: 100%|██████████| 153/153 [00:02<00:00, 73.74it/s, loss=0.482, metrics={'acc': 0.7676}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 120.10it/s, loss=0.375, metrics={'acc': 0.7802}]
    epoch 3: 100%|██████████| 153/153 [00:02<00:00, 73.27it/s, loss=0.42, metrics={'acc': 0.8017}] 
    valid: 100%|██████████| 39/39 [00:00<00:00, 139.83it/s, loss=0.361, metrics={'acc': 0.8082}]
    epoch 4: 100%|██████████| 153/153 [00:02<00:00, 72.46it/s, loss=0.395, metrics={'acc': 0.8108}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 136.43it/s, loss=0.357, metrics={'acc': 0.8158}]
    epoch 5: 100%|██████████| 153/153 [00:02<00:00, 71.34it/s, loss=0.383, metrics={'acc': 0.818}] 
    valid: 100%|██████████| 39/39 [00:00<00:00, 136.50it/s, loss=0.355, metrics={'acc': 0.8218}]
    epoch 6: 100%|██████████| 153/153 [00:02<00:00, 72.82it/s, loss=0.378, metrics={'acc': 0.8181}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 128.62it/s, loss=0.354, metrics={'acc': 0.8219}]
    epoch 7: 100%|██████████| 153/153 [00:02<00:00, 72.72it/s, loss=0.376, metrics={'acc': 0.8218}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 129.85it/s, loss=0.354, metrics={'acc': 0.8249}]
    epoch 8: 100%|██████████| 153/153 [00:02<00:00, 71.97it/s, loss=0.375, metrics={'acc': 0.8209}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 128.82it/s, loss=0.354, metrics={'acc': 0.8243}]
    epoch 9: 100%|██████████| 153/153 [00:02<00:00, 69.46it/s, loss=0.375, metrics={'acc': 0.8185}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 125.86it/s, loss=0.353, metrics={'acc': 0.8223}]
    epoch 10: 100%|██████████| 153/153 [00:02<00:00, 69.66it/s, loss=0.374, metrics={'acc': 0.8202}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 131.28it/s, loss=0.353, metrics={'acc': 0.8238}]



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
     '_backend',
     '_backward_hooks',
     '_buffers',
     '_construct',
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
model.history._history
```




    {'train_loss': [0.7201351490285661,
      0.48229670855734086,
      0.4197782385193445,
      0.3953245247111601,
      0.38271428147951764,
      0.37770096071405346,
      0.375562605515025,
      0.37540602976200627,
      0.3753419857399136,
      0.3741065304653317],
     'train_acc': [0.6214,
      0.7676,
      0.8017,
      0.8108,
      0.818,
      0.8181,
      0.8218,
      0.8209,
      0.8185,
      0.8202],
     'val_loss': [0.4325417692844684,
      0.37463742876664186,
      0.3612546171897497,
      0.35738192154810977,
      0.35490937798451155,
      0.35429174930621415,
      0.35409936461693203,
      0.3539723998461014,
      0.353485290820782,
      0.35325784331712967],
     'val_acc': [0.6602,
      0.7802,
      0.8082,
      0.8158,
      0.8218,
      0.8219,
      0.8249,
      0.8243,
      0.8223,
      0.8238]}




```python
model.lr_history
```




    {'lr_wide_0': [0.001,
      0.001,
      0.001,
      0.0001,
      0.0001,
      0.0001,
      1.0000000000000003e-05,
      1.0000000000000003e-05,
      1.0000000000000003e-05,
      1.0000000000000002e-06],
     'lr_deepdense_0': [0.001,
      0.001,
      0.001,
      0.001,
      0.001,
      0.0001,
      0.0001,
      0.0001,
      0.0001,
      0.0001]}



We can see that the learning rate effectively decreases by a factor of 0.1 (the default) after the corresponding `step_size`. Note that the keys of the dictionary have a suffix `_0`. This is because if you pass different parameter groups to the torch optimizers, these will also be recorded. We'll see this in the `Regression` notebook. 

And I guess one has a good idea of how to use the package. Before we leave this notebook just mentioning that the `WideDeep` class comes with a useful method to "rescue" the learned embeddings. For example, let's say I want to use the embeddings learned for the different levels of the categorical feature `education`


```python
model.get_embeddings(col_name='education', cat_encoding_dict=preprocess_deep.encoding_dict)
```




    {'11th': array([ 0.23203531,  0.22896081, -0.40356618,  0.43150797, -0.24202456,
            -0.15940084, -0.08549729,  0.4564645 ,  0.13446881, -0.16135852,
            -0.1743799 , -0.4434135 , -0.18031678,  0.15880926,  0.02965698,
            -0.22083491], dtype=float32),
     'HS-grad': array([ 0.3114914 ,  0.36132628, -0.42488536, -0.44385988,  0.2485746 ,
             0.01649826,  0.43731764, -0.16036318,  0.22887692, -0.25932702,
            -0.02782134, -0.06970705,  0.19947569, -0.06710748, -0.10803316,
             0.46665302], dtype=float32),
     'Assoc-acdm': array([-0.24408445,  0.05876141, -0.04507849, -0.19578709, -0.14715208,
             0.4983954 , -0.05527269,  0.16526866, -0.307099  ,  0.17685033,
            -0.14156346, -0.06647427, -0.46975732,  0.16181333,  0.02914725,
            -0.49256295], dtype=float32),
     'Some-college': array([ 0.5198188 ,  0.12946902,  0.46062082,  0.44757575, -0.3287289 ,
            -0.12341443, -0.11078605,  0.1642068 ,  0.7651399 ,  0.06216411,
            -0.8117381 ,  0.6532599 , -0.00924105,  0.4417696 ,  0.09026518,
            -0.12002172], dtype=float32),
     '10th': array([ 0.5984684 ,  0.16307777, -0.0040624 ,  0.09925628,  0.20535131,
            -0.15751003, -0.16682488, -0.1383966 , -0.04823777,  0.15658148,
            -0.12845115, -0.1440473 ,  0.35936442,  0.01721832, -0.01479862,
             0.1628472 ], dtype=float32),
     'Prof-school': array([ 0.02319724,  0.3303704 ,  0.08904056, -0.21102089,  0.19608757,
            -0.07665357,  0.15307519,  0.25392193, -0.03555196, -0.01884847,
             0.03365186, -0.11260296, -0.13606223, -0.09259846,  0.27067274,
            -0.16312157], dtype=float32),
     '7th-8th': array([ 0.33061972,  0.1258869 , -0.18391465, -0.5522697 ,  0.14627822,
             0.08831056,  0.00233046, -0.00830169, -0.07232173, -0.3524963 ,
            -0.01687683, -0.28693867,  0.19178541, -0.17721641, -0.24398643,
             0.15801452], dtype=float32),
     'Bachelors': array([ 0.23301753,  0.11734378, -0.02250313, -0.31250337,  0.25254628,
            -0.13198347,  0.32288718, -0.11564187,  0.08262251,  0.00897656,
            -0.04101277,  0.2034123 ,  0.00600741,  0.11451315,  0.08216624,
            -0.18260935], dtype=float32),
     'Masters': array([-0.2682981 ,  0.03703215, -0.25879607, -0.40328467, -0.32078862,
            -0.15390627, -0.00587583,  0.2890941 , -0.2309889 , -0.03192039,
             0.42183968,  0.3534382 , -0.10053465,  0.20614813, -0.00845117,
             0.13243063], dtype=float32),
     'Doctorate': array([ 0.40153244, -0.15173616,  0.2734586 , -0.06986004, -0.14779176,
             0.06517711,  0.43264598, -0.04060874,  0.09469996,  0.04779944,
             0.11410471, -0.61585397, -0.33141896, -0.06763163,  0.19431648,
             0.32619408], dtype=float32),
     '5th-6th': array([-0.06580594, -0.15445694, -0.4775835 ,  0.28082463,  0.21930388,
             0.15399367,  0.08140283,  0.12158986,  0.65451396, -0.3062649 ,
            -0.4490934 ,  0.346769  , -0.36774218,  0.06957038,  0.1303332 ,
             0.07054735], dtype=float32),
     'Assoc-voc': array([ 0.3115598 ,  0.18573369,  0.17958838, -0.30102468, -0.35813195,
             0.11202388,  0.2779358 ,  0.22348149,  0.09943093, -0.53038543,
             0.03727521, -0.04638249, -0.09950424,  0.27130258,  0.07549058,
            -0.49732867], dtype=float32),
     '9th': array([ 0.2628104 , -0.2855187 , -0.25854272, -0.08381794, -0.2020421 ,
            -0.02920138,  0.10086066, -0.10290657, -0.33239442, -0.2356638 ,
            -0.248578  ,  0.01665138,  0.28796577,  0.07396127,  0.01030401,
             0.37545788], dtype=float32),
     '12th': array([ 0.11248264, -0.14112231, -0.18007489,  0.41162646,  0.27112672,
            -0.02875315, -0.16151035, -0.4613239 , -0.41860878, -0.27310988,
            -0.12612441, -0.26779214,  0.46872276,  0.50543463, -0.06184073,
             0.01199363], dtype=float32),
     '1st-4th': array([ 0.10582871, -0.22928524, -0.21345232,  0.27670494, -0.28263775,
             0.06005969, -0.04883407, -0.04386626, -0.18646769, -0.28977564,
             0.3295173 , -0.2891513 ,  0.3165016 , -0.30840456, -0.13870218,
             0.24087428], dtype=float32),
     'Preschool': array([ 0.06730541, -0.12282428, -0.0063521 ,  0.07224482,  0.24416964,
            -0.09476493, -0.12492466, -0.1393237 , -0.36801594, -0.02907634,
             0.44266376, -0.15134929,  0.24314906, -0.15032478,  0.20950297,
            -0.12269441], dtype=float32)}


