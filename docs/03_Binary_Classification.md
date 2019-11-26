## 1. Simple Binary Classification with defaults

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



### 1.1 Preparing the data

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


### 1.2. Defining the model


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
          (embed_dropout): Dropout(p=0.0, inplace=False)
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

### 1.3 Compiling and Running/Fitting
Once the model is built, we just need to compile it and run it


```python
model.compile(method='binary', metrics=[BinaryAccuracy])
```


```python
model.fit(X_wide=X_wide, X_deep=X_deep, target=target, n_epochs=5, batch_size=256, val_split=0.2)
```

      0%|          | 0/153 [00:00<?, ?it/s]

    Training


    epoch 1: 100%|██████████| 153/153 [00:02<00:00, 55.16it/s, loss=0.419, metrics={'acc': 0.7994}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 117.28it/s, loss=0.364, metrics={'acc': 0.8059}]
    epoch 2: 100%|██████████| 153/153 [00:02<00:00, 57.51it/s, loss=0.352, metrics={'acc': 0.8351}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 117.51it/s, loss=0.355, metrics={'acc': 0.835}]
    epoch 3: 100%|██████████| 153/153 [00:02<00:00, 57.71it/s, loss=0.345, metrics={'acc': 0.8379}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 107.89it/s, loss=0.352, metrics={'acc': 0.8375}]
    epoch 4: 100%|██████████| 153/153 [00:02<00:00, 58.28it/s, loss=0.341, metrics={'acc': 0.8396}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 117.59it/s, loss=0.349, metrics={'acc': 0.8391}]
    epoch 5: 100%|██████████| 153/153 [00:02<00:00, 57.90it/s, loss=0.338, metrics={'acc': 0.8408}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 115.33it/s, loss=0.348, metrics={'acc': 0.8406}]


As you can see, you can run a wide and deep model in just a few lines of code

Let's now see how to use `WideDeep` with varying parameters

## 2. Binary Classification with varying parameters

### 2.1 Warm-up

We can choose to warm up each model individually before the joined training begins. To warm up, the models will be trained during `warm_epochs` using a triangular one-cycle learning rate (referred as *slanted triangular learning rates* in [Howard & Ruder 2018](https://arxiv.org/pdf/1801.06146.pdf)) going from `warm_max_lr`/10. to `warm_max_lr` (default is 0.01). 10% of the training steps are used to increase the learning rate which then decreases back to `warm_max_lr`/10. for the remaining 90%.   


```python
wide = Wide(wide_dim=X_wide.shape[1], output_dim=1)
deepdense = DeepDense(hidden_layers=[64,32], 
                      deep_column_idx=preprocess_deep.deep_column_idx,
                      embed_input=preprocess_deep.embeddings_input,
                      continuous_cols=continuous_cols)
model = WideDeep(wide=wide, deepdense=deepdense)
model.compile(method='binary', metrics=[BinaryAccuracy])
model.fit(X_wide=X_wide, X_deep=X_deep, target=target, n_epochs=4, batch_size=256, val_split=0.2, 
          warm_up=True, warm_epochs=1)
```

      0%|          | 0/153 [00:00<?, ?it/s]

    Warming up wide for 1 epochs


    epoch 1: 100%|██████████| 153/153 [00:01<00:00, 141.42it/s, loss=0.45, metrics={'acc': 0.7909}]
      0%|          | 0/153 [00:00<?, ?it/s]

    Warming up deepdense for 1 epochs


    epoch 1: 100%|██████████| 153/153 [00:02<00:00, 63.59it/s, loss=0.382, metrics={'acc': 0.8049}]
      0%|          | 0/153 [00:00<?, ?it/s]

    Training


    epoch 1: 100%|██████████| 153/153 [00:02<00:00, 57.88it/s, loss=0.346, metrics={'acc': 0.8381}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 118.42it/s, loss=0.35, metrics={'acc': 0.8381}]
    epoch 2: 100%|██████████| 153/153 [00:02<00:00, 56.82it/s, loss=0.34, metrics={'acc': 0.8414}] 
    valid: 100%|██████████| 39/39 [00:00<00:00, 116.15it/s, loss=0.349, metrics={'acc': 0.8409}]
    epoch 3: 100%|██████████| 153/153 [00:02<00:00, 57.64it/s, loss=0.338, metrics={'acc': 0.8424}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 114.68it/s, loss=0.348, metrics={'acc': 0.8418}]
    epoch 4: 100%|██████████| 153/153 [00:02<00:00, 58.26it/s, loss=0.336, metrics={'acc': 0.8438}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 111.94it/s, loss=0.347, metrics={'acc': 0.843}]


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

###  2.1 Optimizers, LR schedulers, Initializers and Callbacks


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


    epoch 1: 100%|██████████| 153/153 [00:03<00:00, 48.30it/s, loss=0.911, metrics={'acc': 0.5729}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 117.93it/s, loss=0.521, metrics={'acc': 0.6051}]
    epoch 2: 100%|██████████| 153/153 [00:03<00:00, 47.55it/s, loss=0.578, metrics={'acc': 0.7309}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 117.68it/s, loss=0.402, metrics={'acc': 0.7466}]
    epoch 3: 100%|██████████| 153/153 [00:03<00:00, 48.88it/s, loss=0.477, metrics={'acc': 0.7783}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 117.25it/s, loss=0.373, metrics={'acc': 0.7879}]
    epoch 4: 100%|██████████| 153/153 [00:03<00:00, 48.89it/s, loss=0.428, metrics={'acc': 0.8012}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 115.87it/s, loss=0.365, metrics={'acc': 0.807}]
    epoch 5: 100%|██████████| 153/153 [00:03<00:00, 48.28it/s, loss=0.406, metrics={'acc': 0.8087}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 108.82it/s, loss=0.359, metrics={'acc': 0.814}]
    epoch 6: 100%|██████████| 153/153 [00:03<00:00, 47.57it/s, loss=0.395, metrics={'acc': 0.8126}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 114.21it/s, loss=0.358, metrics={'acc': 0.8171}]
    epoch 7: 100%|██████████| 153/153 [00:03<00:00, 46.31it/s, loss=0.391, metrics={'acc': 0.813}] 
    valid: 100%|██████████| 39/39 [00:00<00:00, 113.62it/s, loss=0.357, metrics={'acc': 0.8175}]
    epoch 8: 100%|██████████| 153/153 [00:03<00:00, 48.37it/s, loss=0.39, metrics={'acc': 0.8169}] 
    valid: 100%|██████████| 39/39 [00:00<00:00, 113.52it/s, loss=0.357, metrics={'acc': 0.8207}]
    epoch 9: 100%|██████████| 153/153 [00:03<00:00, 47.56it/s, loss=0.389, metrics={'acc': 0.8162}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 112.79it/s, loss=0.356, metrics={'acc': 0.8203}]
    epoch 10: 100%|██████████| 153/153 [00:03<00:00, 47.69it/s, loss=0.386, metrics={'acc': 0.8171}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 109.54it/s, loss=0.356, metrics={'acc': 0.8212}]



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
     '_warm_model',
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

    {'train_loss': [0.9105595845023012, 0.5782874746649873, 0.47749636551133945, 0.4281357573527916, 0.4061133719347661, 0.3947428677206725, 0.3914796267849168, 0.38961343983419583, 0.38926642879941104, 0.386191635934356], 'train_acc': [0.5729, 0.7309, 0.7783, 0.8012, 0.8087, 0.8126, 0.813, 0.8169, 0.8162, 0.8171], 'val_loss': [0.5205524701338547, 0.4024948156796969, 0.37283383500881684, 0.36488463634099716, 0.35886253301913923, 0.35753893775817674, 0.35730381042529374, 0.35672229528427124, 0.3563888867696126, 0.35624249776204425], 'val_acc': [0.6051, 0.7466, 0.7879, 0.807, 0.814, 0.8171, 0.8175, 0.8207, 0.8203, 0.8212]}



```python
print(model.lr_history)
```

    {'lr_wide_0': [0.001, 0.001, 0.001, 0.0001, 0.0001, 0.0001, 1.0000000000000003e-05, 1.0000000000000003e-05, 1.0000000000000003e-05, 1.0000000000000002e-06], 'lr_deepdense_0': [0.001, 0.001, 0.001, 0.001, 0.001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]}


We can see that the learning rate effectively decreases by a factor of 0.1 (the default) after the corresponding `step_size`. Note that the keys of the dictionary have a suffix `_0`. This is because if you pass different parameter groups to the torch optimizers, these will also be recorded. We'll see this in the `Regression` notebook. 

And I guess one has a good idea of how to use the package. Before we leave this notebook just mentioning that the `WideDeep` class comes with a useful method to "rescue" the learned embeddings. For example, let's say I want to use the embeddings learned for the different levels of the categorical feature `education`


```python
model.get_embeddings(col_name='education', cat_encoding_dict=preprocess_deep.encoding_dict)
```




    {'11th': array([ 0.41717738,  0.33800027,  0.14558531,  0.23621577,  0.18866219,
             0.41959327, -0.40843502,  0.2764823 , -0.04228376, -0.2770995 ,
            -0.04413053,  0.21518119,  0.54168355,  0.06607324, -0.25075328,
             0.43814617], dtype=float32),
     'HS-grad': array([-0.10619222,  0.08463281, -0.1720976 ,  0.16436636, -0.15246172,
            -0.15992908, -0.07142664,  0.02854085, -0.38450843,  0.3621033 ,
             0.00361137, -0.37991726, -0.00742414, -0.19315098,  0.23940389,
             0.00427438], dtype=float32),
     'Assoc-acdm': array([-0.10346556,  0.5880965 , -0.35604322, -0.28074315,  0.11279969,
            -0.03097979,  0.1316176 ,  0.04005286,  0.22053859,  0.2822993 ,
             0.2548561 , -0.00729926,  0.16980447,  0.00099144, -0.21386623,
            -0.03788675], dtype=float32),
     'Some-college': array([ 1.0774918e-01, -5.1934827e-02,  2.2199769e-01,  1.2707384e-01,
            -6.6714182e-02, -1.2450726e-02, -4.7941156e-02, -1.1758558e-02,
            -1.7372087e-02, -3.7972507e-01, -7.2314329e-03, -1.1348904e-02,
            -2.8346037e-04, -1.8352264e-01,  6.1671283e-02,  2.2232330e-01],
           dtype=float32),
     '10th': array([ 0.29999158, -0.21744487,  0.06491452, -0.23188359, -0.36609316,
            -0.38092315,  0.04983652,  0.32082322, -0.09453602, -0.07210832,
             0.02355519, -0.34295735,  0.243176  , -0.12205487, -0.02939285,
             0.03140339], dtype=float32),
     'Prof-school': array([ 0.21009974, -0.17979464,  0.23510288, -0.4422548 ,  0.19806142,
            -0.08493114,  0.06911367,  0.1785185 ,  0.17035176,  0.26042286,
             0.20824155,  0.28717726, -0.33635965, -0.199471  , -0.00237502,
            -0.15463887], dtype=float32),
     '7th-8th': array([-0.28743556, -0.27534077, -0.2952116 ,  0.35380983,  0.530602  ,
             0.24720307,  0.00427648, -0.35313243, -0.11463641, -0.13932341,
             0.66691613,  0.46317872, -0.2385504 , -0.27184793, -0.14130774,
            -0.18510057], dtype=float32),
     'Bachelors': array([ 0.19901408,  0.13878398,  0.12359496, -0.1516372 ,  0.15461658,
            -0.12157986,  0.28729957, -0.26748437,  0.07945791,  0.0911655 ,
             0.3575531 ,  0.08508369, -0.1413984 , -0.10829177, -0.26311323,
             0.20712389], dtype=float32),
     'Masters': array([-0.02080029,  0.15801647,  0.3071945 ,  0.0232136 ,  0.18986145,
             0.16438615, -0.12542671, -0.04688492, -0.07556052, -0.2942081 ,
             0.05731679,  0.20982188, -0.37307253, -0.27664435,  0.5616179 ,
            -0.13841839], dtype=float32),
     'Doctorate': array([ 0.10035745, -0.08719181, -0.22271474, -0.17451538,  0.20309775,
            -0.0911912 , -0.26586285,  0.09883135,  0.07470689,  0.5613913 ,
            -0.12691443,  0.09585453,  0.08105591, -0.212968  , -0.12663043,
            -0.15639608], dtype=float32),
     '5th-6th': array([-0.19824742,  0.22895677,  0.3450842 ,  0.36915532,  0.02075848,
            -0.18854013,  0.21759517,  0.10949593, -0.29776537,  0.06532905,
             0.43095952, -0.3871383 , -0.13502343,  0.06983275, -0.12452139,
            -0.47077683], dtype=float32),
     'Assoc-voc': array([ 0.03224453, -0.06446365,  0.24354428,  0.15382324,  0.15567093,
             0.03998892,  0.4248653 ,  0.22545348, -0.0560311 ,  0.16399181,
             0.27097237, -0.06783602,  0.28948635, -0.5472932 , -0.06647005,
            -0.02521862], dtype=float32),
     '9th': array([-0.3762758 , -0.02360561,  0.05081445,  0.05898981, -0.02661294,
             0.03272862,  0.14599569, -0.04475676, -0.39210543, -0.62865454,
             0.04343129,  0.44932538, -0.15965058,  0.10564981, -0.12342159,
            -0.21983312], dtype=float32),
     '12th': array([ 0.36367476,  0.02930327, -0.03054986, -0.09188785,  0.08873531,
             0.35568398,  0.37973958,  0.41732144, -0.21674553,  0.00693592,
             0.19131127,  0.26841545, -0.0428647 ,  0.10508669,  0.03650329,
            -0.05330637], dtype=float32),
     '1st-4th': array([-0.71947837, -0.05691843,  0.16745438, -0.00462133, -0.36181843,
            -0.11321898,  0.37750733,  0.3297556 ,  0.3258544 , -0.22029436,
            -0.25121528, -0.04426979,  0.23183182, -0.09465475,  0.15154967,
             0.05574629], dtype=float32),
     'Preschool': array([ 0.8810191 ,  0.11680676,  0.18423152, -0.02020044,  0.20060717,
             0.19240808, -0.21568672, -0.00838439,  0.32876205, -0.02553497,
             0.0844057 ,  0.2446878 ,  0.16002655, -0.46517354,  0.2243812 ,
            -0.2781279 ], dtype=float32)}


