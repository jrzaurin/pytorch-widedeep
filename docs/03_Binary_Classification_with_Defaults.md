## Simple Binary Classification with defaults

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


### Defining the model


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

### Compiling and Running/Fitting
Once the model is built, we just need to compile it and run it


```python
model.compile(method='binary', metrics=[BinaryAccuracy])
```


```python
model.fit(X_wide=X_wide, X_deep=X_deep, target=target, n_epochs=5, batch_size=256, val_split=0.2)
```

      0%|          | 0/153 [00:00<?, ?it/s]

    Training


    epoch 1: 100%|██████████| 153/153 [00:02<00:00, 56.52it/s, loss=0.412, metrics={'acc': 0.7993}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 123.12it/s, loss=0.352, metrics={'acc': 0.8071}]
    epoch 2: 100%|██████████| 153/153 [00:02<00:00, 59.55it/s, loss=0.351, metrics={'acc': 0.8351}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 121.98it/s, loss=0.346, metrics={'acc': 0.8359}]
    epoch 3: 100%|██████████| 153/153 [00:02<00:00, 59.82it/s, loss=0.346, metrics={'acc': 0.8377}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 121.88it/s, loss=0.344, metrics={'acc': 0.8384}]
    epoch 4: 100%|██████████| 153/153 [00:02<00:00, 58.97it/s, loss=0.342, metrics={'acc': 0.8392}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 122.20it/s, loss=0.342, metrics={'acc': 0.84}] 
    epoch 5: 100%|██████████| 153/153 [00:02<00:00, 58.28it/s, loss=0.34, metrics={'acc': 0.8406}] 
    valid: 100%|██████████| 39/39 [00:00<00:00, 116.57it/s, loss=0.341, metrics={'acc': 0.8413}]


As you can see, you can run a wide and deep model in just a few lines of code
