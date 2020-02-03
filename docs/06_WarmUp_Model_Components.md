## The Warm Up option

When we run `.fit`, we can choose to first "warm up" each model individually (similar to fine-tunning if the model was pre-trained, but this is a general functionality, i.e. no need of a pretrained model) before the joined training begins. 

There are 3 warming up routines:

1. Warm up all trainable layers at once with a triangular one-cycle learning rate (referred as slanted triangular learning rates in Howard & Ruder 2018)
2. Gradual warm up inspired by the work of [Felbo et al., 2017](https://arxiv.org/abs/1708.00524) for fine-tunning
3. Gradual warm up inspired by the work of [Howard & Ruder 2018](https://arxiv.org/abs/1801.06146) for fine-tunning

Currently warming up is only supported without a fully connected `DeepHead`, i.e. if `deephead=None`. In addition, `Felbo` and `Howard` routines only applied to `DeepText` and `DeepImage` models. The `Wide` and `DeepDense` components can also be warmed up together, but only in an "all at once" mode.

### Warm up all at once

The models will be trained for `warm_epochs` using a triangular one-cycle learning rate (slanted triangular learning rate) ranging from `warm_max_lr/10` to `warm_max_lr` (default is 0.01). 10% of the training steps are used to increase the learning rate which then decreases for the remaining 90%. 

Here all trainable layers are warmed up.


To use it, simply:


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
# For convenience, we'll replace '-' with '_'
df.columns = [c.replace("-", "_") for c in df.columns]
# binary target
df['income_label'] = (df["income"].apply(lambda x: ">50K" in x)).astype(int)
df.drop('income', axis=1, inplace=True)
```


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
wide = Wide(wide_dim=X_wide.shape[1], output_dim=1)
deepdense = DeepDense(hidden_layers=[64,32], 
                      deep_column_idx=preprocess_deep.deep_column_idx,
                      embed_input=preprocess_deep.embeddings_input,
                      continuous_cols=continuous_cols)
model = WideDeep(wide=wide, deepdense=deepdense)
```


```python
model.compile(method='binary', metrics=[BinaryAccuracy])
```

Up until here is identical to the code in notebook `03_Binary_Classification_with_Defaults`. Now you can warm up via the warm up parameters


```python
model.fit(X_wide=X_wide, X_deep=X_deep, target=target, n_epochs=5, batch_size=256, val_split=0.2, 
          warm_up=True, warm_epochs=5, warm_max_lr=0.01)
```

      0%|          | 0/153 [00:00<?, ?it/s]

    Warming up wide for 5 epochs


    epoch 1: 100%|██████████| 153/153 [00:01<00:00, 131.88it/s, loss=0.471, metrics={'acc': 0.7946}]
    epoch 2: 100%|██████████| 153/153 [00:00<00:00, 154.81it/s, loss=0.373, metrics={'acc': 0.8115}]
    epoch 3: 100%|██████████| 153/153 [00:01<00:00, 151.56it/s, loss=0.364, metrics={'acc': 0.8182}]
    epoch 4: 100%|██████████| 153/153 [00:00<00:00, 154.22it/s, loss=0.362, metrics={'acc': 0.8216}]
    epoch 5: 100%|██████████| 153/153 [00:01<00:00, 152.65it/s, loss=0.36, metrics={'acc': 0.8238}] 
      0%|          | 0/153 [00:00<?, ?it/s]

    Warming up deepdense for 5 epochs


    epoch 1: 100%|██████████| 153/153 [00:02<00:00, 64.04it/s, loss=0.395, metrics={'acc': 0.8222}]
    epoch 2: 100%|██████████| 153/153 [00:02<00:00, 65.34it/s, loss=0.349, metrics={'acc': 0.8242}]
    epoch 3: 100%|██████████| 153/153 [00:02<00:00, 65.05it/s, loss=0.343, metrics={'acc': 0.8262}]
    epoch 4: 100%|██████████| 153/153 [00:02<00:00, 64.93it/s, loss=0.339, metrics={'acc': 0.8279}]
    epoch 5: 100%|██████████| 153/153 [00:02<00:00, 65.15it/s, loss=0.335, metrics={'acc': 0.8295}]
      0%|          | 0/153 [00:00<?, ?it/s]

    Training


    epoch 1: 100%|██████████| 153/153 [00:02<00:00, 58.31it/s, loss=0.345, metrics={'acc': 0.8415}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 120.77it/s, loss=0.346, metrics={'acc': 0.8416}]
    epoch 2: 100%|██████████| 153/153 [00:02<00:00, 58.33it/s, loss=0.335, metrics={'acc': 0.8446}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 117.88it/s, loss=0.344, metrics={'acc': 0.8438}]
    epoch 3: 100%|██████████| 153/153 [00:02<00:00, 58.43it/s, loss=0.331, metrics={'acc': 0.8457}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 109.26it/s, loss=0.343, metrics={'acc': 0.8449}]
    epoch 4: 100%|██████████| 153/153 [00:02<00:00, 58.08it/s, loss=0.329, metrics={'acc': 0.8457}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 120.23it/s, loss=0.344, metrics={'acc': 0.8446}]
    epoch 5: 100%|██████████| 153/153 [00:02<00:00, 58.75it/s, loss=0.327, metrics={'acc': 0.8464}]
    valid: 100%|██████████| 39/39 [00:00<00:00, 119.22it/s, loss=0.344, metrics={'acc': 0.8453}]


### Warm up Gradually: The "felbo"  and the "howard" routines

The Felbo routine can be illustrated as follows:

<p align="center">
  <img width="600" src="figures/felbo_routine.png">
</p>

**Figure 1.** The figure can be described as follows: warm up (or train) the last layer for one epoch using a one cycle triangular learning rate. Then warm up the next deeper layer for one epoch, with a learning rate that is a factor of 2.5 lower than the previous learning rate (the 2.5 factor is fixed) while freezing the already warmed up layer(s). Repeat untill all individual layers are warmed. Then warm one last epoch with all warmed layers trainable. The vanishing color gradient in the figure attempts to illustrate the decreasing learning rate. 

Note that this is not identical to the Fine-Tunning routine described in Felbo et al, 2017, this is why I used the word 'inspired'.

The Howard routine can be illustrated as follows:

<p align="center">
  <img width="600" src="figures/howard_routine.png">
</p>

**Figure 2.** The figure can be described as follows: warm up (or train) the last layer for one epoch using a one cycle triangular learning rate. Then warm up the next deeper layer for one epoch, with a learning rate that is a factor of 2.5 lower than the previous learning rate (the 2.5 factor is fixed) while keeping the already warmed up layer(s) trainable. Repeat. The vanishing color gradient in the figure attempts to illustrate the decreasing learning rate. 

Note that I write "*warm up (or train) the last layer for one epoch [...]*". However, in practice the user will have to specify the order of the layers to be warmed up. This is another reason why I wrote that the warm up routines I have implemented are **inspired** by the work of Felbo and Howard and not identical to their implemenations.

The `felbo` and `howard` routines can be accessed with via the `warm` parameters.


```python
from pytorch_widedeep.preprocessing import TextPreprocessor, ImagePreprocessor
from pytorch_widedeep.models import DeepText, DeepImage
```


```python
df = pd.read_csv('data/airbnb/airbnb_sample.csv')
# There are a number of columns that are already binary. Therefore, no need to one hot encode them
crossed_cols = (['property_type', 'room_type'],)
already_dummies = [c for c in df.columns if 'amenity' in c] + ['has_house_rules']
wide_cols = ['is_location_exact', 'property_type', 'room_type', 'host_gender',
'instant_bookable'] + already_dummies
cat_embed_cols = [(c, 16) for c in df.columns if 'catg' in c] + \
    [('neighbourhood_cleansed', 64), ('cancellation_policy', 16)]
continuous_cols = ['latitude', 'longitude', 'security_deposit', 'extra_people']
# it does not make sense to standarised Latitude and Longitude
already_standard = ['latitude', 'longitude']
# text and image colnames
text_col = 'description'
img_col = 'id'
# path to pretrained word embeddings and the images
word_vectors_path = 'data/glove.6B/glove.6B.100d.txt'
img_path = 'data/airbnb/property_picture'
# target
target_col = 'yield'
```


```python
target = df[target_col].values

prepare_wide = WidePreprocessor(wide_cols=wide_cols, crossed_cols=crossed_cols)
X_wide = prepare_wide.fit_transform(df)

prepare_deep = DeepPreprocessor(embed_cols=cat_embed_cols, continuous_cols=continuous_cols)
X_deep = prepare_deep.fit_transform(df)

text_processor = TextPreprocessor(word_vectors_path=word_vectors_path, text_col=text_col)
X_text = text_processor.fit_transform(df)

image_processor = ImagePreprocessor(img_col=img_col, img_path=img_path)
X_images = image_processor.fit_transform(df)
```

    The vocabulary contains 6400 tokens
    Indexing word vectors...
    Loaded 400000 word vectors
    Preparing embeddings matrix...
    2175 words in the vocabulary had data/glove.6B/glove.6B.100d.txt vectors and appear more than 5 times
    Reading Images from data/airbnb/property_picture


      3%|▎         | 29/1001 [00:00<00:03, 282.66it/s]

    Resizing


    100%|██████████| 1001/1001 [00:03<00:00, 327.44it/s]


    Computing normalisation metrics



```python
wide = Wide(wide_dim=X_wide.shape[1], output_dim=1)
deepdense = DeepDense( hidden_layers=[64,32], dropout=[0.2,0.2],
                      deep_column_idx=prepare_deep.deep_column_idx,
                      embed_input=prepare_deep.embeddings_input,
                      continuous_cols=continuous_cols)
deeptext = DeepText(vocab_size=len(text_processor.vocab.itos),
                    hidden_dim=64, n_layers=3, rnn_dropout=0.5,
                    embedding_matrix=text_processor.embedding_matrix)
deepimage = DeepImage(pretrained=True, head_layers=None)
model = WideDeep(wide=wide, deepdense=deepdense, deeptext=deeptext, deepimage=deepimage)
model.compile(method='regression')
```

let's have a look to the fit method


```python
?model.fit
```

As you will see the `warm` args are: 

```python
    warm_up:bool=False,
    warm_epochs:int=4,
    warm_max_lr:float=0.01,
    warm_deeptext_gradual:bool=False,
    warm_deeptext_max_lr:float=0.01,
    warm_deeptext_layers:Union[List[torch.nn.modules.module.Module], NoneType]=None,
    warm_deepimage_gradual:bool=False,
    warm_deepimage_max_lr:float=0.01,
    warm_deepimage_layers:Union[List[torch.nn.modules.module.Module], NoneType]=None,
    warm_routine:str='howard',
```

We need to explicitly indicate 1) that we want to warm up, 2) that we want `DeepText` and/or `DeepImage` to warm up gradually 3) in that case, the warm up routine and 4) the layers we want to warm up. 

For example, let's have a look to the model


```python
model
```




    WideDeep(
      (wide): Wide(
        (wide_linear): Linear(in_features=356, out_features=1, bias=True)
      )
      (deepdense): Sequential(
        (0): DeepDense(
          (embed_layers): ModuleDict(
            (emb_layer_accommodates_catg): Embedding(3, 16)
            (emb_layer_bathrooms_catg): Embedding(3, 16)
            (emb_layer_bedrooms_catg): Embedding(4, 16)
            (emb_layer_beds_catg): Embedding(4, 16)
            (emb_layer_cancellation_policy): Embedding(5, 16)
            (emb_layer_guests_included_catg): Embedding(3, 16)
            (emb_layer_host_listings_count_catg): Embedding(4, 16)
            (emb_layer_minimum_nights_catg): Embedding(3, 16)
            (emb_layer_neighbourhood_cleansed): Embedding(32, 64)
          )
          (embed_dropout): Dropout(p=0.0, inplace=False)
          (dense): Sequential(
            (dense_layer_0): Sequential(
              (0): Linear(in_features=196, out_features=64, bias=True)
              (1): LeakyReLU(negative_slope=0.01, inplace=True)
              (2): Dropout(p=0.2, inplace=False)
            )
            (dense_layer_1): Sequential(
              (0): Linear(in_features=64, out_features=32, bias=True)
              (1): LeakyReLU(negative_slope=0.01, inplace=True)
              (2): Dropout(p=0.2, inplace=False)
            )
          )
        )
        (1): Linear(in_features=32, out_features=1, bias=True)
      )
      (deeptext): Sequential(
        (0): DeepText(
          (word_embed): Embedding(2192, 100, padding_idx=1)
          (rnn): LSTM(100, 64, num_layers=3, batch_first=True, dropout=0.5)
        )
        (1): Linear(in_features=64, out_features=1, bias=True)
      )
      (deepimage): Sequential(
        (0): DeepImage(
          (backbone): Sequential(
            (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
            (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            (4): Sequential(
              (0): BasicBlock(
                (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU(inplace=True)
                (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (1): BasicBlock(
                (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU(inplace=True)
                (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (5): Sequential(
              (0): BasicBlock(
                (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU(inplace=True)
                (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (downsample): Sequential(
                  (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
                  (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
              )
              (1): BasicBlock(
                (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU(inplace=True)
                (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (6): Sequential(
              (0): BasicBlock(
                (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU(inplace=True)
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (downsample): Sequential(
                  (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
                  (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
              )
              (1): BasicBlock(
                (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU(inplace=True)
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (7): Sequential(
              (0): BasicBlock(
                (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU(inplace=True)
                (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (downsample): Sequential(
                  (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
                  (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
              )
              (1): BasicBlock(
                (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (relu): ReLU(inplace=True)
                (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (8): AdaptiveAvgPool2d(output_size=(1, 1))
          )
        )
        (1): Linear(in_features=512, out_features=1, bias=True)
      )
    )



We can see that the `DeepImage` model is comprised by a `Sequential` model that is a ResNet `backbone` and a `Linear` Layer. I want to warm up the layers in the ResNet `backbone`, apart from the first sequence `[Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d]`, and the `Linear` layer, so let's access them.


```python
first_child = list(model.deepimage.children())[0]
img_layers = list(first_child.backbone.children())[4:8] + [list(model.deepimage.children())[1]]
img_layers
```




    [Sequential(
       (0): BasicBlock(
         (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
         (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (relu): ReLU(inplace=True)
         (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
         (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
       )
       (1): BasicBlock(
         (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
         (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (relu): ReLU(inplace=True)
         (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
         (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
       )
     ), Sequential(
       (0): BasicBlock(
         (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
         (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (relu): ReLU(inplace=True)
         (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
         (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (downsample): Sequential(
           (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
           (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         )
       )
       (1): BasicBlock(
         (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
         (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (relu): ReLU(inplace=True)
         (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
         (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
       )
     ), Sequential(
       (0): BasicBlock(
         (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
         (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (relu): ReLU(inplace=True)
         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
         (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (downsample): Sequential(
           (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
           (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         )
       )
       (1): BasicBlock(
         (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
         (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (relu): ReLU(inplace=True)
         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
         (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
       )
     ), Sequential(
       (0): BasicBlock(
         (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
         (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (relu): ReLU(inplace=True)
         (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
         (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (downsample): Sequential(
           (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
           (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         )
       )
       (1): BasicBlock(
         (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
         (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (relu): ReLU(inplace=True)
         (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
         (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
       )
     ), Linear(in_features=512, out_features=1, bias=True)]



The layers need to be passed in the order that we want them to be warmed up. In the future I might infer this automatically within the `_warmup.py` submodule, but for now, the user needs to specify the warm up order. In this case, is pretty straightforward.


```python
warm_img_layers = img_layers[::-1]
```


```python
model.fit(X_wide=X_wide, X_deep=X_deep, X_text=X_text, X_img=X_images, target=target, n_epochs=1, 
          batch_size=32, val_split=0.2, warm_up=True, warm_epochs=1, warm_deepimage_gradual=True, 
          warm_deepimage_layers=warm_img_layers, warm_deepimage_max_lr=0.01, warm_routine='howard')
```

      0%|          | 0/25 [00:00<?, ?it/s]

    Warming up wide for 1 epochs


    epoch 1: 100%|██████████| 25/25 [00:00<00:00, 57.98it/s, loss=127]
      0%|          | 0/25 [00:00<?, ?it/s]

    Warming up deepdense for 1 epochs


    epoch 1: 100%|██████████| 25/25 [00:00<00:00, 45.81it/s, loss=116]
      0%|          | 0/25 [00:00<?, ?it/s]

    Warming up deeptext for 1 epochs


    epoch 1: 100%|██████████| 25/25 [00:04<00:00,  5.37it/s, loss=132]
      0%|          | 0/25 [00:00<?, ?it/s]

    Warming up deepimage, layer 1 of 5


    epoch 1: 100%|██████████| 25/25 [01:10<00:00,  2.83s/it, loss=119]
      0%|          | 0/25 [00:00<?, ?it/s]

    Warming up deepimage, layer 2 of 5


    epoch 1: 100%|██████████| 25/25 [01:34<00:00,  3.76s/it, loss=108]
      0%|          | 0/25 [00:00<?, ?it/s]

    Warming up deepimage, layer 3 of 5


    epoch 1: 100%|██████████| 25/25 [01:57<00:00,  4.69s/it, loss=106]
      0%|          | 0/25 [00:00<?, ?it/s]

    Warming up deepimage, layer 4 of 5


    epoch 1: 100%|██████████| 25/25 [02:24<00:00,  5.79s/it, loss=105] 
      0%|          | 0/25 [00:00<?, ?it/s]

    Warming up deepimage, layer 5 of 5


    epoch 1: 100%|██████████| 25/25 [03:01<00:00,  7.26s/it, loss=105] 
      0%|          | 0/25 [00:00<?, ?it/s]

    Training


    epoch 1: 100%|██████████| 25/25 [02:05<00:00,  5.03s/it, loss=129]
    valid: 100%|██████████| 7/7 [00:14<00:00,  2.11s/it, loss=103] 


And one would access to the `felbo` routine by changing the `param`, `warm_routine` to `'felbo'` 
