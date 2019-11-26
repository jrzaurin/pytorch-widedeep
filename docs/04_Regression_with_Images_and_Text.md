## 1. Regression with Images and Text

In this notebook we will go through a series of examples on how to combine all Wide & Deep components, the Wide component (`wide`), the stack of dense layers for the "categorical embeddings" and numerical column (`deepdense`), the text data (`deeptext`) and images (`deepimage`). 

To that aim I will use the Airbnb listings dataset for London, which you can download from [here](http://insideairbnb.com/get-the-data.html). I have taken a sample of 1000 listings to keep the data tractable in this notebook. Also, I have preprocess the data and prepared it for this excercise. All preprocessing steps can be found in the notebook `airbnb_data_preprocessing.ipynb` in this `examples` folder. Note that you do not need to go through that notebook to get an understanding on how to use the `pytorch-widedeep` library. 


```python
import numpy as np
import pandas as pd
import os
import torch

from pytorch_widedeep.preprocessing import WidePreprocessor, DeepPreprocessor, TextPreprocessor, ImagePreprocessor
from pytorch_widedeep.models import Wide, DeepDense, DeepText, DeepImage, WideDeep
from pytorch_widedeep.initializers import *
from pytorch_widedeep.callbacks import *
from pytorch_widedeep.optim import RAdam
```


```python
df = pd.read_csv('data/airbnb/airbnb_sample.csv')
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
      <th>id</th>
      <th>host_id</th>
      <th>description</th>
      <th>host_listings_count</th>
      <th>host_identity_verified</th>
      <th>neighbourhood_cleansed</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>is_location_exact</th>
      <th>property_type</th>
      <th>room_type</th>
      <th>accommodates</th>
      <th>bathrooms</th>
      <th>bedrooms</th>
      <th>beds</th>
      <th>guests_included</th>
      <th>minimum_nights</th>
      <th>instant_bookable</th>
      <th>cancellation_policy</th>
      <th>has_house_rules</th>
      <th>host_gender</th>
      <th>accommodates_catg</th>
      <th>guests_included_catg</th>
      <th>minimum_nights_catg</th>
      <th>host_listings_count_catg</th>
      <th>bathrooms_catg</th>
      <th>bedrooms_catg</th>
      <th>beds_catg</th>
      <th>amenity_24-hour_check-in</th>
      <th>amenity__toilet</th>
      <th>amenity_accessible-height_bed</th>
      <th>amenity_accessible-height_toilet</th>
      <th>amenity_air_conditioning</th>
      <th>amenity_air_purifier</th>
      <th>amenity_alfresco_bathtub</th>
      <th>amenity_amazon_echo</th>
      <th>amenity_baby_bath</th>
      <th>amenity_baby_monitor</th>
      <th>amenity_babysitter_recommendations</th>
      <th>amenity_balcony</th>
      <th>amenity_bath_towel</th>
      <th>amenity_bathroom_essentials</th>
      <th>amenity_bathtub</th>
      <th>amenity_bathtub_with_bath_chair</th>
      <th>amenity_bbq_grill</th>
      <th>amenity_beach_essentials</th>
      <th>amenity_beach_view</th>
      <th>amenity_beachfront</th>
      <th>amenity_bed_linens</th>
      <th>amenity_bedroom_comforts</th>
      <th>...</th>
      <th>amenity_roll-in_shower</th>
      <th>amenity_room-darkening_shades</th>
      <th>amenity_safety_card</th>
      <th>amenity_sauna</th>
      <th>amenity_self_check-in</th>
      <th>amenity_shampoo</th>
      <th>amenity_shared_gym</th>
      <th>amenity_shared_hot_tub</th>
      <th>amenity_shared_pool</th>
      <th>amenity_shower_chair</th>
      <th>amenity_single_level_home</th>
      <th>amenity_ski-in_ski-out</th>
      <th>amenity_smart_lock</th>
      <th>amenity_smart_tv</th>
      <th>amenity_smoke_detector</th>
      <th>amenity_smoking_allowed</th>
      <th>amenity_soaking_tub</th>
      <th>amenity_sound_system</th>
      <th>amenity_stair_gates</th>
      <th>amenity_stand_alone_steam_shower</th>
      <th>amenity_standing_valet</th>
      <th>amenity_steam_oven</th>
      <th>amenity_stove</th>
      <th>amenity_suitable_for_events</th>
      <th>amenity_sun_loungers</th>
      <th>amenity_table_corner_guards</th>
      <th>amenity_tennis_court</th>
      <th>amenity_terrace</th>
      <th>amenity_toilet_paper</th>
      <th>amenity_touchless_faucets</th>
      <th>amenity_tv</th>
      <th>amenity_walk-in_shower</th>
      <th>amenity_warming_drawer</th>
      <th>amenity_washer</th>
      <th>amenity_washer_dryer</th>
      <th>amenity_waterfront</th>
      <th>amenity_well-lit_path_to_entrance</th>
      <th>amenity_wheelchair_accessible</th>
      <th>amenity_wide_clearance_to_shower</th>
      <th>amenity_wide_doorway_to_guest_bathroom</th>
      <th>amenity_wide_entrance</th>
      <th>amenity_wide_entrance_for_guests</th>
      <th>amenity_wide_entryway</th>
      <th>amenity_wide_hallways</th>
      <th>amenity_wifi</th>
      <th>amenity_window_guards</th>
      <th>amenity_wine_cooler</th>
      <th>security_deposit</th>
      <th>extra_people</th>
      <th>yield</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>13913.jpg</td>
      <td>54730</td>
      <td>My bright double bedroom with a large window has a relaxed feeling! It comfortably fits one or t...</td>
      <td>4.0</td>
      <td>f</td>
      <td>Islington</td>
      <td>51.56802</td>
      <td>-0.11121</td>
      <td>t</td>
      <td>apartment</td>
      <td>private_room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>f</td>
      <td>moderate</td>
      <td>1</td>
      <td>female</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>100.0</td>
      <td>15.0</td>
      <td>12.00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>15400.jpg</td>
      <td>60302</td>
      <td>Lots of windows and light.  St Luke's Gardens are at the end of the block, and the river not too...</td>
      <td>1.0</td>
      <td>t</td>
      <td>Kensington and Chelsea</td>
      <td>51.48796</td>
      <td>-0.16898</td>
      <td>t</td>
      <td>apartment</td>
      <td>entire_home/apt</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>3</td>
      <td>f</td>
      <td>strict_14_with_grace_period</td>
      <td>1</td>
      <td>female</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>150.0</td>
      <td>0.0</td>
      <td>109.50</td>
    </tr>
    <tr>
      <td>2</td>
      <td>17402.jpg</td>
      <td>67564</td>
      <td>Open from June 2018 after a 3-year break, we are delighted to be welcoming guests again to this ...</td>
      <td>19.0</td>
      <td>t</td>
      <td>Westminster</td>
      <td>51.52098</td>
      <td>-0.14002</td>
      <td>t</td>
      <td>apartment</td>
      <td>entire_home/apt</td>
      <td>6</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>4</td>
      <td>3</td>
      <td>t</td>
      <td>strict_14_with_grace_period</td>
      <td>1</td>
      <td>female</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>350.0</td>
      <td>10.0</td>
      <td>149.65</td>
    </tr>
    <tr>
      <td>3</td>
      <td>24328.jpg</td>
      <td>41759</td>
      <td>Artist house, bright high ceiling rooms, private parking and a communal garden in a conservation...</td>
      <td>2.0</td>
      <td>t</td>
      <td>Wandsworth</td>
      <td>51.47298</td>
      <td>-0.16376</td>
      <td>t</td>
      <td>other</td>
      <td>entire_home/apt</td>
      <td>2</td>
      <td>1.5</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>30</td>
      <td>f</td>
      <td>moderate</td>
      <td>1</td>
      <td>male</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>250.0</td>
      <td>0.0</td>
      <td>215.60</td>
    </tr>
    <tr>
      <td>4</td>
      <td>25023.jpg</td>
      <td>102813</td>
      <td>Large, all comforts, 2-bed flat; first floor; lift; pretty communal gardens + off-street parking...</td>
      <td>1.0</td>
      <td>f</td>
      <td>Wandsworth</td>
      <td>51.44687</td>
      <td>-0.21874</td>
      <td>t</td>
      <td>apartment</td>
      <td>entire_home/apt</td>
      <td>4</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2</td>
      <td>4</td>
      <td>f</td>
      <td>moderate</td>
      <td>1</td>
      <td>female</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>250.0</td>
      <td>11.0</td>
      <td>79.35</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 223 columns</p>
</div>



### 1.1 Regression with the defaults


```python
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

### 1.1.1 Prepare the data

I will focus here on how to prepare the data and run the model. Check notebooks 1 and 2 to see what's going on behind the scences

Preparing the data is rather simple


```python
target = df[target_col].values
```


```python
wide_preprocessor = WidePreprocessor(wide_cols=wide_cols, crossed_cols=crossed_cols)
X_wide = wide_preprocessor.fit_transform(df)
```


```python
deep_preprocessor = DeepPreprocessor(embed_cols=cat_embed_cols, continuous_cols=continuous_cols)
X_deep = deep_preprocessor.fit_transform(df)
```


```python
text_preprocessor = TextPreprocessor(word_vectors_path=word_vectors_path)
X_text = text_preprocessor.fit_transform(df, text_col)
```

    The vocabulary contains 6400 tokens
    Indexing word vectors...
    Loaded 400000 word vectors
    Preparing embeddings matrix...
    2175 words in the vocabulary had data/glove.6B/glove.6B.100d.txt vectors and appear more than 5 times



```python
image_processor = ImagePreprocessor()
X_images = image_processor.fit_transform(df, img_col, img_path)
```

    Reading Images from data/airbnb/property_picture


      9%|▊         | 86/1001 [00:00<00:02, 423.21it/s]

    Resizing


    100%|██████████| 1001/1001 [00:02<00:00, 423.31it/s]


    Computing normalisation metrics


### 1.1.2. Build the model components


```python
# Linear model
wide = Wide(wide_dim=X_wide.shape[1], output_dim=1)
# DeepDense: 2 Dense layers
deepdense = DeepDense(hidden_layers=[128,64], dropout=[0.5, 0.5], 
                      deep_column_idx=deep_preprocessor.deep_column_idx,
                      embed_input=deep_preprocessor.embeddings_input,
                      continuous_cols=continuous_cols)
# DeepText: a stack of 2 LSTMs
deeptext = DeepText(vocab_size=len(text_preprocessor.vocab.itos), hidden_dim=64, 
                    n_layers=2, rnn_dropout=0.5, 
                    embedding_matrix=text_preprocessor.embedding_matrix)
# Pretrained Resnet 18 (default is all but last 2 conv blocks frozen) plus a FC-Head 512->256->128
deepimage = DeepImage(pretrained=True, head_layers=[512, 256, 128])
```


```python
model = WideDeep(wide=wide, deepdense=deepdense, deeptext=deeptext, deepimage=deepimage)
```

### 1.1.3. Compile and fit


```python
model.compile(method='regression')
```


```python
model.fit(X_wide=X_wide, X_deep=X_deep, X_text=X_text, X_img=X_images,
    target=target, n_epochs=1, batch_size=32, val_split=0.2)
```

      0%|          | 0/25 [00:00<?, ?it/s]

    Training


    epoch 1: 100%|██████████| 25/25 [02:03<00:00,  4.93s/it, loss=118]
    valid: 100%|██████████| 7/7 [00:14<00:00,  2.06s/it, loss=99.3]


### 1.1.4. Warming up before training


```python
wide = Wide(wide_dim=X_wide.shape[1], output_dim=1)
deepdense = DeepDense(hidden_layers=[128,64], dropout=[0.5, 0.5], 
                      deep_column_idx=deep_preprocessor.deep_column_idx,
                      embed_input=deep_preprocessor.embeddings_input,
                      continuous_cols=continuous_cols)
deeptext = DeepText(vocab_size=len(text_preprocessor.vocab.itos), hidden_dim=64, 
                    n_layers=2, rnn_dropout=0.5, 
                    embedding_matrix=text_preprocessor.embedding_matrix)
deepimage = DeepImage(pretrained=True, head_layers=[512, 256, 128])
model = WideDeep(wide=wide, deepdense=deepdense, deeptext=deeptext, deepimage=deepimage)
model.compile(method='regression')
model.fit(X_wide=X_wide, X_deep=X_deep, X_text=X_text, X_img=X_images, target=target, n_epochs=1, 
          batch_size=32, val_split=0.2, warm_up=True, warm_epochs=1)
```

      0%|          | 0/25 [00:00<?, ?it/s]

    Warming up wide for 1 epochs


    epoch 1: 100%|██████████| 25/25 [00:00<00:00, 58.17it/s, loss=127]
      0%|          | 0/25 [00:00<?, ?it/s]

    Warming up deepdense for 1 epochs


    epoch 1: 100%|██████████| 25/25 [00:00<00:00, 44.23it/s, loss=115]
      0%|          | 0/25 [00:00<?, ?it/s]

    Warming up deeptext for 1 epochs


    epoch 1: 100%|██████████| 25/25 [00:03<00:00,  7.25it/s, loss=132]
      0%|          | 0/25 [00:00<?, ?it/s]

    Warming up deepimage for 1 epochs


    epoch 1: 100%|██████████| 25/25 [02:00<00:00,  4.83s/it, loss=122]
      0%|          | 0/25 [00:00<?, ?it/s]

    Training


    epoch 1: 100%|██████████| 25/25 [02:03<00:00,  4.95s/it, loss=105]
    valid: 100%|██████████| 7/7 [00:14<00:00,  2.02s/it, loss=91.1]


### 1.2 Regression with varying parameters and a FC-Head receiving the deep side

This would be the second architecture shown in the README file


```python
wide = Wide(wide_dim=X_wide.shape[1], output_dim=1)
deepdense = DeepDense(hidden_layers=[128,64], dropout=[0.5, 0.5], 
                      deep_column_idx=deep_preprocessor.deep_column_idx,
                      embed_input=deep_preprocessor.embeddings_input,
                      continuous_cols=continuous_cols)
deeptext = DeepText(vocab_size=len(text_preprocessor.vocab.itos), hidden_dim=128, 
                    n_layers=2, rnn_dropout=0.5, 
                    embedding_matrix=text_preprocessor.embedding_matrix)
deepimage = DeepImage(pretrained=True, head_layers=[512, 256, 128])
```

The **FC-Head** is passed as a parameter


```python
model = WideDeep(wide=wide, deepdense=deepdense, deeptext=deeptext, deepimage=deepimage, head_layers=[128, 64])
```

Let's have a look to the model


```python
model
```




    WideDeep(
      (wide): Wide(
        (wide_linear): Linear(in_features=356, out_features=1, bias=True)
      )
      (deepdense): DeepDense(
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
            (0): Linear(in_features=196, out_features=128, bias=True)
            (1): LeakyReLU(negative_slope=0.01, inplace=True)
            (2): Dropout(p=0.5, inplace=False)
          )
          (dense_layer_1): Sequential(
            (0): Linear(in_features=128, out_features=64, bias=True)
            (1): LeakyReLU(negative_slope=0.01, inplace=True)
            (2): Dropout(p=0.5, inplace=False)
          )
        )
      )
      (deeptext): DeepText(
        (word_embed): Embedding(2192, 100, padding_idx=1)
        (rnn): LSTM(100, 128, num_layers=2, batch_first=True, dropout=0.5)
      )
      (deepimage): DeepImage(
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
        (imagehead): Sequential(
          (dense_layer_0): Sequential(
            (0): Linear(in_features=512, out_features=256, bias=True)
            (1): LeakyReLU(negative_slope=0.01, inplace=True)
            (2): Dropout(p=0.0, inplace=False)
          )
          (dense_layer_1): Sequential(
            (0): Linear(in_features=256, out_features=128, bias=True)
            (1): LeakyReLU(negative_slope=0.01, inplace=True)
            (2): Dropout(p=0.0, inplace=False)
          )
        )
      )
      (deephead): Sequential(
        (head_layer_0): Sequential(
          (0): Linear(in_features=320, out_features=128, bias=True)
          (1): LeakyReLU(negative_slope=0.01, inplace=True)
          (2): Dropout(p=0.0, inplace=False)
        )
        (head_layer_1): Sequential(
          (0): Linear(in_features=128, out_features=64, bias=True)
          (1): LeakyReLU(negative_slope=0.01, inplace=True)
          (2): Dropout(p=0.0, inplace=False)
        )
        (head_out): Linear(in_features=64, out_features=1, bias=True)
      )
    )



Both, the Text and Image components allow FC-heads on their own (referred very creatively as `texthead` and `imagehead`). Following this nomenclature, the FC-head that receives the concatenation of the whole deep component is called `deephead`. 

Now let's go "kaggle crazy". Let's use different optimizers, initializers and schedulers for different components. Moreover, let's use a different learning rate for different parameter groups, for the `DeepDense` component


```python
deep_params = []
for childname, child in model.named_children():
    if childname == 'deepdense':
        for n,p in child.named_parameters():
            if "emb_layer" in n: deep_params.append({'params': p, 'lr': 1e-4})
            else: deep_params.append({'params': p, 'lr': 1e-3})
```


```python
wide_opt = torch.optim.Adam(model.wide.parameters())
deep_opt = torch.optim.Adam(deep_params)
text_opt = RAdam(model.deeptext.parameters())
img_opt  = RAdam(model.deepimage.parameters())
head_opt = torch.optim.Adam(model.deephead.parameters())
```


```python
wide_sch = torch.optim.lr_scheduler.StepLR(wide_opt, step_size=5)
deep_sch = torch.optim.lr_scheduler.MultiStepLR(deep_opt, milestones=[3,8])
text_sch = torch.optim.lr_scheduler.StepLR(text_opt, step_size=5)
img_sch  = torch.optim.lr_scheduler.MultiStepLR(deep_opt, milestones=[3,8])
head_sch = torch.optim.lr_scheduler.StepLR(head_opt, step_size=5)
```


```python
# remember, one optimizer per model components, for lr_schedures and initializers is not neccesary
optimizers = {'wide': wide_opt, 'deepdense':deep_opt, 'deeptext':text_opt, 'deepimage': img_opt, 'deephead': head_opt}
schedulers = {'wide': wide_sch, 'deepdense':deep_sch, 'deeptext':text_sch, 'deepimage': img_sch, 'deephead': head_sch}

# Now...we have used pretrained word embeddings, so you do not want to
# initialise these  embeddings. However you might still want to initialise the
# other layers in the DeepText component. No probs, you can do that with the
# parameter pattern and your knowledge on regular  expressions. Here we are
# telling to the KaimingNormal initializer to NOT touch the  parameters whose
# name contains the string word_embed. 
initializers = {'wide': KaimingNormal, 'deepdense':KaimingNormal, 
                'deeptext':KaimingNormal(pattern=r"^(?!.*word_embed).*$"), 
                'deepimage':KaimingNormal}

mean = [0.406, 0.456, 0.485]  #BGR
std =  [0.225, 0.224, 0.229]  #BGR
transforms = [ToTensor, Normalize(mean=mean, std=std)]
callbacks = [LRHistory(n_epochs=10), EarlyStopping, ModelCheckpoint(filepath='model_weights/wd_out')]
```


```python
model.compile(method='regression', initializers=initializers, optimizers=optimizers,
    lr_schedulers=schedulers, callbacks=callbacks, transforms=transforms)
```

    /Users/javier/pytorch-widedeep/pytorch_widedeep/initializers.py:32: UserWarning: No initializer found for deephead
      if self.verbose: warnings.warn("No initializer found for {}".format(name))



```python
model.fit(X_wide=X_wide, X_deep=X_deep, X_text=X_text, X_img=X_images,
    target=target, n_epochs=1, batch_size=32, val_split=0.2)
```

      0%|          | 0/25 [00:00<?, ?it/s]

    Training


    epoch 1: 100%|██████████| 25/25 [02:04<00:00,  4.97s/it, loss=129]
    valid: 100%|██████████| 7/7 [00:14<00:00,  2.06s/it, loss=95.6]



```python
# we have only run one epoch, but let's check that the LRHistory callback records the lr values for each group
model.lr_history
```




    {'lr_wide_0': [0.001, 0.001],
     'lr_deepdense_0': [0.0001, 0.0001],
     'lr_deepdense_1': [0.0001, 0.0001],
     'lr_deepdense_2': [0.0001, 0.0001],
     'lr_deepdense_3': [0.0001, 0.0001],
     'lr_deepdense_4': [0.0001, 0.0001],
     'lr_deepdense_5': [0.0001, 0.0001],
     'lr_deepdense_6': [0.0001, 0.0001],
     'lr_deepdense_7': [0.0001, 0.0001],
     'lr_deepdense_8': [0.0001, 0.0001],
     'lr_deepdense_9': [0.001, 0.001],
     'lr_deepdense_10': [0.001, 0.001],
     'lr_deepdense_11': [0.001, 0.001],
     'lr_deepdense_12': [0.001, 0.001],
     'lr_deeptext_0': [0.001, 0.001],
     'lr_deepimage_0': [0.001, 0.001],
     'lr_deephead_0': [0.001, 0.001]}


