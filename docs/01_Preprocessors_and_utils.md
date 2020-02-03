##  Processors and Utils

Description of the main tools and utilities that one needs to prepare the data for the `WideDeep` model constructor.

#### The `preprocessing`  module

There are 4 preprocessors, corresponding to 4 main components of the `WideDeep` model. These are

* `WidePreprocessor`
* `DeepPreprocessor`
* `TextPreprocessor`
* `ImagePreprocessor`

Behind the scenes, these preprocessors use a series of helper funcions and classes that are in the `utils` module.

#### The `utils` module

Initially I did not intend to "expose" them to the user, but I believe can be useful for all sorts of preprocessing tasks, so let me discuss them briefly.

The util tools in the module are:

* `deep_utils.label_encoder`
* `text_utils.simple_preprocess`
* `text_utils.get_texts`
* `text_utils.pad_sequences`
* `text_utils.build_embeddings_matrix`
* `fastai_transforms.Tokenizer`
* `fastai_transforms.Vocab`
* `image_utils.SimplePreprocessor`
* `image_utils.AspectAwarePreprocessor`

They are accessible directly from `utils` (e.g. `wd.utils.label_encoder`).

Let's have a look to what they do and how they might be useful to the user in general.

##  1. WidePreprocessor

This class simply takes a dataset and one-hot encodes it, with a few additional rings and bells.

For example


```python
import numpy as np
import pandas as pd
import pytorch_widedeep as wd

from pytorch_widedeep.preprocessing import WidePreprocessor
```


```python
# have a look to the documentation
?WidePreprocessor
```


```python
df = pd.read_csv("data/adult/adult.csv.zip")
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
wide_cols = ['education', 'relationship','workclass','occupation','native-country','gender']
crossed_cols = [('education', 'occupation'), ('native-country', 'occupation')]
```


```python
wide_preprocessor = WidePreprocessor(wide_cols=wide_cols, crossed_cols=crossed_cols)
X_wide = wide_preprocessor.fit_transform(df)
# From here on, any new observation can be prepared by simply running `.transform`
# new_X_wide = wide_preprocessor.transform(new_df)
```


```python
X_wide
```




    array([[0., 1., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]])



If you had a look to the documentation you will see that there is an option to return a sparse matrix. This will save memory on disk, but due to the batch generation process for `WideDeep` the running time will be notably slow. See [here](https://github.com/jrzaurin/pytorch-widedeep/blob/bfbe6e5d2309857db0dcc5cf3282dfa60504aa52/pytorch_widedeep/models/_wd_dataset.py#L47) for more details.

##  2. DeepPreprocessor

Label encodes the categorical columns and normalises the numerical ones (unless otherwised specified).


```python
from pytorch_widedeep.preprocessing import DeepPreprocessor
```


```python
?DeepPreprocessor
```


```python
# cat_embed_cols = [(column_name, embed_dim), ...]
cat_embed_cols = [('education',10), ('relationship',8), ('workclass',10), ('occupation',10),('native-country',10)]
continuous_cols = ["age","hours-per-week"]
```


```python
deep_preprocessor = DeepPreprocessor(embed_cols=cat_embed_cols, continuous_cols=continuous_cols)
X_deep = deep_preprocessor.fit_transform(df)
# From here on, any new observation can be prepared by simply running `.transform`
# new_X_deep = deep_preprocessor.transform(new_df)
```


```python
X_deep
```




    array([[ 0.        ,  0.        ,  0.        , ...,  0.        ,
            -0.99512893, -0.03408696],
           [ 1.        ,  1.        ,  0.        , ...,  0.        ,
            -0.04694151,  0.77292975],
           [ 2.        ,  1.        ,  1.        , ...,  0.        ,
            -0.77631645, -0.03408696],
           ...,
           [ 1.        ,  3.        ,  0.        , ...,  0.        ,
             1.41180837, -0.03408696],
           [ 1.        ,  0.        ,  0.        , ...,  0.        ,
            -1.21394141, -1.64812038],
           [ 1.        ,  4.        ,  6.        , ...,  0.        ,
             0.97418341, -0.03408696]])



Behing the scenes, `DeepProcessor` uses `label_encoder`, simply a numerical encoder for categorical featurers.

#### 2.1. `dense_utils`

You can access to `label_encoder` from utils


```python
import pytorch_widedeep as wd
```


```python
enc_df, enc_dict = wd.utils.label_encoder(df)
```


```python
enc_df.head()
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
      <td>0</td>
      <td>226802</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>38</td>
      <td>0</td>
      <td>89814</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>28</td>
      <td>1</td>
      <td>336951</td>
      <td>2</td>
      <td>12</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>44</td>
      <td>0</td>
      <td>160323</td>
      <td>3</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>7688</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>18</td>
      <td>2</td>
      <td>103497</td>
      <td>3</td>
      <td>10</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
enc_dict
```




    {'workclass': {'Private': 0,
      'Local-gov': 1,
      '?': 2,
      'Self-emp-not-inc': 3,
      'Federal-gov': 4,
      'State-gov': 5,
      'Self-emp-inc': 6,
      'Without-pay': 7,
      'Never-worked': 8},
     'education': {'11th': 0,
      'HS-grad': 1,
      'Assoc-acdm': 2,
      'Some-college': 3,
      '10th': 4,
      'Prof-school': 5,
      '7th-8th': 6,
      'Bachelors': 7,
      'Masters': 8,
      'Doctorate': 9,
      '5th-6th': 10,
      'Assoc-voc': 11,
      '9th': 12,
      '12th': 13,
      '1st-4th': 14,
      'Preschool': 15},
     'marital-status': {'Never-married': 0,
      'Married-civ-spouse': 1,
      'Widowed': 2,
      'Divorced': 3,
      'Separated': 4,
      'Married-spouse-absent': 5,
      'Married-AF-spouse': 6},
     'occupation': {'Machine-op-inspct': 0,
      'Farming-fishing': 1,
      'Protective-serv': 2,
      '?': 3,
      'Other-service': 4,
      'Prof-specialty': 5,
      'Craft-repair': 6,
      'Adm-clerical': 7,
      'Exec-managerial': 8,
      'Tech-support': 9,
      'Sales': 10,
      'Priv-house-serv': 11,
      'Transport-moving': 12,
      'Handlers-cleaners': 13,
      'Armed-Forces': 14},
     'relationship': {'Own-child': 0,
      'Husband': 1,
      'Not-in-family': 2,
      'Unmarried': 3,
      'Wife': 4,
      'Other-relative': 5},
     'race': {'Black': 0,
      'White': 1,
      'Asian-Pac-Islander': 2,
      'Other': 3,
      'Amer-Indian-Eskimo': 4},
     'gender': {'Male': 0, 'Female': 1},
     'native-country': {'United-States': 0,
      '?': 1,
      'Peru': 2,
      'Guatemala': 3,
      'Mexico': 4,
      'Dominican-Republic': 5,
      'Ireland': 6,
      'Germany': 7,
      'Philippines': 8,
      'Thailand': 9,
      'Haiti': 10,
      'El-Salvador': 11,
      'Puerto-Rico': 12,
      'Vietnam': 13,
      'South': 14,
      'Columbia': 15,
      'Japan': 16,
      'India': 17,
      'Cambodia': 18,
      'Poland': 19,
      'Laos': 20,
      'England': 21,
      'Cuba': 22,
      'Taiwan': 23,
      'Italy': 24,
      'Canada': 25,
      'Portugal': 26,
      'China': 27,
      'Nicaragua': 28,
      'Honduras': 29,
      'Iran': 30,
      'Scotland': 31,
      'Jamaica': 32,
      'Ecuador': 33,
      'Yugoslavia': 34,
      'Hungary': 35,
      'Hong': 36,
      'Greece': 37,
      'Trinadad&Tobago': 38,
      'Outlying-US(Guam-USVI-etc)': 39,
      'France': 40,
      'Holand-Netherlands': 41},
     'income': {'<=50K': 0, '>50K': 1}}



##  3. TextPreprocessor

This preprocessor returns the tokenised, padded sequences that will be directly fed to the stack of LSTMs.


```python
from pytorch_widedeep.preprocessing import TextPreprocessor
```


```python
?TextPreprocessor
```


```python
df=pd.read_csv("data/airbnb/airbnb_sample.csv")
```


```python
texts = df.description.tolist()
texts[:2]
```




    ["My bright double bedroom with a large window has a relaxed feeling! It comfortably fits one or two and is centrally located just two blocks from Finsbury Park. Enjoy great restaurants in the area and easy access to easy transport tubes, trains and buses. Babies and children of all ages are welcome. Hello Everyone, I'm offering my lovely double bedroom in Finsbury Park area (zone 2) for let in a shared apartment.  You will share the apartment with me and it is fully furnished with a self catering kitchen. Two people can easily sleep well as the room has a queen size bed. I also have a travel cot for a baby for guest with small children.  I will require a deposit up front as a security gesture on both our parts and will be given back to you when you return the keys.  I trust anyone who will be responding to this add would treat my home with care and respect .  Best Wishes  Alina Guest will have access to the self catering kitchen and bathroom. There is the flat is equipped wifi internet,",
     "Lots of windows and light.  St Luke's Gardens are at the end of the block, and the river not too far the other way. Ten minutes walk if you go slowly. Buses to everywhere round the corner and shops, restaurants, pubs, the cinema and Waitrose . Bright Chelsea Apartment  This is a bright one bedroom ground floor apartment in an interesting listed building. There is one double bedroom and a living room/kitchen The apartment has a full  bathroom and the kitchen is fully equipped. Two wardrobes are available exclusively for guests and bedside tables and two long drawers. This sunny convenient compact flat is just around the corner from the Waitrose supermarket and all sorts of shops, cinemas, restaurants and pubs.  This is a lovely part of London. There is a fun farmers market in the King's Road at the weekend.  Buses to everywhere are just round the corner, and two underground stations are within ten minutes walk. There is a very nice pub round by St. Luke's gardens, 4 mins slow walk, the "]




```python
text_preprocessor = TextPreprocessor(text_col='description')
X_text = text_preprocessor.fit_transform(df)
# From here on, any new observation can be prepared by simply running `.transform`
# new_X_text = text_preprocessor.transform(new_df)
```

    The vocabulary contains 6400 tokens



```python
print(X_text[0])
```

    [  29   48   37  367  818   17  910   17  177   15  122  349   53  879
     1174  126  393   40  911    0   23  228   71  819    9   53   55 1380
      225   11   18  308   18 1564   10  755    0  942  239   53   55    0
       11   36 1013  277 1974   70   62   15 1475    9  943    5  251    5
        0    5    0    5  177   53   37   75   11   10  294  726   32    9
       42    5   25   12   10   22   12  136  100  145]


`TextPreprocessor` uses the utilities within the `text_utils` and `fastai_utils` modules. Again, utilities there are directly accessible from `utils`.

#### 3.1. `text_utils` and `fastai_utils`

`text_utils` are: `simple_preprocess`, `get_texts`, `pad_sequences`, `build_embeddings_matrix`

`fastai_utils` is the fastai's [transforms](https://github.com/fastai/fastai/blob/master/fastai/text/transform.py) module (with some minor adaptations to function outside the fastai library). Therefore, **all credit to Jeremy Howard and his team**. The reason for using fastai's `transforms` module instead of using directly the library is simply because the library is gigantic, and has a large number of dependencies. I wanted to keep this package as light as possible, and the only functions/classes I need are in that module.

All utilities within that module are available in `pytorch-widedeep` via the `utils` module, so make sure you have a look because some functions there are quite handy. Here I will only focus on two: `Tokenizer` and `Vocab`.

Let's have a look to some of these utils


```python
tokens = wd.utils.get_texts(texts)
vocab = wd.utils.Vocab.create(tokens, max_vocab=2000, min_freq=1)
```


```python
vocab.stoi
```




    defaultdict(int,
                {'xxunk': 0,
                 'xxpad': 1,
                 'xxbos': 2,
                 'xxeos': 3,
                 'xxfld': 4,
                 'xxmaj': 5,
                 'xxup': 6,
                 'xxrep': 7,
                 'xxwrep': 8,
                 'and': 9,
                 'the': 10,
                 'to': 11,
                 'is': 12,
                 'in': 13,
                 'of': 14,
                 'with': 15,
                 'london': 16,
                 'for': 17,
                 'you': 18,
                 'room': 19,
                 'are': 20,
                 'from': 21,
                 'flat': 22,
                 'on': 23,
                 'bedroom': 24,
                 'there': 25,
                 'it': 26,
                 'walk': 27,
                 'double': 28,
                 'bed': 29,
                 'house': 30,
                 'has': 31,
                 'kitchen': 32,
                 'minutes': 33,
                 'apartment': 34,
                 'all': 35,
                 'this': 36,
                 'have': 37,
                 'very': 38,
                 'we': 39,
                 'as': 40,
                 'or': 41,
                 'bathroom': 42,
                 'central': 43,
                 'area': 44,
                 'station': 45,
                 'large': 46,
                 'can': 47,
                 'also': 48,
                 'street': 49,
                 'garden': 50,
                 'your': 51,
                 'away': 52,
                 'will': 53,
                 'two': 54,
                 'be': 55,
                 'one': 56,
                 'park': 57,
                 'quiet': 58,
                 'tube': 59,
                 'an': 60,
                 'restaurants': 61,
                 'home': 62,
                 'which': 63,
                 'just': 64,
                 'floor': 65,
                 'available': 66,
                 'great': 67,
                 'living': 68,
                 'located': 69,
                 'my': 70,
                 'our': 71,
                 'close': 72,
                 'if': 73,
                 'by': 74,
                 'access': 75,
                 'at': 76,
                 'lovely': 77,
                 'tv': 78,
                 'well': 79,
                 'comfortable': 80,
                 'private': 81,
                 'modern': 82,
                 'spacious': 83,
                 'use': 84,
                 'shops': 85,
                 'mins': 86,
                 'guests': 87,
                 'that': 88,
                 'shower': 89,
                 'road': 90,
                 'but': 91,
                 'space': 92,
                 'bus': 93,
                 'transport': 94,
                 'only': 95,
                 'city': 96,
                 'within': 97,
                 'free': 98,
                 'stay': 99,
                 'wifi': 100,
                 'so': 101,
                 'high': 102,
                 'single': 103,
                 'bright': 104,
                 'minute': 105,
                 'not': 106,
                 'min': 107,
                 'location': 108,
                 'easy': 109,
                 'fully': 110,
                 'line': 111,
                 'beautiful': 112,
                 'victorian': 113,
                 'please': 114,
                 'towels': 115,
                 'local': 116,
                 'short': 117,
                 'west': 118,
                 'clean': 119,
                 'out': 120,
                 'own': 121,
                 'small': 122,
                 'underground': 123,
                 'people': 124,
                 'new': 125,
                 'up': 126,
                 'bedrooms': 127,
                 'light': 128,
                 'walking': 129,
                 'provided': 130,
                 'market': 131,
                 'end': 132,
                 'dining': 133,
                 'good': 134,
                 'links': 135,
                 'equipped': 136,
                 'machine': 137,
                 'centre': 138,
                 'distance': 139,
                 'welcome': 140,
                 'family': 141,
                 'parking': 142,
                 'washing': 143,
                 'perfect': 144,
                 'internet': 145,
                 'shared': 146,
                 'place': 147,
                 'east': 148,
                 'no': 149,
                 'other': 150,
                 'coffee': 151,
                 'bars': 152,
                 'sofa': 153,
                 'open': 154,
                 'train': 155,
                 'situated': 156,
                 'green': 157,
                 'into': 158,
                 'south': 159,
                 'hill': 160,
                 'heart': 161,
                 'bath': 162,
                 'zone': 163,
                 'buses': 164,
                 'stations': 165,
                 'shopping': 166,
                 'cafes': 167,
                 'its': 168,
                 'over': 169,
                 'breakfast': 170,
                 'furnished': 171,
                 'need': 172,
                 'friendly': 173,
                 'pubs': 174,
                 'many': 175,
                 'table': 176,
                 'guest': 177,
                 'big': 178,
                 'rooms': 179,
                 'linen': 180,
                 'am': 181,
                 'excellent': 182,
                 'st': 183,
                 'get': 184,
                 'including': 185,
                 'plenty': 186,
                 'lounge': 187,
                 'wi': 188,
                 'wardrobe': 189,
                 'ground': 190,
                 'size': 191,
                 'views': 192,
                 'full': 193,
                 'king': 194,
                 'fridge': 195,
                 'fi': 196,
                 'lots': 197,
                 'around': 198,
                 'about': 199,
                 'enjoy': 200,
                 'building': 201,
                 'ideal': 202,
                 'balcony': 203,
                 'nearby': 204,
                 'night': 205,
                 'amenities': 206,
                 'most': 207,
                 'here': 208,
                 'facilities': 209,
                 'day': 210,
                 'property': 211,
                 'any': 212,
                 'throughout': 213,
                 'where': 214,
                 'cosy': 215,
                 'victoria': 216,
                 'separate': 217,
                 'terrace': 218,
                 'time': 219,
                 'more': 220,
                 'floors': 221,
                 'safe': 222,
                 'like': 223,
                 'microwave': 224,
                 'back': 225,
                 'beds': 226,
                 'residential': 227,
                 'both': 228,
                 'town': 229,
                 'do': 230,
                 'plan': 231,
                 'shoreditch': 232,
                 'bridge': 233,
                 'tea': 234,
                 'neighbourhood': 235,
                 'famous': 236,
                 'live': 237,
                 'top': 238,
                 'who': 239,
                 'suite': 240,
                 'newly': 241,
                 'take': 242,
                 'feel': 243,
                 'river': 244,
                 'etc': 245,
                 'three': 246,
                 'me': 247,
                 'attractions': 248,
                 'first': 249,
                 'some': 250,
                 'best': 251,
                 'windows': 252,
                 'after': 253,
                 'overground': 254,
                 'make': 255,
                 'second': 256,
                 'few': 257,
                 'near': 258,
                 'kensington': 259,
                 'main': 260,
                 'storage': 261,
                 'en': 262,
                 'decorated': 263,
                 'desk': 264,
                 'dryer': 265,
                 'gardens': 266,
                 'studio': 267,
                 'oxford': 268,
                 'square': 269,
                 'stylish': 270,
                 'airy': 271,
                 'nice': 272,
                 'next': 273,
                 'off': 274,
                 'camden': 275,
                 'village': 276,
                 'would': 277,
                 'stop': 278,
                 'peaceful': 279,
                 'thames': 280,
                 'included': 281,
                 'couple': 282,
                 'right': 283,
                 'everything': 284,
                 'door': 285,
                 'old': 286,
                 'sunny': 287,
                 'overlooking': 288,
                 'features': 289,
                 'dalston': 290,
                 'during': 291,
                 'trendy': 292,
                 'share': 293,
                 'self': 294,
                 'sleep': 295,
                 'offer': 296,
                 'heathrow': 297,
                 'stops': 298,
                 'north': 299,
                 'happy': 300,
                 'olympic': 301,
                 'dishwasher': 302,
                 'outside': 303,
                 'includes': 304,
                 'stratford': 305,
                 'extra': 306,
                 'art': 307,
                 'when': 308,
                 'toilet': 309,
                 'leafy': 310,
                 'food': 311,
                 'public': 312,
                 'vibrant': 313,
                 'tower': 314,
                 'ride': 315,
                 'work': 316,
                 'oven': 317,
                 'looking': 318,
                 'business': 319,
                 'want': 320,
                 'find': 321,
                 'airbnb': 322,
                 'll': 323,
                 'than': 324,
                 'direct': 325,
                 'see': 326,
                 'islington': 327,
                 'block': 328,
                 'go': 329,
                 'cross': 330,
                 'wooden': 331,
                 'rent': 332,
                 'been': 333,
                 'airport': 334,
                 'note': 335,
                 'built': 336,
                 'fitted': 337,
                 'long': 338,
                 'part': 339,
                 're': 340,
                 'hackney': 341,
                 'love': 342,
                 'drawers': 343,
                 'person': 344,
                 'canal': 345,
                 'plus': 346,
                 'super': 347,
                 'wharf': 348,
                 'children': 349,
                 'recently': 350,
                 'check': 351,
                 'facing': 352,
                 'original': 353,
                 'between': 354,
                 'piccadilly': 355,
                 'th': 356,
                 'areas': 357,
                 'lane': 358,
                 'fantastic': 359,
                 'luxury': 360,
                 'suitable': 361,
                 'view': 362,
                 'price': 363,
                 'help': 364,
                 'markets': 365,
                 'dvd': 366,
                 'travel': 367,
                 'convenient': 368,
                 'contemporary': 369,
                 'period': 370,
                 'kings': 371,
                 'parks': 372,
                 'wimbledon': 373,
                 'canary': 374,
                 'heating': 375,
                 'corner': 376,
                 'fresh': 377,
                 'broadway': 378,
                 'comfy': 379,
                 'huge': 380,
                 'palace': 381,
                 'offers': 382,
                 'accommodation': 383,
                 'ensuite': 384,
                 'provide': 385,
                 'secure': 386,
                 'set': 387,
                 'host': 388,
                 'westfield': 389,
                 'rail': 390,
                 'hidden': 391,
                 'let': 392,
                 'front': 393,
                 'amazing': 394,
                 'style': 395,
                 'experience': 396,
                 'notting': 397,
                 'screen': 398,
                 'lines': 399,
                 'trains': 400,
                 'hampstead': 401,
                 'portobello': 402,
                 'little': 403,
                 'wireless': 404,
                 'every': 405,
                 'four': 406,
                 'base': 407,
                 'major': 408,
                 'district': 409,
                 'patio': 410,
                 'greenwich': 411,
                 'us': 412,
                 'making': 413,
                 'sharing': 414,
                 'nearest': 415,
                 'entire': 416,
                 'chest': 417,
                 'onto': 418,
                 'water': 419,
                 'broadband': 420,
                 'really': 421,
                 'too': 422,
                 'bedding': 423,
                 'freezer': 424,
                 'sleeps': 425,
                 'supermarkets': 426,
                 'circus': 427,
                 'whole': 428,
                 'summer': 429,
                 'roof': 430,
                 'takes': 431,
                 'supermarket': 432,
                 'bathrooms': 433,
                 'service': 434,
                 'sq': 435,
                 'standard': 436,
                 'refurbished': 437,
                 'each': 438,
                 'relaxing': 439,
                 'renovated': 440,
                 'warm': 441,
                 'brick': 442,
                 'mattress': 443,
                 'cinema': 444,
                 'sized': 445,
                 'cooking': 446,
                 'connections': 447,
                 'places': 448,
                 'their': 449,
                 'window': 450,
                 'sitting': 451,
                 'across': 452,
                 'along': 453,
                 'per': 454,
                 'wood': 455,
                 'they': 456,
                 'washer': 457,
                 'term': 458,
                 'down': 459,
                 'fast': 460,
                 'ceilings': 461,
                 'brand': 462,
                 'centrally': 463,
                 'furniture': 464,
                 'doorstep': 465,
                 'communal': 466,
                 'come': 467,
                 'stunning': 468,
                 'year': 469,
                 'chair': 470,
                 'smoking': 471,
                 'comes': 472,
                 'working': 473,
                 'ask': 474,
                 'such': 475,
                 'yourself': 476,
                 'staying': 477,
                 'quality': 478,
                 'hour': 479,
                 'loft': 480,
                 'wonderful': 481,
                 'car': 482,
                 'tennis': 483,
                 'waterloo': 484,
                 'visit': 485,
                 'don': 486,
                 'explore': 487,
                 'twin': 488,
                 'tourist': 489,
                 'popular': 490,
                 'wc': 491,
                 'listing': 492,
                 'via': 493,
                 'outdoor': 494,
                 'five': 495,
                 'smart': 496,
                 'chairs': 497,
                 'liverpool': 498,
                 'chelsea': 499,
                 'then': 500,
                 'before': 501,
                 'phone': 502,
                 'kettle': 503,
                 'freeview': 504,
                 'much': 505,
                 'court': 506,
                 'contained': 507,
                 'comfortably': 508,
                 'queen': 509,
                 'several': 510,
                 'was': 511,
                 'being': 512,
                 'friends': 513,
                 'master': 514,
                 'welcoming': 515,
                 'pretty': 516,
                 'world': 517,
                 'families': 518,
                 'know': 519,
                 'rd': 520,
                 'charming': 521,
                 'unique': 522,
                 'speed': 523,
                 'those': 524,
                 'hammersmith': 525,
                 'week': 526,
                 'connection': 527,
                 'morning': 528,
                 'years': 529,
                 'newington': 530,
                 'appliances': 531,
                 'converted': 532,
                 'couples': 533,
                 'brixton': 534,
                 'theatre': 535,
                 'give': 536,
                 'museum': 537,
                 'book': 538,
                 'quite': 539,
                 'clothes': 540,
                 'third': 541,
                 'hyde': 542,
                 'now': 543,
                 'entrance': 544,
                 'iron': 545,
                 'unlimited': 546,
                 'busy': 547,
                 'may': 548,
                 'borough': 549,
                 'booking': 550,
                 'terraced': 551,
                 'always': 552,
                 'beautifully': 553,
                 'what': 554,
                 'designed': 555,
                 'include': 556,
                 'doors': 557,
                 'number': 558,
                 'mod': 559,
                 'battersea': 560,
                 'lift': 561,
                 'fields': 562,
                 'cable': 563,
                 'overlooks': 564,
                 'eat': 565,
                 'exploring': 566,
                 'tree': 567,
                 'player': 568,
                 'additional': 569,
                 'natural': 570,
                 'through': 571,
                 'bank': 572,
                 'cons': 573,
                 'accommodate': 574,
                 'way': 575,
                 'used': 576,
                 'northern': 577,
                 'angel': 578,
                 'relax': 579,
                 'rest': 580,
                 'days': 581,
                 'visiting': 582,
                 'easily': 583,
                 'bush': 584,
                 'holiday': 585,
                 'even': 586,
                 'stoke': 587,
                 'development': 588,
                 'look': 589,
                 'gorgeous': 590,
                 'journey': 591,
                 'gatwick': 592,
                 'power': 593,
                 'common': 594,
                 'pm': 595,
                 'real': 596,
                 'airports': 597,
                 'charge': 598,
                 'required': 599,
                 'relaxed': 600,
                 'variety': 601,
                 'request': 602,
                 'travelling': 603,
                 'french': 604,
                 'clubs': 605,
                 'ft': 606,
                 'music': 607,
                 'hotel': 608,
                 'hairdryer': 609,
                 'could': 610,
                 'clapham': 611,
                 'ealing': 612,
                 'meals': 613,
                 'flooring': 614,
                 'upper': 615,
                 'wide': 616,
                 'sheets': 617,
                 'stadium': 618,
                 'columbia': 619,
                 'life': 620,
                 'child': 621,
                 'less': 622,
                 'while': 623,
                 'things': 624,
                 'apartments': 625,
                 'sun': 626,
                 'eye': 627,
                 'either': 628,
                 'under': 629,
                 'rear': 630,
                 'needed': 631,
                 'lined': 632,
                 'arrival': 633,
                 'tesco': 634,
                 'travellers': 635,
                 'yet': 636,
                 'cafe': 637,
                 'flower': 638,
                 'books': 639,
                 'heath': 640,
                 'cool': 641,
                 'side': 642,
                 'royal': 643,
                 'questions': 644,
                 'privacy': 645,
                 'offering': 646,
                 'another': 647,
                 'complete': 648,
                 'further': 649,
                 'range': 650,
                 'reception': 651,
                 'fulham': 652,
                 'hair': 653,
                 'traditional': 654,
                 'made': 655,
                 'whilst': 656,
                 'hi': 657,
                 'designer': 658,
                 'seating': 659,
                 'late': 660,
                 'opposite': 661,
                 'cat': 662,
                 'key': 663,
                 'junction': 664,
                 'directly': 665,
                 'tidy': 666,
                 'read': 667,
                 'able': 668,
                 'homely': 669,
                 'regent': 670,
                 'hot': 671,
                 'selection': 672,
                 'diner': 673,
                 'surrounded': 674,
                 'highbury': 675,
                 'ben': 676,
                 'cooker': 677,
                 'pancras': 678,
                 'international': 679,
                 'ten': 680,
                 'reach': 681,
                 'quick': 682,
                 'lot': 683,
                 'bbq': 684,
                 'filled': 685,
                 'historic': 686,
                 'routes': 687,
                 'fireplace': 688,
                 'consists': 689,
                 'ideally': 690,
                 'gallery': 691,
                 'covent': 692,
                 'hand': 693,
                 'hob': 694,
                 'level': 695,
                 'non': 696,
                 'jubilee': 697,
                 'houses': 698,
                 'toaster': 699,
                 'same': 700,
                 'conveniently': 701,
                 'maisonette': 702,
                 'uk': 703,
                 'georgian': 704,
                 'bethnal': 705,
                 'paddington': 706,
                 'hoxton': 707,
                 'miles': 708,
                 'cozy': 709,
                 'site': 710,
                 'air': 711,
                 'stays': 712,
                 'pleasant': 713,
                 'finsbury': 714,
                 'office': 715,
                 'bar': 716,
                 'richmond': 717,
                 'british': 718,
                 'pool': 719,
                 'channels': 720,
                 'calm': 721,
                 'vintage': 722,
                 'galleries': 723,
                 'accessible': 724,
                 'opens': 725,
                 'catering': 726,
                 'ceiling': 727,
                 'cleaning': 728,
                 'spaces': 729,
                 'italian': 730,
                 'design': 731,
                 'lively': 732,
                 'sightseeing': 733,
                 'professional': 734,
                 'comfort': 735,
                 'going': 736,
                 'cost': 737,
                 'soho': 738,
                 'far': 739,
                 'choice': 740,
                 'leisure': 741,
                 'inside': 742,
                 'benefits': 743,
                 'restaurant': 744,
                 'english': 745,
                 'character': 746,
                 'trip': 747,
                 'however': 748,
                 'gym': 749,
                 'whitechapel': 750,
                 'utensils': 751,
                 'numerous': 752,
                 'national': 753,
                 'longer': 754,
                 'keys': 755,
                 'listed': 756,
                 'conversion': 757,
                 'sink': 758,
                 'decor': 759,
                 'swimming': 760,
                 'toiletries': 761,
                 'hub': 762,
                 'split': 763,
                 'extremely': 764,
                 'although': 765,
                 'continental': 766,
                 'system': 767,
                 'gate': 768,
                 'them': 769,
                 'gas': 770,
                 'wish': 771,
                 'sky': 772,
                 'fashionable': 773,
                 'wardrobes': 774,
                 'allowed': 775,
                 'times': 776,
                 'theatres': 777,
                 'library': 778,
                 've': 779,
                 'de': 780,
                 'moments': 781,
                 'heated': 782,
                 'boasts': 783,
                 'downstairs': 784,
                 'availability': 785,
                 'cats': 786,
                 'steps': 787,
                 'courtyard': 788,
                 'electric': 789,
                 'edwardian': 790,
                 'services': 791,
                 'cook': 792,
                 'possible': 793,
                 'hello': 794,
                 'interesting': 795,
                 'pub': 796,
                 'conservation': 797,
                 'contains': 798,
                 'metro': 799,
                 'based': 800,
                 'sites': 801,
                 'museums': 802,
                 'renting': 803,
                 'railway': 804,
                 'evening': 805,
                 'got': 806,
                 'concierge': 807,
                 'getting': 808,
                 'westminster': 809,
                 'arranged': 810,
                 'parliament': 811,
                 'dlr': 812,
                 'known': 813,
                 'reviews': 814,
                 'holland': 815,
                 'shop': 816,
                 'tubes': 817,
                 'cot': 818,
                 'parts': 819,
                 'ready': 820,
                 'convenience': 821,
                 'excel': 822,
                 'still': 823,
                 'hanging': 824,
                 'cupboard': 825,
                 'store': 826,
                 'leading': 827,
                 'provides': 828,
                 'bike': 829,
                 'antique': 830,
                 'connected': 831,
                 'perfectly': 832,
                 'buckingham': 833,
                 'cotton': 834,
                 'found': 835,
                 'exclusive': 836,
                 'upstairs': 837,
                 'piano': 838,
                 'tate': 839,
                 'kingsize': 840,
                 'middle': 841,
                 'lived': 842,
                 'run': 843,
                 'express': 844,
                 'bookings': 845,
                 'stores': 846,
                 'months': 847,
                 'throw': 848,
                 'simple': 849,
                 'myself': 850,
                 'eurostar': 851,
                 'shepherds': 852,
                 'bathtub': 853,
                 'sure': 854,
                 'course': 855,
                 'sights': 856,
                 'history': 857,
                 'owner': 858,
                 'luxurious': 859,
                 'young': 860,
                 'glass': 861,
                 'dinner': 862,
                 'croydon': 863,
                 'these': 864,
                 'ironing': 865,
                 'board': 866,
                 'keep': 867,
                 'website': 868,
                 'foot': 869,
                 'itself': 870,
                 'nd': 871,
                 'wash': 872,
                 'nights': 873,
                 'earls': 874,
                 'taking': 875,
                 'meet': 876,
                 'streatham': 877,
                 'expect': 878,
                 'require': 879,
                 'desirable': 880,
                 'inch': 881,
                 'community': 882,
                 'multi': 883,
                 'drive': 884,
                 'different': 885,
                 'mall': 886,
                 'forest': 887,
                 'looks': 888,
                 'kilburn': 889,
                 'adults': 890,
                 'regents': 891,
                 'grove': 892,
                 'prefer': 893,
                 'white': 894,
                 'atmosphere': 895,
                 'group': 896,
                 'spare': 897,
                 'comprises': 898,
                 'queens': 899,
                 'wine': 900,
                 'hall': 901,
                 'seven': 902,
                 'hosts': 903,
                 'visitors': 904,
                 'put': 905,
                 'because': 906,
                 'riverside': 907,
                 'flats': 908,
                 'everyone': 909,
                 'baby': 910,
                 'security': 911,
                 'bedside': 912,
                 'arty': 913,
                 'mattresses': 914,
                 'gated': 915,
                 'stone': 916,
                 'noise': 917,
                 'contact': 918,
                 'attractive': 919,
                 'elegant': 920,
                 'information': 921,
                 'maximum': 922,
                 'approx': 923,
                 'sash': 924,
                 'tram': 925,
                 'television': 926,
                 'fibre': 927,
                 'walks': 928,
                 'sofas': 929,
                 'upon': 930,
                 'study': 931,
                 'bit': 932,
                 'metres': 933,
                 'winter': 934,
                 'arena': 935,
                 'tooting': 936,
                 'below': 937,
                 'storey': 938,
                 'boutiques': 939,
                 'dog': 940,
                 'hours': 941,
                 'anyone': 942,
                 'respect': 943,
                 'cinemas': 944,
                 'makes': 945,
                 'cul': 946,
                 'colourful': 947,
                 'stroll': 948,
                 'furnishings': 949,
                 'photos': 950,
                 'called': 951,
                 'mansion': 952,
                 'though': 953,
                 'streets': 954,
                 'fabulous': 955,
                 'linens': 956,
                 'quirky': 957,
                 'early': 958,
                 'completely': 959,
                 'cottage': 960,
                 'pillows': 961,
                 'touch': 962,
                 'laundry': 963,
                 'kennington': 964,
                 'six': 965,
                 'adjacent': 966,
                 'usually': 967,
                 'currently': 968,
                 'radio': 969,
                 'together': 970,
                 'transportation': 971,
                 'tub': 972,
                 'fun': 973,
                 'artist': 974,
                 'complex': 975,
                 'comforts': 976,
                 'bustle': 977,
                 'tourists': 978,
                 'ample': 979,
                 'persons': 980,
                 'kids': 981,
                 'sac': 982,
                 'traffic': 983,
                 'europe': 984,
                 'stairs': 985,
                 'minimum': 986,
                 'chiswick': 987,
                 'meal': 988,
                 'enough': 989,
                 'circle': 990,
                 'point': 991,
                 'behind': 992,
                 'description': 993,
                 'sleeping': 994,
                 'mirror': 995,
                 'semi': 996,
                 'weekends': 997,
                 'necessary': 998,
                 'oak': 999,
                 ...})




```python
sequences = [vocab.numericalize(t) for t in tokens]
padded_seq = [wd.utils.pad_sequences(s, maxlen=200, pad_idx=1) for s in sequences]
```


```python
padded_seq[0]
```




    array([   1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
              1,    1,    5,   70,  104,   28,   24,   15,   46,  450,   31,
            600, 1173,    5,   26,  508, 1299,   56,   41,   54,    9,   12,
            463,   69,   64,   54, 1474,   21,    5,  714,    5,   57,    5,
            200,   67,   61,   13,   10,   44,    9,  109,   75,   11,  109,
             94,  817,  400,    9,  164,    5, 1973,    9,  349,   14,   35,
              0,   20,  140,    5,  794,    5,  909,  646,   70,   77,   28,
             24,   13,    5,  714,    5,   57,   44,  163,   17,  392,   13,
            146,   34,    5,   18,   53,  293,   10,   34,   15,  247,    9,
             26,   12,  110,  171,   15,  294,  726,   32,    5,   54,  124,
             47,  583,  295,   79,   40,   10,   19,   31,  509,  191,   29,
             48,   37,  367,  818,   17,  910,   17,  177,   15,  122,  349,
             53,  879, 1174,  126,  393,   40,  911,    0,   23,  228,   71,
            819,    9,   53,   55, 1380,  225,   11,   18,  308,   18, 1564,
             10,  755,    0,  942,  239,   53,   55,    0,   11,   36, 1013,
            277, 1974,   70,   62,   15, 1475,    9,  943,    5,  251,    5,
              0,    5,    0,    5,  177,   53,   37,   75,   11,   10,  294,
            726,   32,    9,   42,    5,   25,   12,   10,   22,   12,  136,
            100,  145], dtype=int32)



## 4. ImagePreprocessor

`ImagePreprocessor` simply resizes the images, being aware of the aspect ratio.


```python
image_preprocessor = wd.preprocessing.ImagePreprocessor(img_col='id', img_path="data/airbnb/property_picture/")
X_images = image_preprocessor.fit_transform(df)
# From here on, any new observation can be prepared by simply running `.transform`
# new_X_images = image_preprocessor.transform(new_df)
```

    Reading Images from data/airbnb/property_picture/


      4%|▍         | 40/1001 [00:00<00:02, 391.71it/s]

    Resizing


    100%|██████████| 1001/1001 [00:02<00:00, 415.12it/s]


    Computing normalisation metrics



```python
X_images[0].shape
```




    (224, 224, 3)



`ImagePreprocessor` uses two helpers: `SimplePreprocessor` and `AspectAwarePreprocessor`

These two classes are directly taken from Adrian Rosebrock's fantastic book ["Deep Learning for Computer Vision"](https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/). Therefore, **all credit to Adrian**.

Let's see what they do


```python
import cv2
from os import listdir
from random import sample

prop_img_path = "data/airbnb/property_picture/"
prop_imgnames = listdir(prop_img_path)
prop_imgnames = sample(prop_imgnames, 10)
print(prop_imgnames)
```

    ['512853.jpg', '460396.jpg', '92352.jpg', '472203.jpg', '534665.jpg', '529070.jpg', '549281.jpg', '499163.jpg', '218915.jpg', '526627.jpg']



```python
prop_imgs = [cv2.imread(str(prop_img_path+img)) for img in prop_imgnames]
prop_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in prop_imgs]
```


```python
import matplotlib.pyplot as plt
%matplotlib inline

def show_img(im, figsize=(5,5), ax=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.set_axis_off()
    return ax
```


```python
fig,axes = plt.subplots(2, 5, figsize=(16,5))
for i,im in enumerate(prop_imgs):
    show_img(im, ax=axes.flat[i])
```


![png](figures/01_Preprocessors_and_utils_40_0.png)



```python
print([im.shape for im in prop_imgs])
```

    [(426, 639, 3), (426, 639, 3), (426, 639, 3), (426, 639, 3), (426, 360, 3), (426, 639, 3), (426, 639, 3), (426, 639, 3), (426, 639, 3), (426, 639, 3)]



```python
spp = wd.utils.SimplePreprocessor(224,224)
prop_resized_imgs = [spp.preprocess(im) for im in prop_imgs]
```


```python
fig,axes = plt.subplots(2, 5, figsize=(16,5))
for i,im in enumerate(prop_resized_imgs):
    show_img(im, ax=axes.flat[i])
```


![png](figures/01_Preprocessors_and_utils_43_0.png)



```python
print([im.shape for im in prop_resized_imgs])
```

    [(224, 224, 3), (224, 224, 3), (224, 224, 3), (224, 224, 3), (224, 224, 3), (224, 224, 3), (224, 224, 3), (224, 224, 3), (224, 224, 3), (224, 224, 3)]



```python
aap = wd.utils.AspectAwarePreprocessor(width=224, height=224)
prop_aap_resized_imgs = [aap.preprocess(im) for im in prop_imgs]
```


```python
fig,axes = plt.subplots(2, 5, figsize=(16,5))
for i,im in enumerate(prop_aap_resized_imgs):
    show_img(im, ax=axes.flat[i])
```


![png](figures/01_Preprocessors_and_utils_46_0.png)



```python
print([im.shape for im in prop_aap_resized_imgs])
```

    [(224, 224, 3), (224, 224, 3), (224, 224, 3), (224, 224, 3), (224, 224, 3), (224, 224, 3), (224, 224, 3), (224, 224, 3), (224, 224, 3), (224, 224, 3)]

