# Wide-and-Deep-PyTorch
PyTorch implementation of Tensorflow's Wide and Deep Algorithm

This is a PyTorch implementation of Tensorflow's Wide and Deep Algorithm. Details of the algorithm can be found [here](https://www.tensorflow.org/tutorials/wide_and_deep) and the very nice research paper can be found [here](https://arxiv.org/abs/1606.07792). A `Keras` (quick and relatively dirty) implementation of the algorithm can be found [here](https://github.com/jrzaurin/Wide-and-Deep-Keras). 

## Requirements:

The algorithm was built using `python 2.7.13` and the required packages are:
```
pandas
numpy
scipy
sklearn
pytorch
```

## How to use it.

I have included 3 demos to explain how the data needs to be prepared, how the algorithm is built (the wide and deep parts separately) and how to use it. If you are familiar with the algorithm and you just want to give it a go, you can directly go to demo3 or have a look to main.py (which can be run as `python main.py` and has a few more details). Using it is as simple as this: 

### 1. Prepare the data
The wide-part and deep-part need to be specified as follows:
```    
import numpy as np
import pandas as pd

# Read the data
DF = pd.read_csv('data/adult_data.csv')

# target for logistic regression
DF['income_label'] = (DF["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

# Experiment set up
# WIDE PART
wide_cols = ['age','hours_per_week','education', 'relationship','workclass',
             'occupation','native_country','gender']
crossed_cols = (['education', 'occupation'], ['native_country', 'occupation'])

# DEEP PART
# columns that will be passed as embeddings and their corresponding number of embeddings (optional, see demo3). 
embeddings_cols = [('education',10), ('relationship',8), ('workclass',10),
                    ('occupation',10),('native_country',10)]
continuous_cols = ["age","hours_per_week"]

# TARGET AND METHOD
target = 'income_label'
method = 'logistic'

# PREPARE DATA
from wide_deep.data_utils import prepare_data
wd_dataset = prepare_data(DF, wide_cols,crossed_cols,embeddings_cols,continuous_cols,target)
```
### 2. Build the model
The model is built with the `WideDeep` class. To run on a GPU simply do `model = model.cuda()`.
``` 
# Network set up
wide_dim = wd_dataset['train_dataset'].wide.shape[1]
n_unique = len(np.unique(wd_dataset['train_dataset'].labels))
deep_column_idx = wd_dataset['deep_column_idx']
embeddings_input= wd_dataset['embeddings_input']
encoding_dict   = wd_dataset['encoding_dict']
hidden_layers = [100,50]
dropout = [0.5,0.2]
n_class=1

# Build the model
import torch
from wide_deep.torch_model import WideDeep

model = WideDeep(wide_dim,embeddings_input,continuous_cols,deep_column_idx,hidden_layers,dropout,encoding_dict,n_class)
model.compile(method=method)
if torch.cuda.is_available():
    model = model.cuda()
```

### 3. Fit and predict
```    
# Fit and predict as with your usual sklearn model
train_dataset = wd_dataset['train_dataset']
model.fit(dataset=train_dataset, n_epochs=10, batch_size=64)
test_dataset  = wd_dataset['test_dataset']
pred = model.predict(test_dataset)

# save
torch.save(model.state_dict(), 'model/logistic.pkl')
```

And that's it. 

## COMMENTS

Here I have illustrated how to use the model for binary classification. In `demo3` you can find how to use it for linear regression and multiclass classification. In the [research paper](https://arxiv.org/pdf/1606.07792.pdf) they show how they used it in the context of recommendation algorithms. 

In my experience, this algorithm shines where the data is complex (multiple categorical features with many levels) and rich (lots of observations). Normally, categorical features are preprocessed before passing them to a `sklearn` model using `LabeEncoder` (or `DictVectorizer`) or `OneHotEncoder`. `LabeEncoder` can turn `['sun', 'rain', wind', ...]` into `[1,2,3,..]` but the imposed ordinality means that the average between `sun` and `wind` will be `rain`. Some algorithms (e.g. decision trees) still work well with these label-encoded features, but it might be a problem. One option is to use `OneHotEncoder`, which results in binary features "living" in an orthogonal vector space. However, if you have many levels per feature the number of resulting one-hot encoded features might be too large. Instead, another option would be to use `Wide and Deep`, represent some categorical features with embeddings, and let the the model learn their representation through the deep layers.

A "real-world" example would be a recommendation algorithm that I recently implemented which relied mostly on a multiclass classification using `XGBoost`. `XGBoost` was challenged with a series of matrix factorization algorithms (`pyFM, lightFM, fastFM` in python or `MF and ALS` in pySpark) and always performed better. The only algorithm that obtained better metrics than XGBoost (meaning better `Mean Average Precision` on the ranked recommendations) was `Wide and Deep`.

Finally, a very interesting use of this algorithm consists of passing user and item features through the wide part and the user and item emebeddings through the deep part. Although **mathematically different**, this set up is conceptually similar to the Rendle's [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) in the sense that we can infer interest/score/ratings for unseen user-item interactions from the sparse set of seen interactions, using also user and item features. As a byproduct, we will also get the user and item embeddings. Essentially the two algorithms are trying to learn the embeddings based on existing user-item interactions, although the relations learned through the dense layers can be more "expressive" than those using Matrix Factorization. 

## TO DO:

1. Add some error message/handling.
2. Perhaps adding the Wide and Deep functionalities separately
3. Add a `keras_model.py` module to `wide_deep`
