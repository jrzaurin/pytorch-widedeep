import numpy as np
import torch
import pandas as pd
import pytest

from pytorch_widedeep.preprocessing import TextPreprocessor
from pytorch_widedeep.models import BasicRNN, WideDeep
from pytorch_widedeep.training import Trainer

from pytorch_widedeep.preprocessing.base_preprocessor import (
    BasePreprocessor,
    check_is_fitted,
)

from pytorch_widedeep.stream.preprocessing.text_preprocessor import StreamTextPreprocessor

from sklearn.datasets import fetch_20newsgroups

'''
Do experiments based on auto-regressive models - can generate samples to check if it works

1. Train text / language model and perform entire pipeline through the original API
2. Repeat same test for the stream API

'''
from typing import Union, List

def pipeline(X: pd.DataFrame, preprocessor: BasePreprocessor):
    if isinstance(preprocessor, StreamTextPreprocessor):
        X = preprocessor.fit_transform(X, chunksize=16) 
    else:
        X = preprocessor.fit_transform(X)
    basic_rnn = BasicRNN(vocab_size=len(preprocessor.vocab.itos), hidden_dim=20, n_layers=2, padding_idx=0, embed_dim=80)
    wd = WideDeep(deeptext=basic_rnn, pred_dim=len(categories))
    trainer = Trainer(model=wd, objective='multiclass')
    trainer.fit(X_text=X, target=y, n_epochs=10)
    pred = trainer.predict(X_text=X)
    return (sum(pred == y) / len(y))


twenty_train = fetch_20newsgroups()
X = pd.DataFrame(twenty_train['data'], columns=['text_col'])
y = twenty_train.target
categories = twenty_train.target_names


# tpp = TextPreprocessor('text_col')

stream_tpp = StreamTextPreprocessor('text_col')

acc = pipeline(X, stream_tpp)


#len(tpp.vocab.itos)


import pdb; pdb.set_trace()