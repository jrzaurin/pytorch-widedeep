import numpy as np
import torch
import pandas as pd
from typing import Union, List
import pytest

from pytorch_widedeep.preprocessing import TextPreprocessor
from pytorch_widedeep.models import BasicRNN, WideDeep
from pytorch_widedeep.training import Trainer

from pytorch_widedeep.preprocessing.base_preprocessor import (
    BasePreprocessor,
    check_is_fitted,
)

from pytorch_widedeep.stream.preprocessing.text_preprocessor import StreamTextPreprocessor, Vocab, VocabBuilder
from pytorch_widedeep.stream.training.trainer import StreamTrainer

from sklearn.datasets import fetch_20newsgroups

'''
Do experiments based on auto-regressive models - can generate samples to check if it works

1. Train text / language model and perform entire pipeline through the original API
2. Repeat same test for the stream API

'''

def batch_pipeline(X: pd.DataFrame, preprocessor: BasePreprocessor):
    X = preprocessor.fit_transform(X)
    basic_rnn = BasicRNN(vocab_size=len(preprocessor.vocab.itos), hidden_dim=20, n_layers=2, padding_idx=0, embed_dim=80)
    wd = WideDeep(deeptext=basic_rnn, pred_dim=len(categories))
    trainer = Trainer(model=wd, objective='multiclass')

    trainer.fit(X_text=X, target=y, n_epochs=10)
    pred = trainer.predict(X_text=X)
    return (sum(pred == y) / len(y))

def stream_pipeline(X_path: str, preprocessor: BasePreprocessor, chunksize: int):
    spp = preprocessor.fit(X_path, chunksize=chunksize) 

    basic_rnn = BasicRNN(vocab_size=len(preprocessor.vocab.itos), hidden_dim=20, n_layers=2, padding_idx=0, embed_dim=80)
    wd = WideDeep(deeptext=basic_rnn, pred_dim=len(categories))
    trainer = Trainer(model=wd, objective='multiclass')

    import pdb; pdb.set_trace()
    # trainer.fit(X_text=X, target=y, n_epochs=10)
    # pred = trainer.predict(X_text=X)
    # return (sum(pred == y) / len(y))


# tpp = TextPreprocessor('text_col')
# acc = pipeline(X=None, X_path='stream_mc_test.csv', preprocessor=stream_tpp, chunksize=4096)

# Investigate IterableDataset in dataloaders 
# We want this to be inside the trainer API


# assert len(stream_tpp.vocab.itos) == 28474

#len(tpp.vocab.itos)

twenty_train = fetch_20newsgroups()
X = pd.DataFrame(twenty_train['data'], columns=['text_col'])
# X.to_csv('stream_mc_test.csv')
y = twenty_train.target
categories = twenty_train.target_names

X_path = './stream_mc_test.csv'

# batch_pipeline(X, TextPreprocessor('text_col'))

text_preproc = StreamTextPreprocessor('text_col')
text_preproc.fit(X_path, 1024*100)

basic_rnn = BasicRNN(vocab_size=len(text_preproc.vocab.itos), hidden_dim=20, n_layers=2, padding_idx=0, embed_dim=80)
wd = WideDeep(deeptext=basic_rnn, pred_dim=len(categories))
trainer = StreamTrainer(model=wd, objective='multiclass')
trainer.fit(
    X_train_path=X_path, 
    target=y, 
    preprocessor=text_preproc, 
    n_epochs=1
)

# from pytorch_widedeep.stream._stream_ds import StreamTextDataset
# from torch.utils.data import DataLoader

# l = DataLoader(
#             StreamTextDataset(X_path, preprocessor=text_preproc), 
#             batch_size=5,
#             drop_last=True
#         )


# print(next(enumerate(l)))
# print(next(enumerate(l)))

# import pdb; pdb.set_trace()