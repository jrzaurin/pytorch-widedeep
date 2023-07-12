import numpy as np
import torch
import pandas as pd
from typing import Union, List
import pytest

from pytorch_widedeep.models import BasicRNN, WideDeep, Vision

from pytorch_widedeep.stream.preprocessing.image_preprocessor import StreamImagePreprocessor
from pytorch_widedeep.stream.preprocessing.text_preprocessor import StreamTextPreprocessor, Vocab, VocabBuilder
from pytorch_widedeep.stream.training.trainer import StreamTrainer

from pathlib import Path


data_path = '~/airbnb/airbnb/airbnb_sample.csv'
image_path = '/home/krish/airbnb/airbnb/property_picture'

df = next(pd.read_csv(data_path, chunksize=1000))

crossed_cols = [("property_type", "room_type")]
already_dummies = [c for c in df.columns if "amenity" in c] + ["has_house_rules"]
wide_cols = [
    "is_location_exact",
    "property_type",
    "room_type",
    "host_gender",
    "instant_bookable",
] + already_dummies
cat_embed_cols = [(c, 16) for c in df.columns if "catg" in c] + [
    ("neighbourhood_cleansed", 64),
    ("cancellation_policy", 16),
]
continuous_cols = ["latitude", "longitude", "security_deposit", "extra_people"]
already_standard = ["latitude", "longitude"]
text_col = "description"
# word_vectors_path = str(DATA_PATH / "glove.6B/glove.6B.100d.txt")
img_col = "id"
# img_path = str(DATA_PATH / "airbnb/property_picture")
target_col = "yield"

# target = df[target].values


# Fit preprocessors
text_preproc = StreamTextPreprocessor(text_col)
text_preproc.fit(data_path, 1024*100)

image_processor = StreamImagePreprocessor(img_col=img_col, img_path=image_path)
image_processor.fit()

# Choose models
deepimage = Vision(
     pretrained_model_name="resnet18", n_trainable=0, head_hidden_dims=[200, 100]
 )
basic_rnn = BasicRNN(vocab_size=len(text_preproc.vocab.itos), hidden_dim=20, n_layers=2, padding_idx=0, embed_dim=80)
wd = WideDeep(deeptext=basic_rnn, deepimage=deepimage)

# Train
trainer = StreamTrainer(
    model=wd, 
    objective='regression',
    X_path=data_path,
    img_col=img_col,
    text_col=text_col,
    target_col=target_col,
    text_preprocessor=text_preproc,
    img_preprocessor=image_processor,
    fetch_size=250
)
trainer.fit(
     n_epochs=20,
     batch_size=64
 )

from pytorch_widedeep.stream._stream_ds import StreamTextDataset, StreamWideDeepDataset
# from torch.utils.data import DataLoader

# l = DataLoader(
#             StreamTextDataset(X_path, preprocessor=text_preproc, chunksize=5), 
#             batch_size=1,
#             drop_last=True
#         )

# text_preproc = StreamTextPreprocessor(text_col)
# text_preproc.fit(data_path, 1024*100)

# image_processor = StreamImagePreprocessor(img_col=img_col, img_path=image_path)
# image_processor.fit()

# wd = StreamWideDeepDataset(
#     data_path, 
#     img_col, 
#     text_col,
#     text_preproc,
#     image_processor
# )

# next(enumerate(wd))

# print(next(enumerate(l)))
# print(next(enumerate(l)))

# import pdb; pdb.set_trace()