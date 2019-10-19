import numpy as np
import pandas as pd
import os
import torch

from pytorch_widedeep.preprocessing import (WideProcessor, DeepProcessor,
    TextProcessor, ImageProcessor)

import pdb

use_cuda = torch.cuda.is_available()

if __name__ == '__main__':


    df = pd.read_csv('../data/airbnb/tmp_df.csv')

    crossed_cols = (['property_type', 'room_type'],)
    already_dummies = [c for c in df.columns if 'amenity' in c] + ['has_house_rules']
    wide_cols = ['is_location_exact', 'property_type', 'room_type', 'host_gender',
    'instant_bookable'] + already_dummies
    cat_embed_cols = [(c, 16) for c in df.columns if 'catg' in c] + \
        [('neighbourhood_cleansed', 64), ('cancellation_policy', 16)]
    continuous_cols = ['latitude', 'longitude', 'security_deposit', 'extra_people']
    already_standard = ['latitude', 'longitude']
    text_col = 'description'
    word_vectors_path = '../data/glove.6B/glove.6B.100d.txt'
    img_col = 'id'
    img_path = '../data/airbnb/property_picture'
    target = 'yield'

    pdb.set_trace()
    text_processor = TextProcessor(word_vectors_path=word_vectors_path)
    X_text = text_processor.fit_transform(df, text_col)
    image_processor = ImageProcessor()
    X_images = image_processor.fit_transform(df, img_col, img_path)

    deeptext = DeepText(
        vocab_size=len(text_processor.vocab.itos),
        hidden_dim=64,
        n_layers=3,
        rnn_dropout=0.5,
        spatial_dropout=0.5,
        padding_idx=1,
        output_dim=1,
        embedding_matrix=text_processor.embedding_matrix
        )
    deepimage = DeepImage()

    # model = WideDeep(output_dim=1, wide_dim=wd.wide.shape[1],
    #     cat_embed_input = wd.cat_embed_input,
    #     cat_embed_encoding_dict=wd.cat_embed_encoding_dict,
    #     continuous_cols=wd.continuous_cols,
    #     deep_column_idx=wd.deep_column_idx, add_text=True,
    #     vocab_size=len(wd.vocab.itos),
    #     word_embed_matrix = wd.word_embed_matrix,
    #     add_image=True)

    # initializers = {'wide': Normal, 'deepdense':Normal, 'deeptext':Normal, 'deepimage':Normal}
    # optimizers = {'wide': Adam, 'deepdense':Adam, 'deeptext':RAdam, 'deepimage':Adam}
    # schedulers = {'wide': StepLR(step_size=5), 'deepdense':StepLR(step_size=5), 'deeptext':MultiStepLR(milestones=[5,8]),
    #     'deepimage':MultiStepLR(milestones=[5,8])}
    # mean = [0.406, 0.456, 0.485]  #BGR
    # std =  [0.225, 0.224, 0.229]  #BGR
    # transforms = [ToTensor, Normalize(mean=mean, std=std)]
    # callbacks = [EarlyStopping, ModelCheckpoint(filepath='model_weights/wd_out.pt')]

    # model.compile(method='regression', initializers=initializers, optimizers=optimizers,
    #     lr_schedulers=schedulers, callbacks=callbacks, transforms=transforms)

    # model.fit(X_wide=wd.wide, X_deep=wd.deepdense, X_text=wd.deeptext, X_img=wd.deepimage,
    #     target=wd.target, n_epochs=1, batch_size=32, val_split=0.2)