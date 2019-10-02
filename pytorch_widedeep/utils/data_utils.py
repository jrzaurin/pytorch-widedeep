import numpy as np
import pandas as pd
import pickle
import cv2
import os

from pathlib import Path
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split

from .wide_utils  import prepare_wide
from .deep_utils  import prepare_deep
from .image_utils import prepare_image
from .text_utils  import prepare_text

from ..wdtypes import *


pd.options.mode.chained_assignment = None


def prepare_data(df:pd.DataFrame, target:str, wide_cols:List[str],
        crossed_cols:List[Tuple[str,str]], cat_embed_cols:List[Union[str,
        Tuple[str,int]]], continuous_cols:List[str],
        already_dummies:Optional[List[str]]=None,
        already_standard:Optional[List[str]]=None, scale:bool=True,
        default_embed_dim:int=8, padded_sequences:Optional[np.ndarray]=None,
        vocab:Optional[Any]=None, word_embed_matrix:Optional[np.ndarray]=None,
        text_col:Optional[str]=None, max_vocab:int=30000, min_freq:int=5,
        maxlen:int=80, word_vectors_path:Optional[PosixPath]=None,
        img_col:Optional[str]=None, img_path:Optional[PosixPath]=None,
        width:int=224, height:int=224,
        processed_images:Optional[np.ndarray]=None,
        filepath:Optional[str]=None, seed:int=1,verbose:int=1) -> Bunch:

    # Target
    y = df[target].values

    # Wide
    X_wide = prepare_wide(df, wide_cols, crossed_cols, already_dummies)

    # Deep Dense Layers
    X_deep, cat_embed_input, cat_embed_encoding_dict, deep_column_idx = \
        prepare_deep(df, cat_embed_cols, continuous_cols, already_standard,
            scale, default_embed_dim)

    # sklearn's Bunch as Container for the dataset
    wd_dataset = Bunch(target=y, wide=X_wide.astype('float32'),
        deepdense=X_deep, cat_embed_input=cat_embed_input,
        cat_embed_encoding_dict = cat_embed_encoding_dict,
        continuous_cols = continuous_cols,
        deep_column_idx=deep_column_idx)

    # Deep Text
    if padded_sequences is not None:
        assert vocab is not None, 'A vocabulary object is missing'
        wd_dataset.deeptext, wd_dataset.vocab = padded_sequences, vocab
        if word_embed_matrix is not None:
            wd_dataset.word_embed_matrix = word_embed_matrix
    elif text_col:
        X_text, word_embed_matrix, vocab = \
            prepare_text(df, text_col, max_vocab, min_freq, maxlen, word_vectors_path, verbose)
        wd_dataset.deeptext, wd_dataset.vocab = X_text, vocab
        if word_embed_matrix is not None:
            wd_dataset.word_embed_matrix = word_embed_matrix

    # Deep Image
    if processed_images is not None:
        X_images = processed_images
    elif img_col:
        X_images = prepare_image(df, img_col, img_path, width, height, verbose)
        mean_R, mean_G, mean_B = [], [], []
        std_R, std_G, std_B = [], [], []
    try:
        for img in X_images:
            (mean_b, mean_g, mean_r), (std_b, std_g, std_r) = cv2.meanStdDev(img)
            mean_R.append(mean_r), mean_G.append(mean_g), mean_B.append(mean_b)
            std_R.append(std_r), std_G.append(std_g), std_B.append(std_b)
        normalise_metrics = dict(
            mean = {"R": np.mean(mean_R)/255., "G": np.mean(mean_G)/255., "B": np.mean(mean_B)/255.},
            std = {"R": np.mean(std_R)/255., "G": np.mean(std_G)/255., "B": np.mean(std_B)/255.}
            )
        wd_dataset.deepimage, wd_dataset.normalise_metrics = X_images, normalise_metrics
    except NameError:
        pass

    if filepath is not None:
        assert not os.path.isdir(filepath), "filepath is a directory. Please provide full path including filename"
        file_dir, file_name = filepath.split("/")[:-1], filepath.split("/")[-1]
        if len(file_dir)==0:
            pickle.dump(wd_dataset, open(filepath, 'wb'))
        elif not os.path.exists(file_dir[0]):
            os.makedirs(file_dir)
            pickle.dump(wd_dataset, open(filepath, 'wb'))
        else:
            pickle.dump(wd_dataset, open(filepath, 'wb'))

    if verbose: print('Wide and Deep data preparation completed.')
    return wd_dataset