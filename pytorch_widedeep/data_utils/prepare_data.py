import numpy as np
import pandas as pd
import pickle
import cv2

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch

from ..wdtypes import *

from .wide_utils import prepare_wide
from .deep_utils import prepare_deep
from .image_utils import prepare_image
from .text_utils import prepare_text

pd.options.mode.chained_assignment = None


def prepare_data(df:pd.DataFrame, target:str, wide_cols:List[str],
                crossed_cols:List[Tuple[str,str]],
                embeddings_cols:List[Union[str, Tuple[str,int]]],
                continuous_cols:List[str],
                already_dummies:Optional[List[str]]=None,
                standardize_cols:Optional[List[str]]=None, scale:bool=True,
                default_emb_dim:int=8, text_col:Optional[str]=None,
                padded_sequences:Optional[np.ndarray]=None,
                word_embeddings_matrix:Optional[np.ndarray]=None,
                vocabulary:Optional[Any]=None, max_vocab:int=30000,
                min_freq:int=5, maxlen:int=80,
                word_vectors_path:Optional[PosixPath]=None,
                img_col:Optional[str]=None, img_path:Optional[PosixPath]=None,
                width:int=224, height:int=224,
                processed_images:Optional[np.ndarray]=None,
                output_dir:Optional[PosixPath]=None, seed:int=1,verbose:int=1) -> Bunch:

    # Target
    y = df[target].values

    # Wide
    X_wide = prepare_wide(df, wide_cols, crossed_cols, already_dummies)

    # Deep Dense Layers
    X_deep, cat_embed_inp, cat_embed_encoding_dict, deep_column_idx = \
        prepare_deep(df, embeddings_cols, continuous_cols, standardize_cols, scale, default_emb_dim)

    # Using sklearn's Bunch as Container for the dataset
    wd_dataset = Bunch(wide=X_wide, deep_dense=X_deep,
        cat_embeddings_input=cat_embed_inp,
        cat_embeddings_encoding_dict = cat_embed_encoding_dict,
        continuous_cols = continuous_cols,
        deep_column_idx=deep_column_idx)

    # Deep Text
    if padded_sequences is not None:
        assert vocab is not None, 'A vocabulary object is missing'
        X_text = padded_sequences
    elif text_col:
        X_text, word_embeddings_matrix, vocab = \
            prepare_text(df, text_col, max_vocab, min_freq, maxlen, word_vectors_path, verbose)
    try:
        wd_dataset.deep_text, wd_dataset.word_embeddings_matrix, wd_dataset.vocab = \
            X_text, word_embeddings_matrix, vocab
    except NameError:
        pass

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
        wd_dataset.deep_img, wd_dataset.normalise_metrics = X_images, normalise_metrics
    except NameError:
        pass

    if output_dir is not None: pickle.dump(wd_dataset, open(output_dir/'wd_dataset.p', 'wb'))
    if verbose: print('Wide and Deep data preparation completed.')
    return wd_dataset