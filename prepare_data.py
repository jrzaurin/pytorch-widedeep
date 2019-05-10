# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import argparse
import pickle
import json
import pdb
import cv2

from os import listdir
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from fastai.text import Vocab

from widedeep.utils.data_utils import label_encode
from widedeep.utils.image_utils import resize_images
from widedeep.utils.text_utils import (get_texts_gensim, pad_sequences,
    build_embeddings_matrix)

from widedeep.models.wdtypes import *


pd.options.mode.chained_assignment = None


def prepare_images(df:pd.DataFrame, img_id:str, img_path:PosixPath):
    imgnames = listdir(img_path)
    ids_with_images = [int(s.split('.')[0]) for s in imgnames]
    # subseting only those ids with images
    dfo = df[df[img_id].isin(ids_with_images)]
    # Make sure the order of the images is the same as the rest of the
    # features and text
    id_order = [str(i)+'.jpg' for i in dfo[img_id]]
    imgs = resize_images(img_path, id_order)
    return imgs, dfo


def prepare_text(df:pd.DataFrame, text_col:str, max_vocab:int, min_freq:int, maxlen:int,
    word_vectors_path:str):
    texts = df[text_col].tolist()
    # texts = [t.lower() for t in texts]
    tokens = get_texts_gensim(texts)
    vocab = Vocab.create(tokens, max_vocab=max_vocab, min_freq=min_freq)
    sequences = [vocab.numericalize(t) for t in tokens]
    padded_seq = np.array([pad_sequences(s, maxlen=maxlen) for s in sequences])
    print("Our vocabulary contains {} words".format(len(vocab.stoi)))
    embedding_matrix = build_embeddings_matrix(vocab, word_vectors_path)
    return padded_seq, vocab, embedding_matrix


def prepare_deep(df:pd.DataFrame, embeddings_cols:List[Union[str, Tuple[str,int]]],
    continuous_cols:List[str], standardize_cols:List[str], scale:bool=True, def_dim:int=8):
    """
    Highly customised function to prepare the features that will be passed
    through the "Deep-Dense" model.

    Parameters:
    ----------
    df: pd.Dataframe
    embeddings_cols: List
        List containing just the name of the columns that will be represented
        with embeddings or a Tuple with the name and the embedding dimension.
        e.g.:  [('education',32), ('relationship',16)
    continuous_cols: List
        List with the name of the so called continuous cols
    standardize_cols: List
        List with the name of the continuous cols that will be Standarised.
        Only included because the Airbnb dataset includes Longitude and
        Latitude and does not make sense to normalise that
    scale: bool
        whether or not to scale/Standarise continuous cols. Should almost
        always be True.
    def_dim: int
        Default dimension for the embeddings used in the Deep-Dense model

    Returns:
    df_deep.values: np.ndarray
        array with the prepare input data for the Deep-Dense model
    embeddings_input: List of Tuples
        List containing Tuples with the name of embedding col, number of unique values
        and embedding dimension. e.g. : [(education, 11, 32), ...]
    embeddings_encoding_dict: Dict
        Dict containing the encoding mappings that will be required to recover the
        embeddings once the model has trained
    deep_column_idx: Dict
        Dict containing the index of the embedding columns that will be required to
        slice the tensors when training the model
    """
    # If embeddings_cols does not include the embeddings dimensions it will be
    # set as def_dim (8)
    if type(embeddings_cols[0]) is tuple:
        emb_dim = dict(embeddings_cols)
        embeddings_coln = [emb[0] for emb in embeddings_cols]
    else:
        emb_dim = {e:def_dim for e in embeddings_cols}
        embeddings_coln = embeddings_cols
    deep_cols = embeddings_coln+continuous_cols

    # copy the df so it does not change internally
    df_deep = df.copy()[deep_cols]

    # Extract the categorical column names that will be label_encoded
    categorical_columns = list(df_deep.select_dtypes(include=['object']).columns)
    categorical_columns+= list(set([c for c in df_deep.columns if 'catg' in c]))

    # Encode the dataframe and get the encoding dictionary
    df_deep, encoding_dict = label_encode(df_deep, cols=categorical_columns)
    embeddings_encoding_dict = {k:encoding_dict[k] for k in encoding_dict if k in deep_cols}
    embeddings_input = []
    for k,v in embeddings_encoding_dict.items():
        embeddings_input.append((k, len(v), emb_dim[k]))

    # select the deep_cols and get the column index that will be use later
    # to slice the tensors
    deep_column_idx = {k:v for v,k in enumerate(df_deep.columns)}

    # The continous columns will be concatenated with the embeddings, so you
    # probably want to normalize them
    if scale:
        scaler = StandardScaler()
        for cc in standardize_cols:
            df_deep[cc]  = scaler.fit_transform(df_deep[cc].values.reshape(-1,1).astype(float))

    return df_deep.values, embeddings_input, embeddings_encoding_dict, deep_column_idx


def prepare_wide(df:pd.DataFrame, target:str, wide_cols:List[str],
    crossed_cols:List[Tuple[str,str]], already_dummies:Optional[List[str]]=None):
    """
    Highly customised function to prepare the features that will be passed
    through the "Wide" model.

    Parameters:
    ----------
    df: pd.Dataframe
    target: str
    wide_cols: List
        List with the name of the columns that will be one-hot encoded and
        pass through the Wide model
    crossed_cols: List
        List of Tuples with the name of the columns that will be "crossed"
        and then one-hot encoded. e.g. (['education', 'occupation'], ...)
    already_dummies: List
        List of columns that are already dummies/one-hot encoded

    Returns:
    df_wide.values: np.ndarray
        values that will be passed through the Wide Model
    y: np.ndarray
        target
    """
    y = np.array(df[target])
    df_wide = df.copy()[wide_cols]

    crossed_columns = []
    for cols in crossed_cols:
        colname = '_'.join(cols)
        df_wide[colname] = df_wide[cols].apply(lambda x: '-'.join(x), axis=1)
        crossed_columns.append(colname)

    if already_dummies:
        dummy_cols = [c for c in wide_cols+crossed_columns if c not in already_dummies]
    else:
        dummy_cols = wide_cols+crossed_columns
    df_wide = pd.get_dummies(df_wide, columns=dummy_cols)

    return df_wide.values, y

# I believe this and the next functions do not need comments
def prepare_data_airbnb(df, img_id, img_path, text_col, max_vocab, min_freq, maxlen,
    word_vectors_path, embeddings_cols, continuous_cols, standardize_cols, target,
    wide_cols, crossed_cols, already_dummies, out_dir, scale=True, def_dim=8, seed=1,
    save=True):

    dfc = df.copy()

    X_images, dfo = prepare_images(dfc, img_id, img_path)

    X_text, vocab, word_embed_mtx = prepare_text(dfo, text_col, max_vocab, min_freq,
        maxlen, word_vectors_path)

    X_deep, cat_embed_inp, cat_embed_encoding_dict, deep_column_idx = \
        prepare_deep(dfo, embeddings_cols, continuous_cols, standardize_cols, scale=True,
            def_dim=8)

    X_wide, y = prepare_wide(dfo, target, wide_cols, crossed_cols, already_dummies)

    # train/valid/test split
    X_tr_wide, X_val_wide = train_test_split(X_wide, test_size=0.4, random_state=seed)
    X_tr_deep, X_val_deep = train_test_split(X_deep, test_size=0.4, random_state=seed)
    X_tr_text, X_val_text = train_test_split(X_text, test_size=0.4, random_state=seed)
    X_tr_img, X_val_img = train_test_split(X_images, test_size=0.4, random_state=seed)
    y_tr, y_val = train_test_split(y, test_size=0.4, random_state=seed)

    X_val_wide, X_te_wide = train_test_split(X_val_wide, test_size=0.5, random_state=seed)
    X_val_deep, X_te_deep = train_test_split(X_val_deep, test_size=0.5, random_state=seed)
    X_val_text, X_te_text = train_test_split(X_val_text, test_size=0.5, random_state=seed)
    X_val_img, X_te_img = train_test_split(X_val_img, test_size=0.5, random_state=seed)
    y_val, y_te = train_test_split(y_val, test_size=0.5, random_state=seed)

    # Computing the average of the RGB channels
    mean_R, mean_G, mean_B = [], [], []
    std_R, std_G, std_B = [], [], []
    for img in X_tr_img:
        # remember, cv2 reads BGR
        (mean_b, mean_g, mean_r), (std_b, std_g, std_r) = cv2.meanStdDev(img)
        mean_R.append(mean_r), mean_G.append(mean_g), mean_B.append(mean_b)
        std_R.append(std_r), std_G.append(std_g), std_B.append(std_b)
    normalise_metrics = dict(
        mean = {"R": np.mean(mean_R)/255., "G": np.mean(mean_G)/255., "B": np.mean(mean_B)/255.},
        std = {"R": np.mean(std_R)/255., "G": np.mean(std_G)/255., "B": np.mean(std_B)/255.}
        )
    f = open(out_dir/'normalise_metrics.json', "w")
    f.write(json.dumps(normalise_metrics))
    f.close()

    # store all the datasets in a Dictionary
    wd_dataset = dict(
        train = dict(
            wide = X_tr_wide.astype('float32'),
            deep_dense = X_tr_deep,
            deep_text = X_tr_text.astype('int64'),
            deep_img = X_tr_img,
            target = y_tr,
            ),
        valid = dict(
            wide = X_val_wide.astype('float32'),
            deep_dense = X_val_deep,
            deep_text = X_val_text.astype('int64'),
            deep_img = X_val_img,
            target = y_val,
            ),
        test = dict(
            wide = X_te_wide.astype('float32'),
            deep_dense = X_te_deep,
            deep_text = X_te_text.astype('int64'),
            deep_img = X_te_img,
            target = y_te,
            ),
        vocab = vocab,
        word_embeddings_matrix = word_embed_mtx,
        cat_embeddings_input = cat_embed_inp,
        cat_embeddings_encoding_dict = cat_embed_encoding_dict,
        continuous_cols = continuous_cols,
        deep_column_idx = deep_column_idx
        )
    if save: pickle.dump(wd_dataset, open(out_dir/'wd_dataset.p', 'wb'))
    print('Wide and Deep airbnb data preparation completed.')
    return wd_dataset


def prepare_data_adult(df, wide_cols, crossed_cols, embeddings_cols, continuous_cols,
    standardize_cols, target, out_dir, scale=True, def_dim=8, seed=1, save=True):

    dfc = df.copy()

    X_deep, cat_embed_inp, cat_embed_encoding_dict, deep_column_idx = \
        prepare_deep(dfc, embeddings_cols, continuous_cols, standardize_cols, scale=True,
            def_dim=8)

    X_wide, y = prepare_wide(dfc, target, wide_cols, crossed_cols)

    # train/valid/test split
    X_tr_wide, X_val_wide = train_test_split(X_wide, test_size=0.4, random_state=seed)
    X_tr_deep, X_val_deep = train_test_split(X_deep, test_size=0.4, random_state=seed)
    y_tr, y_val = train_test_split(y, test_size=0.4, random_state=seed)

    X_val_wide, X_te_wide = train_test_split(X_val_wide, test_size=0.5, random_state=seed)
    X_val_deep, X_te_deep = train_test_split(X_val_deep, test_size=0.5, random_state=seed)
    y_val, y_te = train_test_split(y_val, test_size=0.5, random_state=seed)

    wd_dataset = dict(
        train = dict(
            wide = X_tr_wide.astype('float32'),
            deep_dense = X_tr_deep,
            target = y_tr,
            ),
        valid = dict(
            wide = X_val_wide.astype('float32'),
            deep_dense = X_val_deep,
            target = y_val,
            ),
        test = dict(
            wide = X_te_wide.astype('float32'),
            deep_dense = X_te_deep,
            target = y_te,
            ),
        cat_embeddings_input = cat_embed_inp,
        cat_embeddings_encoding_dict = cat_embed_encoding_dict,
        continuous_cols = continuous_cols,
        deep_column_idx = deep_column_idx
        )
    if save: pickle.dump(wd_dataset, open(out_dir/'wd_dataset.p', 'wb'))
    print('Wide and Deep adult data preparation completed.')
    return wd_dataset


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--wordvectors", type=str, default='glove', required=False,
        help="glove or fasttext")
    ap.add_argument("--imtype", type=str, default='property', required=False,
        help="property or host")
    args = vars(ap.parse_args())

    DATA_PATH=Path('data')

    if args['dataset'] == 'adult':
        print("preparing the adult/census dataset")
        DF = pd.read_csv(DATA_PATH/'adult/adult.csv')
        DF.columns = [c.replace("-", "_") for c in DF.columns]
        DF['income_label'] = (DF["income"].apply(lambda x: ">50K" in x)).astype(int)
        DF.drop("income", axis=1, inplace=True)
        DF['age_buckets'] = pd.cut(DF.age, bins=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65],
            labels=np.arange(9))
        out_dir = DATA_PATH/'adult/wide_deep_data/'
        wide_cols = ['age_buckets', 'education', 'relationship','workclass','occupation',
            'native_country','gender']
        crossed_cols = (['education', 'occupation'], ['native_country', 'occupation'])
        embeddings_cols = [('education',10), ('relationship',8), ('workclass',10),
            ('occupation',10),('native_country',10)]
        continuous_cols = ["age","hours_per_week"]
        standardize_cols = continuous_cols
        target = 'income_label'

        wd_dataset_adult = prepare_data_adult(
            DF, wide_cols,
            crossed_cols,
            embeddings_cols,
            continuous_cols,
            standardize_cols,
            target, out_dir,
            scale=True
            )

    if args['dataset'] == 'airbnb':
        print("preparing the airbnb dataset")
        DF = pd.read_csv(DATA_PATH/'airbnb/listings_processed.csv')
        DF = DF[DF.description.apply(lambda x: len(x.split(' '))>=10)]
        out_dir = DATA_PATH/'airbnb/wide_deep_data/'
        if args['imtype'] == 'host':
            img_id = 'host_id'
            img_path = DATA_PATH/'airbnb/host_picture'
        else:
            img_id = 'id'
            img_path = DATA_PATH/'airbnb/property_picture'
        if args['wordvectors'] == 'glove':
            word_vectors_path = 'data/glove.6B/glove.6B.300d.txt'
        else:
            word_vectors_path = 'data/fasttext.cc/cc.en.300.vec'
        text_col = 'description'
        crossed_cols = (['property_type', 'room_type'],)
        embeddings_cols = [(c, 16) for c in DF.columns if 'catg' in c] + \
            [('neighbourhood_cleansed', 64)]
        continuous_cols = ['latitude', 'longitude', 'security_deposit', 'extra_people']
        standardize_cols = ['security_deposit', 'extra_people']
        already_dummies = [c for c in DF.columns if 'amenity' in c] + ['has_house_rules']
        wide_cols = ['is_location_exact', 'property_type', 'room_type', 'host_gender'] +\
            already_dummies
        target = 'yield'

        wd_dataset_airbnb = prepare_data_airbnb(
            df = DF.sample(30000),
            img_id = img_id,
            img_path = img_path,
            text_col = text_col,
            max_vocab = 30000,
            min_freq = 5,
            maxlen = 170,
            word_vectors_path = word_vectors_path,
            embeddings_cols = embeddings_cols,
            continuous_cols = continuous_cols,
            standardize_cols = standardize_cols,
            target = target,
            wide_cols = wide_cols,
            crossed_cols = crossed_cols,
            already_dummies = already_dummies,
            out_dir = out_dir,
            scale=True,
            def_dim=8,
            seed=1
            )