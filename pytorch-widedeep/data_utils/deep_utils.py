import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from ..wdtypes import *

pd.options.mode.chained_assignment = None


def prepare_deep(df:pd.DataFrame, embeddings_cols:List[Union[str, Tuple[str,int]]],
    continuous_cols:List[str], standardize_cols:List[str], scale:bool=True, def_dim:int=8):
    """
    Function to prepare the features that will be passed through the "Deep-Dense" model.

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


def label_encode(df_inp:pd.DataFrame, cols:Optional[List[str]]=None,
    val_to_idx:Optional[Dict[str,Dict[str,int]]]=None):
    """
    Helper function to label-encode some features of a given dataset.

    Parameters:
    -----------
    df_inp: pd.Dataframe
        input dataframe
    cols: List
        optional - columns to be label-encoded
    val_to_idx: Dict
        optional - dictionary with the encodings

    Returns:
    --------
    df: pd.Dataframe
        df with Label-encoded features.
    val_to_idx: Dict
        Dictionary with the encoding information
    """
    df = df_inp.copy()

    if cols == None:
        cols = list(df.select_dtypes(include=['object']).columns)

    if not val_to_idx:

        val_types = dict()
        for c in cols:
            val_types[c] = df[c].unique()

        val_to_idx = dict()
        for k, v in val_types.items():
            val_to_idx[k] = {o: i for i, o in enumerate(val_types[k])}

    for k, v in val_to_idx.items():
        df[k] = df[k].apply(lambda x: v[x])

    return df, val_to_idx
