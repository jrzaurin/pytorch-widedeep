import numpy as np
import pandas as pd
import warnings
import cv2

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
from tqdm import tqdm

from ..wdtypes import *
from ..utils.fastai_transforms import Vocab
from ..utils.dense_utils import label_encoder
from ..utils.text_utils import get_texts, pad_sequences, build_embeddings_matrix
from ..utils.image_utils import AspectAwarePreprocessor, SimplePreprocessor


class BasePreprocessor(object):
    def fit(self, df: pd.DataFrame):
        raise NotImplementedError("Preprocessor must implement this method")

    def transform(self, df: pd.DataFrame):
        raise NotImplementedError("Preprocessor must implement this method")

    def fit_transform(self, df: pd.DataFrame):
        raise NotImplementedError("Preprocessor must implement this method")


class WidePreprocessor(BasePreprocessor):
    r"""Preprocessor to prepare the wide input dataset

    Parameters
    ----------
    wide_cols: List
        List with the name of the columns that will be one-hot encoded and
        pass through the Wide model
    crossed_cols: List
        List of Tuples with the name of the columns that will be "crossed"
        and then one-hot encoded. e.g. (['education', 'occupation'], ...)
    already_dummies: List
        List of columns that are already dummies/one-hot encoded

    Attributes
    ----------
    one_hot_enc: sklearn's OneHotEncoder
    wide_crossed_cols: List
        List with the names of all columns that will be one-hot encoded

    Example
    --------
    Assuming we have a dataset loaded in memory as a pd.DataFrame

    >>> wide_cols = ['age_buckets', 'education', 'relationship','workclass','occupation',
    ... 'native_country','gender']
    >>> crossed_cols = [('education', 'occupation'), ('native_country', 'occupation')]
    >>> wide_preprocessor = WidePreprocessor(wide_cols=wide_cols, crossed_cols=crossed_cols)
    >>> X_wide = wide_preprocessor.fit_transform(df)

    From there on, for new data (loaded as a dataframe)
    >>> new_X_wide = wide_preprocessor.transform(new_df)
    """

    def __init__(
        self,
        wide_cols: List[str],
        crossed_cols=None,
        already_dummies: Optional[List[str]] = None,
        sparse=False,
    ):
        super(WidePreprocessor, self).__init__()
        self.wide_cols = wide_cols
        self.crossed_cols = crossed_cols
        self.already_dummies = already_dummies
        self.one_hot_enc = OneHotEncoder(sparse=sparse)

    def _cross_cols(self, df: pd.DataFrame):
        crossed_colnames = []
        for cols in self.crossed_cols:
            cols = list(cols)
            for c in cols:
                df[c] = df[c].astype("str")
            colname = "_".join(cols)
            df[colname] = df[cols].apply(lambda x: "-".join(x), axis=1)
            crossed_colnames.append(colname)
        return df, crossed_colnames

    def fit(self, df: pd.DataFrame) -> BasePreprocessor:
        df_wide = df.copy()[self.wide_cols]
        if self.crossed_cols is not None:
            df_wide, crossed_colnames = self._cross_cols(df_wide)
            self.wide_crossed_cols = self.wide_cols + crossed_colnames
        else:
            self.wide_crossed_cols = self.wide_cols

        if self.already_dummies:
            dummy_cols = [
                c for c in self.wide_crossed_cols if c not in self.already_dummies
            ]
            self.one_hot_enc.fit(df_wide[dummy_cols])
        else:
            self.one_hot_enc.fit(df_wide[self.wide_crossed_cols])
        return self

    def transform(self, df: pd.DataFrame) -> Union[sparse_matrix, np.ndarray]:
        try:
            self.one_hot_enc.categories_
        except:
            raise NotFittedError(
                "This WidePreprocessor instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )
        df_wide = df.copy()[self.wide_cols]
        if self.crossed_cols is not None:
            df_wide, _ = self._cross_cols(df_wide)
        if self.already_dummies:
            X_oh_1 = df_wide[self.already_dummies].values
            dummy_cols = [
                c for c in self.wide_crossed_cols if c not in self.already_dummies
            ]
            X_oh_2 = self.one_hot_enc.transform(df_wide[dummy_cols])
            return np.hstack((X_oh_1, X_oh_2))
        else:
            return self.one_hot_enc.transform(df_wide[self.wide_crossed_cols])

    def fit_transform(self, df: pd.DataFrame) -> Union[sparse_matrix, np.ndarray]:
        return self.fit(df).transform(df)


class DeepPreprocessor(BasePreprocessor):
    r"""Preprocessor to prepare the deepdense input dataset

    Parameters
    ----------
    embed_cols: List
        List containing the name of the columns that will be represented with
        embeddings or a Tuple with the name and the embedding dimension. e.g.:
         [('education',32), ('relationship',16)
    continuous_cols: List
        List with the name of the so called continuous cols
    scale: Bool
        Bool indicating whether or not to scale/Standarise continuous cols.
        Should "almost always" be True.
    default_embed_dim: Int, default=8
        Dimension for the embeddings used in the Deep-Dense model
    already_standard: List, Optional,
        List with the name of the continuous cols that do not need to be
        Standarised.

    Attributes
    ----------
    encoding_dict: Dict
        Dict with the categorical encoding
    embed_cols: List
        List with the columns that will be represented with embeddings
    embed_dim: Dict
        Dict where keys are the embed cols and values are the embed dimensions
    standardize_cols: List
        List of the columns that will be standarized
    deep_column_idx: Dict
        Dict where keys are column names and values are column indexes. This
        will be neccesary to slice tensors
    scaler: sklearn's StandardScaler

    Example
    --------
    Assuming we have a dataset loaded in memory as a pd.DataFrame

    >>> cat_embed_cols = [('education',10), ('relationship',8), ('workclass',10),
    ... ('occupation',10),('native_country',10)]
    >>> continuous_cols = ["age","hours_per_week"]
    >>> deep_preprocessor = DeepPreprocessor(embed_cols=cat_embed_cols, continuous_cols=continuous_cols)
    >>> X_deep = deep_preprocessor.fit_transform(df)

    From there on, for new data (loaded as a dataframe)
    >>> new_X_deep = deep_preprocessor.transform(new_df)
    """

    def __init__(
        self,
        embed_cols: List[Union[str, Tuple[str, int]]] = None,
        continuous_cols: List[str] = None,
        scale: bool = True,
        default_embed_dim: int = 8,
        already_standard: Optional[List[str]] = None,
    ):
        super(DeepPreprocessor, self).__init__()

        self.embed_cols = embed_cols
        self.continuous_cols = continuous_cols
        self.already_standard = already_standard
        self.scale = scale
        self.default_embed_dim = default_embed_dim

        assert (self.embed_cols is not None) or (
            self.continuous_cols is not None
        ), "'embed_cols' and 'continuous_cols' are 'None'. Please, define at least one of the two."

    def _prepare_embed(self, df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(self.embed_cols[0], tuple):
            self.embed_dim = dict(self.embed_cols)  # type: ignore
            embed_colname = [emb[0] for emb in self.embed_cols]
        else:
            self.embed_dim = {e: self.default_embed_dim for e in self.embed_cols}  # type: ignore
            embed_colname = self.embed_cols  # type: ignore
        return df.copy()[embed_colname]

    def _prepare_continuous(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.scale:
            if self.already_standard is not None:
                self.standardize_cols = [
                    c for c in self.continuous_cols if c not in self.already_standard
                ]
            else:
                self.standardize_cols = self.continuous_cols
        return df.copy()[self.continuous_cols]

    def fit(self, df: pd.DataFrame) -> BasePreprocessor:
        if self.embed_cols is not None:
            df_emb = self._prepare_embed(df)
            _, self.encoding_dict = label_encoder(df_emb, cols=df_emb.columns.tolist())
            self.embeddings_input: List = []
            for k, v in self.encoding_dict.items():
                self.embeddings_input.append((k, len(v), self.embed_dim[k]))
        if self.continuous_cols is not None:
            df_cont = self._prepare_continuous(df)
            if self.scale:
                df_std = df_cont[self.standardize_cols]
                self.scaler = StandardScaler().fit(df_std.values)
            else:
                warnings.warn("Continuous columns will not be normalised")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.embed_cols is not None:
            df_emb = self._prepare_embed(df)
            df_emb, _ = label_encoder(
                df_emb, cols=df_emb.columns.tolist(), val_to_idx=self.encoding_dict
            )
        if self.continuous_cols is not None:
            df_cont = self._prepare_continuous(df)
            if self.scale:
                try:
                    self.scaler.mean_
                except:
                    raise NotFittedError(
                        "This DeepPreprocessor instance is not fitted yet. "
                        "Call 'fit' with appropriate arguments before using this estimator."
                    )
                df_std = df_cont[self.standardize_cols]
                df_cont[self.standardize_cols] = self.scaler.transform(df_std.values)
        try:
            df_deep = pd.concat([df_emb, df_cont], axis=1)
        except:
            try:
                df_deep = df_emb.copy()
            except:
                df_deep = df_cont.copy()
        self.deep_column_idx = {k: v for v, k in enumerate(df_deep.columns)}
        return df_deep.values

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)


class TextPreprocessor(BasePreprocessor):
    r"""Preprocessor to prepare the deepdense input dataset

    Parameters
    ----------
    text_col: str
        column in the input pd.DataFrame containing the texts
    max_vocab: Int, default=30000
        Maximum number of token in the vocabulary
    min_freq: Int, default=5
        Minimum frequency for a token to be part of the vocabulary
    maxlen: Int, default=80
        Maximum length of the tokenized sequences
    word_vectors_path: Optional, str
        Path to the pretrained word vectors
    verbose: Int, Default 1
        Enable verbose output.

    Attributes
    ----------
    vocab: fastai Vocab object. See https://docs.fast.ai/text.transform.html#Vocab
        Vocab object containing the information of the vocabulary
    tokens: List
        List with Lists of str containing the tokenized texts
    embedding_matrix: np.ndarray
        Array with the pretrained embeddings

    Example
    --------
    Assuming we have a dataset loaded in memory as a pd.DataFrame

    >>> text_preprocessor = TextPreprocessor()
    >>> X_text = text_preprocessor.fit_transform(df, text_col)

    from there on

    From there on, for new data (loaded as a dataframe)
    >>> new_X_text = text_preprocessor.transform(new_df)
    """

    def __init__(
        self,
        text_col: str,
        max_vocab: int = 30000,
        min_freq: int = 5,
        maxlen: int = 80,
        word_vectors_path: Optional[str] = None,
        verbose: int = 1,
    ):
        super(TextPreprocessor, self).__init__()
        self.text_col = text_col
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.maxlen = maxlen
        self.word_vectors_path = word_vectors_path
        self.verbose = verbose

    def fit(self, df: pd.DataFrame) -> BasePreprocessor:
        texts = df[self.text_col].tolist()
        tokens = get_texts(texts)
        self.vocab = Vocab.create(
            tokens, max_vocab=self.max_vocab, min_freq=self.min_freq
        )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        try:
            self.vocab
        except:
            raise NotFittedError(
                "This TextPreprocessor instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )
        texts = df[self.text_col].tolist()
        self.tokens = get_texts(texts)
        sequences = [self.vocab.numericalize(t) for t in self.tokens]
        padded_seq = np.array([pad_sequences(s, maxlen=self.maxlen) for s in sequences])
        if self.verbose:
            print("The vocabulary contains {} tokens".format(len(self.vocab.stoi)))
        if self.word_vectors_path is not None:
            self.embedding_matrix = build_embeddings_matrix(
                self.vocab, self.word_vectors_path, self.min_freq
            )
        return padded_seq

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)


class ImagePreprocessor(BasePreprocessor):
    r"""Preprocessor to prepare the deepdense input dataset

    Parameters
    ----------
    img_col: str
        name of the column with the images filenames
    img_path: str
        path to the dicrectory where the images are stored
    width: Int, default=224
        width of the resulting processed image. 224 because the default
        architecture used by WideDeep is ResNet
    height: Int, default=224
        width of the resulting processed image. 224 because the default
        architecture used by WideDeep is ResNet
    verbose: Int, Default 1
        Enable verbose output.

    Attributes
    ----------
    aap: Class, AspectAwarePreprocessor()
        Preprocessing tool taken from Adrian Rosebrock's book "Deep Learning
        for Computer Vision".
    spp: Class, SimplePreprocessor()
        Preprocessing tool taken from Adrian Rosebrock's book "Deep Learning
        for Computer Vision".
    normalise_metrics: Dict
        Dict containing the normalisation metrics of the image dataset, i.e.
        mean and std for the R, G and B channels

    Example
    --------
    Assuming we have a dataset loaded in memory as a pd.DataFrame

    >>> image_preprocessor = ImagePreprocessor()
    >>> img_path = 'path/to/my_images'
    >>> X_images = image_preprocessor.fit_transform(df, img_col, img_path)

    from there on

    From there on, for new data (loaded as a dataframe)
    >>> next_X_images = image_preprocessor.transform(new_df)
    """

    def __init__(
        self,
        img_col: str,
        img_path: str,
        width: int = 224,
        height: int = 224,
        verbose: int = 1,
    ):
        super(ImagePreprocessor, self).__init__()
        self.img_col = img_col
        self.img_path = img_path
        self.width = width
        self.height = height
        self.verbose = verbose

    def fit(self, df: pd.DataFrame) -> BasePreprocessor:
        self.aap = AspectAwarePreprocessor(self.width, self.height)
        self.spp = SimplePreprocessor(self.width, self.height)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        try:
            self.aap
        except:
            raise NotFittedError(
                "This ImagePreprocessor instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )
        image_list = df[self.img_col].tolist()
        if self.verbose:
            print("Reading Images from {}".format(self.img_path))
        imgs = [cv2.imread("/".join([self.img_path, img])) for img in image_list]

        # finding images with different height and width
        aspect = [(im.shape[0], im.shape[1]) for im in imgs]
        aspect_r = [a[0] / a[1] for a in aspect]
        diff_idx = [i for i, r in enumerate(aspect_r) if r != 1.0]

        if self.verbose:
            print("Resizing")
        resized_imgs = []
        for i, img in tqdm(enumerate(imgs), total=len(imgs), disable=self.verbose != 1):
            if i in diff_idx:
                resized_imgs.append(self.aap.preprocess(img))
            else:
                resized_imgs.append(self.spp.preprocess(img))

        if self.verbose:
            print("Computing normalisation metrics")
        mean_R, mean_G, mean_B = [], [], []
        std_R, std_G, std_B = [], [], []
        for rsz_img in resized_imgs:
            (mean_b, mean_g, mean_r), (std_b, std_g, std_r) = cv2.meanStdDev(rsz_img)
            mean_R.append(mean_r)
            mean_G.append(mean_g)
            mean_B.append(mean_b)
            std_R.append(std_r)
            std_G.append(std_g)
            std_B.append(std_b)
        self.normalise_metrics = dict(
            mean={
                "R": np.mean(mean_R) / 255.0,
                "G": np.mean(mean_G) / 255.0,
                "B": np.mean(mean_B) / 255.0,
            },
            std={
                "R": np.mean(std_R) / 255.0,
                "G": np.mean(std_G) / 255.0,
                "B": np.mean(std_B) / 255.0,
            },
        )
        return np.asarray(resized_imgs)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)
