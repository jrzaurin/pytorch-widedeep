import warnings
from abc import ABC, abstractmethod

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..wdtypes import *
from ..utils.text_utils import (
    get_texts,
    pad_sequences,
    build_embeddings_matrix,
)
from ..utils.dense_utils import LabelEncoder
from ..utils.image_utils import SimplePreprocessor, AspectAwarePreprocessor
from ..utils.fastai_transforms import Vocab


class BasePreprocessor(ABC):
    """Base Abstract Class of All Preprocessors."""

    @abstractmethod
    def __init__(self, *args):
        pass

    @abstractmethod
    def fit(self, df: pd.DataFrame):
        raise NotImplementedError("Preprocessor must implement this method")

    @abstractmethod
    def transform(self, df: pd.DataFrame):
        raise NotImplementedError("Preprocessor must implement this method")

    @abstractmethod
    def fit_transform(self, df: pd.DataFrame):
        raise NotImplementedError("Preprocessor must implement this method")


class WidePreprocessor(BasePreprocessor):
    r"""Preprocessor to prepare the wide input dataset

    Parameters
    ----------
    wide_cols: List[str]
        List with the name of the columns that will be one-hot encoded and
        passed through the Wide model
    crossed_cols: List[Tuple[str, str]]
        List of Tuples with the name of the columns that will be `'crossed'`
        and then one-hot encoded. e.g. [('education', 'occupation'), ...]
    already_dummies: List[str]
        List of columns that are already dummies/one-hot encoded, and
        therefore do not need to be processed

    Attributes
    ----------
    one_hot_enc: :obj:`OneHotEncoder`
        an instance of :class:`sklearn.preprocessing.OneHotEncoder`
    wide_crossed_cols: :obj:`List`
        List with the names of all columns that will be one-hot encoded

    Examples
    --------
    >>> import pandas as pd
    >>> from pytorch_widedeep.preprocessing import WidePreprocessor
    >>> df = pd.DataFrame({'color': ['r', 'b', 'g'], 'size': ['s', 'n', 'l']})
    >>> wide_cols = ['color']
    >>> crossed_cols = [('color', 'size')]
    >>> wide_preprocessor = WidePreprocessor(wide_cols=wide_cols, crossed_cols=crossed_cols)
    >>> wide_preprocessor.fit_transform(df)
    array([[0., 0., 1., 0., 0., 1.],
           [1., 0., 0., 1., 0., 0.],
           [0., 1., 0., 0., 1., 0.]])
    """

    def __init__(
        self,
        wide_cols: List[str],
        crossed_cols=None,
        already_dummies: Optional[List[str]] = None,
        sparse=False,
        handle_unknown="ignore",
    ):
        super(WidePreprocessor, self).__init__()
        self.wide_cols = wide_cols
        self.crossed_cols = crossed_cols
        self.already_dummies = already_dummies
        self.one_hot_enc = OneHotEncoder(sparse=sparse, handle_unknown=handle_unknown)

    def fit(self, df: pd.DataFrame) -> BasePreprocessor:
        """Fits the Preprocessor and creates required attributes
        """
        df_wide = self._prepare_wide(df)
        self.wide_crossed_cols = df_wide.columns.tolist()
        if self.already_dummies:
            dummy_cols = [
                c for c in self.wide_crossed_cols if c not in self.already_dummies
            ]
            self.one_hot_enc.fit(df_wide[dummy_cols])
        else:
            self.one_hot_enc.fit(df_wide[self.wide_crossed_cols])
        return self

    def transform(self, df: pd.DataFrame) -> Union[sparse_matrix, np.ndarray]:
        """Returns the processed dataframe as a one hot encoded dense or
        sparse matrix
        """
        try:
            self.one_hot_enc.categories_
        except:
            raise NotFittedError(
                "This WidePreprocessor instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )
        df_wide = self._prepare_wide(df)
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
        """Combines ``fit`` and ``transform``
        """
        return self.fit(df).transform(df)

    def _cross_cols(self, df: pd.DataFrame):
        df_cc = df.copy()
        crossed_colnames = []
        for cols in self.crossed_cols:
            cols = list(cols)
            for c in cols:
                df_cc[c] = df_cc[c].astype("str")
            colname = "_".join(cols)
            df_cc[colname] = df_cc[cols].apply(lambda x: "-".join(x), axis=1)
            crossed_colnames.append(colname)
        return df_cc[crossed_colnames]

    def _prepare_wide(self, df: pd.DataFrame):
        if self.crossed_cols is not None:
            df_cc = self._cross_cols(df)
            return pd.concat([df[self.wide_cols], df_cc], axis=1)
        else:
            return df.copy()[self.wide_cols]


class DensePreprocessor(BasePreprocessor):
    r"""Preprocessor to prepare the deepdense input dataset

    Parameters
    ----------
    embed_cols: List[Union[str, Tuple[str, int]]]
        List containing the name of the columns that will be represented by
        embeddings or a Tuple with the name and the embedding dimension. e.g.:
        [('education',32), ('relationship',16)
    continuous_cols: List[str]
        List with the name of the so called continuous cols
    scale: bool
        Bool indicating whether or not to scale/Standarise continuous cols.
        Should "almost always" be True.
    default_embed_dim: Int, Default=8
        Dimension for the embeddings used in the Deep-Dense model
    already_standard: List[str], Optional,
        List with the name of the continuous cols that do not need to be
        Standarised.

    Attributes
    ----------
    label_encoder: :obj:`LabelEncoder`
        see :class:`pytorch_widedeep.utils.dense_utils.LabelEncder`
    embed_cols: :obj:`List`
        List with the columns that will be represented by embeddings
    embed_dim: :obj:`Dict`
        Dictionary where keys are the embed cols and values are the embed
        dimensions
    standardize_cols: :obj:`List`
        List of the columns that will be standarized
    deep_column_idx: :obj:`Dict`
        Dictionary where keys are column names and values are column indexes.
        This will be neccesary to slice tensors
    scaler: :obj:`StandardScaler`
        an instance of :class:`sklearn.preprocessing.StandardScaler`

    Examples
    --------
    >>> import pandas as pd
    >>> from pytorch_widedeep.preprocessing import DensePreprocessor
    >>> df = pd.DataFrame({'color': ['r', 'b', 'g'], 'size': ['s', 'n', 'l'], 'age': [25, 40, 55]})
    >>> embed_cols = [('color',5), ('size',5)]
    >>> cont_cols = ['age']
    >>> deep_preprocessor = DensePreprocessor(embed_cols=embed_cols, continuous_cols=cont_cols)
    >>> deep_preprocessor.fit_transform(df)
    array([[ 0.        ,  0.        , -1.22474487],
           [ 1.        ,  1.        ,  0.        ],
           [ 2.        ,  2.        ,  1.22474487]])
    >>> deep_preprocessor.embed_dim
    {'color': 5, 'size': 5}
    >>> deep_preprocessor.deep_column_idx
    {'color': 0, 'size': 1, 'age': 2}
    """

    def __init__(
        self,
        embed_cols: List[Union[str, Tuple[str, int]]] = None,
        continuous_cols: List[str] = None,
        scale: bool = True,
        default_embed_dim: int = 8,
        already_standard: Optional[List[str]] = None,
    ):
        super(DensePreprocessor, self).__init__()

        self.embed_cols = embed_cols
        self.continuous_cols = continuous_cols
        self.already_standard = already_standard
        self.scale = scale
        self.default_embed_dim = default_embed_dim

        assert (self.embed_cols is not None) or (
            self.continuous_cols is not None
        ), "'embed_cols' and 'continuous_cols' are 'None'. Please, define at least one of the two."

    def fit(self, df: pd.DataFrame) -> BasePreprocessor:
        """Fits the Preprocessor and creates required attributes
        """
        if self.embed_cols is not None:
            df_emb = self._prepare_embed(df)
            self.label_encoder = LabelEncoder(df_emb.columns.tolist()).fit(df_emb)
            self.embeddings_input: List = []
            for k, v in self.label_encoder.encoding_dict.items():
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
        """Returns the processed ``dataframe`` as a np.ndarray
        """
        if self.embed_cols is not None:
            df_emb = self._prepare_embed(df)
            df_emb = self.label_encoder.transform(df_emb)
        if self.continuous_cols is not None:
            df_cont = self._prepare_continuous(df)
            if self.scale:
                try:
                    self.scaler.mean_
                except:
                    raise NotFittedError(
                        "This DensePreprocessor instance is not fitted yet. "
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
        """Combines ``fit`` and ``transform``
        """
        return self.fit(df).transform(df)

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


class TextPreprocessor(BasePreprocessor):
    r"""Preprocessor to prepare the deeptext input dataset

    Parameters
    ----------
    text_col: str
        column in the input dataframe containing the texts
    max_vocab: int, default=30000
        Maximum number of token in the vocabulary
    min_freq: int, default=5
        Minimum frequency for a token to be part of the vocabulary
    maxlen: int, default=80
        Maximum length of the tokenized sequences
    word_vectors_path: str, Optional
        Path to the pretrained word vectors
    verbose: int, default 1
        Enable verbose output.

    Attributes
    ----------
    vocab: :obj:`Vocab`
        an instance of :class:`pytorch_widedeep.utils.fastai_transforms.Vocab`
    tokens: :obj:`List`
        List with Lists of str containing the tokenized texts
    embedding_matrix: :obj:`np.ndarray`
        Array with the pretrained embeddings

    Examples
    ---------
    >>> import pandas as pd
    >>> from pytorch_widedeep.preprocessing import TextPreprocessor
    >>> df_train = pd.DataFrame({'text_column': ["life was like a box of chocolates",
    ... "You never know what you're gonna get"]})
    >>> text_preprocessor = TextPreprocessor(text_col='text_column', max_vocab=25, min_freq=1, maxlen=10)
    >>> text_preprocessor.fit_transform(df_train)
    The vocabulary contains 24 tokens
    array([[ 1,  1,  1,  1, 10, 11, 12, 13, 14, 15],
           [ 5,  9, 16, 17, 18,  9, 19, 20, 21, 22]], dtype=int32)
    >>> df_te = pd.DataFrame({'text_column': ['you never know what is in the box']})
    >>> text_preprocessor.transform(df_te)
    array([[ 1,  1,  9, 16, 17, 18,  0,  0,  0, 13]], dtype=int32)
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
        """Builds the vocabulary
        """
        texts = df[self.text_col].tolist()
        tokens = get_texts(texts)
        self.vocab = Vocab.create(
            tokens, max_vocab=self.max_vocab, min_freq=self.min_freq
        )
        if self.verbose:
            print("The vocabulary contains {} tokens".format(len(self.vocab.stoi)))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Returns the padded, `numericalised` sequences
        """
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
        if self.word_vectors_path is not None:
            self.embedding_matrix = build_embeddings_matrix(
                self.vocab, self.word_vectors_path, self.min_freq
            )
        return padded_seq

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Combines ``fit`` and ``transform``
        """
        return self.fit(df).transform(df)


class ImagePreprocessor(BasePreprocessor):
    r"""Preprocessor to prepare the deepimage input dataset. The Preprocessing
    consists simply on resizing according to their aspect ratio

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
    verbose: Int, default 1
        Enable verbose output.

    Attributes
    ----------
    aap: :obj:`AspectAwarePreprocessor`
        an instance of :class:`pytorch_widedeep.utils.image_utils.AspectAwarePreprocessor`
    spp: :obj:`SimplePreprocessor`
        an instance of :class:`pytorch_widedeep.utils.image_utils.SimplePreprocessor`
    normalise_metrics: :obj:`Dict`
        Dict containing the normalisation metrics of the image dataset, i.e.
        mean and std for the R, G and B channels

    Examples
    --------
    >>> import pandas as pd
    >>> from pytorch_widedeep.preprocessing import ImagePreprocessor
    >>> df_train = pd.DataFrame({'images_column': ['galaxy1.png', 'galaxy2.png']})
    >>> df_test = pd.DataFrame({'images_column': ['galaxy3.png']})
    >>> img_preprocessor = ImagePreprocessor(img_col='images_column', img_path='.', verbose=0)
    >>> resized_images = img_preprocessor.fit_transform(df_train)
    >>> new_resized_images = img_preprocessor.transform(df_train)


    .. note:: Normalising metrics will only be computed when the
        ``fit_transform`` method is run. Running ``transform`` only will not
        change the computed metrics and running ``fit`` only simply
        instantiates the resizing functions.

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
        r"""Simply instantiates the Preprocessors
        :obj:`AspectAwarePreprocessor`` and :obj:`SimplePreprocessor` for image
        resizing.

        See
        :class:`pytorch_widedeep.utils.image_utils.AspectAwarePreprocessor`
        and :class:`pytorch_widedeep.utils.image_utils.SimplePreprocessor`.

        """
        self.aap = AspectAwarePreprocessor(self.width, self.height)
        self.spp = SimplePreprocessor(self.width, self.height)
        self._compute_normalising_metrics = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Resizes the images to the input height and width.
        """
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
                # if aspect ratio is 1:1, no need for AspectAwarePreprocessor
                resized_imgs.append(self.spp.preprocess(img))

        if self._compute_normalising_metrics:
            if self.verbose:
                print("Computing normalisation metrics")
            # mean and std deviation will only be computed when the fit method
            # is called
            mean_R, mean_G, mean_B = [], [], []
            std_R, std_G, std_B = [], [], []
            for rsz_img in resized_imgs:
                (mean_b, mean_g, mean_r), (std_b, std_g, std_r) = cv2.meanStdDev(
                    rsz_img
                )
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
            self._compute_normalising_metrics = False
        return np.asarray(resized_imgs)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Combines ``fit`` and ``transform``
        """
        return self.fit(df).transform(df)
