import warnings

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.utils.text_utils import (
    get_texts,
    pad_sequences,
    build_embeddings_matrix,
)
from pytorch_widedeep.utils.image_utils import (
    SimplePreprocessor,
    AspectAwarePreprocessor,
)
from pytorch_widedeep.utils.deeptabular_utils import LabelEncoder
from pytorch_widedeep.utils.fastai_transforms import Vocab

__all__ = [
    "WidePreprocessor",
    "TabPreprocessor",
    "TextPreprocessor",
    "ImagePreprocessor",
]


# This class does not represent any sctructural advantage, but I keep it to
# keep things tidy and also as guidance for contribution
class BasePreprocessor:
    """Base Class of All Preprocessors."""

    def __init__(self, *args):
        pass

    def fit(self, df: pd.DataFrame):
        raise NotImplementedError("Preprocessor must implement this method")

    def transform(self, df: pd.DataFrame):
        raise NotImplementedError("Preprocessor must implement this method")

    def fit_transform(self, df: pd.DataFrame):
        raise NotImplementedError("Preprocessor must implement this method")


def check_is_fitted(
    estimator: BasePreprocessor,
    attributes: List[str] = None,
    all_or_any: str = "all",
    condition: bool = True,
):
    r"""Checks if an estimator is fitted

    Parameters
    ----------
    estimator: ``BasePreprocessor``,
        An object of type ``BasePreprocessor``
    attributes: List, default = None
        List of strings with the attributes to check for
    all_or_any: str, default = "all"
        whether all or any of the attributes in the list must be present
    condition: bool, default = True,
        If not attribute list is passed, this condition that must be True for
        the estimator to be considered as fitted
    """

    estimator_name: str = estimator.__class__.__name__
    error_msg = (
        "This {} instance is not fitted yet. Call 'fit' with appropriate "
        "arguments before using this estimator.".format(estimator_name)
    )
    if attributes is not None and all_or_any == "all":
        if not all([hasattr(estimator, attr) for attr in attributes]):
            raise NotFittedError(error_msg)
    elif attributes is not None and all_or_any == "any":
        if not any([hasattr(estimator, attr) for attr in attributes]):
            raise NotFittedError(error_msg)
    elif not condition:
        raise NotFittedError(error_msg)


class WidePreprocessor(BasePreprocessor):
    def __init__(
        self, wide_cols: List[str], crossed_cols: List[Tuple[str, str]] = None
    ):
        r"""Preprocessor to prepare the wide input dataset

        This Preprocessor prepares the data for the wide, linear component.
        This linear model is implemented via an Embedding layer that is
        connected to the output neuron. ``WidePreprocessor`` numerically
        encodes all the unique values of all categorical columns ``wide_cols +
        crossed_cols``. See the Example below.

        Parameters
        ----------
        wide_cols: List
            List of strings with the name of the columns that will label
            encoded and passed through the ``wide`` component
        crossed_cols: List, default = None
            List of Tuples with the name of the columns that will be `'crossed'`
            and then label encoded. e.g. [('education', 'occupation'), ...]

        Attributes
        ----------
        wide_crossed_cols: List
            List with the names of all columns that will be label encoded
        encoding_dict: Dict
            Dictionary where the keys are the result of pasting `colname + '_' +
            column value` and the values are the corresponding mapped integer.

        Example
        -------
        >>> import pandas as pd
        >>> from pytorch_widedeep.preprocessing import WidePreprocessor
        >>> df = pd.DataFrame({'color': ['r', 'b', 'g'], 'size': ['s', 'n', 'l']})
        >>> wide_cols = ['color']
        >>> crossed_cols = [('color', 'size')]
        >>> wide_preprocessor = WidePreprocessor(wide_cols=wide_cols, crossed_cols=crossed_cols)
        >>> X_wide = wide_preprocessor.fit_transform(df)
        >>> X_wide
        array([[1, 4],
               [2, 5],
               [3, 6]])
        >>> wide_preprocessor.encoding_dict
        {'color_r': 1, 'color_b': 2, 'color_g': 3, 'color_size_r-s': 4, 'color_size_b-n': 5, 'color_size_g-l': 6}
        >>> wide_preprocessor.inverse_transform(X_wide)
          color color_size
        0     r        r-s
        1     b        b-n
        2     g        g-l
        """
        super(WidePreprocessor, self).__init__()

        self.wide_cols = wide_cols
        self.crossed_cols = crossed_cols

    def fit(self, df: pd.DataFrame) -> BasePreprocessor:
        r"""Fits the Preprocessor and creates required attributes"""
        df_wide = self._prepare_wide(df)
        self.wide_crossed_cols = df_wide.columns.tolist()
        glob_feature_list = self._make_global_feature_list(
            df_wide[self.wide_crossed_cols]
        )
        # leave 0 for padding/"unseen" categories
        self.encoding_dict = {v: i + 1 for i, v in enumerate(glob_feature_list)}
        self.inverse_encoding_dict = {k: v for v, k in self.encoding_dict.items()}
        self.inverse_encoding_dict[0] = "unseen"
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        r"""Returns the processed dataframe"""
        check_is_fitted(self, attributes=["encoding_dict"])
        df_wide = self._prepare_wide(df)
        encoded = np.zeros([len(df_wide), len(self.wide_crossed_cols)])
        for col_i, col in enumerate(self.wide_crossed_cols):
            encoded[:, col_i] = df_wide[col].apply(
                lambda x: self.encoding_dict[col + "_" + str(x)]
                if col + "_" + str(x) in self.encoding_dict
                else 0
            )
        return encoded.astype("int64")

    def inverse_transform(self, encoded: np.ndarray) -> pd.DataFrame:
        r"""Takes as input the output from the ``transform`` method and it will
        return the original values.
        """
        decoded = pd.DataFrame(encoded, columns=self.wide_crossed_cols)
        decoded = decoded.applymap(lambda x: self.inverse_encoding_dict[x])
        for col in decoded.columns:
            rm_str = "".join([col, "_"])
            decoded[col] = decoded[col].apply(lambda x: x.replace(rm_str, ""))
        return decoded

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Combines ``fit`` and ``transform``"""
        return self.fit(df).transform(df)

    def _make_global_feature_list(self, df: pd.DataFrame) -> List:
        glob_feature_list = []
        for column in df.columns:
            glob_feature_list += self._make_column_feature_list(df[column])
        return glob_feature_list

    def _make_column_feature_list(self, s: pd.Series) -> List:
        return [s.name + "_" + str(x) for x in s.unique()]

    def _cross_cols(self, df: pd.DataFrame):
        df_cc = df.copy()
        crossed_colnames = []
        for cols in self.crossed_cols:
            for c in cols:
                df_cc[c] = df_cc[c].astype("str")
            colname = "_".join(cols)
            df_cc[colname] = df_cc[list(cols)].apply(lambda x: "-".join(x), axis=1)
            crossed_colnames.append(colname)
        return df_cc[crossed_colnames]

    def _prepare_wide(self, df: pd.DataFrame):
        if self.crossed_cols is not None:
            df_cc = self._cross_cols(df)
            return pd.concat([df[self.wide_cols], df_cc], axis=1)
        else:
            return df.copy()[self.wide_cols]


class TabPreprocessor(BasePreprocessor):
    def __init__(
        self,
        embed_cols: Union[List[str], List[Tuple[str, int]]] = None,
        continuous_cols: List[str] = None,
        scale: bool = True,
        default_embed_dim: int = 16,
        already_standard: List[str] = None,
        for_tabtransformer: bool = False,
        verbose: int = 1,
    ):
        r"""Preprocessor to prepare the ``deeptabular`` component input dataset

        Parameters
        ----------
        embed_cols: List, default = None
            List containing the name of the columns that will be represented by
            embeddings or a Tuple with the name and the embedding dimension. e.g.:
            [('education',32), ('relationship',16), ...]
        continuous_cols: List, default = None
            List with the name of the so called continuous cols
        scale: bool, default = True
            Bool indicating whether or not to scale/standarise continuous
            cols. The user should bear in mind that all the ``deeptabular``
            components available within ``pytorch-widedeep`` they also include
            the possibility of normalising the input continuous features via a
            ``BatchNorm`` or a ``LayerNorm`` layer. see
            :class:`pytorch_widedeep.models`
        default_embed_dim: int, default=16
            Dimension for the embeddings used for the ``deeptabular``
            component if the embed_dim is not provided in the ``embed_cols``
            parameter
        already_standard: List, default = None
            List with the name of the continuous cols that do not need to be
            Standarised. For example, you might have Long and Lat in your
            dataset and might want to encode them somehow (e.g. see the
            ``LatLongScalarEnc`` available in the `autogluon
            <https://github.com/awslabs/autogluon/tree/master/tabular/src/autogluon/tabular>`_
            tabular library) and NOT standarize them any further
        for_tabtransformer: bool, default = False
            Boolean indicating whether the preprocessed data will be passed to
            a ``TabTransformer`` model. If ``True``, the param ``embed_cols``
            must just be a list containing the categorical columns: e.g.:
            ['education', 'relationship', ...] This is because following the
            results in the `paper <https://arxiv.org/pdf/2012.06678.pdf>`_,
            they will all be encoded using embeddings of the same dim (32 by
            default). See
            :class:`pytorch_widedeep.models.tab_transformer.TabTransformer`
        verbose: int, default = 1

        Attributes
        ----------
        label_encoder: LabelEncoder
            see :class:`pytorch_widedeep.utils.dense_utils.LabelEncder`
        embed_cols: List
            List with the columns that will be represented by embeddings
        embed_dim: Dict
            Dictionary where keys are the embed cols and values are the embedding
            dimensions. If ``for_tabtransformer`` is set to ``True`` the embedding
            dimensions are the same for all columns and this attributes is not
            generated during the ``fit`` process
        standardize_cols: List
            List of the columns that will be standarized
        column_idx: Dict
            Dictionary where keys are column names and values are column indexes.
            This is be neccesary to slice tensors
        scaler: StandardScaler
            an instance of :class:`sklearn.preprocessing.StandardScaler`

        Examples
        --------
        >>> import pandas as pd
        >>> from pytorch_widedeep.preprocessing import TabPreprocessor
        >>> df = pd.DataFrame({'color': ['r', 'b', 'g'], 'size': ['s', 'n', 'l'], 'age': [25, 40, 55]})
        >>> embed_cols = [('color',5), ('size',5)]
        >>> cont_cols = ['age']
        >>> deep_preprocessor = TabPreprocessor(embed_cols=embed_cols, continuous_cols=cont_cols)
        >>> deep_preprocessor.fit_transform(df)
        array([[ 1.        ,  1.        , -1.22474487],
               [ 2.        ,  2.        ,  0.        ],
               [ 3.        ,  3.        ,  1.22474487]])
        >>> deep_preprocessor.embed_dim
        {'color': 5, 'size': 5}
        >>> deep_preprocessor.column_idx
        {'color': 0, 'size': 1, 'age': 2}
        """
        super(TabPreprocessor, self).__init__()

        self.embed_cols = embed_cols
        self.continuous_cols = continuous_cols
        self.scale = scale
        self.default_embed_dim = default_embed_dim
        self.already_standard = already_standard
        self.for_tabtransformer = for_tabtransformer
        self.verbose = verbose

        self.is_fitted = False

        if (self.embed_cols is None) and (self.continuous_cols is None):
            raise ValueError(
                "'embed_cols' and 'continuous_cols' are 'None'. Please, define at least one of the two."
            )

        tabtransformer_error_message = (
            "If for_tabtransformer is 'True' embed_cols must be a list "
            " of strings with the columns to be encoded as embeddings."
        )
        if self.for_tabtransformer and self.embed_cols is None:
            raise ValueError(tabtransformer_error_message)
        if self.for_tabtransformer and isinstance(self.embed_cols[0], tuple):  # type: ignore[index]
            raise ValueError(tabtransformer_error_message)

    def fit(self, df: pd.DataFrame) -> BasePreprocessor:
        """Fits the Preprocessor and creates required attributes"""
        if self.embed_cols is not None:
            df_emb = self._prepare_embed(df)
            self.label_encoder = LabelEncoder(df_emb.columns.tolist()).fit(df_emb)
            self.embeddings_input: List = []
            for k, v in self.label_encoder.encoding_dict.items():
                if self.for_tabtransformer:
                    self.embeddings_input.append((k, len(v)))
                else:
                    self.embeddings_input.append((k, len(v), self.embed_dim[k]))
        if self.continuous_cols is not None:
            df_cont = self._prepare_continuous(df)
            if self.scale:
                df_std = df_cont[self.standardize_cols]
                self.scaler = StandardScaler().fit(df_std.values)
            elif self.verbose:
                warnings.warn("Continuous columns will not be normalised")
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Returns the processed ``dataframe`` as a np.ndarray"""
        check_is_fitted(self, condition=self.is_fitted)
        if self.embed_cols is not None:
            df_emb = self._prepare_embed(df)
            df_emb = self.label_encoder.transform(df_emb)
        if self.continuous_cols is not None:
            df_cont = self._prepare_continuous(df)
            if self.scale:
                df_std = df_cont[self.standardize_cols]
                df_cont[self.standardize_cols] = self.scaler.transform(df_std.values)
        try:
            df_deep = pd.concat([df_emb, df_cont], axis=1)
        except NameError:
            try:
                df_deep = df_emb.copy()
            except NameError:
                df_deep = df_cont.copy()
        self.column_idx = {k: v for v, k in enumerate(df_deep.columns)}
        return df_deep.values

    def inverse_transform(self, encoded: np.ndarray) -> pd.DataFrame:
        r"""Takes as input the output from the ``transform`` method and it will
        return the original values.

        Parameters
        ----------
        encoded: np.ndarray
            array with the output of the ``transform`` method
        """
        decoded = pd.DataFrame(encoded, columns=self.column_idx.keys())
        # embeddings back to original category
        if self.embed_cols is not None:
            if isinstance(self.embed_cols[0], tuple):
                emb_c: List = [c[0] for c in self.embed_cols]
            else:
                emb_c = self.embed_cols.copy()
            for c in emb_c:
                decoded[c] = decoded[c].map(self.label_encoder.inverse_encoding_dict[c])
        # continuous_cols back to non-standarised
        try:
            decoded[self.continuous_cols] = self.scaler.inverse_transform(
                decoded[self.continuous_cols]
            )
        except AttributeError:
            pass
        return decoded

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Combines ``fit`` and ``transform``"""
        return self.fit(df).transform(df)

    def _prepare_embed(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.for_tabtransformer:
            return df.copy()[self.embed_cols]
        else:
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
    def __init__(
        self,
        text_col: str,
        max_vocab: int = 30000,
        min_freq: int = 5,
        maxlen: int = 80,
        word_vectors_path: Optional[str] = None,
        verbose: int = 1,
    ):
        r"""Preprocessor to prepare the ``deeptext`` input dataset

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
        vocab: Vocab
            an instance of :class:`pytorch_widedeep.utils.fastai_transforms.Vocab`
        tokens: List
            List with Lists of str containing the tokenized texts
        embedding_matrix: np.ndarray
            Array with the pretrained embeddings

        Examples
        ---------
        >>> import pandas as pd
        >>> from pytorch_widedeep.preprocessing import TextPreprocessor
        >>> df_train = pd.DataFrame({'text_column': ["life is like a box of chocolates",
        ... "You never know what you're gonna get"]})
        >>> text_preprocessor = TextPreprocessor(text_col='text_column', max_vocab=25, min_freq=1, maxlen=10)
        >>> text_preprocessor.fit_transform(df_train)
        The vocabulary contains 24 tokens
        array([[ 1,  1,  1,  1, 10, 11, 12, 13, 14, 15],
               [ 5,  9, 16, 17, 18,  9, 19, 20, 21, 22]], dtype=int32)
        >>> df_te = pd.DataFrame({'text_column': ['you never know what is in the box']})
        >>> text_preprocessor.transform(df_te)
        array([[ 1,  1,  9, 16, 17, 18, 11,  0,  0, 13]], dtype=int32)
        """
        super(TextPreprocessor, self).__init__()

        self.text_col = text_col
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.maxlen = maxlen
        self.word_vectors_path = word_vectors_path
        self.verbose = verbose

    def fit(self, df: pd.DataFrame) -> BasePreprocessor:
        """Builds the vocabulary"""
        texts = df[self.text_col].tolist()
        tokens = get_texts(texts)
        self.vocab = Vocab.create(
            tokens, max_vocab=self.max_vocab, min_freq=self.min_freq
        )
        if self.verbose:
            print("The vocabulary contains {} tokens".format(len(self.vocab.stoi)))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Returns the padded, `numericalised` sequences"""
        check_is_fitted(self, attributes=["vocab"])
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
        """Combines ``fit`` and ``transform``"""
        return self.fit(df).transform(df)

    def inverse_transform(self, padded_seq: np.ndarray) -> pd.DataFrame:
        """Returns the original text plus the added 'special' tokens"""
        texts = [self.vocab.textify(num) for num in padded_seq]
        return pd.DataFrame({self.text_col: texts})


class ImagePreprocessor(BasePreprocessor):
    def __init__(
        self,
        img_col: str,
        img_path: str,
        width: int = 224,
        height: int = 224,
        verbose: int = 1,
    ):
        r"""Preprocessor to prepare the ``deepimage`` input dataset.

        The Preprocessing consists simply on resizing according to their
        aspect ratio

        Parameters
        ----------
        img_col: str
            name of the column with the images filenames
        img_path: str
            path to the dicrectory where the images are stored
        width: int, default=224
            width of the resulting processed image. 224 because the default
            architecture used by WideDeep is ResNet
        height: int, default=224
            width of the resulting processed image. 224 because the default
            architecture used by WideDeep is ResNet
        verbose: int, default 1
            Enable verbose output.

        Attributes
        ----------
        aap: AspectAwarePreprocessor
            an instance of :class:`pytorch_widedeep.utils.image_utils.AspectAwarePreprocessor`
        spp: SimplePreprocessor
            an instance of :class:`pytorch_widedeep.utils.image_utils.SimplePreprocessor`
        normalise_metrics: Dict
            Dict containing the normalisation metrics of the image dataset, i.e.
            mean and std for the R, G and B channels

        Examples
        --------
        >>> import pandas as pd
        >>>
        >>> from pytorch_widedeep.preprocessing import ImagePreprocessor
        >>>
        >>> path_to_image1 = 'tests/test_data_utils/images/galaxy1.png'
        >>> path_to_image2 = 'tests/test_data_utils/images/galaxy2.png'
        >>>
        >>> df_train = pd.DataFrame({'images_column': [path_to_image1]})
        >>> df_test = pd.DataFrame({'images_column': [path_to_image2]})
        >>> img_preprocessor = ImagePreprocessor(img_col='images_column', img_path='.', verbose=0)
        >>> resized_images = img_preprocessor.fit_transform(df_train)
        >>> new_resized_images = img_preprocessor.transform(df_train)

        .. note:: Normalising metrics will only be computed when the
            ``fit_transform`` method is run. Running ``transform`` only will not
            change the computed metrics and running ``fit`` only simply
            instantiates the resizing functions.
        """
        super(ImagePreprocessor, self).__init__()

        self.img_col = img_col
        self.img_path = img_path
        self.width = width
        self.height = height
        self.verbose = verbose

    def fit(self, df: pd.DataFrame) -> BasePreprocessor:
        r"""Instantiates the Preprocessors
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
        """Resizes the images to the input height and width."""
        check_is_fitted(self, attributes=["aap"])
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
        """Combines ``fit`` and ``transform``"""
        return self.fit(df).transform(df)

    def inverse_transform(self, transformed_image):
        raise NotImplementedError(
            "'inverse_transform' method is not implemented for 'ImagePreprocessor'"
        )
