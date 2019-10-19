import numpy as np
import pandas as pd
import warnings


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .wdtypes import *
from .utils.deep_utils import label_encoder
from .utils.text_utils import *
from .utils.image_utils import *


class DataProcessor(object):

    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def fit_transform(self):
        pass


class WideProcessor(DataProcessor):
    def __init__(self, wide_cols:List[str], crossed_cols=None,
        already_dummies:Optional[List[str]]=None):
        super(WideProcessor, self).__init__()
        self.wide_cols = wide_cols
        self.crossed_cols = crossed_cols
        self.already_dummies = already_dummies
        self.one_hot_enc = OneHotEncoder(sparse=False)

    def _cross_cols(self, df:pd.DataFrame):
        crossed_colnames = []
        for cols in self.crossed_cols:
            cols = list(cols)
            for c in cols: df[c] = df[c].astype('str')
            colname = '_'.join(cols)
            df[colname] = df[cols].apply(lambda x: '-'.join(x), axis=1)
            crossed_colnames.append(colname)
        return df, crossed_colnames

    def fit(self, df:pd.DataFrame)->DataProcessor:
        df_wide = df.copy()[self.wide_cols]
        if self.crossed_cols is not None:
            df_wide, crossed_colnames = self._cross_cols(df_wide)
            self.wide_crossed_cols = self.wide_cols + crossed_colnames
        else:
            self.wide_crossed_cols = self.wide_cols

        if self.already_dummies:
            dummy_cols = [c for c in self.wide_crossed_cols if c not in self.already_dummies]
            self.one_hot_enc.fit(df_wide[dummy_cols])
        else:
            self.one_hot_enc.fit(df_wide[self.wide_crossed_cols])
        return self

    def transform(self, df:pd.DataFrame)->np.ndarray:
        check_is_fitted(self.one_hot_enc, 'categories_')
        df_wide = df.copy()[self.wide_cols]
        if self.crossed_cols is not None:
            df_wide, _ = self._cross_cols(df_wide)
        if self.already_dummies:
            X_oh_1 = self.df_wide[self.already_dummies].values
            dummy_cols = [c for c in self.wide_crossed_cols if c not in self.already_dummies]
            X_oh_2=self.one_hot_enc.transform(df_wide[dummy_cols])
            return np.hstack((X_oh_1, X_oh_2))
        else:
            return (self.one_hot_enc.transform(df_wide[self.wide_crossed_cols]))

    def fit_transform(self, df:pd.DataFrame)->np.ndarray:
        return self.fit(df).transform(df)


class DeepProcessor(DataProcessor):
    def __init__(self,
        embed_cols:List[Union[str,Tuple[str,int]]]=None,
        continuous_cols:List[str]=None,
        already_standard:Optional[List[str]]=None,
        scale:bool=True,
        default_embed_dim:int=8):
        super(DeepProcessor, self).__init__()

        self.embed_cols=embed_cols
        self.continuous_cols=continuous_cols
        self.already_standard=already_standard
        self.scale=scale
        self.default_embed_dim=default_embed_dim

        assert (self.embed_cols is not None) or (self.continuous_cols is not None), \
        'Either the embedding columns or continuous columns must not be passed'

    def _prepare_embed(self, df:pd.DataFrame)->pd.DataFrame:
        if isinstance(self.embed_cols[0], Tuple):
            self.embed_dim = dict(self.embed_cols)
            embed_colname = [emb[0] for emb in self.embed_cols]
        else:
            self.embed_dim = {e:self.default_embed_dim for e in self.embed_cols}
            embed_colname = self.embed_cols
        return df.copy()[embed_colname]

    def _prepare_continuous(self, df:pd.DataFrame)->pd.DataFrame:
        if self.scale:
            if self.already_standard is not None:
                self.standardize_cols = [c for c in self.continuous_cols if c not in self.already_standard]
            else: self.standardize_cols = self.continuous_cols
        return df.copy()[self.continuous_cols]

    def fit(self, df:pd.DataFrame)->DataProcessor:
        if self.embed_cols is not None:
            df_emb = self._prepare_embed(df)
            _, self.encoding_dict = label_encoder(df_emb, cols=df_emb.columns.tolist())
            self.embeddings_input = []
            for k,v in self.encoding_dict.items():
                self.embeddings_input.append((k, len(v), self.embed_dim[k]))
        if self.continuous_cols is not None:
            df_cont = self._prepare_continuous(df)
            if self.scale:
                df_std = df_cont[self.standardize_cols]
                self.scaler = StandardScaler().fit(df_std.values)
            else:
                warnings.warn('Continuous columns will not be normalised')
        return self

    def transform(self, df:pd.DataFrame)->np.ndarray:
        if self.embed_cols is not None:
            df_emb = self._prepare_embed(df)
            df_emb, _ = label_encoder(df_emb, cols=df_emb.columns.tolist(),
                val_to_idx=self.encoding_dict)
        if self.continuous_cols is not None:
            df_cont = self._prepare_continuous(df)
            if self.scale:
                check_is_fitted(self.scaler, 'mean_')
                df_std = df_cont[self.standardize_cols]
                df_cont[self.standardize_cols] = self.scaler.transform(df_std.values)
        try:
            df_deep = pd.concat([df_emb, df_cont], axis=1)
        except:
            try:
                df_deep = df_emb.copy()
            except:
                df_deep = df_cont.copy()
        self.deep_column_idx = {k:v for v,k in enumerate(df_deep.columns)}
        return df_deep.values

    def fit_transform(self, df:pd.DataFrame)->np.ndarray:
        return self.fit(df).transform(df)


class TextProcessor(DataProcessor):
    """docstring for TextProcessor"""
    def __init__(self, max_vocab:int=30000, min_freq:int=5,
        maxlen:int=80, word_vectors_path:Optional[str]=None,
        verbose:int=1):
        super(TextProcessor, self).__init__()
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.maxlen = maxlen
        self.word_vectors_path = word_vectors_path
        self.verbose = verbose

    def fit(self, df:pd.DataFrame, text_col:str)->DataProcessor:
        text_col = text_col
        texts = df[text_col].tolist()
        tokens = get_texts(texts)
        self.vocab = Vocab.create(tokens, max_vocab=self.max_vocab, min_freq=self.min_freq)
        return self

    def transform(self, df:pd.DataFrame, text_col:str)->np.ndarray:
        check_is_fitted(self, 'vocab')
        self.text_col = text_col
        texts = df[self.text_col].tolist()
        self.tokens = get_texts(texts)
        sequences = [self.vocab.numericalize(t) for t in self.tokens]
        padded_seq = np.array([pad_sequences(s, maxlen=self.maxlen) for s in sequences])
        if self.verbose:
            print("The vocabulary contains {} words".format(len(self.vocab.stoi)))
        if self.word_vectors_path is not None:
            self.embedding_matrix = build_embeddings_matrix(self.vocab, self.word_vectors_path)
        return padded_seq

    def fit_transform(self, df:pd.DataFrame, text_col:str)->np.ndarray:
        return self.fit(df, text_col).transform(df, text_col)


class ImageProcessor(DataProcessor):
    """docstring for ImageProcessor"""
    def __init__(self, width:int=224, height:int=224, verbose:int=1):
        super(ImageProcessor, self).__init__()
        self.width = width
        self.height = height
        self.verbose = verbose

    def fit(self)->DataProcessor:
        self.aap = AspectAwarePreprocessor(self.width, self.height)
        self.spp = SimplePreprocessor(self.width, self.height)
        return self

    def transform(self, df, img_col:str, img_path:str)->np.ndarray:
        check_is_fitted(self, 'aap')
        self.img_col = img_col
        image_list = df[self.img_col].tolist()
        if self.verbose: print('Reading Images from {}'.format(img_path))
        imgs = [cv2.imread("/".join([img_path,img])) for img in image_list]

        # finding images with different height and width
        aspect = [(im.shape[0], im.shape[1]) for im in imgs]
        aspect_r = [a[0]/a[1] for a in aspect]
        diff_idx = [i for i,r in enumerate(aspect_r) if r!=1.]

        if self.verbose: print('Resizing')
        resized_imgs = []
        for i,img in tqdm(enumerate(imgs), total=len(imgs), disable=self.verbose != 1):
            if i in diff_idx:
                resized_imgs.append(self.aap.preprocess(img))
            else:
                resized_imgs.append(self.spp.preprocess(img))

        if self.verbose: print('Computing normalisation metrics')
        mean_R, mean_G, mean_B = [], [], []
        std_R, std_G, std_B = [], [], []
        for rsz_img in resized_imgs:
            (mean_b, mean_g, mean_r), (std_b, std_g, std_r) = cv2.meanStdDev(rsz_img)
            mean_R.append(mean_r), mean_G.append(mean_g), mean_B.append(mean_b)
            std_R.append(std_r), std_G.append(std_g), std_B.append(std_b)
        self.normalise_metrics = dict(
            mean = {"R": np.mean(mean_R)/255., "G": np.mean(mean_G)/255., "B": np.mean(mean_B)/255.},
            std = {"R": np.mean(std_R)/255., "G": np.mean(std_G)/255., "B": np.mean(std_B)/255.}
            )
        return np.asarray(resized_imgs)

    def fit_transform(self, df, img_col:str, img_path:str)->np.ndarray:
        return self.fit().transform(df, img_col, img_path)
