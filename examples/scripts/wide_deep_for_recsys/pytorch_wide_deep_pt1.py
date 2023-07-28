# In this script I illustrate how one coould use our library to reproduce
# almost exactly the same model used in the Kaggle Notebook

from pathlib import Path

import numpy as np
import torch
import pandas as pd
from torch import nn
from scipy.sparse import coo_matrix

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import TabMlp, BasicRNN, WideDeep
from pytorch_widedeep.preprocessing import TabPreprocessor

device = "cuda" if torch.cuda.is_available() else "cpu"

save_path = Path("prepared_data")

PAD_IDX = 0


def get_coo_indexes(lil):
    rows = []
    cols = []
    for i, el in enumerate(lil):
        if type(el) != list:
            el = [el]
        for j in el:
            rows.append(i)
            cols.append(j)
    return rows, cols


def get_sparse_features(series, shape):
    coo_indexes = get_coo_indexes(series.tolist())
    sparse_df = coo_matrix(
        (np.ones(len(coo_indexes[0])), (coo_indexes[0], coo_indexes[1])), shape=shape
    )
    return sparse_df


def sparse_to_idx(data, pad_idx=-1):
    indexes = data.nonzero()
    indexes_df = pd.DataFrame()
    indexes_df["rows"] = indexes[0]
    indexes_df["cols"] = indexes[1]
    mdf = indexes_df.groupby("rows").apply(lambda x: x["cols"].tolist())
    max_len = mdf.apply(lambda x: len(x)).max()
    return mdf.apply(lambda x: pd.Series(x + [pad_idx] * (max_len - len(x)))).values


id_cols = ["user_id", "movie_id"]

df_train = pd.read_pickle(save_path / "df_train.pkl")
df_valid = pd.read_pickle(save_path / "df_valid.pkl")
df_test = pd.read_pickle(save_path / "df_test.pkl")
df_test = pd.concat([df_valid, df_test], ignore_index=True)

# here is another caveat, using all dataset to build 'train_movies_watched'
# when in reality one should use only the training
max_movie_index = max(df_train.movie_id.max(), df_test.movie_id.max())

X_train = df_train.drop(id_cols + ["rating", "prev_movies", "target"], axis=1)
y_train = np.array(df_train.target.values, dtype="int64")
train_movies_watched = get_sparse_features(
    df_train["prev_movies"], (len(df_train), max_movie_index + 1)
)

X_test = df_test.drop(id_cols + ["rating", "prev_movies", "target"], axis=1)
y_test = np.array(df_test.target.values, dtype="int64")
test_movies_watched = get_sparse_features(
    df_test["prev_movies"], (len(df_test), max_movie_index + 1)
)

cat_cols = ["gender", "occupation", "zip_code"]
cont_cols = [c for c in X_train if c not in cat_cols]
tab_preprocessor = TabPreprocessor(
    cat_embed_cols=cat_cols,
    continuous_cols=cont_cols,
)

# The sparse matrices need to be turned into dense whether at array or tensor
# stage. This is one of the reasons why the wide component in our library is
# implemented as Embeddings. However, our implementation is still not
# suitable for the type of pre-processing that the author of the Kaggle
# notebook did to come up with the what it would be the wide component
# (a sparse martrix with 1s at those locations corresponding to the movies
# that a user has seen at a point in time). Therefore, we will have to code a
# Wide model (fairly simple since it is a linear layer)
X_train_wide = np.array(train_movies_watched.todense())
X_test_wide = np.array(test_movies_watched.todense())

# Here our tabular component is a bit more elaborated than that in the
# notebook, just a bit...
X_train_tab = tab_preprocessor.fit_transform(X_train.fillna(0))
X_test_tab = tab_preprocessor.transform(X_test.fillna(0))

# The text component are the sequences of movies wacthed. There is an element
# of information redundancy here in my opinion. This is because the wide and
# text components have implicitely the same information, but in different
# form. Anyway, we want to reproduce the Kaggle notebook as close as
# possible.
X_train_text = sparse_to_idx(train_movies_watched, pad_idx=PAD_IDX)
X_test_text = sparse_to_idx(test_movies_watched, pad_idx=PAD_IDX)


class Wide(nn.Module):
    def __init__(self, input_dim: int, pred_dim: int):
        super().__init__()

        self.input_dim = input_dim
        self.pred_dim = pred_dim

        # The way I coded the library I never though that someone would ever
        # wanted to code their own wide component. However, if you do, the
        # wide component must have a 'wide_linear' attribute. In other words,
        # the linear layer must be called 'wide_linear'
        self.wide_linear = nn.Linear(input_dim, pred_dim)

    def forward(self, X):
        out = self.wide_linear(X.type(torch.float32))
        return out


wide = Wide(X_train_wide.shape[1], max_movie_index + 1)


class SimpleEmbed(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, pad_idx: int):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.pad_idx = pad_idx

        # The sequences of movies watched are simply embedded in the Kaggle
        # notebook. No RNN, Transformer or any model is used
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

    def forward(self, X):
        embed = self.embed(X)
        embed_mean = torch.mean(embed, dim=1)
        return embed_mean

    @property
    def output_dim(self) -> int:
        return self.embed_dim


# In the notebook the author uses simply embeddings
simple_embed = SimpleEmbed(max_movie_index + 1, 16, 0)
# but maybe one would like to use an RNN to account for the sequence nature of
# the problem formulation
basic_rnn = BasicRNN(
    vocab_size=max_movie_index + 1,
    embed_dim=16,
    hidden_dim=32,
    n_layers=2,
    rnn_type="gru",
)

tab_mlp = TabMlp(
    column_idx=tab_preprocessor.column_idx,
    cat_embed_input=tab_preprocessor.cat_embed_input,
    continuous_cols=tab_preprocessor.continuous_cols,
    cont_norm_layer=None,
    mlp_hidden_dims=[1024, 512, 256],
    mlp_activation="relu",
)

# The main difference between this wide and deep model and the Wide and Deep
# model in the Kaggle notebook is that in that notebook, the author
# concatenates the embedings and the tabular features(which he refers
# as 'continuous'), then passes this concatenation through a stack of
# linear + Relu layers. Then concatenates this output with the binary
# features and connects this concatenation with the final linear layer. Our
# implementation follows the notation of the original paper and instead of
# concatenating the tabular, text and wide components, we first compute their
# output, and then add it (see here: https://arxiv.org/pdf/1606.07792.pdf,
# their Eq 3). Note that this is effectively the same with the caveat that
# while in one case we initialise a big weight matrix at once, in our
# implementation we initialise different matrices for different components.
# Anyway, let's give it a go.
wide_deep_model = WideDeep(
    wide=wide, deeptabular=tab_mlp, deeptext=simple_embed, pred_dim=max_movie_index + 1
)
# # To use an RNN, simply
# wide_deep_model = WideDeep(
#     wide=wide, deeptabular=tab_mlp, deeptext=basic_rnn, pred_dim=max_movie_index + 1
# )

trainer = Trainer(
    model=wide_deep_model,
    objective="multiclass",
    custom_loss_function=nn.CrossEntropyLoss(ignore_index=PAD_IDX),
    optimizers=torch.optim.Adam(wide_deep_model.parameters(), lr=1e-3),
)

trainer.fit(
    X_train={
        "X_wide": X_train_wide,
        "X_tab": X_train_tab,
        "X_text": X_train_text,
        "target": y_train,
    },
    X_val={
        "X_wide": X_test_wide,
        "X_tab": X_test_tab,
        "X_text": X_test_text,
        "target": y_test,
    },
    n_epochs=10,
    batch_size=512,
    shuffle=False,
)
