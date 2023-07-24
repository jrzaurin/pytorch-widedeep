from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, cat, mean
from scipy.sparse import coo_matrix

device = "cuda" if torch.cuda.is_available() else "cpu"

save_path = Path("prepared_data")


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


def idx_to_sparse(idx, sparse_dim):
    sparse = np.zeros(sparse_dim)
    sparse[int(idx)] = 1
    return pd.Series(sparse, dtype=int)


def process_cats_as_kaggle_notebook(df):
    df["gender"] = (df["gender"] == "M").astype(int)
    df = pd.concat(
        [
            df.drop("occupation", axis=1),
            pd.get_dummies(df["occupation"]).astype(int),
        ],
        axis=1,
    )
    df.drop("other", axis=1, inplace=True)
    df.drop("zip_code", axis=1, inplace=True)

    return df


id_cols = ["user_id", "movie_id"]

df_train = pd.read_pickle(save_path / "df_train.pkl")
df_valid = pd.read_pickle(save_path / "df_valid.pkl")
df_test = pd.read_pickle(save_path / "df_test.pkl")
df_test = pd.concat([df_valid, df_test], ignore_index=True)

df_train = process_cats_as_kaggle_notebook(df_train)
df_test = process_cats_as_kaggle_notebook(df_test)

# here is another caveat, using all dataset to build 'train_movies_watched'
# when in reality one should use only the training
max_movie_index = max(df_train.movie_id.max(), df_test.movie_id.max())

X_train = df_train.drop(id_cols + ["prev_movies", "target"], axis=1)
y_train = df_train.target.values
train_movies_watched = get_sparse_features(
    df_train["prev_movies"], (len(df_train), max_movie_index + 1)
)

X_test = df_test.drop(id_cols + ["prev_movies", "target"], axis=1)
y_test = df_test.target.values
test_movies_watched = get_sparse_features(
    df_test["prev_movies"], (len(df_test), max_movie_index + 1)
)

PAD_IDX = 0

X_train_tensor = torch.Tensor(X_train.fillna(0).values).to(device)
train_movies_watched_tensor = (
    torch.sparse_coo_tensor(
        indices=train_movies_watched.nonzero(),
        values=[1] * len(train_movies_watched.nonzero()[0]),
        size=train_movies_watched.shape,
    )
    .to_dense()
    .to(device)
)
movies_train_sequences = (
    torch.Tensor(
        sparse_to_idx(train_movies_watched, pad_idx=PAD_IDX),
    )
    .long()
    .to(device)
)
target_train = torch.Tensor(y_train).long().to(device)


X_test_tensor = torch.Tensor(X_test.fillna(0).values).to(device)
test_movies_watched_tensor = (
    torch.sparse_coo_tensor(
        indices=test_movies_watched.nonzero(),
        values=[1] * len(test_movies_watched.nonzero()[0]),
        size=test_movies_watched.shape,
    )
    .to_dense()
    .to(device)
)
movies_test_sequences = (
    torch.Tensor(
        sparse_to_idx(test_movies_watched, pad_idx=PAD_IDX),
    )
    .long()
    .to(device)
)
target_test = torch.Tensor(y_test).long().to(device)


class WideAndDeep(nn.Module):
    def __init__(
        self,
        continious_feature_shape,  # number of continious features
        embed_size,  # size of embedding for binary features
        embed_dict_len,  # number of unique binary features
        pad_idx,  # padding index
    ):
        super(WideAndDeep, self).__init__()
        self.embed = nn.Embedding(embed_dict_len, embed_size, padding_idx=pad_idx)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(embed_size + continious_feature_shape, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dict_len + 256, embed_dict_len),
        )

    def forward(self, continious, binary, binary_idx):
        # get embeddings for sequence of indexes
        binary_embed = self.embed(binary_idx)
        binary_embed_mean = mean(binary_embed, dim=1)
        # get logits for "deep" part: continious features + binary embeddings
        deep_logits = self.linear_relu_stack(
            cat((continious, binary_embed_mean), dim=1)
        )
        # get final softmax logits for "deep" part and raw binary features
        total_logits = self.head(cat((deep_logits, binary), dim=1))
        return total_logits


model = WideAndDeep(X_train.shape[1], 16, max_movie_index + 1, PAD_IDX).to(device)
print(model)


EPOCHS = 10
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for t in range(EPOCHS):
    model.train()
    pred_train = model(
        X_train_tensor, train_movies_watched_tensor, movies_train_sequences
    )
    loss_train = loss_fn(pred_train, target_train)

    # Backpropagation
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_test = model(
            X_test_tensor, test_movies_watched_tensor, movies_test_sequences
        )
        loss_test = loss_fn(pred_test, target_test)

    print(f"Epoch {t}")
    print(f"Train loss: {loss_train:>7f}")
    print(f"Test loss: {loss_test:>7f}")
