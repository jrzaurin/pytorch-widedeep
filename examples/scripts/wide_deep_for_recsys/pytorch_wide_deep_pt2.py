from pathlib import Path

import numpy as np
import torch
import pandas as pd
from torch import nn

from pytorch_widedeep import Trainer
from pytorch_widedeep.utils import pad_sequences
from pytorch_widedeep.models import TabMlp, WideDeep, Transformer
from pytorch_widedeep.preprocessing import TabPreprocessor

save_path = Path("prepared_data")

PAD_IDX = 0

id_cols = ["user_id", "movie_id"]

df_train = pd.read_pickle(save_path / "df_train.pkl")
df_valid = pd.read_pickle(save_path / "df_valid.pkl")
df_test = pd.read_pickle(save_path / "df_test.pkl")
df_test = pd.concat([df_valid, df_test], ignore_index=True)

# sequence length. Shorter sequences will be padded to this length. This is
# identical to the Kaggle's implementation
maxlen = max(
    df_train.prev_movies.apply(lambda x: len(x)).max(),
    df_test.prev_movies.apply(lambda x: len(x)).max(),
)

# Here there is a caveat. In pple, we are using (as in the Kaggle notebook)
# all indexes to compute the number of tokens in the dataset. To do this
# properly, one would have to use ONLY train tokens and add a token for new
# unknown/unseen movies in the test set. This can also be done with this
# library and manually, so I will leave it to the reader to implement that
# tokenzation appraoch
max_movie_index = max(df_train.movie_id.max(), df_test.movie_id.max())

# From now one things are pretty simple, moreover bearing in mind that in this
# example we are not going to use a wide component since, in pple, I believe
# the information in that component is also 'carried' by the movie sequences
# (also in previous scripts one can see that most prediction power comes from
# the linear, wide model)
df_train_user_item = df_train[["user_id", "movie_id", "rating"]]
train_movies_sequences = df_train.prev_movies.apply(
    lambda x: [int(el) for el in x]
).to_list()
y_train = df_train.target.values.astype(int)

df_test_user_item = df_train[["user_id", "movie_id", "rating"]]
test_movies_sequences = df_test.prev_movies.apply(
    lambda x: [int(el) for el in x]
).to_list()
y_test = df_test.target.values.astype(int)

# As a tabular component we are going to encode simply the triplets
# (user, items, rating)
tab_preprocessor = tab_preprocessor = TabPreprocessor(
    cat_embed_cols=["user_id", "movie_id", "rating"],
)
X_train_tab = tab_preprocessor.fit_transform(df_train_user_item)
X_test_tab = tab_preprocessor.transform(df_test_user_item)

# And here we pad the sequences and define a transformer model for the text
# component that is, in this case, the sequences of movies watched
X_train_text = np.array(
    [
        pad_sequences(
            s,
            maxlen=maxlen,
            pad_first=False,
            pad_idx=PAD_IDX,
        )
        for s in train_movies_sequences
    ]
)
X_test_text = np.array(
    [
        pad_sequences(
            s,
            maxlen=maxlen,
            pad_first=False,
            pad_idx=0,
        )
        for s in test_movies_sequences
    ]
)

tab_mlp = TabMlp(
    column_idx=tab_preprocessor.column_idx,
    cat_embed_input=tab_preprocessor.cat_embed_input,
    mlp_hidden_dims=[1024, 512, 256],
    mlp_activation="relu",
)

# plenty of options here, see the docs
transformer = Transformer(
    vocab_size=max_movie_index + 1,
    embed_dim=16,
    n_heads=2,
    n_blocks=2,
    seq_length=maxlen,
)

wide_deep_model = WideDeep(
    deeptabular=tab_mlp, deeptext=transformer, pred_dim=max_movie_index + 1
)

trainer = Trainer(
    model=wide_deep_model,
    objective="multiclass",
    custom_loss_function=nn.CrossEntropyLoss(ignore_index=PAD_IDX),
    optimizers=torch.optim.Adam(wide_deep_model.parameters(), lr=1e-3),
)

trainer.fit(
    X_train={
        "X_tab": X_train_tab,
        "X_text": X_train_text,
        "target": y_train,
    },
    X_val={
        "X_tab": X_test_tab,
        "X_text": X_test_text,
        "target": y_test,
    },
    n_epochs=10,
    batch_size=521,
    shuffle=False,
)
