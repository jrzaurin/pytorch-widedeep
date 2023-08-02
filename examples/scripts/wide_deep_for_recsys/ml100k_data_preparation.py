# This script is mostly a copy/paste from the Kaggle notebook
# https://www.kaggle.com/code/matanivanov/wide-deep-learning-for-recsys-with-pytorch.
# Is a response to the issue:
# https://github.com/jrzaurin/pytorch-widedeep/issues/133 In this script we
# simply prepare the data that will later be used for a custom Wide and Deep
# model and for Wide and Deep models created using this library
from pathlib import Path

from sklearn.model_selection import train_test_split

from pytorch_widedeep.datasets import load_movielens100k

data, user, items = load_movielens100k(as_frame=True)

# Alternatively, as specified in the docs: 'The last 19 fields are the genres' so:
# list_of_genres = items.columns.tolist()[-19:]
list_of_genres = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


# adding a column with the number of movies watched per user
dataset = data.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
dataset["one"] = 1
dataset["num_watched"] = dataset.groupby("user_id")["one"].cumsum()
dataset.drop("one", axis=1, inplace=True)

# adding a column with the mean rating at a point in time per user
dataset["mean_rate"] = (
    dataset.groupby("user_id")["rating"].cumsum() / dataset["num_watched"]
)

# In this particular exercise the problem is formulating as predicting the
# next movie that will be watched (in consequence the last interactions will be discarded)
dataset["target"] = dataset.groupby("user_id")["movie_id"].shift(-1)

# Here the author builds the sequences
dataset["prev_movies"] = dataset["movie_id"].apply(lambda x: str(x))
dataset["prev_movies"] = (
    dataset.groupby("user_id")["prev_movies"]
    .apply(lambda x: (x + " ").cumsum().str.strip())
    .reset_index(drop=True)
)
dataset["prev_movies"] = dataset["prev_movies"].apply(lambda x: x.split())

# Adding a genre_rate as the mean of all movies rated for a given genre per
# user
dataset = dataset.merge(items[["movie_id"] + list_of_genres], on="movie_id", how="left")
for genre in list_of_genres:
    dataset[f"{genre}_rate"] = dataset[genre] * dataset["rating"]
    dataset[genre] = dataset.groupby("user_id")[genre].cumsum()
    dataset[f"{genre}_rate"] = (
        dataset.groupby("user_id")[f"{genre}_rate"].cumsum() / dataset[genre]
    )
dataset[list_of_genres] = dataset[list_of_genres].apply(
    lambda x: x / dataset["num_watched"]
)

# Again, we use the same settings as those in the Kaggle notebook,
# but 'COLD_START_TRESH' is pretty aggressive
COLD_START_TRESH = 5

filtred_data = dataset[
    (dataset["num_watched"] >= COLD_START_TRESH) & ~(dataset["target"].isna())
].sort_values("timestamp")
train_data, _test_data = train_test_split(filtred_data, test_size=0.2, shuffle=False)
valid_data, test_data = train_test_split(_test_data, test_size=0.5, shuffle=False)

cols_to_drop = [
    # "rating",
    "timestamp",
    "num_watched",
]

df_train = train_data.drop(cols_to_drop, axis=1)
df_valid = valid_data.drop(cols_to_drop, axis=1)
df_test = test_data.drop(cols_to_drop, axis=1)

save_path = Path("prepared_data")
if not save_path.exists():
    save_path.mkdir(parents=True, exist_ok=True)

df_train.to_pickle(save_path / "df_train.pkl")
df_valid.to_pickle(save_path / "df_valid.pkl")
df_test.to_pickle(save_path / "df_test.pkl")
