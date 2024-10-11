import re
from typing import Dict, List, Tuple, Union, Literal, Optional

import numpy as np
import pandas as pd

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import WideDeep
from pytorch_widedeep.metrics import Accuracy
from pytorch_widedeep.datasets import load_movielens100k
from pytorch_widedeep.preprocessing import TabPreprocessor, embed_sz_rule
from pytorch_widedeep.models.rec.din import DeepInterestNetwork
from pytorch_widedeep.utils.text_utils import pad_sequences
from pytorch_widedeep.utils.deeptabular_utils import LabelEncoder
from pytorch_widedeep.preprocessing.base_preprocessor import (
    BasePreprocessor,
    check_is_fitted,
)


class DINPreprocessor(BasePreprocessor):
    def __init__(
        self,
        *,
        user_id_col: str,
        target_col: str,
        item_embed_col: Union[str, Tuple[str, int]],
        max_seq_length: int,
        action_col: Optional[str] = None,
        other_seq_embed_cols: Optional[List[str] | List[Tuple[str, int]]] = None,
        cat_embed_cols: Optional[Union[List[str], List[Tuple[str, int]]]] = None,
        continuous_cols: Optional[List[str]] = None,
        quantization_setup: Optional[
            Union[int, Dict[str, Union[int, List[float]]]]
        ] = None,
        cols_to_scale: Optional[Union[List[str], str]] = None,
        auto_embed_dim: bool = True,
        embedding_rule: Literal["google", "fastai_old", "fastai_new"] = "fastai_new",
        default_embed_dim: int = 16,
        verbose: int = 1,
        scale: bool = False,
        already_standard: Optional[List[str]] = None,
        **kwargs,
    ):

        self.user_id_col = user_id_col
        self.item_embed_col = item_embed_col
        self.max_seq_length = max_seq_length
        self.target_col = target_col if target_col is not None else "target"
        self.action_col = action_col
        self.other_seq_embed_cols = other_seq_embed_cols
        self.cat_embed_cols = cat_embed_cols
        self.continuous_cols = continuous_cols
        self.quantization_setup = quantization_setup
        self.cols_to_scale = cols_to_scale
        self.auto_embed_dim = auto_embed_dim
        self.embedding_rule = embedding_rule
        self.default_embed_dim = default_embed_dim
        self.verbose = verbose
        self.scale = scale
        self.already_standard = already_standard
        self.kwargs = kwargs

        self.has_standard_tab_data = False
        if self.cat_embed_cols or self.continuous_cols:
            self.has_standard_tab_data = True
            self.tab_preprocessor = TabPreprocessor(
                cat_embed_cols=self.cat_embed_cols,
                continuous_cols=self.continuous_cols,
                quantization_setup=self.quantization_setup,
                cols_to_scale=self.cols_to_scale,
                auto_embed_dim=self.auto_embed_dim,
                embedding_rule=self.embedding_rule,
                default_embed_dim=self.default_embed_dim,
                verbose=self.verbose,
                scale=self.scale,
                already_standard=self.already_standard,
                **self.kwargs,
            )

        self.is_fitted = False

    def fit(self, df: pd.DataFrame):

        if self.has_standard_tab_data:
            self.tab_preprocessor.fit(df)
            self.din_columns_idx = {
                col: i for i, col in enumerate(self.tab_preprocessor.column_idx.keys())
            }
        else:
            self.din_columns_idx = {}

        self.item_le = LabelEncoder(
            columns_to_encode=[
                (
                    self.item_embed_col[0]
                    if isinstance(self.item_embed_col, tuple)
                    else self.item_embed_col
                )
            ]
        )
        self.item_le.fit(df)
        self.n_items = max(self.item_le.encoding_dict[self.item_embed_col[0]].values())
        user_behaviour_embed_size = (
            self.item_embed_col[1]
            if isinstance(self.item_embed_col, tuple)
            else embed_sz_rule(self.n_items)
        )
        self.user_behaviour_config: Tuple[List[str], int, int] = (
            [f"item_{i+1}" for i in range(self.max_seq_length)],
            self.n_items,
            user_behaviour_embed_size,
        )

        _current_len = len(self.din_columns_idx)
        self.din_columns_idx.update(
            {f"item_{i+1}": i + _current_len for i in range(self.max_seq_length)}
        )
        self.din_columns_idx.update(
            {
                "target_item": len(self.din_columns_idx),
            }
        )

        if self.action_col is not None:
            self.action_le = LabelEncoder(columns_to_encode=[self.action_col])
            self.action_le.fit(df)

            self.n_actions = len(self.action_le.encoding_dict[self.action_col])
            self.action_seq_config: Tuple[List[str], int] = (
                [f"action_{i+1}" for i in range(self.max_seq_length)],
                self.n_actions,
            )

            _current_len = len(self.din_columns_idx)
            self.din_columns_idx.update(
                {f"action_{i+1}": i + _current_len for i in range(self.max_seq_length)}
            )

        if self.other_seq_embed_cols is not None:
            self.other_seq_le = LabelEncoder(
                columns_to_encode=[
                    col[0] if isinstance(col, tuple) else col
                    for col in self.other_seq_embed_cols
                ]
            )
            self.other_seq_le.fit(df)

            other_seq_cols = [
                col[0] if isinstance(col, tuple) else col
                for col in self.other_seq_embed_cols
            ]
            self.n_other_seq_cols: Dict[str, int] = {
                col: max(self.other_seq_le.encoding_dict[col].values())
                for col in other_seq_cols
            }

            if isinstance(self.other_seq_embed_cols[0], tuple):
                other_seq_embed_sizes: Dict[str, int] = {
                    tp[0]: tp[1] for tp in self.other_seq_embed_cols  # type: ignore[misc]
                }
            else:
                other_seq_embed_sizes = {
                    col: embed_sz_rule(self.n_other_seq_cols[col])
                    for col in other_seq_cols
                }

            self.other_seq_config: List[Tuple[List[str], int, int]] = [
                (
                    [f"{col}_{i+1}" for i in range(self.max_seq_length)],
                    self.n_other_seq_cols[col],
                    other_seq_embed_sizes[col],
                )
                for col in other_seq_cols
            ]

            _current_len = len(self.din_columns_idx)
            for col in other_seq_cols:
                self.din_columns_idx.update(
                    {
                        f"{col}_{i+1}": i + _current_len
                        for i in range(self.max_seq_length)
                    }
                )
                self.din_columns_idx.update(
                    {f"target_{col}": len(self.din_columns_idx)}
                )
                _current_len = len(self.din_columns_idx)

        self.is_fitted = True

        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:

        _df = self._pre_transform(df)

        df_w_sequences = self._build_sequences(_df)

        user_behaviour_seq = df_w_sequences["items_sequence"].tolist()
        X_user_behaviour = np.vstack(
            [
                pad_sequences(seq, self.max_seq_length, pad_first=False, pad_idx=0)
                for seq in user_behaviour_seq
            ]
        )

        X_target_item = np.array(df_w_sequences["target_item"].tolist()).reshape(-1, 1)

        X_all = np.concatenate([X_user_behaviour, X_target_item], axis=1)

        if self.has_standard_tab_data:
            X_tab = self.tab_preprocessor.transform(df_w_sequences)
            X_all = np.concatenate([X_tab, X_all], axis=1)

        if self.action_col is not None:
            action_seq = df_w_sequences["actions_sequence"].tolist()
            X_actions = np.vstack(
                [
                    pad_sequences(seq, self.max_seq_length, pad_first=False, pad_idx=0)
                    for seq in action_seq
                ]
            )
            X_all = np.concatenate([X_all, X_actions], axis=1)

        if self.other_seq_embed_cols is not None:
            other_seq_cols = [
                col[0] if isinstance(col, tuple) else col
                for col in self.other_seq_embed_cols
            ]
            other_seq = {
                col: df_w_sequences[f"{col}_sequence"].tolist()
                for col in other_seq_cols
            }
            other_seq_target = {
                col: df_w_sequences[f"target_{col}"].tolist() for col in other_seq_cols
            }
            X_other_seq_arrays: List[np.ndarray] = []
            # to obsessively make sure that the order of the columns is the
            # same
            for col in other_seq_cols:
                X_other_seq_arrays.append(
                    np.vstack(
                        [
                            pad_sequences(
                                s, self.max_seq_length, pad_first=False, pad_idx=0
                            )
                            for s in other_seq[col]
                        ]
                    )
                )
                X_other_seq_arrays.append(
                    np.array(other_seq_target[col]).reshape(-1, 1)
                )

            X_all = np.concatenate([X_all] + X_other_seq_arrays, axis=1)

        assert len(self.din_columns_idx) == X_all.shape[1], (
            f"Something went wrong. The number of columns in the final array "
            f"({X_all.shape[1]}) is different from the number of columns in "
            f"self.din_columns_idx ({len(self.din_columns_idx)})"
        )

        return X_all, np.array(df_w_sequences["target"].tolist())

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        self.fit(df)
        return self.transform(df)

    def _pre_transform(self, df: pd.DataFrame) -> pd.DataFrame:

        check_is_fitted(self, attributes=["din_columns_idx"])

        df = self.item_le.transform(df)

        if self.action_col is not None:
            df = self.action_le.transform(df)

        if self.other_seq_embed_cols is not None:
            df = self.other_seq_le.transform(df)

        return df

    def _build_sequences(self, df: pd.DataFrame) -> pd.DataFrame:

        res_df = (
            df.groupby(self.user_id_col)
            .apply(self._group_sequences)
            .reset_index(drop=True)
        )

        return res_df

    def _group_sequences(self, group: pd.DataFrame) -> pd.DataFrame:

        item_col = (
            self.item_embed_col[0]
            if isinstance(self.item_embed_col, tuple)
            else self.item_embed_col
        )
        items = group[item_col].tolist()

        targets = group[self.target_col].tolist()
        drop_cols = [item_col, self.target_col]

        sequences: List[Dict[str, str | List[int]]] = []
        for i in range(len(items) - self.max_seq_length):
            item_sequences = items[i : i + self.max_seq_length]
            target_item = items[i + self.max_seq_length]
            target = targets[i + self.max_seq_length]

            sequence = {
                "user_id": group.name,
                "items_sequence": item_sequences,
                "target_item": target_item,
            }

            if self.action_col is not None:
                if self.action_col != self.target_col:
                    actions = group[self.action_col].tolist()
                    action_sequences = actions[i : i + self.max_seq_length]
                    drop_cols.append(self.action_col)
                else:
                    target -= (
                        1  # the 'transform' method adds 1 as it saves 0 for padding
                    )
                    action_sequences = targets[i : i + self.max_seq_length]

                sequence["target"] = target
                sequence["actions_sequence"] = action_sequences

            if self.other_seq_embed_cols is not None:
                other_seq_cols = [
                    col[0] if isinstance(col, tuple) else col
                    for col in self.other_seq_embed_cols
                ]
                drop_cols += other_seq_cols
                other_seqs: Dict[str, List[int]] = {
                    col: group[col].tolist() for col in other_seq_cols
                }
                other_seqs_sequences = {
                    col: other_seqs[col][i : i + self.max_seq_length]
                    for col in other_seq_cols
                }

                for col in other_seq_cols:
                    sequence[f"{col}_sequence"] = other_seqs_sequences[col]
                    sequence[f"target_{col}"] = other_seqs[col][i + self.max_seq_length]

            sequences.append(sequence)

        seq_df = pd.DataFrame(sequences)

        non_seq_cols = group.drop_duplicates(["user_id"]).drop(drop_cols, axis=1)

        return pd.merge(seq_df, non_seq_cols, on="user_id")

    def inverse_transform(self, X: np.ndarray) -> pd.DataFrame:
        # trasform mutates the df and is complex to revert the transformation
        raise NotImplementedError(
            "inverse_transform is not implemented for this preprocessor"
        )


if __name__ == "__main__":

    def clean_genre_list(genre_list):
        return "_".join(
            sorted([re.sub(r"[^a-z0-9]", "", genre.lower()) for genre in genre_list])
        )

    data, users, items = load_movielens100k(as_frame=True)

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

    assert (
        isinstance(items, pd.DataFrame)
        and isinstance(data, pd.DataFrame)
        and isinstance(users, pd.DataFrame)
    )
    items["genre_list"] = items[list_of_genres].apply(
        lambda x: [genre for genre in list_of_genres if x[genre] == 1], axis=1
    )

    items["genre_list"] = items["genre_list"].apply(clean_genre_list)

    df = pd.merge(data, items[["movie_id", "genre_list"]], on="movie_id")
    df = pd.merge(
        df,
        users[["user_id", "age", "gender", "occupation"]],
        on="user_id",
    )

    df["rating"] = df["rating"].apply(lambda x: 1 if x >= 4 else 0)

    df = df.sort_values(by=["timestamp"]).reset_index(drop=True)
    df = df.drop("timestamp", axis=1)

    din_preprocessor = DINPreprocessor(
        user_id_col="user_id",
        target_col="rating",
        item_embed_col=("movie_id", 16),
        max_seq_length=5,
        action_col="rating",
        other_seq_embed_cols=[("genre_list", 16)],
        cat_embed_cols=["user_id", "age", "gender", "occupation"],
    )

    X, y = din_preprocessor.fit_transform(df)

    din = DeepInterestNetwork(
        column_idx=din_preprocessor.din_columns_idx,
        user_behavior_confiq=din_preprocessor.user_behaviour_config,
        action_seq_config=din_preprocessor.action_seq_config,
        other_seq_cols_confiq=din_preprocessor.other_seq_config,
        cat_embed_input=din_preprocessor.tab_preprocessor.cat_embed_input,  # type: ignore[attr-defined]
        mlp_hidden_dims=[128, 64],
    )

    # And from here on, everything is standard
    model = WideDeep(deeptabular=din)

    trainer = Trainer(model=model, objective="binary", metrics=[Accuracy()])

    # in the real world you would have to split the data into train, val and test
    trainer.fit(
        X_tab=X,
        target=y,
        n_epochs=5,
        batch_size=512,
    )
