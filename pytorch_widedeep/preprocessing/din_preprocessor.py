# import re
import warnings
from typing import Dict, List, Tuple, Union, Literal, Optional

import numpy as np
import pandas as pd

# from pytorch_widedeep.models import WideDeep
# from pytorch_widedeep.metrics import Accuracy
# from pytorch_widedeep.datasets import load_movielens100k
# from pytorch_widedeep.training import Trainer
# from pytorch_widedeep.models.rec.din import DeepInterestNetwork
from pytorch_widedeep.utils.text_utils import pad_sequences
from pytorch_widedeep.utils.general_utils import alias
from pytorch_widedeep.utils.deeptabular_utils import LabelEncoder
from pytorch_widedeep.preprocessing.tab_preprocessor import (
    TabPreprocessor,
    embed_sz_rule,
)
from pytorch_widedeep.preprocessing.base_preprocessor import (
    BasePreprocessor,
    check_is_fitted,
)


class DINPreprocessor(BasePreprocessor):
    """
    Preprocessor for Deep Interest Network (DIN) models.

    This preprocessor handles the preparation of data for DIN models,
    including sequence building, label encoding, and handling of various
    types of input columns (categorical, continuous, and sequential).

    Parameters:
    -----------
    user_id_col : str
        Name of the column containing user IDs.
    item_embed_col : Union[str, Tuple[str, int]]
        Name of the column containing item IDs to be embedded, or a tuple of
        (column_name, embedding_dim).
    target_col : str
        Name of the column containing the target variable.
    max_seq_length : int
        Maximum length of sequences to be created.
    action_col : Optional[str], default=None
        Name of the column containing user actions (if applicable).
    other_seq_embed_cols : Optional[List[str] | List[Tuple[str, int]]], default=None
        List of other columns to be treated as sequences.
    cat_embed_cols : Optional[Union[List[str], List[Tuple[str, int]]]], default=None
        List of categorical columns to be represented by embeddings.
    continuous_cols: List, default = None
        List with the name of the continuous cols.
    quantization_setup: int or Dict[str, Union[int, List[float]]], default=None
        Continuous columns can be turned into categorical via `pd.cut`.
    cols_to_scale: List or str, default = None,
        List with the names of the columns that will be standardized via
        sklearn's `StandardScaler`.
    auto_embed_dim: bool, default = True
        Boolean indicating whether the embedding dimensions will be
        automatically defined via rule of thumb.
    embedding_rule: str, default = 'fastai_new'
        Rule of thumb for embedding size.
    default_embed_dim: int, default=16
        Default dimension for the embeddings.
    verbose : int, default=1
        Verbosity level.
    scale: bool, default = False
        Boolean indicating whether or not to scale/standardize continuous cols.
    already_standard: List, default = None
        List with the name of the continuous cols that do not need to be
        scaled/standardized.
    **kwargs :
        Additional keyword arguments to be passed to the TabPreprocessor.

    Attributes:
    -----------
    is_fitted : bool
        Whether the preprocessor has been fitted.
    has_standard_tab_data : bool
        Whether the data includes standard tabular data.
    tab_preprocessor : TabPreprocessor
        Preprocessor for standard tabular data.
    din_columns_idx : Dict[str, int]
        Dictionary mapping column names to their indices in the processed data.
    item_le : LabelEncoder
        Label encoder for item IDs.
    n_items : int
        Number of unique items.
    user_behaviour_config : Tuple[List[str], int, int]
        Configuration for user behavior sequences.
    action_le : LabelEncoder
        Label encoder for action column (if applicable).
    n_actions : int
        Number of unique actions (if applicable).
    action_seq_config : Tuple[List[str], int]
        Configuration for action sequences (if applicable).
    other_seq_le : LabelEncoder
        Label encoder for other sequence columns (if applicable).
    n_other_seq_cols : Dict[str, int]
        Number of unique values in each other sequence column.
    other_seq_config : List[Tuple[List[str], int, int]]
        Configuration for other sequence columns.
    """

    @alias("item_embed_col", ["item_id_col"])
    def __init__(
        self,
        *,
        user_id_col: str,
        item_embed_col: Union[str, Tuple[str, int]],
        target_col: str,
        max_seq_length: int,
        action_col: Optional[str] = None,
        other_seq_embed_cols: Optional[Union[List[str], List[Tuple[str, int]]]] = None,
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

        self.has_standard_tab_data = bool(self.cat_embed_cols or self.continuous_cols)
        if self.has_standard_tab_data:
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

    def fit(self, df: pd.DataFrame) -> "DINPreprocessor":
        if self.has_standard_tab_data:
            self.tab_preprocessor.fit(df)
            self.din_columns_idx = {
                col: i for i, col in enumerate(self.tab_preprocessor.column_idx.keys())
            }
        else:
            self.din_columns_idx = {}

        self._fit_label_encoders(df)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        _df = self._pre_transform(df)
        df_w_sequences = self._build_sequences(_df)
        X_all = self._concatenate_features(df_w_sequences)
        return X_all, np.array(df_w_sequences[self.target_col].tolist())

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        self.fit(df)
        return self.transform(df)

    def inverse_transform(self, X: np.ndarray) -> pd.DataFrame:
        raise NotImplementedError(
            "inverse_transform is not implemented for this preprocessor"
        )

    def _fit_label_encoders(self, df: pd.DataFrame) -> None:
        self.item_le = LabelEncoder(columns_to_encode=[self._get_item_col()])
        self.item_le.fit(df)
        self.n_items = max(self.item_le.encoding_dict[self._get_item_col()].values())
        user_behaviour_embed_size = self._get_embed_size(
            self.item_embed_col, self.n_items
        )
        self.user_behaviour_config = (
            [f"item_{i+1}" for i in range(self.max_seq_length)],
            self.n_items,
            user_behaviour_embed_size,
        )

        self._update_din_columns_idx("item", self.max_seq_length)
        self.din_columns_idx["target_item"] = len(self.din_columns_idx)

        if self.action_col:
            self._fit_action_label_encoder(df)
        if self.other_seq_embed_cols:
            self._fit_other_seq_label_encoders(df)

    def _fit_action_label_encoder(self, df: pd.DataFrame) -> None:
        self.action_le = LabelEncoder(columns_to_encode=[self.action_col])
        self.action_le.fit(df)
        self.n_actions = len(self.action_le.encoding_dict[self.action_col])
        self.action_seq_config = (
            [f"action_{i+1}" for i in range(self.max_seq_length)],
            self.n_actions,
        )
        self._update_din_columns_idx("action", self.max_seq_length)

    def _other_seq_cols_float_warning(self, df: pd.DataFrame) -> None:
        other_seq_cols = [
            col[0] if isinstance(col, tuple) else col
            for col in self.other_seq_embed_cols
        ]
        for col in other_seq_cols:
            if df[col].dtype == float:
                warnings.warn(
                    f"{col} is a float column. It will be converted to integers. "
                    "If this is not what you want, please convert it beforehand.",
                    UserWarning,
                )

    def _convert_other_seq_cols_to_int(self, df: pd.DataFrame) -> None:
        other_seq_cols = [
            col[0] if isinstance(col, tuple) else col
            for col in self.other_seq_embed_cols
        ]
        for col in other_seq_cols:
            if df[col].dtype == float:
                df[col] = df[col].astype(int)

    def _fit_other_seq_label_encoders(self, df: pd.DataFrame) -> None:
        self._other_seq_cols_float_warning(df)
        self._convert_other_seq_cols_to_int(df)
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
        self.n_other_seq_cols = {
            col: max(self.other_seq_le.encoding_dict[col].values())
            for col in other_seq_cols
        }
        other_seq_embed_sizes = {
            col: self._get_embed_size(col, self.n_other_seq_cols[col])
            for col in other_seq_cols
        }
        self.other_seq_config = [
            (
                self._get_seq_col_names(col),
                self.n_other_seq_cols[col],
                other_seq_embed_sizes[col],
            )
            for col in other_seq_cols
        ]
        self._update_din_columns_idx_for_other_seq(other_seq_cols)

    def _get_item_col(self) -> str:
        return (
            self.item_embed_col[0]
            if isinstance(self.item_embed_col, tuple)
            else self.item_embed_col
        )

    def _get_embed_size(self, col: Union[str, Tuple[str, int]], n_unique: int) -> int:
        return col[1] if isinstance(col, tuple) else embed_sz_rule(n_unique)

    def _get_seq_col_names(self, col: str) -> List[str]:
        return [f"{col}_{i+1}" for i in range(self.max_seq_length)]

    def _update_din_columns_idx(self, prefix: str, length: int) -> None:
        current_len = len(self.din_columns_idx)
        self.din_columns_idx.update(
            {f"{prefix}_{i+1}": i + current_len for i in range(length)}
        )

    def _update_din_columns_idx_for_other_seq(self, other_seq_cols: List[str]) -> None:
        current_len = len(self.din_columns_idx)
        for col in other_seq_cols:
            self.din_columns_idx.update(
                {f"{col}_{i+1}": i + current_len for i in range(self.max_seq_length)}
            )
            self.din_columns_idx[f"target_{col}"] = len(self.din_columns_idx)
            current_len = len(self.din_columns_idx)

    def _pre_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, attributes=["din_columns_idx"])
        df = self.item_le.transform(df)
        if self.action_col:
            df = self.action_le.transform(df)
        if self.other_seq_embed_cols:
            df = self.other_seq_le.transform(df)
        return df

    def _build_sequences(self, df: pd.DataFrame) -> pd.DataFrame:
        # TO DO: do something to avoid warning here
        return (
            df.groupby(self.user_id_col)
            .apply(self._group_sequences)
            .reset_index(drop=True)
        )

    def _group_sequences(self, group: pd.DataFrame) -> pd.DataFrame:
        item_col = self._get_item_col()
        items = group[item_col].tolist()
        targets = group[self.target_col].tolist()
        drop_cols = [item_col, self.target_col]

        # sequences cannot be built with user, item pairs with only one
        # interaction
        if len(items) <= 1:
            return pd.DataFrame()

        sequences = [
            self._create_sequence(group, items, targets, i, drop_cols)
            for i in range(max(1, len(items) - self.max_seq_length))
        ]

        seq_df = pd.DataFrame(sequences)
        non_seq_cols = group.drop_duplicates([self.user_id_col]).drop(drop_cols, axis=1)
        return pd.merge(seq_df, non_seq_cols, on=self.user_id_col)

    def _create_sequence(
        self,
        group: pd.DataFrame,
        items: List[int],
        targets: List[int],
        i: int,
        drop_cols: List[str],
    ) -> Dict[str, Union[str, List[int]]]:
        end_idx = min(i + self.max_seq_length, len(items) - 1)
        item_sequences = items[i:end_idx]
        target_item = items[end_idx]
        target = targets[end_idx]

        sequence = {
            self.user_id_col: group.name,
            "items_sequence": item_sequences,
            "target_item": target_item,
        }

        if self.action_col:
            sequence.update(
                self._create_action_sequence(group, targets, i, target, drop_cols)
            )
        else:
            sequence[self.target_col] = target
            drop_cols.append(self.target_col)

        if self.other_seq_embed_cols:
            sequence.update(self._create_other_seq_sequence(group, i, drop_cols))

        return sequence

    def _create_action_sequence(
        self,
        group: pd.DataFrame,
        targets: List[int],
        i: int,
        target: int,
        drop_cols: List[str],
    ) -> Dict[str, Union[int, List[int]]]:
        if self.action_col != self.target_col:
            actions = group[self.action_col].tolist()
            action_sequences = (
                actions[i : i + self.max_seq_length]
                if i + self.max_seq_length < len(actions)
                else actions[i:-1]
            )
            drop_cols.append(self.action_col)
        else:
            target -= 1  # the 'transform' method adds 1 as it saves 0 for padding
            action_sequences = (
                targets[i : i + self.max_seq_length]
                if i + self.max_seq_length < len(targets)
                else targets[i:-1]
            )

        return {self.target_col: target, "actions_sequence": action_sequences}

    def _create_other_seq_sequence(
        self, group: pd.DataFrame, i: int, drop_cols: List[str]
    ) -> Dict[str, Union[int, List[int]]]:
        other_seq_cols = [
            col[0] if isinstance(col, tuple) else col
            for col in self.other_seq_embed_cols
        ]
        drop_cols += other_seq_cols
        other_seqs = {col: group[col].tolist() for col in other_seq_cols}

        other_seqs_sequences: Dict[str, Union[int, List[int]]] = {}
        for col in other_seq_cols:
            other_seqs_sequences[col] = (
                other_seqs[col][i : i + self.max_seq_length]
                if i + self.max_seq_length < len(other_seqs[col])
                else other_seqs[col][i:-1]
            )

        sequence: Dict[str, Union[int, List[int]]] = {}
        for col in other_seq_cols:
            sequence[f"{col}_sequence"] = other_seqs_sequences[col]
            sequence[f"target_{col}"] = (
                other_seqs[col][i + self.max_seq_length]
                if i + self.max_seq_length < len(other_seqs[col])
                else other_seqs[col][-1]
            )

        return sequence

    def _concatenate_features(self, df_w_sequences: pd.DataFrame) -> np.ndarray:
        user_behaviour_seq = df_w_sequences["items_sequence"].tolist()
        X_user_behaviour = np.vstack(
            [
                pad_sequences(seq, self.max_seq_length, pad_idx=0)
                for seq in user_behaviour_seq
            ]
        )
        X_target_item = np.array(df_w_sequences["target_item"].tolist()).reshape(-1, 1)
        X_all = np.concatenate([X_user_behaviour, X_target_item], axis=1)

        if self.has_standard_tab_data:
            X_tab = self.tab_preprocessor.transform(df_w_sequences)
            X_all = np.concatenate([X_tab, X_all], axis=1)

        if self.action_col:
            action_seq = df_w_sequences["actions_sequence"].tolist()
            X_actions = np.vstack(
                [
                    pad_sequences(seq, self.max_seq_length, pad_idx=0)
                    for seq in action_seq
                ]
            )
            X_all = np.concatenate([X_all, X_actions], axis=1)

        if self.other_seq_embed_cols:
            X_all = self._concatenate_other_seq_features(df_w_sequences, X_all)

        assert len(self.din_columns_idx) == X_all.shape[1], (
            f"Something went wrong. The number of columns in the final array "
            f"({X_all.shape[1]}) is different from the number of columns in "
            f"self.din_columns_idx ({len(self.din_columns_idx)})"
        )

        return X_all

    def _concatenate_other_seq_features(
        self, df_w_sequences: pd.DataFrame, X_all: np.ndarray
    ) -> np.ndarray:
        other_seq_cols = [
            col[0] if isinstance(col, tuple) else col
            for col in self.other_seq_embed_cols
        ]
        other_seq = {
            col: df_w_sequences[f"{col}_sequence"].tolist() for col in other_seq_cols
        }
        other_seq_target = {
            col: df_w_sequences[f"target_{col}"].tolist() for col in other_seq_cols
        }
        X_other_seq_arrays = []

        for col in other_seq_cols:
            X_other_seq_arrays.append(
                np.vstack(
                    [
                        pad_sequences(s, self.max_seq_length, pad_idx=0)
                        for s in other_seq[col]
                    ]
                )
            )
            X_other_seq_arrays.append(np.array(other_seq_target[col]).reshape(-1, 1))

        return np.concatenate([X_all] + X_other_seq_arrays, axis=1)


# if __name__ == "__main__":

#     def clean_genre_list(genre_list):
#         return "_".join(
#             sorted([re.sub(r"[^a-z0-9]", "", genre.lower()) for genre in genre_list])
#         )

#     data, users, items = load_movielens100k(as_frame=True)

#     list_of_genres = [
#         "unknown",
#         "Action",
#         "Adventure",
#         "Animation",
#         "Children's",
#         "Comedy",
#         "Crime",
#         "Documentary",
#         "Drama",
#         "Fantasy",
#         "Film-Noir",
#         "Horror",
#         "Musical",
#         "Mystery",
#         "Romance",
#         "Sci-Fi",
#         "Thriller",
#         "War",
#         "Western",
#     ]

#     assert (
#         isinstance(items, pd.DataFrame)
#         and isinstance(data, pd.DataFrame)
#         and isinstance(users, pd.DataFrame)
#     )
#     items["genre_list"] = items[list_of_genres].apply(
#         lambda x: [genre for genre in list_of_genres if x[genre] == 1], axis=1
#     )

#     items["genre_list"] = items["genre_list"].apply(clean_genre_list)

#     df = pd.merge(data, items[["movie_id", "genre_list"]], on="movie_id")
#     df = pd.merge(
#         df,
#         users[["user_id", "age", "gender", "occupation"]],
#         on="user_id",
#     )

#     df["rating"] = df["rating"].apply(lambda x: 1 if x >= 4 else 0)

#     df = df.sort_values(by=["timestamp"]).reset_index(drop=True)
#     df = df.drop("timestamp", axis=1)

#     din_preprocessor = DINPreprocessor(
#         user_id_col="user_id",
#         target_col="rating",
#         item_embed_col=("movie_id", 16),
#         max_seq_length=5,
#         action_col="rating",
#         other_seq_embed_cols=[("genre_list", 16)],
#         cat_embed_cols=["user_id", "age", "gender", "occupation"],
#     )

#     X, y = din_preprocessor.fit_transform(df)

#     din = DeepInterestNetwork(
#         column_idx=din_preprocessor.din_columns_idx,
#         user_behavior_confiq=din_preprocessor.user_behaviour_config,
#         action_seq_config=din_preprocessor.action_seq_config,
#         other_seq_cols_confiq=din_preprocessor.other_seq_config,
#         cat_embed_input=din_preprocessor.tab_preprocessor.cat_embed_input,  # type: ignore[attr-defined]
#         mlp_hidden_dims=[128, 64],
#     )

#     # And from here on, everything is standard
#     model = WideDeep(deeptabular=din)

#     trainer = Trainer(model=model, objective="binary", metrics=[Accuracy()])

#     # in the real world you would have to split the data into train, val and test
#     trainer.fit(
#         X_tab=X,
#         target=y,
#         n_epochs=5,
#         batch_size=512,
#     )