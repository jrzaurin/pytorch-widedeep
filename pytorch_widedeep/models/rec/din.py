from typing import Any, Dict, List, Tuple, Union, Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pytorch_widedeep.models.tabular.mlp._layers import MLP
from pytorch_widedeep.models._base_wd_model_component import BaseWDModelComponent
from pytorch_widedeep.models.tabular._base_tabular_model import (
    BaseTabularModelWithAttention,
    BaseTabularModelWithoutAttention,
)


class Dice(nn.Module):
    def __init__(self, input_dim: int):
        super(Dice, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim, eps=1e-9)
        self.alpha = nn.Parameter(torch.zeros(input_dim))

    def forward(self, X: Tensor) -> Tensor:
        # This implementation assumes X has n_dim = 3
        x_p = self.bn(X.transpose(1, 2)).transpose(1, 2)
        p = torch.sigmoid(x_p)
        return p * X + (1 - p) * self.alpha * X


class ActivationUnit(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        activation: Literal["prelu", "dice"],
        proj_dim: Optional[int] = None,
    ):
        super(ActivationUnit, self).__init__()
        self.proj_dim = proj_dim if proj_dim is not None else embed_dim
        self.linear_in = nn.Linear(embed_dim * 4, self.proj_dim)
        if activation == "prelu":
            self.activation: Union[nn.PReLU, Dice] = nn.PReLU()
        elif activation == "dice":
            self.activation = Dice(self.proj_dim)
        self.linear_out = nn.Linear(self.proj_dim, 1)

    def forward(self, item: Tensor, user_behavior: Tensor) -> Tensor:
        # in this implementation:
        # item: [batch_size, 1, embedding_dim]
        # user_behavior: [batch_size, seq_len, embedding_dim]
        item_expanded = item.expand(-1, user_behavior.size(1), -1)
        attn_input = torch.cat(
            [
                item_expanded,
                user_behavior,
                item_expanded - user_behavior,
                item_expanded * user_behavior,
            ],
            dim=-1,
        )
        attn_output = self.activation(self.linear_in(attn_input))
        attn_output = self.linear_out(attn_output).squeeze(-1)
        return F.softmax(attn_output, dim=1)


class DeepInterestNetwork(BaseWDModelComponent):
    """
    Adaptation of the Deep Interest Network (DIN) for recommendation systems
    as described in the paper: 'Deep Interest Network for Click-Through Rate
    Prediction' by Guorui Zhou et al. 2018.

    Note that all the categorical- and continuous-related parameters refer to
    the categorical and continuous columns that are not part of the
    sequential columns and will be treated as standard tabular data.

    This model requires some specific data preparation that allows for quite a
    lot of flexibility. Therefore, I have included a preprocessor
    (`DINPreprocessor`) in the preprocessing module that will take care of
    the data preparation.

    Parameters
    ----------
    column_idx : Dict[str, int]
        Dictionary mapping column names to their corresponding index.
    target_item_col : str
        Name of the target item column. Note that this is not the target
        column. This algorithm relies on a sequence representation of
        interactions. The target item would be the next item in the sequence
        of interactions (e.g. item 6th in a sequence of 5 items), and our
        goal is to predict a given action on it.
    user_behavior_confiq : Tuple[List[str], int, int]
        Configuration for user behavior sequence columns. Tuple containing:
        - List of column names that correspond to the user behavior sequence<br/>
        - Number of unique feature values (n_tokens)<br/>
        - Embedding dimension<br/>
        Example: `(["item_1", "item_2", "item_3"], 5, 8)`
    action_seq_config : Optional[Tuple[List[str], int]], default=None
        Configuration for a so-called action sequence columns (for example a
        rating, or purchased/not-purchased, etc). Tuple containing:<br/>
        - List of column names<br/>
        - Number of unique feature values (n_tokens)<br/>
        This action will **always** be learned as a 1d embedding and will be
        combined with the user behaviour. For example, imagine that the
        action is purchased/not-purchased. then per item in the user
        behaviour sequence there will be a binary action to learn 0/1. Such
        action will be represented by a float number that will multiply the
        corresponding item embedding in the user behaviour sequence.<br/>
        Example: `(["rating_1", "rating_2", "rating_3"], 5)`<br/>
        Internally, the embedding dimension will be set to 1
    other_seq_cols_confiq : Optional[List[Tuple[List[str], int, int]]], default=None
        Configuration for other sequential columns. List of tuples containing:<br/>
        - List of column names that correspond to the sequential column<br/>
        - Number of unique feature values (n_tokens)<br/>
        - Embedding dimension<br/>
        Example: `[(["seq1_col1", "seq1_col2"], 5, 8), (["seq2_col1", "seq2_col2"], 5, 8)]`
    attention_unit_activation : Literal["prelu", "dice"], default="prelu"
        Activation function to use in the attention unit.
    cat_embed_input : Optional[List[Tuple[str, int, int]]], default=None
        Configuration for other columns. List of tuples containing:<br/>
        - Column name<br/>
        - Number of unique feature values (n_tokens)<br/>
        - Embedding dimension<br/>

        **Note**: From here in advance the remaining parameters are related to
        the categorical and continuous columns that are not part of the
        sequential columns and will be treated as standard tabular data.

    cat_embed_dropout : Optional[float], default=None
        Categorical embeddings dropout. If `None`, it will default
        to 0.
    use_cat_bias : Optional[bool], default=None
        Boolean indicating if bias will be used for the categorical embeddings.
        If `None`, it will default to 'False'.
    cat_embed_activation : Optional[str], default=None
        Activation function for the categorical embeddings, if any. Currently
        _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported
    continuous_cols : Optional[List[str]], default=None
        List with the name of the numeric (aka continuous) columns
    cont_norm_layer : Optional[Literal["batchnorm", "layernorm"]], default=None
        Type of normalization layer applied to the continuous features.
        Options are: _'layernorm'_ and _'batchnorm'_. if `None`, no
        normalization layer will be used.
    embed_continuous : Optional[bool], default=None
        Boolean indicating if the continuous columns will be embedded using
        one of the available methods: _'standard'_, _'periodic'_
        or _'piecewise'_. If `None`, it will default to 'False'.<br/>
        :information_source: **NOTE**: This parameter is deprecated and it
         will be removed in future releases. Please, use the
         `embed_continuous_method` parameter instead.
    embed_continuous_method : Optional[Literal["piecewise", "periodic", "standard"]], default=None
        Method to use to embed the continuous features. Options are:
        _'standard'_, _'periodic'_ or _'piecewise'_. The _'standard'_
        embedding method is based on the FT-Transformer implementation
        presented in the paper: [Revisiting Deep Learning Models for
        Tabular Data](https://arxiv.org/abs/2106.11959v5). The _'periodic'_
        and_'piecewise'_ methods were presented in the paper: [On Embeddings for
        Numerical Features in Tabular Deep Learning](https://arxiv.org/abs/2203.05556).
        Please, read the papers for details.
    cont_embed_dim : Optional[int], default=None
        Size of the continuous embeddings. If the continuous columns are
        embedded, `cont_embed_dim` must be passed.
    cont_embed_dropout : Optional[float], default=None
        Dropout for the continuous embeddings. If `None`, it will default to 0.0
    cont_embed_activation : Optional[str], default=None
        Activation function for the continuous embeddings if any. Currently
        _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported.
        If `None`, no activation function will be applied.
    quantization_setup : Optional[Dict[str, List[float]]], default=None
        This parameter is used when the _'piecewise'_ method is used to embed
        the continuous cols. It is a dict where keys are the name of the continuous
        columns and values are lists with the boundaries for the quantization
        of the continuous_cols. See the examples for details. If
        If the _'piecewise'_ method is used, this parameter is required.
    n_frequencies : Optional[int], default=None
        This is the so called _'k'_ in their paper [On Embeddings for
        Numerical Features in Tabular Deep Learning](https://arxiv.org/abs/2203.05556),
        and is the number of 'frequencies' that will be used to represent each
        continuous column. See their Eq 2 in the paper for details. If
        the _'periodic'_ method is used, this parameter is required.
    sigma : Optional[float], default=None
        This is the sigma parameter in the paper mentioned when describing the
        previous parameters and it is used to initialise the 'frequency
        weights'. See their Eq 2 in the paper for details. If
        the _'periodic'_ method is used, this parameter is required.
    share_last_layer : Optional[bool], default=None
        This parameter is not present in the before mentioned paper but it is implemented in
        the [official repo](https://github.com/yandex-research/rtdl-num-embeddings/tree/main).
        If `True` the linear layer that turns the frequencies into embeddings
        will be shared across the continuous columns. If `False` a different
        linear layer will be used for each continuous column.
        If the _'periodic'_ method is used, this parameter is required.
    full_embed_dropout: bool, Optional, default = None,
        If `True`, the full embedding corresponding to a column will be masked
        out/dropout. If `None`, it will default to `False`.
    mlp_hidden_dims: List, default = [200, 100]
        List with the number of neurons per dense layer in the mlp.
    mlp_activation: str, default = "relu"
        Activation function for the dense layers of the MLP. Currently
        _'tanh'_, _'relu'_, _'leaky_relu'_, _'gelu'_ and _'preglu'_ are
          supported
    mlp_dropout: float or List, default = 0.1
        float or List of floats with the dropout between the dense layers.
        e.g: _[0.5,0.5]_
    mlp_batchnorm: bool, default = False
        Boolean indicating whether or not batch normalization will be applied
        to the dense layers
    mlp_batchnorm_last: bool, default = False
        Boolean indicating whether or not batch normalization will be applied
        to the last of the dense layers
    mlp_linear_first: bool, default = False
        Boolean indicating the order of the operations in the dense
        layer. If `True: [LIN -> ACT -> BN -> DP]`. If `False: [BN -> DP ->
        LIN -> ACT]`

    Attributes
    ----------
    user_behavior_indexes: List[int]
        List with the indexes of the user behavior columns
    user_behavior_embed: BaseTabularModelWithAttention
        Embedding layer for the user
    action_seq_indexes: List[int]
        List with the indexes of the rating sequence columns if the
        action_seq_config parameter is not None
    action_embed: BaseTabularModelWithAttention
        Embedding layer for the rating sequence columns if the
        action_seq_config parameter is not None
    other_seq_cols_indexes: Dict[str, List[int]]
        Dictionary with the indexes of the other sequential columns if the
        other_seq_cols_confiq parameter is not None
    other_seq_cols_embed: nn.ModuleDict
        Dictionary with the embedding layers for the other sequential columns
        if the other_seq_cols_confiq parameter is not None
    other_cols_idx: List[int]
        List with the indexes of the other columns if the other_cols_config
        parameter is not None
    other_col_embed: BaseTabularModel
        Embedding layer for the other columns if the other_cols_config
        parameter is not None
    mlp: Optional[MLP]
        MLP component of the model. If None, no MLP will be used. This should
        almost always be not None.

    Examples
    --------
    >>> import torch
    >>> import numpy as np
    >>> from torch import Tensor
    >>> from typing import Dict, List, Tuple
    >>> from pytorch_widedeep.models.rec import DeepInterestNetwork
    >>> np_seed = np.random.seed(42)
    >>> torch_seed = torch.manual_seed(42)
    >>> num_users = 10
    >>> num_items = 5
    >>> num_contexts = 3
    >>> seq_length = 3
    >>> num_samples = 10
    >>> user_ids = np.random.randint(0, num_users, num_samples)
    >>> target_item_ids = np.random.randint(0, num_items, num_samples)
    >>> context_ids = np.random.randint(0, num_contexts, num_samples)
    >>> user_behavior = np.array(
    ...     [
    ...         np.random.choice(num_items, seq_length, replace=False)
    ...         for _ in range(num_samples)
    ...     ]
    ... )
    >>> X_arr = np.column_stack((user_ids, target_item_ids, context_ids, user_behavior))
    >>> X = torch.tensor(X_arr, dtype=torch.long)
    >>> column_idx: Dict[str, int] = {
    ...     "user_id": 0,
    ...     "target_item": 1,
    ...     "context": 2,
    ...     "item_1": 3,
    ...     "item_2": 4,
    ...     "item_3": 5,
    ... }
    >>> user_behavior_config: Tuple[List[str], int, int] = (
    ...     ["item_1", "item_2", "item_3"],
    ...     num_items,
    ...     8,
    ... )
    >>> cat_embed_input: List[Tuple[str, int, int]] = [
    ...     ("user_id", num_users, 8),
    ...     ("context", num_contexts, 4),
    ... ]
    >>> model = DeepInterestNetwork(
    ...     column_idx=column_idx,
    ...     target_item_col="target_item",
    ...     user_behavior_confiq=user_behavior_config,
    ...     cat_embed_input=cat_embed_input,
    ...     mlp_hidden_dims=[16, 8],
    ... )
    >>> output = model(X)
    """

    def __init__(
        self,
        *,
        column_idx: Dict[str, int],
        user_behavior_confiq: Tuple[List[str], int, int],
        target_item_col: str = "target_item",
        action_seq_config: Optional[Tuple[List[str], int]] = None,
        other_seq_cols_confiq: Optional[List[Tuple[List[str], int, int]]] = None,
        attention_unit_activation: Literal["prelu", "dice"] = "prelu",
        cat_embed_input: Optional[List[Tuple[str, int, int]]] = None,
        cat_embed_dropout: Optional[float] = None,
        use_cat_bias: Optional[bool] = None,
        cat_embed_activation: Optional[str] = None,
        continuous_cols: Optional[List[str]] = None,
        cont_norm_layer: Optional[Literal["batchnorm", "layernorm"]] = None,
        embed_continuous: Optional[bool] = None,
        embed_continuous_method: Optional[
            Literal["standard", "piecewise", "periodic"]
        ] = None,
        cont_embed_dim: Optional[int] = None,
        cont_embed_dropout: Optional[float] = None,
        cont_embed_activation: Optional[str] = None,
        quantization_setup: Optional[Dict[str, List[float]]] = None,
        n_frequencies: Optional[int] = None,
        sigma: Optional[float] = None,
        share_last_layer: Optional[bool] = None,
        full_embed_dropout: Optional[bool] = None,
        mlp_hidden_dims: Optional[List[int]] = None,
        mlp_activation: Optional[str] = None,
        mlp_dropout: Optional[float] = None,
        mlp_batchnorm: Optional[bool] = None,
        mlp_batchnorm_last: Optional[bool] = None,
        mlp_linear_first: Optional[bool] = None,
    ):
        super(DeepInterestNetwork, self).__init__()

        self.column_idx = {
            k: v for k, v in sorted(column_idx.items(), key=lambda x: x[1])
        }

        self.column_idx = column_idx
        self.target_item_col = target_item_col
        self.user_behavior_confiq = user_behavior_confiq
        self.action_seq_config = (
            (action_seq_config[0], action_seq_config[1], 1)
            if action_seq_config is not None
            else None
        )
        self.other_seq_cols_confiq = other_seq_cols_confiq
        self.cat_embed_input = cat_embed_input
        self.attention_unit_activation = attention_unit_activation

        self.cat_embed_dropout = cat_embed_dropout
        self.use_cat_bias = use_cat_bias
        self.cat_embed_activation = cat_embed_activation
        self.continuous_cols = continuous_cols
        self.cont_norm_layer = cont_norm_layer
        self.embed_continuous = embed_continuous
        self.embed_continuous_method = embed_continuous_method
        self.cont_embed_dim = cont_embed_dim
        self.cont_embed_dropout = cont_embed_dropout
        self.cont_embed_activation = cont_embed_activation
        self.quantization_setup = quantization_setup
        self.n_frequencies = n_frequencies
        self.sigma = sigma
        self.share_last_layer = share_last_layer
        self.full_embed_dropout = full_embed_dropout

        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.mlp_dropout = mlp_dropout
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.mlp_linear_first = mlp_linear_first

        self.target_item_idx = self.column_idx[target_item_col]

        self.user_behavior_indexes, self.user_behavior_embed = (
            self.set_user_behavior_indexes_and_embed(user_behavior_confiq)
        )
        self.user_behavior_dim = user_behavior_confiq[2]

        if self.action_seq_config is not None:
            self.action_seq_indexes, self.action_embed = (
                self._set_rating_indexes_and_embed(self.action_seq_config)
            )
        else:
            self.action_embed = None

        if self.other_seq_cols_confiq is not None:
            (
                self.other_seq_cols_indexes,
                self.other_seq_cols_embed,
                self.other_seq_dim,
            ) = self._set_other_seq_cols_indexes_embed_and_dim(
                self.other_seq_cols_confiq
            )
        else:
            self.other_seq_cols_embed = None
            self.other_seq_dim = 0

        if self.cat_embed_input is not None or self.continuous_cols is not None:
            self.other_cols_idx, self.other_col_embed, self.other_cols_dim = (
                self._set_other_cols_idx_embed_and_dim(
                    self.cat_embed_input, self.continuous_cols
                )
            )
        else:
            self.other_col_embed = None
            self.other_cols_dim = 0

        self.attention = ActivationUnit(
            user_behavior_confiq[2], attention_unit_activation
        )

        if self.mlp_hidden_dims is not None:
            mlp_input_dim = (
                self.user_behavior_dim * 2 + self.other_seq_dim + self.other_cols_dim
            )
            self.mlp = MLP(
                d_hidden=[mlp_input_dim] + self.mlp_hidden_dims,
                activation=(
                    "relu" if self.mlp_activation is None else self.mlp_activation
                ),
                dropout=0.0 if self.mlp_dropout is None else self.mlp_dropout,
                batchnorm=False if self.mlp_batchnorm is None else self.mlp_batchnorm,
                batchnorm_last=(
                    False
                    if self.mlp_batchnorm_last is None
                    else self.mlp_batchnorm_last
                ),
                linear_first=(
                    False if self.mlp_linear_first is None else self.mlp_linear_first
                ),
            )
        else:
            self.mlp = None

    def forward(self, X: Tensor) -> Tensor:

        X_target_item = X[:, [self.target_item_idx]]
        item_embed = self.user_behavior_embed.cat_embed.embed(X_target_item.long())

        X_user_behavior = X[:, self.user_behavior_indexes]
        user_behavior_embed = self.user_behavior_embed._get_embeddings(X_user_behavior)
        # 0 is the padding index
        mask = (X_user_behavior != 0).float().to(X.device)

        if self.action_embed is not None:
            X_rating = X[:, self.action_seq_indexes]
            action_embed = self.action_embed._get_embeddings(X_rating)
            user_behavior_embed = user_behavior_embed * action_embed

        attention_scores = self.attention(item_embed, user_behavior_embed)
        attention_scores = attention_scores * mask
        user_interest = (attention_scores.unsqueeze(-1) * user_behavior_embed).sum(1)

        deep_out = torch.cat([item_embed.squeeze(1), user_interest], dim=1)

        if self.other_seq_cols_embed is not None:
            X_other_seq: Dict[str, Tensor] = {
                col: X[:, idx] for col, idx in self.other_seq_cols_indexes.items()
            }
            other_seq_embed = torch.cat(
                [
                    self.other_seq_cols_embed[col]._get_embeddings(X_other_seq[col])
                    for col in self.other_seq_cols_indexes.keys()
                ],
                dim=-1,
            ).sum(1)
            deep_out = torch.cat([deep_out, other_seq_embed], dim=1)

        if self.other_col_embed is not None:
            X_other_cols = X[:, self.other_cols_idx]
            other_cols_embed = self.other_col_embed._get_embeddings(X_other_cols)
            deep_out = torch.cat([deep_out, other_cols_embed], dim=1)

        if self.mlp is not None:
            deep_out = self.mlp(deep_out)

        return deep_out

    @staticmethod
    def _get_seq_cols_embed_confiq(tup: Tuple[List[str], int, int]) -> Dict[str, Any]:
        # tup[0] is the list of columns
        # tup[1] is the number of unique feat value or "n_tokens"
        # tup[2] is the embedding dimension

        # Once sliced, the indexes will go from 0 to len(tup[0]) it is assumed
        # that the columns in tup[0] are ordered of appearance in the input
        # data
        column_idx = {col: i for i, col in enumerate(tup[0])}

        # This is a hack so that I can use any BaseTabularModelWithAttention.
        # For this model to work 'cat_embed_input' is normally a List of
        # Tuples where the first element is the column name and the second is
        # the number of unique values for that column. That second elements
        # is added internally to compute what one could call "n_tokens". Here
        # I'm passing that value as the second element of the first tuple
        # and then adding 0s for the rest of the columns
        cat_embed_input = [(tup[0][0], tup[1])] + [(col, 0) for col in tup[0][1:]]

        input_dim = tup[2]

        col_config = {
            "column_idx": column_idx,
            "input_dim": input_dim,
            "cat_embed_input": cat_embed_input,
            "cat_embed_dropout": None,
            "use_cat_bias": None,
            "cat_embed_activation": None,
            "shared_embed": None,
            "add_shared_embed": None,
            "frac_shared_embed": None,
            "continuous_cols": None,
            "cont_norm_layer": None,
            "embed_continuous_method": None,
            "cont_embed_dropout": None,
            "cont_embed_activation": None,
            "quantization_setup": None,
            "n_frequencies": None,
            "sigma": None,
            "share_last_layer": None,
            "full_embed_dropout": None,
        }

        return col_config

    def _get_other_cols_embed_config(
        self,
        cat_embed_input: Optional[List[Tuple[str, int, int]]],
        continuous_cols: Optional[List[str]],
        column_idx: Dict[str, int],
    ) -> Dict[str, Any]:

        cols_config = {
            "column_idx": {col: i for i, col in enumerate(column_idx.keys())},
            "cat_embed_input": cat_embed_input,
            "cat_embed_dropout": self.cat_embed_dropout,
            "use_cat_bias": self.use_cat_bias,
            "cat_embed_activation": self.cat_embed_activation,
            "continuous_cols": continuous_cols,
            "cont_norm_layer": self.cont_norm_layer,
            "embed_continuous": self.embed_continuous,
            "embed_continuous_method": self.embed_continuous_method,
            "cont_embed_dim": self.cont_embed_dim,
            "cont_embed_dropout": self.cont_embed_dropout,
            "cont_embed_activation": self.cont_embed_activation,
            "quantization_setup": self.quantization_setup,
            "n_frequencies": self.n_frequencies,
            "sigma": self.sigma,
            "share_last_layer": self.share_last_layer,
            "full_embed_dropout": self.full_embed_dropout,
        }

        return cols_config

    def set_user_behavior_indexes_and_embed(
        self, user_behavior_confiq: Tuple[List[str], int, int]
    ) -> Tuple[List[int], BaseTabularModelWithAttention]:
        user_behavior_indexes = [
            self.column_idx[col] for col in user_behavior_confiq[0]
        ]
        user_behavior_embed = BaseTabularModelWithAttention(
            **self._get_seq_cols_embed_confiq(user_behavior_confiq)
        )
        return user_behavior_indexes, user_behavior_embed

    def _set_rating_indexes_and_embed(
        self, action_seq_config: Tuple[List[str], int, int]
    ) -> Tuple[List[int], BaseTabularModelWithAttention]:
        action_seq_indexes = [self.column_idx[col] for col in action_seq_config[0]]
        action_embed = BaseTabularModelWithAttention(
            **self._get_seq_cols_embed_confiq(action_seq_config)
        )
        return action_seq_indexes, action_embed

    def _set_other_seq_cols_indexes_embed_and_dim(
        self, other_seq_cols_confiq: List[Tuple[List[str], int, int]]
    ) -> Tuple[Dict[str, List[int]], nn.ModuleDict, int]:
        other_seq_cols_indexes: Dict[str, List[int]] = {}
        for i, el in enumerate(other_seq_cols_confiq):
            key = f"seq_{i}"
            idxs = [self.column_idx[col] for col in el[0]]
            other_seq_cols_indexes[key] = idxs
        other_seq_cols_config = {
            f"seq_{i}": self._get_seq_cols_embed_confiq(el)
            for i, el in enumerate(other_seq_cols_confiq)
        }
        other_seq_cols_embed = nn.ModuleDict(
            {
                key: BaseTabularModelWithAttention(**config)
                for key, config in other_seq_cols_config.items()
            }
        )
        other_seq_dim = sum([el[2] for el in other_seq_cols_confiq])

        return other_seq_cols_indexes, other_seq_cols_embed, other_seq_dim

    def _set_other_cols_idx_embed_and_dim(
        self,
        cat_embed_input: Optional[List[Tuple[str, int, int]]],
        continuous_cols: Optional[List[str]],
    ) -> Tuple[List[int], BaseTabularModelWithoutAttention, int]:
        other_cols_idx: Dict[str, int] = {}
        if cat_embed_input is not None:
            other_cols_idx = {
                col: self.column_idx[col] for col in [el[0] for el in cat_embed_input]
            }
        if continuous_cols is not None:
            other_cols_idx.update(
                {col: self.column_idx[col] for col in continuous_cols}
            )

        sorted_other_cols_idx = {
            k: v for k, v in sorted(other_cols_idx.items(), key=lambda x: x[1])
        }

        other_col_embed = BaseTabularModelWithoutAttention(
            **self._get_other_cols_embed_config(
                cat_embed_input, continuous_cols, sorted_other_cols_idx
            )
        )

        other_cols_dim = other_col_embed.output_dim

        return list(other_cols_idx.values()), other_col_embed, other_cols_dim

    @property
    def output_dim(self) -> int:
        if self.mlp_hidden_dims is not None:
            return self.mlp_hidden_dims[-1]
        else:
            return self.user_behavior_dim * 2 + self.other_seq_dim + self.other_cols_dim
