from typing import Any, Dict, List, Tuple, Optional

import torch
from torch import Tensor, nn

from pytorch_widedeep.models.tabular.mlp._layers import MLP
from pytorch_widedeep.models.tabular._base_tabular_model import (
    BaseTabularModelWithAttention,
)


class DeepFieldAwareFactorizationMachine(BaseTabularModelWithAttention):
    """
    Deep Field Aware Factorization Machine (DeepFFM) for recommendation
    systems. Adaptation of the paper 'Field-aware Factorization Machines in a
    Real-world Online Advertising System', Juan et al. 2017.

    This class implements only the 'Deep' component of the model described in
    the paper. The linear component is not implemented 'internally' and, if
    one wants to include it, it can be easily added using the 'wide'/linear
    component in this library. See the examples in the examples folder.

    Note that in this case, only categorical features are accepted. This is
    because the embeddings of each feature will be learned using all other
    features. Therefore these embeddings have to be all of the same nature.
    This does not occur if we mix categorical and continuous features.

    Parameters
    ----------
    column_idx : Dict[str, int]
        Dictionary mapping column names to their corresponding index.
    num_factors : int
        Number of factors for the factorization machine.
    reduce_sum : bool, default=True
        Whether to reduce the sum in the factorization machine output.
    cat_embed_input : Optional[List[Tuple[str, int]]], default=None
        List of tuples with categorical column names and number of unique values.
    cat_embed_dropout : Optional[float], default=None
        Categorical embeddings dropout. If `None`, it will default
        to 0.
    use_cat_bias : Optional[bool], default=None
        Boolean indicating if bias will be used for the categorical embeddings.
        If `None`, it will default to 'False'.
    cat_embed_activation : Optional[str], default=None
        Activation function for the categorical embeddings, if any. Currently
        _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported
    mlp_hidden_dims: List, default = [200, 100]
        List with the number of neurons per dense layer in the mlp.
    mlp_activation: str, default = "relu"
        Activation function for the dense layers of the MLP. Currently
        _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported
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
    n_features: int
        Number of unique features/columns
    n_tokens: int
        Number of unique values (tokens) in the full dataset (corpus)
    encoders: nn.ModuleList
        List of `BaseTabularModelWithAttention` instances. One per categorical
        column
    mlp: nn.Module
        Multi-layer perceptron. If `None` the output will be the output of the
        factorization machine (i.e. the sum of the interactions)

    Examples
    --------
    >>> import torch
    >>> from torch import Tensor
    >>> from typing import Dict, List, Tuple
    >>> from pytorch_widedeep.models.rec import DeepFieldAwareFactorizationMachine
    >>> X = torch.randint(0, 10, (16, 2))
    >>> column_idx: Dict[str, int] = {"col1": 0, "col2": 1}
    >>> cat_embed_input: List[Tuple[str, int]] = [("col1", 10), ("col2", 10)]
    >>> ffm = DeepFieldAwareFactorizationMachine(
    ...     column_idx=column_idx,
    ...     num_factors=4,
    ...     cat_embed_input=cat_embed_input,
    ...     mlp_hidden_dims=[16, 8]
    ... )
    >>> output = ffm(X)
    """

    def __init__(
        self,
        *,
        column_idx: Dict[str, int],
        num_factors: int,
        cat_embed_input: List[Tuple[str, int]],
        reduce_sum: bool = True,
        cat_embed_dropout: Optional[float] = None,
        use_cat_bias: Optional[bool] = None,
        cat_embed_activation: Optional[str] = None,
        mlp_hidden_dims: Optional[List[int]] = None,
        mlp_activation: Optional[str] = None,
        mlp_dropout: Optional[float] = None,
        mlp_batchnorm: Optional[bool] = None,
        mlp_batchnorm_last: Optional[bool] = None,
        mlp_linear_first: Optional[bool] = None,
    ):
        super(DeepFieldAwareFactorizationMachine, self).__init__(
            column_idx=column_idx,
            input_dim=num_factors,
            cat_embed_input=cat_embed_input,
            cat_embed_dropout=cat_embed_dropout,
            use_cat_bias=use_cat_bias,
            cat_embed_activation=cat_embed_activation,
            shared_embed=False,
            add_shared_embed=None,
            frac_shared_embed=None,
            continuous_cols=None,
            cont_norm_layer=None,
            embed_continuous_method=None,
            cont_embed_dropout=None,
            cont_embed_activation=None,
            quantization_setup=None,
            n_frequencies=None,
            sigma=None,
            share_last_layer=None,
            full_embed_dropout=None,
        )

        self.reduce_sum = reduce_sum

        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.mlp_dropout = mlp_dropout
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.mlp_linear_first = mlp_linear_first

        self.n_features = len(self.column_idx)
        self.n_tokens = sum([ei[1] for ei in cat_embed_input])

        self.encoders = nn.ModuleList(
            [
                BaseTabularModelWithAttention(**config)
                for config in self._get_encoder_configs()
            ]
        )

        if self.mlp_hidden_dims is not None:
            d_hidden = [
                self.n_features * (self.n_features - 1) // 2 * num_factors
            ] + self.mlp_hidden_dims
            self.mlp = MLP(
                d_hidden=d_hidden,
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

        interactions_l: List[Tensor] = []
        for i in range(len(self.column_idx)):
            for j in range(i + 1, len(self.column_idx)):
                # the syntax [i] and [j] is to keep the shape of the tensors
                # as they are sliced within '_get_embeddings'. This will
                # return a tensor of shape (b, 1, embed_dim). Then it has to
                # be squeezed to (b, embed_dim)  before multiplied
                embed_i = self.encoders[i]._get_embeddings(X[:, [i]]).squeeze(1)
                embed_j = self.encoders[j]._get_embeddings(X[:, [j]]).squeeze(1)
                interactions_l.append(embed_i * embed_j)

        interactions = torch.cat(interactions_l, dim=1)

        if self.mlp is not None:
            interactions = interactions.view(X.size(0), -1)
            deep_out = self.mlp(interactions)
        else:
            deep_out = interactions

        if self.reduce_sum:
            deep_out = deep_out.sum(dim=1, keepdim=True)

        return deep_out

    def _get_encoder_configs(self) -> List[Dict[str, Any]]:
        config: List[Dict[str, Any]] = []
        for col, _ in self.column_idx.items():
            cat_embed_input = [(col, self.n_tokens)]
            _config = {
                "column_idx": {col: 0},
                "input_dim": self.input_dim,
                "cat_embed_input": cat_embed_input,
                "cat_embed_dropout": self.cat_embed_dropout,
                "use_cat_bias": self.use_cat_bias,
                "cat_embed_activation": self.cat_embed_activation,
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

            config.append(_config)

        return config

    @property
    def output_dim(self) -> int:
        if self.reduce_sum:
            return 1
        elif self.mlp_hidden_dims is not None:
            return self.mlp_hidden_dims[-1]
        else:
            return self.n_features * (self.n_features - 1) // 2 * self.input_dim
