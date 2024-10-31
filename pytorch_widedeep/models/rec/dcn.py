from typing import Dict, List, Tuple, Union, Literal, Optional

import torch
from torch import nn

from pytorch_widedeep.models.tabular.mlp._layers import MLP
from pytorch_widedeep.models.tabular._base_tabular_model import (
    BaseTabularModelWithoutAttention,
)


class CrossNetwork(nn.Module):
    def __init__(self, input_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.cross_layers = nn.ModuleList(
            [nn.Linear(input_dim, input_dim, bias=True) for _ in range(num_layers)]
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x0 = X
        xi = X
        for layer in self.cross_layers:
            xi = x0 * layer(xi) + xi
        return xi


class DeepCrossNetwork(BaseTabularModelWithoutAttention):
    r"""Defines a `DeepCrossNetwork` model that can be used as the `deeptabular`
    component of a Wide & Deep model or independently by itself.

    This class implements the [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)
    architecture, which automatically combines features to generate feature interactions
    in an explicit fashion and at each layer.

    The cross layer implements the following equation:

    $$x_{l+1} = x_0 \odot (W_l x_l + b_l) + x_l$$

    where:

    * $\odot$ represents element-wise multiplication
    * $x_l$, $x_{l+1}$ are the outputs from the $l^{th}$ and $(l+1)^{th}$ cross layers
    * $W_l$, $b_l$ are the weight and bias parameters

    Parameters
    ----------
    column_idx: Dict
        Dict containing the index of the columns that will be passed through
        the `DeepCrossNetwork` model. Required to slice the tensors. e.g.
        _{'education': 0, 'relationship': 1, 'workclass': 2, ...}_.
    n_cross_layers: int, default = 3
        Number of cross layers in the cross network
    cat_embed_input: List, Optional, default = None
        List of Tuples with the column name, number of unique values and
        embedding dimension. e.g. _[(education, 11, 32), ...]_
    cat_embed_dropout: float, Optional, default = None
        Categorical embeddings dropout. If `None`, it will default to 0.
    use_cat_bias: bool, Optional, default = None,
        Boolean indicating if bias will be used for the categorical embeddings
    cat_embed_activation: Optional, str, default = None,
        Activation function for the categorical embeddings
    continuous_cols: List, Optional, default = None
        List with the name of the numeric (aka continuous) columns
    cont_norm_layer: str, Optional, default = None
        Type of normalization layer applied to the continuous features.
        Options are: _'layernorm'_ and _'batchnorm'_
    embed_continuous: bool, Optional, default = None,
        Boolean indicating if the continuous columns will be embedded
    embed_continuous_method: Optional, str, default = None,
        Method to use to embed the continuous features. Options are:
        _'standard'_, _'periodic'_ or _'piecewise'_
    cont_embed_dim: int, Optional, default = None,
        Size of the continuous embeddings
    cont_embed_dropout: float, Optional, default = None,
        Dropout for the continuous embeddings
    cont_embed_activation: Optional, str, default = None,
        Activation function for the continuous embeddings
    quantization_setup: Dict[str, List[float]], Optional, default = None,
        Setup for the piecewise embeddings quantization
    n_frequencies: int, Optional, default = None,
        Number of frequencies for periodic embeddings
    sigma: float, Optional, default = None,
        Sigma parameter for periodic embeddings
    share_last_layer: bool, Optional, default = None,
        Whether to share the last layer in periodic embeddings
    full_embed_dropout: bool, Optional, default = None,
        If True, drops the entire embedding for a column
    mlp_hidden_dims: List, default = [200, 100]
        List with the number of neurons per dense layer in the deep network
    mlp_activation: str, default = "relu"
        Activation function for the dense layers
    mlp_dropout: float or List, default = 0.1
        Dropout between the dense layers
    mlp_batchnorm: bool, default = False
        If True, applies batch normalization in the dense layers
    mlp_batchnorm_last: bool, default = False
        If True, applies batch normalization in the last dense layer
    mlp_linear_first: bool, default = True
        If True, applies the order: [Linear -> Activation -> BatchNorm -> Dropout]
        If False: [BatchNorm -> Dropout -> Linear -> Activation]

    Attributes
    ----------
    cross_network: nn.Module
        The cross network component
    deep_network: nn.Module
        The deep network component (MLP)

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.models.rec import DeepCrossNetwork
    >>> X_tab = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
    >>> colnames = ["a", "b", "c", "d", "e"]
    >>> cat_embed_input = [(u, i, j) for u, i, j in zip(colnames[:4], [4] * 4, [8] * 4)]
    >>> column_idx = {k: v for v, k in enumerate(colnames)}
    >>> model = DeepCrossNetwork(
    ...     column_idx=column_idx,
    ...     cat_embed_input=cat_embed_input,
    ...     continuous_cols=["e"],
    ...     n_cross_layers=2,
    ...     mlp_hidden_dims=[16, 8]
    ... )
    >>> out = model(X_tab)
    """

    def __init__(
        self,
        *,
        column_idx: Dict[str, int],
        n_cross_layers: int = 3,
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
        mlp_hidden_dims: List[int] = [200, 100],
        mlp_activation: str = "relu",
        mlp_dropout: Union[float, List[float]] = 0.1,
        mlp_batchnorm: bool = False,
        mlp_batchnorm_last: bool = False,
        mlp_linear_first: bool = True,
    ):
        super(DeepCrossNetwork, self).__init__(
            column_idx=column_idx,
            cat_embed_input=cat_embed_input,
            cat_embed_dropout=cat_embed_dropout,
            use_cat_bias=use_cat_bias,
            cat_embed_activation=cat_embed_activation,
            continuous_cols=continuous_cols,
            cont_norm_layer=cont_norm_layer,
            embed_continuous=embed_continuous,
            embed_continuous_method=embed_continuous_method,
            cont_embed_dim=cont_embed_dim,
            cont_embed_dropout=cont_embed_dropout,
            cont_embed_activation=cont_embed_activation,
            quantization_setup=quantization_setup,
            n_frequencies=n_frequencies,
            sigma=sigma,
            share_last_layer=share_last_layer,
            full_embed_dropout=full_embed_dropout,
        )

        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.mlp_dropout = mlp_dropout
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.mlp_linear_first = mlp_linear_first

        embeddings_output_dim = self.cat_out_dim + self.cont_out_dim
        self.deep_network = MLP(
            [embeddings_output_dim] + self.mlp_hidden_dims,
            self.mlp_activation,
            self.mlp_dropout,
            self.mlp_batchnorm,
            self.mlp_batchnorm_last,
            self.mlp_linear_first,
        )
        self.cross_network = CrossNetwork(embeddings_output_dim, n_cross_layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x = self._get_embeddings(X)
        cross_output = self.cross_network(x)
        deep_output = self.deep_network(x)
        return torch.cat([cross_output, deep_output], dim=1)

    @property
    def output_dim(self) -> int:
        return self.mlp_hidden_dims[-1] + self.cat_out_dim + self.cont_out_dim
