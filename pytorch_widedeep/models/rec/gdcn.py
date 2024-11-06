from typing import Dict, List, Tuple, Union, Literal, Optional

import torch
from torch import nn

from pytorch_widedeep.models.tabular.mlp._layers import MLP
from pytorch_widedeep.models.tabular._base_tabular_model import (
    BaseTabularModelWithoutAttention,
)


class GatedCrossLayer(nn.Module):
    """
    Single Gated Cross Layer implementing:
    c_(i+1) = c_0 ⊙ (W^c × c_i + b) ⊙ σ(W^g × c_i) + c_i
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.cross_weight = nn.Parameter(torch.empty(input_dim, input_dim))
        self.gate_weight = nn.Parameter(torch.empty(input_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize layer parameters"""
        nn.init.xavier_uniform_(self.cross_weight)
        nn.init.xavier_uniform_(self.gate_weight)

    def forward(self, x0: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        cross_term = torch.einsum("bi,ij->bj", xi, self.cross_weight)
        cross_term = cross_term + self.bias
        cross_term = x0 * cross_term
        gate_term = torch.einsum("bi,ij->bj", xi, self.gate_weight)
        gate_term = torch.sigmoid(gate_term)
        return cross_term * gate_term + xi


class GatedCrossNetwork(nn.Module):
    def __init__(self, input_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.layers = nn.ModuleList(
            [GatedCrossLayer(input_dim) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        xi = x

        for layer in self.layers:
            xi = layer(x0, xi)

        return xi


class GatedDeepCrossNetwork(BaseTabularModelWithoutAttention):
    r"""Defines a `GatedDeepCrossNetwork` model that can be used as the `deeptabular`
    component of a Wide & Deep model or independently by itself.

    This class implements the Gated Deep & Cross Network (GDCN) architecture
    as described in the paper
    [Towards Deeper, Lighter and Interpretable Cross Network for CTR Prediction](https://arxiv.org/pdf/2311.04635).
    The GDCN enhances the original DCN by introducing a gating mechanism in
    the cross network. The gating mechanism controls feature interactions by
    learning which interactions are more important.

    The cross layer implements the following equation:

    $$c_{i+1} = c_0 \odot (W^c \times c_i + b) \odot \sigma(W^g \times c_i) + c_i$$

    where:

    * $\odot$ represents element-wise multiplication
    * $W^c$ and $W^g$ are the cross and gate weight matrices respectively
    * $\sigma$ is the sigmoid activation function

    Parameters
    ----------
    column_idx: Dict
        Dict containing the index of the columns that will be passed through
        the `GatedDeepCrossNetwork` model. Required to slice the tensors. e.g.
        _{'education': 0, 'relationship': 1, 'workclass': 2, ...}_.
    num_cross_layers: int, default = 3
        Number of cross layers in the cross network
    structure: str, default = "parallel"
        Structure of the model. Either _'parallel'_ or _'stacked'_. If _'parallel'_,
        the output will be the concatenation of the cross network and deep network
        outputs. If _'stacked'_, the cross network output will be fed into the deep
        network.
    cat_embed_input: List, Optional, default = None
        List of Tuples with the column name, number of unique values and
        embedding dimension. e.g. _[(education, 11, 32), ...]_
    cat_embed_dropout: float, Optional, default = None
        Categorical embeddings dropout. If `None`, it will default to 0.
    use_cat_bias: bool, Optional, default = None,
        Boolean indicating if bias will be used for the categorical embeddings.
        If `None`, it will default to 'False'.
    cat_embed_activation: Optional, str, default = None,
        Activation function for the categorical embeddings, if any. Currently
        _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported
    continuous_cols : Optional[List[str]], default=None
        List with the name of the numeric (aka continuous) columns
    cont_norm_layer : Optional[Literal["batchnorm", "layernorm"]], default=None
        Type of normalization layer applied to the continuous features.
        Options are: _'layernorm'_ and _'batchnorm'_. if `None`, no
        normalization layer will be used.
    embed_continuous_method: Optional[Literal["piecewise", "periodic", "standard"]], default="standard"
        Method to use to embed the continuous features. Options are:
        _'standard'_, _'periodic'_ or _'piecewise'_. The _'standard'_
        embedding method is based on the FT-Transformer implementation
        presented in the paper: [Revisiting Deep Learning Models for
        Tabular Data](https://arxiv.org/abs/2106.11959v5). The _'periodic'_
        and_'piecewise'_ methods were presented in the paper: [On Embeddings for
        Numerical Features in Tabular Deep Learning](https://arxiv.org/abs/2203.05556).
        Please, read the papers for details.
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
    cross_network: nn.Module
        The gated cross network component
    deep_network: nn.Module
        The deep network component (MLP)

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.models.rec import GatedDeepCrossNetwork
    >>> X_tab = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
    >>> colnames = ["a", "b", "c", "d", "e"]
    >>> cat_embed_input = [(u, i, j) for u, i, j in zip(colnames[:4], [4] * 4, [8] * 4)]
    >>> column_idx = {k: v for v, k in enumerate(colnames)}
    >>> model = GatedDeepCrossNetwork(
    ...     column_idx=column_idx,
    ...     cat_embed_input=cat_embed_input,
    ...     continuous_cols=["e"],
    ...     num_cross_layers=2,
    ...     mlp_hidden_dims=[16, 8]
    ... )
    >>> out = model(X_tab)
    """

    def __init__(
        self,
        *,
        column_idx: Dict[str, int],
        num_cross_layers: int = 3,
        structure: Literal["stacked", "parallel"] = "parallel",
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
        super(GatedDeepCrossNetwork, self).__init__(
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

        self.num_cross_layers = num_cross_layers
        self.structure = structure

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

        self.cross_network = GatedCrossNetwork(embeddings_output_dim, num_cross_layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x = self._get_embeddings(X)
        cross_output = self.cross_network(x)

        if self.structure == "stacked":
            deep_output = self.deep_network(cross_output)
            return deep_output
        else:  # parallel
            deep_output = self.deep_network(x)
            return torch.cat([cross_output, deep_output], dim=1)

    @property
    def output_dim(self) -> int:
        if self.structure == "stacked":
            return self.mlp_hidden_dims[-1]
        else:  # parallel
            return self.mlp_hidden_dims[-1] + self.cat_out_dim + self.cont_out_dim
