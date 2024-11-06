from typing import Dict, List, Tuple, Literal, Optional

import torch
from torch import Tensor, nn

from pytorch_widedeep.models.tabular.mlp._layers import MLP
from pytorch_widedeep.models.tabular._base_tabular_model import (
    BaseTabularModelWithAttention,
)


class CompressedInteractionNetwork(nn.Module):
    def __init__(self, num_cols: int, cin_layer_dims: List[int]):
        super(CompressedInteractionNetwork, self).__init__()
        self.num_cols = num_cols
        self.cin_layer_dims = cin_layer_dims
        self.cin_layers = nn.ModuleList()

        prev_layer_dim = num_cols
        for layer_dim in cin_layer_dims:
            self.cin_layers.append(
                nn.Conv1d(prev_layer_dim * num_cols, layer_dim, kernel_size=1)
            )
            prev_layer_dim = layer_dim

    def forward(self, X: Tensor) -> Tensor:
        batch_size, embed_dim = X.shape[0], X.shape[-1]
        prev_x = X
        cin_outs = []
        for cin_layer in self.cin_layers:
            x_i = torch.einsum("b m d, b h d  -> b m h d", X, prev_x)
            x_i = x_i.reshape(batch_size, self.num_cols * prev_x.shape[1], embed_dim)
            x_i = cin_layer(x_i)
            cin_outs.append(x_i.sum(2))
            prev_x = x_i

        return torch.cat(cin_outs, dim=1)


class ExtremeDeepFactorizationMachine(BaseTabularModelWithAttention):
    """
    Adaptation of 'xDeepFM implementation: xDeepFM: Combining Explicit and
    Implicit Feature Interactions for Recommender Systems' by Jianxun Lian,
    Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie, Guangzhong Sun and
    Enhong Chen, 2018

    The implementation in this library takes advantage of all the
    functionalities available to encode categorical and continuous features.
    The model can be used with only the factorization machine

    Note that this class implements only the 'Deep' component of the model
    described in the paper. The linear component is not
    implemented 'internally' and, if one wants to include it, it can be
    easily added using the 'wide'/linear component in this library. See the
    examples in the examples folder.

    Parameters
    ----------
    column_idx : Dict[str, int]
        Dictionary mapping column names to their corresponding index.
    input_dim : int
        Embedding input dimensions
    reduce_sum : bool, default=True
        Whether to reduce the sum in the factorization machine output.
    cin_layer_dims : List[int]
        List with the number of units per CIN layer. e.g: _[128, 64]_
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
    n_features: int
        Number of unique features/columns
    cin: CompressedInteractionNetwork
        Instance of the `CompressedInteractionNetwork` class
    mlp: MLP
        Instance of the `MLP` class if `mlp_hidden_dims` is not None. If None,
        the model will return directly the output of the `CIN`

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.models.rec import ExtremeDeepFactorizationMachine
    >>> X_tab = torch.randint(0, 10, (16, 2))
    >>> column_idx = {"col1": 0, "col2": 1}
    >>> cat_embed_input = [("col1", 10), ("col2", 10)]
    >>> xdeepfm = ExtremeDeepFactorizationMachine(
    ...     column_idx=column_idx,
    ...     input_dim=4,
    ...     cin_layer_dims=[8, 16],
    ...     cat_embed_input=cat_embed_input,
    ...     mlp_hidden_dims=[16, 8]
    ... )
    >>> output = xdeepfm(X_tab)
    """

    def __init__(
        self,
        *,
        column_idx: Dict[str, int],
        input_dim: int,
        reduce_sum: bool = True,
        cin_layer_dims: List[int],
        cat_embed_input: List[Tuple[str, int]],
        cat_embed_dropout: Optional[float] = None,
        use_cat_bias: Optional[bool] = None,
        cat_embed_activation: Optional[str] = None,
        continuous_cols: Optional[List[str]] = None,
        cont_norm_layer: Optional[Literal["batchnorm", "layernorm"]] = None,
        embed_continuous_method: Optional[
            Literal["piecewise", "periodic", "standard"]
        ] = "standard",
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
        super(ExtremeDeepFactorizationMachine, self).__init__(
            column_idx=column_idx,
            input_dim=input_dim,
            cat_embed_input=cat_embed_input,
            cat_embed_dropout=cat_embed_dropout,
            use_cat_bias=use_cat_bias,
            cat_embed_activation=cat_embed_activation,
            shared_embed=False,
            add_shared_embed=None,
            frac_shared_embed=None,
            continuous_cols=continuous_cols,
            cont_norm_layer=cont_norm_layer,
            embed_continuous_method=embed_continuous_method,
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

        self.reduce_sum = reduce_sum
        self.cin_layer_dims = cin_layer_dims

        self.n_features = len(self.column_idx)

        self.cin = CompressedInteractionNetwork(
            num_cols=self.n_features, cin_layer_dims=self.cin_layer_dims
        )

        if self.mlp_hidden_dims is not None:
            if (
                self.mlp_hidden_dims[-1] != sum(self.cin_layer_dims)
                and not self.reduce_sum
            ):
                d_hidden = (
                    [sum(self.cin_layer_dims)]
                    + self.mlp_hidden_dims
                    + [sum(self.cin_layer_dims)]
                )
            else:
                d_hidden = [sum(self.cin_layer_dims)] + self.mlp_hidden_dims

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

        embeddings = self._get_embeddings(X)
        cin_out = self.cin(embeddings)

        if self.mlp is None:
            if self.reduce_sum:
                return cin_out.sum(dim=1, keepdim=True)
            return cin_out

        mlp_out = self.mlp(cin_out)

        if self.reduce_sum:
            cin_out = cin_out.sum(dim=1, keepdim=True)
            mlp_out = mlp_out.sum(dim=1, keepdim=True)

        return mlp_out + cin_out

    @property
    def output_dim(self):
        if self.reduce_sum:
            return 1
        else:
            return sum(self.cin_layer_dims)
