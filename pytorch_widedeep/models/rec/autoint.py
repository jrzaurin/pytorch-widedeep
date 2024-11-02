from typing import Dict, List, Tuple, Literal, Optional

import torch
from torch import nn

from pytorch_widedeep.models.tabular._base_tabular_model import (
    BaseTabularModelWithAttention,
)


class AutoInt(BaseTabularModelWithAttention):
    r"""Defines an `AutoInt` model that can be used as the `deeptabular` component
    of a Wide & Deep model or independently by itself.

    This class implements the [AutoInt: Automatic Feature Interaction Learning via Self-Attentive
    Neural Networks](https://arxiv.org/abs/1810.11921) architecture, which learns feature
    interactions through multi-head self-attention networks.

    Parameters
    ----------
    column_idx: Dict
        Dict containing the index of the columns that will be passed through
        the `AutoInt` model. Required to slice the tensors. e.g.
        _{'education': 0, 'relationship': 1, 'workclass': 2, ...}_.
    input_dim: int
        Dimension of the input embeddings
    num_heads: int, default = 4
        Number of attention heads
    num_layers: int, default = 2
        Number of interacting layers (attention + residual)
    reduction: str, default = "mean"
        How to reduce the output of the attention layers. Options are:
        _'mean'_: mean of attention outputs
        _'cat'_: concatenation of attention outputs
    cat_embed_input: List, Optional, default = None
        List of Tuples with the column name, number of unique values and
        embedding dimension. e.g. _[(education, 11, 32), ...]_
    cat_embed_dropout: float, Optional, default = None
        Categorical embeddings dropout. If `None`, it will default to 0.
    use_cat_bias: bool, Optional, default = None,
        Boolean indicating if bias will be used for the categorical embeddings
    cat_embed_activation: Optional, str, default = None,
        Activation function for the categorical embeddings
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

    Attributes
    ----------
    attention_layers: nn.ModuleList
        List of multi-head attention layers

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.models.rec import AutoInt
    >>> X_tab = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
    >>> colnames = ["a", "b", "c", "d", "e"]
    >>> cat_embed_input = [(u, i, j) for u, i, j in zip(colnames[:4], [4] * 4, [8] * 4)]
    >>> column_idx = {k: v for v, k in enumerate(colnames)}
    >>> model = AutoInt(
    ...     column_idx=column_idx,
    ...     input_dim=32,
    ...     cat_embed_input=cat_embed_input,
    ...     continuous_cols=["e"],
    ...     embed_continuous_method="standard",
    ...     num_heads=4,
    ...     num_layers=2
    ... )
    >>> out = model(X_tab)
    """

    def __init__(
        self,
        *,
        column_idx: Dict[str, int],
        input_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        reduction: Literal["mean", "cat"] = "mean",
        cat_embed_input: Optional[List[Tuple[str, int]]] = None,
        cat_embed_dropout: Optional[float] = None,
        use_cat_bias: Optional[bool] = None,
        cat_embed_activation: Optional[str] = None,
        continuous_cols: Optional[List[str]] = None,
        cont_norm_layer: Optional[Literal["batchnorm", "layernorm"]] = None,
        embed_continuous_method: Optional[
            Literal["standard", "piecewise", "periodic"]
        ] = None,
        cont_embed_dropout: Optional[float] = None,
        cont_embed_activation: Optional[str] = None,
        quantization_setup: Optional[Dict[str, List[float]]] = None,
        n_frequencies: Optional[int] = None,
        sigma: Optional[float] = None,
        share_last_layer: Optional[bool] = None,
        full_embed_dropout: Optional[bool] = None,
    ):
        super(AutoInt, self).__init__(
            column_idx=column_idx,
            input_dim=input_dim,
            cat_embed_input=cat_embed_input,
            cat_embed_dropout=cat_embed_dropout,
            use_cat_bias=use_cat_bias,
            cat_embed_activation=cat_embed_activation,
            shared_embed=None,
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

        self.num_heads = num_heads
        self.num_layers = num_layers
        self.reduction = reduction

        self.attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=input_dim,
                    num_heads=num_heads,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x = self._get_embeddings(X)

        for layer in self.attention_layers:
            attn_output, _ = layer(x, x, x)
            x = attn_output + x

        if self.reduction == "mean":
            out = x.mean(dim=1)
        else:
            out = x.reshape(x.size(0), -1)

        return out

    @property
    def output_dim(self) -> int:
        if self.reduction == "mean":
            return self.input_dim
        else:
            return self.input_dim * len(self.column_idx)
