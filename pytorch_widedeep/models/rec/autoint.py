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
    continuous_cols: List, Optional, default = None
        List with the name of the numeric (aka continuous) columns
    cont_norm_layer: str, Optional, default = None
        Type of normalization layer applied to the continuous features.
        Options are: _'layernorm'_ and _'batchnorm'_
    embed_continuous_method: Optional, str, default = None,
        Method to use to embed the continuous features. Options are:
        _'standard'_, _'periodic'_ or _'piecewise'_
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
