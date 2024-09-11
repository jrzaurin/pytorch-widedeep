from typing import Dict, List, Tuple, Optional

import torch
from torch import Tensor

from pytorch_widedeep.models.rec._layers import (
    PseudoLinear,
    CompressedInteractionNetwork,
)
from pytorch_widedeep.models.tabular.mlp._layers import MLP
from pytorch_widedeep.models.tabular._base_tabular_model import (
    BaseTabularModelWithAttention,
)


class xDeepFM(BaseTabularModelWithAttention):
    def __init__(
        self,
        column_idx: Dict[str, int],
        cat_embed_input: List[Tuple[str, int]],
        input_dim: int,
        cin_layer_dims: List[int],
        with_pseudo_linear: bool,
        *,
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
        super(xDeepFM, self).__init__(
            column_idx=column_idx,
            input_dim=input_dim,
            cat_embed_input=cat_embed_input,
            cat_embed_dropout=cat_embed_dropout,
            use_cat_bias=use_cat_bias,
            cat_embed_activation=cat_embed_activation,
            shared_embed=False,
            add_shared_embed=None,
            frac_shared_embed=None,
            continuous_cols=None,
            cont_norm_layer=None,
            embed_continuous=None,
            embed_continuous_method=None,
            cont_embed_dropout=None,
            cont_embed_activation=None,
            quantization_setup=None,
            n_frequencies=None,
            sigma=None,
            share_last_layer=None,
            full_embed_dropout=None,
        )

        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.mlp_dropout = mlp_dropout
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.mlp_linear_first = mlp_linear_first

        self.n_features = len(self.column_idx)

        self.with_pseudo_linear = with_pseudo_linear

        if with_pseudo_linear:
            self.linear = PseudoLinear(
                column_idx=column_idx,
                cat_embed_input=cat_embed_input,
                cat_embed_dropout=cat_embed_dropout,
                use_cat_bias=use_cat_bias,
                cat_embed_activation=cat_embed_activation,
                continuous_cols=None,
                cont_norm_layer=None,
                embed_continuous=None,
                embed_continuous_method=None,
                cont_embed_dropout=None,
                cont_embed_activation=None,
                quantization_setup=None,
                n_frequencies=None,
                sigma=None,
                share_last_layer=None,
                full_embed_dropout=None,
            )
        else:
            self.linear = None

        self.cin = CompressedInteractionNetwork(
            num_cols=self.n_features, cin_layer_dims=cin_layer_dims
        )

        if self.mlp_hidden_dims is not None:
            self.mlp = MLP(
                d_hidden=[sum(cin_layer_dims)] + self.mlp_hidden_dims,
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

        if self.linear is not None:
            linear_out = self.linear(X)
        else:
            linear_out = torch.tensor(0).to(X.device)

        cin_out = self.cin(self._get_embeddings(X))

        if self.mlp is not None:
            mlp_out = self.mlp(cin_out)
        else:
            mlp_out = torch.tensor(0).to(X.device)

        return linear_out + cin_out + mlp_out
