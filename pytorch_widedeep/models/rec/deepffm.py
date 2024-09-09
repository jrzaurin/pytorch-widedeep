from typing import Any, Dict, List, Tuple, Optional

import torch
from torch import Tensor, nn

from pytorch_widedeep.models.rec._layers import PseudoLinear
from pytorch_widedeep.models.tabular.mlp._layers import MLP
from pytorch_widedeep.models.tabular._base_tabular_model import (
    BaseTabularModelWithAttention,
)


class DeepFieldAwareFactorizationMachine(BaseTabularModelWithAttention):
    def __init__(
        self,
        column_idx: Dict[str, int],
        cat_embed_input: List[Tuple[str, int]],
        num_factors: int,
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
        self.n_tokens = sum([ei[1] for ei in cat_embed_input])

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

        self.encoders = nn.ModuleList(
            [
                BaseTabularModelWithAttention(**config)
                for config in self._get_encoder_configs()
            ]
        )

        if self.mlp_hidden_dims is not None:
            first_layer_dim = self.n_features * (self.n_features - 1) // 2 * num_factors
            self.mlp = MLP(
                d_hidden=[first_layer_dim] + self.mlp_hidden_dims,
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

        linear_output = self.linear(X)

        interactions: List[Tensor] = []
        for i in range(len(self.column_idx)):
            for j in range(i + 1, len(self.column_idx)):
                # the syntax [i] and [j] is to keep the shape of the tensors
                # as they are sliced within '_get_embeddings'. This will
                # return a tensor of shape (b, 1, embed_dim). Then it has to
                # be squeezed to (b, embed_dim)  before multiplied
                embed_i = self.encoders[i]._get_embeddings(X[:, [i]]).squeeze(1)
                embed_j = self.encoders[j]._get_embeddings(X[:, [j]]).squeeze(1)
                interactions.append(embed_i * embed_j)

        if self.mlp is not None:
            mlp_input = torch.cat(interactions, dim=1).view(X.size(0), -1)
            interactions_output = self.mlp(mlp_input)
        else:
            interactions_output = torch.cat(interactions, dim=1).sum(
                dim=1, keepdim=True
            )

        return linear_output + interactions_output

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
                "embed_continuous": None,
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
        return 1
