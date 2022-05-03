import torch
from torch import Tensor, nn
from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models.tabular.mlp._layers import MLP


class SameSizeCatMlp(nn.Module):
    def __init__(self, model, activation, reduction):
        super(SameSizeCatMlp, self).__init__()

        self.model = model
        self.activation = activation
        self.reduction = reduction

        self.num_class = model.cat_and_cont_embed.cat_embed.n_tokens

        mlp_hidden_dims = [
            model.encoder_output_dim,
            self.num_class * 2,
            self.num_class,
        ]

        self.mlp = MLP(
            d_hidden=mlp_hidden_dims,
            activation=activation,
        )

    def forward(self, X: Tensor) -> Tuple[Tensor, Tensor]:

        y_true = torch.stack(
            [
                X[:, self.model.column_idx[col]]
                for col, _, _ in self.model.cat_embed_input
            ],
            dim=1,
        )

        y_pred = self.mlp(y_true)

        return y_true, y_pred


class DiffSizeCatMlp(nn.Module):
    def __init__(self, model, activation, reduction):
        super(DiffSizeMlp, self).__init__()

        self.model = model
        self.activation = activation
        self.reduction = reduction

        self.mlp = nn.ModuleDict(
            {
                "mlp_"
                + col: MLP(
                    d_hidden=[
                        model.encoder_output_dim,
                        model.encoder_output_dim * 2,
                        val,
                    ],
                    activation=activation,
                )
                for col, val, _ in model.embed_input
            }
        )

    def forward(self, X: Tensor) -> Tuple[List[Tensor], List[Tensor]]:

        y_true = [
            X[:, self.model.column_idx[col]] for col, _, _ in self.model.cat_embed_input
        ]

        y_pred = [
            self.mlp["mlp_" + col](X[:, self.model.column_idx[col]])
            for col, _, dim in self.model.cat_embed_input
        ]

        return y_true, y_pred


class ContEmbedded(nn.Module):
    def __init__(self, model, activation, reduction):
        super(ContEmbedded, self).__init__()

        self.model = model
        self.activation = activation
        self.reduction = reduction

        self.mlp = MLP(
            d_hidden=[
                model.encoder_output_dim,
                model.encoder_output_dim * 2,
                1,
            ],
            activation=activation,
        )

    def forward(self, X: Tensor) -> Tuple[Tensor, Tensor]:

        y_true = torch.stack(
            [
                X[:, self.model.column_idx[col]]
                for col in self.model.continuous_cols
            ],
            dim=1,
        )

        y_pred = self.mlp(y_true)

        return y_true, y_pred


class ContNotEmbedded(nn.Module):
    def __init__(self, model, activation, reduction):
        super(DiffSizeMlp, self).__init__()

        self.model = model
        self.activation = activation
        self.reduction = reduction

        self.mlp = nn.ModuleDict(
            {
                "mlp_"
                + col: MLP(
                    d_hidden=[
                        model.encoder_output_dim,
                        model.encoder_output_dim * 2,
                        1,
                    ],
                    activation=activation,
                )
                for col in model.continuous_cols
            }
        )

    def forward(self, X: Tensor) -> Tuple[List[Tensor], List[Tensor]]:

        y_true = [
            X[:, self.model.column_idx[col]] for col in self.model.continuous_cols
        ]

        y_pred = [
            self.mlp["mlp_" + col](X[:, self.model.column_idx[col]])
            for col in self.model.continuous_cols
        ]

        return y_true, y_pred
