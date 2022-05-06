import torch
from torch import Tensor, nn

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models.tabular.mlp._layers import MLP


class CatSingleMlp(nn.Module):
    def __init__(self, model, activation):
        super(CatSingleMlp, self).__init__()

        self.column_idx = model.column_idx
        self.cat_embed_input = model.cat_embed_input
        self.num_class = model.cat_and_cont_embed.cat_embed.n_tokens

        self.activation = activation

        mlp_hidden_dims = [
            model.input_dim,
            self.num_class * 4,
            self.num_class,
        ]

        self.mlp = MLP(
            d_hidden=mlp_hidden_dims,
            activation=activation,
            dropout=0.0,
            batchnorm=False,
            batchnorm_last=False,
            linear_first=False,
        )

    def forward(self, X: Tensor, r_: Tensor) -> Tuple[Tensor, Tensor]:

        # '-1' because the Label Encoder is designed to leave 0 for unseen
        #   categories (or padding) during the supervised training. Here we
        #   are predicting directly the categories, and a categorical target
        #   has to start from 0
        x = torch.stack(
            [
                X[:, self.column_idx[col]].long() - 1
                for col, _, _ in self.cat_embed_input
            ],
            dim=1,
        )

        x_ = self.mlp(r_)

        return x, x_


class CatFeaturesMlp(nn.Module):
    def __init__(self, model, activation):
        super(CatFeaturesMlp, self).__init__()

        self.column_idx = model.column_idx
        self.cat_embed_input = model.cat_embed_input

        self.activation = activation

        self.mlp = nn.ModuleDict(
            {
                "mlp_"
                + col: MLP(
                    d_hidden=[
                        model.input_dim,
                        val * 4,
                        val,
                    ],
                    activation=activation,
                    dropout=0.0,
                    batchnorm=False,
                    batchnorm_last=False,
                    linear_first=False,
                )
                for col, val, _ in model.cat_embed_input
            }
        )

    def forward(self, X: Tensor, r_: Tensor) -> List[Tuple[Tensor, Tensor]]:

        # '-1' because the Label Encoder is designed to leave 0 for unseen
        #   categories (or padding) during the supervised training. Here we
        #   are predicting directly the categories, and a categorical target
        #   has to start from 0
        x = [
            X[:, self.column_idx[col]].long() - 1 for col, _, _ in self.cat_embed_input
        ]

        x_ = [
            self.mlp["mlp_" + col](r_[:, self.column_idx[col], :])
            for col, _, _ in self.cat_embed_input
        ]

        return list(zip(x, x_))


class ContSingleMlp(nn.Module):
    def __init__(self, model, activation):
        super(ContSingleMlp, self).__init__()

        self.column_idx = model.column_idx
        self.continuous_cols = model.continuous_cols

        self.activation = activation

        self.mlp = MLP(
            d_hidden=[
                model.input_dim,
                model.input_dim * 2,
                1,
            ],
            activation=activation,
            dropout=0.0,
            batchnorm=False,
            batchnorm_last=False,
            linear_first=False,
        )

    def forward(self, X: Tensor, r_: Tensor) -> Tuple[Tensor, Tensor]:

        x = torch.stack(
            [X[:, self.column_idx[col]] for col in self.continuous_cols],
            dim=1,
        )

        x_ = self.mlp(r_)

        return x, x_


class ContFeaturesMlp(nn.Module):
    def __init__(self, model, activation):
        super(ContFeaturesMlp, self).__init__()

        self.column_idx = model.column_idx
        self.continuous_cols = model.continuous_cols

        self.activation = activation

        self.mlp = nn.ModuleDict(
            {
                "mlp_"
                + col: MLP(
                    d_hidden=[
                        model.input_dim,
                        model.input_dim * 2,
                        1,
                    ],
                    activation=activation,
                    dropout=0.0,
                    batchnorm=False,
                    batchnorm_last=False,
                    linear_first=False,
                )
                for col in model.continuous_cols
            }
        )

    def forward(self, X: Tensor, r_: Tensor) -> List[Tuple[Tensor, Tensor]]:

        x = [
            X[:, self.column_idx[col]].unsqueeze(1).float()
            for col in self.continuous_cols
        ]

        x_ = [
            self.mlp["mlp_" + col](r_[:, self.column_idx[col]])
            for col in self.continuous_cols
        ]

        return list(zip(x, x_))
