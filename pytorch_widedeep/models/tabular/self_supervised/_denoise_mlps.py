import torch
from torch import nn

from pytorch_widedeep.wdtypes import Dict, List, Tuple, Tensor
from pytorch_widedeep.models.tabular.mlp._layers import MLP


class CatSingleMlp(nn.Module):
    r"""Single MLP will be applied to all categorical features"""

    def __init__(
        self,
        input_dim: int,
        cat_embed_input: List[Tuple[str, int]],
        column_idx: Dict[str, int],
        activation: str,
    ):
        super(CatSingleMlp, self).__init__()

        self.input_dim = input_dim
        self.column_idx = column_idx
        self.cat_embed_input = cat_embed_input
        self.activation = activation

        self.num_class = sum([ei[1] for ei in cat_embed_input if ei[0] != "cls_token"])

        self.mlp = MLP(
            d_hidden=[input_dim, self.num_class * 4, self.num_class],
            activation=activation,
            dropout=0.0,
            batchnorm=False,
            batchnorm_last=False,
            linear_first=False,
        )

    def forward(self, X: Tensor, r_: Tensor) -> Tuple[Tensor, Tensor]:
        x = torch.cat(
            [
                X[:, self.column_idx[col]].long()
                for col, _ in self.cat_embed_input
                if col != "cls_token"
            ]
        )

        cat_r_ = torch.cat(
            [
                r_[:, self.column_idx[col], :]
                for col, _ in self.cat_embed_input
                if col != "cls_token"
            ]
        )

        x_ = self.mlp(cat_r_)

        return x, x_


class CatMlpPerFeature(nn.Module):
    r"""Dedicated MLP per categorical feature"""

    def __init__(
        self,
        input_dim: int,
        cat_embed_input: List[Tuple[str, int]],
        column_idx: Dict[str, int],
        activation: str,
    ):
        super(CatMlpPerFeature, self).__init__()

        self.input_dim = input_dim
        self.column_idx = column_idx
        self.cat_embed_input = cat_embed_input
        self.activation = activation

        self.mlp = nn.ModuleDict(
            {
                "mlp_"
                + col: MLP(
                    d_hidden=[
                        input_dim,
                        val * 4,
                        val,
                    ],
                    activation=activation,
                    dropout=0.0,
                    batchnorm=False,
                    batchnorm_last=False,
                    linear_first=False,
                )
                for col, val in self.cat_embed_input
                if col != "cls_token"
            }
        )

    def forward(self, X: Tensor, r_: Tensor) -> List[Tuple[Tensor, Tensor]]:
        x = [
            X[:, self.column_idx[col]].long()
            for col, _ in self.cat_embed_input
            if col != "cls_token"
        ]

        x_ = [
            self.mlp["mlp_" + col](r_[:, self.column_idx[col], :])
            for col, _ in self.cat_embed_input
            if col != "cls_token"
        ]

        return list(zip(x, x_))


class ContSingleMlp(nn.Module):
    r"""Single MLP will be applied to all continuous features"""

    def __init__(
        self,
        input_dim: int,
        continuous_cols: List[str],
        column_idx: Dict[str, int],
        activation: str,
    ):
        super(ContSingleMlp, self).__init__()

        self.input_dim = input_dim
        self.column_idx = column_idx
        self.continuous_cols = continuous_cols
        self.activation = activation

        self.mlp = MLP(
            d_hidden=[input_dim, input_dim * 2, 1],
            activation=activation,
            dropout=0.0,
            batchnorm=False,
            batchnorm_last=False,
            linear_first=False,
        )

    def forward(self, X: Tensor, r_: Tensor) -> Tuple[Tensor, Tensor]:
        x = torch.cat(
            [X[:, self.column_idx[col]].float() for col in self.continuous_cols]
        ).unsqueeze(1)

        cont_r_ = torch.cat(
            [r_[:, self.column_idx[col], :] for col in self.continuous_cols]
        )

        x_ = self.mlp(cont_r_)

        return x, x_


class ContMlpPerFeature(nn.Module):
    r"""Dedicated MLP per continuous feature"""

    def __init__(
        self,
        input_dim: int,
        continuous_cols: List[str],
        column_idx: Dict[str, int],
        activation: str,
    ):
        super(ContMlpPerFeature, self).__init__()

        self.input_dim = input_dim
        self.column_idx = column_idx
        self.continuous_cols = continuous_cols
        self.activation = activation

        self.mlp = nn.ModuleDict(
            {
                "mlp_"
                + col: MLP(
                    d_hidden=[
                        input_dim,
                        input_dim * 2,
                        1,
                    ],
                    activation=activation,
                    dropout=0.0,
                    batchnorm=False,
                    batchnorm_last=False,
                    linear_first=False,
                )
                for col in self.continuous_cols
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
