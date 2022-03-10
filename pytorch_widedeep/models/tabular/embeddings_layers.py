"""
SharedEmbeddings is inspired by the TabTransformer available in AutoGluon:
https://github.com/awslabs/autogluon/tree/master/tabular/src/autogluon/tabular/models/tab_transformer
"""

import math
import warnings

import numpy as np
import torch
import einops
import torch.nn.functional as F
from torch import nn

from pytorch_widedeep.wdtypes import *  # noqa: F403


class FullEmbeddingDropout(nn.Module):
    def __init__(self, p: float):
        super(FullEmbeddingDropout, self).__init__()

        if p < 0 or p > 1:
            raise ValueError(f"p probability has to be between 0 and 1, but got {p}")
        self.p = p

    def forward(self, X: Tensor) -> Tensor:
        if self.training:
            mask = X.new().resize_((X.size(1), 1)).bernoulli_(1 - self.p).expand_as(
                X
            ) / (1 - self.p)
            return mask * X
        else:
            return X

    def extra_repr(self) -> str:
        return f"p={self.p}"


DropoutLayers = Union[nn.Dropout, FullEmbeddingDropout]


class ContEmbeddings(nn.Module):
    def __init__(
        self,
        n_cont_cols: int,
        embed_dim: int,
        embed_dropout: float,
        use_bias: bool,
    ):
        super(ContEmbeddings, self).__init__()

        self.n_cont_cols = n_cont_cols
        self.embed_dim = embed_dim
        self.embed_dropout = embed_dropout
        self.use_bias = use_bias

        self.weight = nn.init.kaiming_uniform_(
            nn.Parameter(torch.Tensor(n_cont_cols, embed_dim)), a=math.sqrt(5)
        )

        self.bias = (
            nn.init.kaiming_uniform_(
                nn.Parameter(torch.Tensor(n_cont_cols, embed_dim)), a=math.sqrt(5)
            )
            if use_bias
            else None
        )

    def forward(self, X: Tensor) -> Tensor:
        x = self.weight.unsqueeze(0) * X.unsqueeze(2)
        if self.bias is not None:
            x = x + self.bias.unsqueeze(0)
        return F.dropout(x, self.embed_dropout, self.training)

    def extra_repr(self) -> str:
        s = "{n_cont_cols}, {embed_dim}, embed_dropout={embed_dropout}, use_bias={use_bias}"
        return s.format(**self.__dict__)


class SharedEmbeddings(nn.Module):
    def __init__(
        self,
        n_embed: int,
        embed_dim: int,
        embed_dropout: float,
        full_embed_dropout: bool = False,
        add_shared_embed: bool = False,
        frac_shared_embed=0.25,
    ):
        super(SharedEmbeddings, self).__init__()

        assert frac_shared_embed < 1, "'frac_shared_embed' must be less than 1"
        self.add_shared_embed = add_shared_embed
        self.embed = nn.Embedding(n_embed, embed_dim, padding_idx=0)
        self.embed.weight.data.clamp_(-2, 2)
        if add_shared_embed:
            col_embed_dim = embed_dim
        else:
            col_embed_dim = int(embed_dim * frac_shared_embed)
        self.shared_embed = nn.Parameter(torch.empty(1, col_embed_dim).uniform_(-1, 1))

        if full_embed_dropout:
            self.dropout: DropoutLayers = FullEmbeddingDropout(embed_dropout)
        else:
            self.dropout = nn.Dropout(embed_dropout)

    def forward(self, X: Tensor) -> Tensor:
        out = self.dropout(self.embed(X))
        shared_embed = self.shared_embed.expand(out.shape[0], -1)
        if self.add_shared_embed:
            out += shared_embed
        else:
            out[:, : shared_embed.shape[1]] = shared_embed
        return out


class DiffSizeCatEmbeddings(nn.Module):
    def __init__(
        self,
        column_idx: Dict[str, int],
        embed_input: List[Tuple[str, int, int]],
        embed_dropout: float,
        use_bias: bool,
    ):
        super(DiffSizeCatEmbeddings, self).__init__()

        self.column_idx = column_idx
        self.embed_input = embed_input
        self.use_bias = use_bias

        # Categorical: val + 1 because 0 is reserved for padding/unseen cateogories.
        self.embed_layers = nn.ModuleDict(
            {
                "emb_layer_" + col: nn.Embedding(val + 1, dim, padding_idx=0)
                for col, val, dim in self.embed_input
            }
        )
        self.embedding_dropout = nn.Dropout(embed_dropout)

        if use_bias:
            self.biases = nn.ParameterDict()
            for col, _, dim in self.embed_input:
                # no major reason for this bound, I just want them to be
                # small, and related to the number of embeddings for that
                # particular feature
                bound = 1 / math.sqrt(dim)
                self.biases["bias_" + col] = nn.Parameter(
                    nn.init.uniform_(torch.Tensor(dim), -bound, bound)
                )

        self.emb_out_dim: int = int(np.sum([embed[2] for embed in self.embed_input]))

    def forward(self, X: Tensor) -> Tensor:
        embed = [
            self.embed_layers["emb_layer_" + col](X[:, self.column_idx[col]].long())
            + (
                self.biases["bias_" + col].unsqueeze(0)
                if self.use_bias
                else torch.zeros(1, dim, device=X.device)
            )
            for col, _, dim in self.embed_input
        ]
        x = torch.cat(embed, 1)
        x = self.embedding_dropout(x)
        return x


class SameSizeCatEmbeddings(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        column_idx: Dict[str, int],
        embed_input: Optional[List[Tuple[str, int]]],
        embed_dropout: float,
        use_bias: bool,
        full_embed_dropout: bool,
        shared_embed: bool,
        add_shared_embed: bool,
        frac_shared_embed: float,
    ):
        super(SameSizeCatEmbeddings, self).__init__()

        self.n_tokens = sum([ei[1] for ei in embed_input])
        self.column_idx = column_idx
        self.embed_input = embed_input
        self.shared_embed = shared_embed
        self.with_cls_token = "cls_token" in column_idx

        categorical_cols = [ei[0] for ei in embed_input]
        self.cat_idx = [self.column_idx[col] for col in categorical_cols]

        if use_bias:
            if shared_embed:
                warnings.warn(
                    "The current implementation of 'SharedEmbeddings' does not use bias",
                    UserWarning,
                )
            n_cat = (
                len(categorical_cols) - 1
                if self.with_cls_token
                else len(categorical_cols)
            )
            self.bias = nn.init.kaiming_uniform_(
                nn.Parameter(torch.Tensor(n_cat, embed_dim)), a=math.sqrt(5)
            )
        else:
            self.bias = None

        # Categorical: val + 1 because 0 is reserved for padding/unseen cateogories.
        if self.shared_embed:
            self.embed: Union[nn.ModuleDict, nn.Embedding] = nn.ModuleDict(
                {
                    "emb_layer_"
                    + col: SharedEmbeddings(
                        val if col == "cls_token" else val + 1,
                        embed_dim,
                        embed_dropout,
                        full_embed_dropout,
                        add_shared_embed,
                        frac_shared_embed,
                    )
                    for col, val in self.embed_input
                }
            )
        else:
            n_tokens = sum([ei[1] for ei in embed_input])
            self.embed = nn.Embedding(n_tokens + 1, embed_dim, padding_idx=0)
            if full_embed_dropout:
                self.dropout: DropoutLayers = FullEmbeddingDropout(embed_dropout)
            else:
                self.dropout = nn.Dropout(embed_dropout)

    def forward(self, X: Tensor) -> Tensor:
        if self.shared_embed:
            cat_embed = [
                self.embed["emb_layer_" + col](  # type: ignore[index]
                    X[:, self.column_idx[col]].long()
                ).unsqueeze(1)
                for col, _ in self.embed_input
            ]
            x = torch.cat(cat_embed, 1)
        else:
            x = self.embed(X[:, self.cat_idx].long())
            if self.bias is not None:
                if self.with_cls_token:
                    # no bias to be learned for the [CLS] token
                    bias = torch.cat(
                        [torch.zeros(1, self.bias.shape[1], device=x.device), self.bias]
                    )
                else:
                    bias = self.bias
                x = x + bias.unsqueeze(0)

            x = self.dropout(x)
        return x


class DiffSizeCatAndContEmbeddings(nn.Module):
    def __init__(
        self,
        column_idx: Dict[str, int],
        cat_embed_input: List[Tuple[str, int, int]],
        cat_embed_dropout: float,
        use_cat_bias: bool,
        continuous_cols: Optional[List[str]],
        cont_norm_layer: str,
        embed_continuous: bool,
        cont_embed_dim: int,
        cont_embed_dropout: float,
        use_cont_bias: bool,
    ):
        super(DiffSizeCatAndContEmbeddings, self).__init__()

        self.cat_embed_input = cat_embed_input
        self.continuous_cols = continuous_cols
        self.embed_continuous = embed_continuous
        self.cont_embed_dim = cont_embed_dim

        # Categorical
        if self.cat_embed_input is not None:
            self.cat_embed = DiffSizeCatEmbeddings(
                column_idx, cat_embed_input, cat_embed_dropout, use_cat_bias
            )
            self.cat_out_dim = int(np.sum([embed[2] for embed in self.cat_embed_input]))
        else:
            self.cat_out_dim = 0

        # Continuous
        if continuous_cols is not None:
            self.cont_idx = [column_idx[col] for col in continuous_cols]
            if cont_norm_layer == "layernorm":
                self.cont_norm: NormLayers = nn.LayerNorm(len(continuous_cols))
            elif cont_norm_layer == "batchnorm":
                self.cont_norm = nn.BatchNorm1d(len(continuous_cols))
            else:
                self.cont_norm = nn.Identity()
            if self.embed_continuous:
                self.cont_embed = ContEmbeddings(
                    len(continuous_cols),
                    cont_embed_dim,
                    cont_embed_dropout,
                    use_cont_bias,
                )
                self.cont_out_dim = len(continuous_cols) * cont_embed_dim
            else:
                self.cont_out_dim = len(continuous_cols)
        else:
            self.cont_out_dim = 0

        self.output_dim = self.cat_out_dim + self.cont_out_dim

    def forward(self, X: Tensor) -> Tuple[Tensor, Any]:

        if self.cat_embed_input is not None:
            x_cat = self.cat_embed(X)
        else:
            x_cat = None

        if self.continuous_cols is not None:
            x_cont = self.cont_norm((X[:, self.cont_idx].float()))
            if self.embed_continuous:
                x_cont = self.cont_embed(x_cont)
                x_cont = einops.rearrange(x_cont, "b s d -> b (s d)")
        else:
            x_cont = None

        return x_cat, x_cont


class SameSizeCatAndContEmbeddings(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        column_idx: Dict[str, int],
        cat_embed_input: Optional[List[Tuple[str, int]]],
        cat_embed_dropout: float,
        use_cat_bias: bool,
        full_embed_dropout: bool,
        shared_embed: bool,
        add_shared_embed: bool,
        frac_shared_embed: float,
        continuous_cols: Optional[List[str]],
        cont_norm_layer: str,
        embed_continuous: bool,
        cont_embed_dropout: float,
        use_cont_bias: bool,
    ):
        super(SameSizeCatAndContEmbeddings, self).__init__()

        self.embed_dim = embed_dim
        self.cat_embed_input = cat_embed_input
        self.continuous_cols = continuous_cols
        self.embed_continuous = embed_continuous

        # Categorical
        if cat_embed_input is not None:
            self.cat_embed = SameSizeCatEmbeddings(
                embed_dim,
                column_idx,
                cat_embed_input,
                cat_embed_dropout,
                use_cat_bias,
                full_embed_dropout,
                shared_embed,
                add_shared_embed,
                frac_shared_embed,
            )
        # Continuous
        if continuous_cols is not None:
            self.cont_idx = [column_idx[col] for col in continuous_cols]
            if cont_norm_layer == "layernorm":
                self.cont_norm: NormLayers = nn.LayerNorm(len(continuous_cols))
            elif cont_norm_layer == "batchnorm":
                self.cont_norm = nn.BatchNorm1d(len(continuous_cols))
            else:
                self.cont_norm = nn.Identity()
            if self.embed_continuous:
                self.cont_embed = ContEmbeddings(
                    len(continuous_cols),
                    embed_dim,
                    cont_embed_dropout,
                    use_cont_bias,
                )

    def forward(self, X: Tensor) -> Tuple[Tensor, Any]:

        if self.cat_embed_input is not None:
            x_cat = self.cat_embed(X)
        else:
            x_cat = None

        if self.continuous_cols is not None:
            x_cont = self.cont_norm((X[:, self.cont_idx].float()))
            if self.embed_continuous:
                x_cont = self.cont_embed(x_cont)
        else:
            x_cont = None

        return x_cat, x_cont
