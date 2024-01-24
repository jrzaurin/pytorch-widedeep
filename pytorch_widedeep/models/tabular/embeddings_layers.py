"""
SharedEmbeddings is inspired by the TabTransformer available in AutoGluon:
https://github.com/awslabs/autogluon/tree/master/tabular/src/autogluon/tabular/models/tab_transformer

PiecewiseContEmbeddings and PeriodicContEmbeddings are presented in the paper:
'On Embeddings for Numerical Features in Tabular Deep Learning'

The implementation here is inspired by and a combination of:
1. The original implementation here: https://github.com/yandex-research/rtdl-num-embeddings/tree/main?tab=readme-ov-file
2. And the implementation at pytorch-frame here: https://pytorch-frame.readthedocs.io/en/latest/_modules/torch_frame/nn/encoder/stype_encoder.html
"""

import math
import warnings

import numpy as np
import torch
from torch import nn

from pytorch_widedeep.wdtypes import Dict, List, Tuple, Union, Tensor, Optional
from pytorch_widedeep.models._get_activation_fn import get_activation_fn

__all__ = [
    "ContEmbeddings",
    "PiecewiseContEmbeddings",
    "PeriodicContEmbeddings",
    "SharedEmbeddings",
    "DiffSizeCatEmbeddings",
    "SameSizeCatEmbeddings",
]


class NLinear(nn.Module):
    def __init__(self, n_feat: int, inp_units: int, out_units: int) -> None:
        super().__init__()

        self.n_feat = n_feat
        self.inp_units = inp_units
        self.out_units = out_units

        self.weights = nn.Parameter(torch.Tensor(n_feat, inp_units, out_units))
        self.bias = nn.Parameter(torch.Tensor(n_feat, out_units))

        self.reset_parameters()

    def reset_parameters(self):
        d_in_rsqrt = self.weights.shape[-2] ** -0.5
        nn.init.uniform_(self.weights, -d_in_rsqrt, d_in_rsqrt)
        nn.init.uniform_(self.bias, -d_in_rsqrt, d_in_rsqrt)

    def forward(self, X: Tensor) -> torch.Tensor:
        assert X.ndim == 3
        assert X.shape[1] == self.n_feat
        assert X.shape[2] == self.inp_units

        x = (X[..., None, :] @ self.weights).squeeze(-2)
        x = x + self.bias
        return x

    def extra_repr(self) -> str:
        s = "n_feat={n_feat}, inp_units={inp_units}, out_units={out_units}"
        return s.format(**self.__dict__)


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
NormLayers = Union[nn.Identity, nn.LayerNorm, nn.BatchNorm1d]


class ContEmbeddings(nn.Module):
    def __init__(
        self,
        n_cont_cols: int,
        embed_dim: int,
        embed_dropout: float,
        full_embed_dropout: bool,
        activation_fn: Optional[str] = None,
    ):
        super(ContEmbeddings, self).__init__()

        self.n_cont_cols = n_cont_cols
        self.embed_dim = embed_dim
        self.embed_dropout = embed_dropout
        self.activation_fn_name = activation_fn

        self.weight = nn.Parameter(torch.Tensor(n_cont_cols, embed_dim))
        self.bias = nn.Parameter(torch.Tensor(n_cont_cols, embed_dim))

        self.reset_parameters()

        self.activation_fn = (
            get_activation_fn(activation_fn) if activation_fn is not None else None
        )

        if full_embed_dropout:
            self.dropout: DropoutLayers = FullEmbeddingDropout(embed_dropout)
        else:
            self.dropout = nn.Dropout(embed_dropout)

    def reset_parameters(self) -> None:
        # see here https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, X: Tensor) -> Tensor:
        # same as torch.einsum("ij,jk->ijk", X, weight)
        x = self.weight.unsqueeze(0) * X.unsqueeze(2)
        x = x + self.bias.unsqueeze(0)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        x = self.dropout(x)
        return x

    def extra_repr(self) -> str:
        all_params = "INFO: [ContLinear = weight(n_cont_cols, embed_dim) + bias(n_cont_cols, embed_dim)]\n"
        all_params += (
            "(linear): ContLinear(n_cont_cols={n_cont_cols}, embed_dim={embed_dim}"
        )
        all_params += ", embed_dropout={embed_dropout})"
        return f"{all_params.format(**self.__dict__)}"


class PiecewiseContEmbeddings(nn.Module):
    def __init__(
        self,
        column_idx: Dict[str, int],
        quantization_setup: Dict[str, List[float]],
        embed_dim: int,
        embed_dropout: float,
        full_embed_dropout: bool,
        activation_fn: Optional[str] = None,
    ) -> None:
        super(PiecewiseContEmbeddings, self).__init__()

        self.column_idx = column_idx
        self.quantization_setup = quantization_setup
        self.embed_dim = embed_dim
        self.embed_dropout = embed_dropout

        self.n_cont_cols = len(quantization_setup)
        self.max_num_buckets = max([len(qs) - 1 for qs in quantization_setup.values()])

        boundaries: Dict[str, Tensor] = {}
        for col, qs in quantization_setup.items():
            boundaries[col] = torch.tensor(qs)
            self.register_buffer("boundaries_" + col, boundaries[col])

        self.weight = nn.Parameter(
            torch.empty(self.n_cont_cols, self.max_num_buckets, self.embed_dim)
        )
        self.bias = nn.Parameter(torch.empty(self.n_cont_cols, self.embed_dim))

        self.reset_parameters()

        self.activation_fn = (
            get_activation_fn(activation_fn) if activation_fn is not None else None
        )

        if full_embed_dropout:
            self.dropout: DropoutLayers = FullEmbeddingDropout(embed_dropout)
        else:
            self.dropout = nn.Dropout(embed_dropout)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, std=0.01)
        nn.init.zeros_(self.bias)

    def forward(self, X: Tensor) -> Tensor:
        # b: batch size
        # n: n_cont_cols
        # k: max_num_buckets
        # e: embed_dim
        encoded_values = []
        for col, index in self.column_idx.items():
            feat = X[:, index].contiguous()

            col_boundaries = getattr(self, "boundaries_" + col)
            bucket_indices = torch.bucketize(feat, col_boundaries[1:-1])

            boundary_start = col_boundaries[bucket_indices]
            boundary_end = col_boundaries[bucket_indices + 1]
            frac = (feat - boundary_start) / (boundary_end - boundary_start + 1e-8)

            greater_mask = (feat.view(-1, 1) > col_boundaries[:-1]).float()
            greater_mask[
                torch.arange(len(bucket_indices), device=greater_mask.device),
                bucket_indices,
            ] = frac.float()

            encoded_values.append(greater_mask)

        out = torch.stack(encoded_values, dim=1)
        x = torch.einsum("b n k, n k e -> b n e", out, self.weight)
        x = x + self.bias
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        x = self.dropout(x)
        return x

    def extra_repr(self) -> str:
        all_params = (
            "INFO: [BucketLinear = weight(n_cont_cols, max_num_buckets, embed_dim) "
        )
        all_params += "+ bias(n_cont_cols, embed_dim)]\n"
        all_params += "(linear): BucketLinear(n_cont_cols={n_cont_cols}, max_num_buckets={max_num_buckets}"
        all_params += ", embed_dim={embed_dim})"
        return f"{all_params.format(**self.__dict__)}"


class PeriodicContEmbeddings(nn.Module):
    def __init__(
        self,
        n_cont_cols: int,
        embed_dim: int,
        embed_dropout: float,
        full_embed_dropout: bool,
        n_frequencies: int,
        sigma: float,
        share_last_layer: bool,
        activation_fn: Optional[str] = None,
    ) -> None:
        super(PeriodicContEmbeddings, self).__init__()

        self.n_cont_cols = n_cont_cols
        self.embed_dim = embed_dim
        self.embed_dropout = embed_dropout
        self.n_frequencies = n_frequencies
        self.sigma = sigma
        self.share_last_layer = share_last_layer

        self.weight_perdiodic = nn.Parameter(
            torch.empty((self.n_cont_cols, self.n_frequencies))
        )
        self.reset_parameters()

        if self.share_last_layer:
            assert activation_fn is not None, (
                "If 'share_last_layer' is True, 'activation_fn' must be "
                "provided (preferably 'relu')"
            )
            self.linear: Union[nn.Linear, NLinear] = nn.Linear(
                2 * self.n_frequencies, self.embed_dim
            )
        else:
            self.linear = NLinear(
                self.n_cont_cols, 2 * self.n_frequencies, self.embed_dim
            )

        self.activation_fn = (
            get_activation_fn(activation_fn) if activation_fn is not None else None
        )

        if full_embed_dropout:
            self.dropout: DropoutLayers = FullEmbeddingDropout(embed_dropout)
        else:
            self.dropout = nn.Dropout(embed_dropout)

    def reset_parameters(self):
        bound = self.sigma * 3
        nn.init.trunc_normal_(self.weight_perdiodic, 0.0, self.sigma, a=-bound, b=bound)

    def forward(self, X) -> Tensor:
        x = 2 * math.pi * self.weight_perdiodic * X[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], -1)
        x = self.linear(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        x = self.dropout(x)
        return x

    def extra_repr(self) -> str:
        s = "INFO: [NLinear: inp_units = (n_frequencies * 2), out_units = embed_dim]"
        return f"{s.format(**self.__dict__)}"


class SharedEmbeddings(nn.Module):
    def __init__(
        self,
        n_embed: int,
        embed_dim: int,
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

    def forward(self, X: Tensor) -> Tensor:
        out = self.embed(X)
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
        activation_fn: Optional[str] = None,
    ):
        super(DiffSizeCatEmbeddings, self).__init__()

        self.column_idx = column_idx
        self.embed_input = embed_input
        self.use_bias = use_bias

        self.embed_layers_names: Dict[str, str] = {
            e[0]: e[0].replace(".", "_") for e in self.embed_input
        }

        # Categorical: val + 1 because 0 is reserved for padding/unseen cateogories.
        self.embed_layers = nn.ModuleDict(
            {
                "emb_layer_"
                + self.embed_layers_names[col]: nn.Embedding(
                    val + 1, dim, padding_idx=0
                )
                for col, val, dim in self.embed_input
            }
        )
        self.activation_fn = (
            get_activation_fn(activation_fn) if activation_fn is not None else None
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
            self.embed_layers["emb_layer_" + self.embed_layers_names[col]](
                X[:, self.column_idx[col]].long()
            )
            + (
                self.biases["bias_" + col].unsqueeze(0)
                if self.use_bias
                else torch.zeros(1, dim, device=X.device)
            )
            for col, _, dim in self.embed_input
        ]
        x = torch.cat(embed, 1)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        x = self.embedding_dropout(x)
        return x


class SameSizeCatEmbeddings(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        column_idx: Dict[str, int],
        embed_input: List[Tuple[str, int]],
        embed_dropout: float,
        full_embed_dropout: bool,
        use_bias: bool,
        shared_embed: bool,
        add_shared_embed: bool,
        frac_shared_embed: float,
        activation_fn: Optional[str] = None,
    ):
        super(SameSizeCatEmbeddings, self).__init__()

        self.n_tokens = sum([ei[1] for ei in embed_input])
        self.column_idx = column_idx
        self.embed_input = embed_input
        self.shared_embed = shared_embed
        self.with_cls_token = "cls_token" in column_idx

        self.embed_layers_names: Dict[str, str] = {
            e[0]: e[0].replace(".", "_") for e in self.embed_input
        }

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

        self.activation_fn = (
            get_activation_fn(activation_fn) if activation_fn is not None else None
        )

        # Categorical: val + 1 because 0 is reserved for padding/unseen cateogories.
        if self.shared_embed:
            self.embed: Union[nn.ModuleDict, nn.Embedding] = nn.ModuleDict(
                {
                    "emb_layer_"
                    + self.embed_layers_names[col]: SharedEmbeddings(
                        val if col == "cls_token" else val + 1,
                        embed_dim,
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
                self.embed["emb_layer_" + self.embed_layers_names[col]](  # type: ignore[index]
                    X[:, self.column_idx[col]].long()
                ).unsqueeze(
                    1
                )
                for col, _ in self.embed_input
            ]
            x = torch.cat(cat_embed, 1)

            if self.activation_fn is not None:
                x = self.activation_fn(x)
            x = self.dropout(x)
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

            if self.activation_fn is not None:
                x = self.activation_fn(x)
            x = self.dropout(x)
        return x
