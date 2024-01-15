import numpy as np
import torch
import einops
from torch import nn

from pytorch_widedeep.wdtypes import (
    Dict,
    List,
    Tuple,
    Union,
    Tensor,
    Literal,
    Optional,
)
from pytorch_widedeep.models._base_wd_model_component import (
    BaseWDModelComponent,
)
from pytorch_widedeep.models.tabular.embeddings_layers import (
    NormLayers,
    ContEmbeddings,
    DiffSizeCatEmbeddings,
    SameSizeCatEmbeddings,
    PeriodicContEmbeddings,
    PiecewiseContEmbeddings,
)

TContEmbeddings = Union[ContEmbeddings, PiecewiseContEmbeddings, PeriodicContEmbeddings]


def _set_continous_normalization_layer(
    continuous_cols: List[str], cont_norm_layer: str
) -> NormLayers:
    if cont_norm_layer == "layernorm":
        cont_norm: NormLayers = nn.LayerNorm(len(continuous_cols))
    elif cont_norm_layer == "batchnorm":
        cont_norm = nn.BatchNorm1d(len(continuous_cols))
    else:
        cont_norm = nn.Identity()

    return cont_norm


def _set_continous_embeddings_layer(
    column_idx: Dict[str, int],
    continuous_cols: List[str],
    embed_continuous_method: Literal["standard", "piecewise", "periodic"],
    cont_embed_dim: int,
    cont_embed_dropout: float,
    cont_embed_activation: Optional[str],
    quantization_setup: Optional[Dict[str, List[float]]],
    n_frequencies: Optional[int],
    sigma: Optional[float],
    share_last_layer: Optional[bool],
) -> TContEmbeddings:
    if embed_continuous_method == "standard":
        cont_embed: TContEmbeddings = ContEmbeddings(
            len(continuous_cols),
            cont_embed_dim,
            cont_embed_dropout,
            cont_embed_activation,
        )

    elif embed_continuous_method == "piecewise":
        assert (
            quantization_setup is not None
        ), "If 'embed_continuous_method' is 'piecewise', 'quantization_setup' must be provided"
        cont_embed = PiecewiseContEmbeddings(
            column_idx,
            quantization_setup,
            cont_embed_dim,
            cont_embed_dropout,
            cont_embed_activation,
        )

    else:
        assert (
            n_frequencies is not None
            and sigma is not None
            and share_last_layer is not None
        ), (
            "If 'embed_continuous_method' is 'periodic', 'n_frequencies', 'sigma' and "
            "'share_last_layer' must be provided"
        )
        cont_embed = PeriodicContEmbeddings(
            len(continuous_cols),
            cont_embed_dim,
            cont_embed_dropout,
            n_frequencies,
            sigma,
            share_last_layer,
            cont_embed_activation,
        )

    return cont_embed


# TO DO: change these two functions to generate Cat and Cont embeddings in
# here as opposed to using DiffSizeCatAndContEmbeddings and
# SameSizeCatAndContEmbeddings
class BaseTabularModelWithoutAttention(BaseWDModelComponent):
    def __init__(
        self,
        column_idx: Dict[str, int],
        cat_embed_input: Optional[List[Tuple[str, int, int]]],
        cat_embed_dropout: float,
        use_cat_bias: bool,
        cat_embed_activation: Optional[str],
        continuous_cols: Optional[List[str]],
        cont_norm_layer: str,
        embed_continuous: Optional[bool],
        embed_continuous_method: Literal["standard", "piecewise", "periodic"],
        cont_embed_dim: int,
        cont_embed_dropout: float,
        cont_embed_activation: Optional[str],
        *,
        quantization_setup: Optional[Dict[str, List[float]]],
        n_frequencies: Optional[int],
        sigma: Optional[float],
        share_last_layer: Optional[bool],
    ):
        super().__init__()

        self.column_idx = column_idx
        self.cat_embed_input = cat_embed_input
        self.cat_embed_dropout = cat_embed_dropout
        self.use_cat_bias = use_cat_bias
        self.cat_embed_activation = cat_embed_activation

        self.continuous_cols = continuous_cols
        self.cont_norm_layer = cont_norm_layer
        self.embed_continuous_method = embed_continuous_method
        self.embed_continuous = embed_continuous
        self.cont_embed_dim = cont_embed_dim
        self.cont_embed_dropout = cont_embed_dropout
        self.cont_embed_activation = cont_embed_activation
        self.quantization_setup = quantization_setup
        self.n_frequencies = n_frequencies
        self.sigma = sigma
        self.share_last_layer = share_last_layer

        # Categorical
        if self.cat_embed_input is not None:
            _cat_embed_input = (
                self.cat_embed_input
            )  # to avoid mypy error since cat_embed_input is Optional
            self.cat_embed = DiffSizeCatEmbeddings(
                column_idx,
                _cat_embed_input,
                cat_embed_dropout,
                use_cat_bias,
                cat_embed_activation,
            )
            self.cat_out_dim = int(np.sum([embed[2] for embed in self.cat_embed_input]))
        else:
            self.cat_out_dim = 0

        # Continuous
        if continuous_cols is not None:
            self.cont_idx = [column_idx[col] for col in continuous_cols]

            self.cont_norm = _set_continous_normalization_layer(
                continuous_cols, cont_norm_layer
            )

            if not self.embed_continuous:
                self.cont_out_dim = len(continuous_cols)
            else:
                self.cont_out_dim = len(continuous_cols) * cont_embed_dim
                self.cont_embed = _set_continous_embeddings_layer(
                    column_idx=column_idx,
                    continuous_cols=continuous_cols,
                    embed_continuous_method=embed_continuous_method,
                    cont_embed_dim=cont_embed_dim,
                    cont_embed_dropout=cont_embed_dropout,
                    cont_embed_activation=cont_embed_activation,
                    quantization_setup=quantization_setup,
                    n_frequencies=n_frequencies,
                    sigma=sigma,
                    share_last_layer=share_last_layer,
                )
        else:
            self.cont_out_dim = 0

    def _get_embeddings(self, X: Tensor) -> Tensor:
        tensors_to_concat: List[Tensor] = []
        if self.cat_embed_input is not None:
            x_cat = self.cat_embed(X)
            tensors_to_concat.append(x_cat)

        if self.continuous_cols is not None:
            x_cont = self.cont_norm((X[:, self.cont_idx].float()))
            if self.embed_continuous_method != "none":
                x_cont = self.cont_embed(x_cont)
                x_cont = einops.rearrange(x_cont, "b s d -> b (s d)")
            tensors_to_concat.append(x_cont)

        x = torch.cat(tensors_to_concat, 1)

        return x


class BaseTabularModelWithAttention(BaseWDModelComponent):
    def __init__(
        self,
        column_idx: Dict[str, int],
        cat_embed_input: Optional[List[Tuple[str, int]]],
        cat_embed_dropout: float,
        use_cat_bias: bool,
        cat_embed_activation: Optional[str],
        full_embed_dropout: bool,
        shared_embed: bool,
        add_shared_embed: bool,
        frac_shared_embed: float,
        continuous_cols: Optional[List[str]],
        cont_norm_layer: str,
        embed_continuous: Optional[bool],
        embed_continuous_method: Literal["standard", "piecewise", "periodic"],
        cont_embed_dim: int,
        cont_embed_dropout: float,
        cont_embed_activation: Optional[str],
        input_dim: int,
        *,
        quantization_setup: Optional[Dict[str, List[float]]],
        n_frequencies: Optional[int],
        sigma: Optional[float],
        share_last_layer: Optional[bool],
    ):
        super().__init__()

        self.column_idx = column_idx
        self.cat_embed_input = cat_embed_input
        self.cat_embed_dropout = cat_embed_dropout
        self.use_cat_bias = use_cat_bias
        self.cat_embed_activation = cat_embed_activation
        self.full_embed_dropout = full_embed_dropout
        self.shared_embed = shared_embed
        self.add_shared_embed = add_shared_embed
        self.frac_shared_embed = frac_shared_embed

        self.continuous_cols = continuous_cols
        self.cont_norm_layer = cont_norm_layer
        self.embed_continuous_method = embed_continuous_method
        self.embed_continuous = embed_continuous
        self.cont_embed_dim = cont_embed_dim
        self.cont_embed_dropout = cont_embed_dropout
        self.cont_embed_activation = cont_embed_activation
        self.quantization_setup = quantization_setup
        self.n_frequencies = n_frequencies
        self.sigma = sigma
        self.share_last_layer = share_last_layer

        self.input_dim = input_dim

        # Categorical
        if self.cat_embed_input is not None:
            _cat_embed_input = self.cat_embed_input
            self.cat_embed = SameSizeCatEmbeddings(
                input_dim,
                column_idx,
                _cat_embed_input,
                cat_embed_dropout,
                use_cat_bias,
                full_embed_dropout,
                shared_embed,
                add_shared_embed,
                frac_shared_embed,
                cat_embed_activation,
            )

        # Continuous
        if continuous_cols is not None:
            self.cont_idx = [column_idx[col] for col in continuous_cols]

            self.cont_norm = _set_continous_normalization_layer(
                continuous_cols, cont_norm_layer
            )

            if self.embed_continuous:
                self.cont_embed = _set_continous_embeddings_layer(
                    column_idx=column_idx,
                    continuous_cols=continuous_cols,
                    embed_continuous_method=embed_continuous_method,
                    cont_embed_dim=cont_embed_dim,
                    cont_embed_dropout=cont_embed_dropout,
                    cont_embed_activation=cont_embed_activation,
                    quantization_setup=quantization_setup,
                    n_frequencies=n_frequencies,
                    sigma=sigma,
                    share_last_layer=share_last_layer,
                )

    def _get_embeddings(self, X: Tensor) -> Tensor:
        tensors_to_concat: List[Tensor] = []
        if self.cat_embed_input is not None:
            x_cat = self.cat_embed(X)
            tensors_to_concat.append(x_cat)

        if self.continuous_cols is not None:
            x_cont = self.cont_norm((X[:, self.cont_idx].float()))
            if self.embed_continuous_method != "none":
                x_cont = self.cont_embed(x_cont)
            tensors_to_concat.append(x_cont)

        x = torch.cat(tensors_to_concat, 1)

        return x

    @property
    def attention_weights(self):
        raise NotImplementedError
