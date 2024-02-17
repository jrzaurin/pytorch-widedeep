import warnings

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
from pytorch_widedeep.utils.general_utils import alias
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


class BaseTabularModelWithoutAttention(BaseWDModelComponent):
    def __init__(
        self,
        column_idx: Dict[str, int],
        *,
        cat_embed_input: Optional[List[Tuple[str, int, int]]],
        cat_embed_dropout: Optional[float],
        use_cat_bias: Optional[bool],
        cat_embed_activation: Optional[str],
        continuous_cols: Optional[List[str]],
        cont_norm_layer: Optional[Literal["batchnorm", "layernorm"]],
        embed_continuous: Optional[bool],
        embed_continuous_method: Optional[Literal["standard", "piecewise", "periodic"]],
        cont_embed_dim: Optional[int],
        cont_embed_dropout: Optional[float],
        cont_embed_activation: Optional[str],
        quantization_setup: Optional[Dict[str, List[float]]],
        n_frequencies: Optional[int],
        sigma: Optional[float],
        share_last_layer: Optional[bool],
        full_embed_dropout: Optional[bool],
    ):
        super().__init__()

        self.column_idx = column_idx

        # datasets can have anything, only categorical cols, only continuous
        # cols or both, therefore, all parameters below are optional inputs

        # Categorical Parameters
        self.cat_embed_input = cat_embed_input
        self.cat_embed_dropout = cat_embed_dropout
        self.use_cat_bias = use_cat_bias
        self.cat_embed_activation = cat_embed_activation

        # Continuous Parameters
        self.continuous_cols = continuous_cols
        self.cont_norm_layer = cont_norm_layer
        self.embed_continuous_method = embed_continuous_method
        self.cont_embed_dim = cont_embed_dim
        self.cont_embed_dropout = cont_embed_dropout
        self.cont_embed_activation = cont_embed_activation
        self.quantization_setup = quantization_setup
        self.n_frequencies = n_frequencies
        self.sigma = sigma
        self.share_last_layer = share_last_layer
        self.full_embed_dropout = full_embed_dropout
        if embed_continuous is not None:
            self.embed_continuous = embed_continuous
            warnings.warn(
                "The 'embed_continuous' parameter is deprecated and will be removed in "
                "the next release. Please use 'embed_continuous_method' instead "
                "See the documentation for more details.",
                DeprecationWarning,
                stacklevel=2,
            )
            if embed_continuous and embed_continuous_method is None:
                raise ValueError(
                    "If 'embed_continuous' is True, 'embed_continuous_method' must be "
                    "one of 'standard', 'piecewise' or 'periodic'."
                )
        elif embed_continuous_method is not None:
            self.embed_continuous = True
        else:
            self.embed_continuous = False

        # Categorical Embeddings
        if self.cat_embed_input is not None:
            self.cat_embed = DiffSizeCatEmbeddings(
                column_idx=self.column_idx,
                embed_input=self.cat_embed_input,
                embed_dropout=(
                    0.0 if self.cat_embed_dropout is None else self.cat_embed_dropout
                ),
                use_bias=False if self.use_cat_bias is None else self.use_cat_bias,
                activation_fn=self.cat_embed_activation,
            )
            self.cat_out_dim = int(np.sum([embed[2] for embed in self.cat_embed_input]))
        else:
            self.cat_out_dim = 0

        # Continuous cols can be embedded or not
        if self.continuous_cols is not None:
            self.cont_idx = [column_idx[col] for col in self.continuous_cols]
            self.cont_norm = _set_continous_normalization_layer(
                self.continuous_cols, self.cont_norm_layer
            )
            if not self.embed_continuous:
                self.cont_out_dim = len(self.continuous_cols)
            else:
                assert (
                    self.cont_embed_dim
                    is not None  # assertion to avoid typing conflicts
                ), "If continuous features are embedded, 'cont_embed_dim' must be provided"
                assert (
                    self.embed_continuous_method
                    is not None  # assertion to avoid typing conflicts
                ), (
                    "If continuous features are embedded, 'embed_continuous_method' must be "
                    "one of 'standard', 'piecewise' or 'periodic'"
                )
                self.cont_out_dim = len(self.continuous_cols) * self.cont_embed_dim
                self.cont_embed = _set_continous_embeddings_layer(
                    column_idx=self.column_idx,
                    continuous_cols=self.continuous_cols,
                    embed_continuous_method=self.embed_continuous_method,
                    cont_embed_dim=self.cont_embed_dim,
                    cont_embed_dropout=self.cont_embed_dropout,
                    full_embed_dropout=self.full_embed_dropout,
                    cont_embed_activation=self.cont_embed_activation,
                    quantization_setup=self.quantization_setup,
                    n_frequencies=self.n_frequencies,
                    sigma=self.sigma,
                    share_last_layer=self.share_last_layer,
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
            if self.embed_continuous:
                x_cont = self.cont_embed(x_cont)
                x_cont = einops.rearrange(x_cont, "b s d -> b (s d)")
            tensors_to_concat.append(x_cont)

        x = torch.cat(tensors_to_concat, 1)

        return x


class BaseTabularModelWithAttention(BaseWDModelComponent):
    @alias("shared_embed", ["cat_embed_shared"])
    def __init__(
        self,
        column_idx: Dict[str, int],
        input_dim: int,
        *,
        cat_embed_input: Optional[List[Tuple[str, int]]],
        cat_embed_dropout: Optional[float],
        use_cat_bias: Optional[bool],
        cat_embed_activation: Optional[str],
        shared_embed: Optional[bool],
        add_shared_embed: Optional[bool],
        frac_shared_embed: Optional[float],
        continuous_cols: Optional[List[str]],
        cont_norm_layer: Optional[Literal["batchnorm", "layernorm"]],
        embed_continuous: Optional[bool],
        embed_continuous_method: Optional[Literal["standard", "piecewise", "periodic"]],
        cont_embed_dropout: Optional[float],
        cont_embed_activation: Optional[str],
        quantization_setup: Optional[Dict[str, List[float]]],
        n_frequencies: Optional[int],
        sigma: Optional[float],
        share_last_layer: Optional[bool],
        full_embed_dropout: Optional[bool],
    ):
        super().__init__()

        self.column_idx = column_idx

        # datasets can have anything, only categorical cols, only continuous
        # cols or both, therefore, all parameters below are optional inputs

        # Categorical Parameters
        self.cat_embed_input = cat_embed_input
        self.cat_embed_dropout = cat_embed_dropout
        self.use_cat_bias = use_cat_bias
        self.cat_embed_activation = cat_embed_activation
        self.shared_embed = shared_embed
        self.add_shared_embed = add_shared_embed
        self.frac_shared_embed = frac_shared_embed

        # Continuous Parameters
        self.continuous_cols = continuous_cols
        self.cont_norm_layer = cont_norm_layer
        self.embed_continuous_method = embed_continuous_method
        self.cont_embed_dropout = cont_embed_dropout
        self.cont_embed_activation = cont_embed_activation
        self.quantization_setup = quantization_setup
        self.n_frequencies = n_frequencies
        self.sigma = sigma
        self.share_last_layer = share_last_layer
        if embed_continuous is not None:
            self.embed_continuous = embed_continuous
            warnings.warn(
                "The 'embed_continuous' parameter is deprecated and will be removed in "
                "the next release. Please use 'embed_continuous_method' instead "
                "See the documentation for more details.",
                DeprecationWarning,
                stacklevel=2,
            )
            if embed_continuous and embed_continuous_method is None:
                raise ValueError(
                    "If 'embed_continuous' is True, 'embed_continuous_method' must be "
                    "one of 'standard', 'piecewise' or 'periodic'."
                )
        elif embed_continuous_method is not None:
            self.embed_continuous = True
        else:
            self.embed_continuous = False

        # if full_embed_dropout is not None will be applied to all cont and cat
        self.full_embed_dropout = full_embed_dropout

        # They are going to be stacked over dim 1 so cat and cont need the
        # same embedding dim
        self.input_dim = input_dim

        # Categorical Embeddings
        if self.cat_embed_input is not None:
            assert (
                self.input_dim is not None
            ), "If 'cat_embed_input' is not None, 'input_dim' must be provided"
            self.cat_embed = SameSizeCatEmbeddings(
                embed_dim=self.input_dim,
                column_idx=self.column_idx,
                embed_input=self.cat_embed_input,
                embed_dropout=(
                    0.0 if self.cat_embed_dropout is None else self.cat_embed_dropout
                ),
                full_embed_dropout=(
                    False
                    if self.full_embed_dropout is None
                    else self.full_embed_dropout
                ),
                use_bias=False if self.use_cat_bias is None else self.use_cat_bias,
                shared_embed=False if self.shared_embed is None else self.shared_embed,
                add_shared_embed=(
                    False if self.add_shared_embed is None else self.add_shared_embed
                ),
                frac_shared_embed=(
                    0.0 if self.frac_shared_embed is None else self.frac_shared_embed
                ),
                activation_fn=self.cat_embed_activation,
            )

        # Continuous cols can be embedded or not
        if self.continuous_cols is not None:
            self.cont_idx = [self.column_idx[col] for col in self.continuous_cols]
            self.cont_norm = _set_continous_normalization_layer(
                self.continuous_cols, self.cont_norm_layer
            )
            if self.embed_continuous:
                assert (
                    self.embed_continuous_method
                    is not None  # assertion to avoid typing conflicts
                ), (
                    "If continuous features are embedded, 'embed_continuous_method' must be "
                    "one of 'standard', 'piecewise' or 'periodic'"
                )
                self.cont_embed = _set_continous_embeddings_layer(
                    column_idx=self.column_idx,
                    continuous_cols=self.continuous_cols,
                    embed_continuous_method=self.embed_continuous_method,
                    cont_embed_dim=self.input_dim,
                    cont_embed_dropout=self.cont_embed_dropout,
                    full_embed_dropout=self.full_embed_dropout,
                    cont_embed_activation=self.cont_embed_activation,
                    quantization_setup=self.quantization_setup,
                    n_frequencies=self.n_frequencies,
                    sigma=self.sigma,
                    share_last_layer=self.share_last_layer,
                )

    def _get_embeddings(self, X: Tensor) -> Tensor:
        tensors_to_concat: List[Tensor] = []
        if self.cat_embed_input is not None:
            x_cat = self.cat_embed(X)
            tensors_to_concat.append(x_cat)

        if self.continuous_cols is not None:
            x_cont = self.cont_norm((X[:, self.cont_idx].float()))
            if self.embed_continuous:
                x_cont = self.cont_embed(x_cont)
            tensors_to_concat.append(x_cont)

        x = torch.cat(tensors_to_concat, 1)

        return x

    @property
    def attention_weights(self):
        raise NotImplementedError


# Eventually these two floating functions will be part of a base class
# BaseContEmbeddings. For now is easier to keep them this way in terms of
# keeping typing consistency
def _set_continous_normalization_layer(
    continuous_cols: List[str],
    cont_norm_layer: Optional[Literal["batchnorm", "layernorm"]],
) -> NormLayers:
    if cont_norm_layer is None:
        cont_norm: NormLayers = nn.Identity()
    else:
        if cont_norm_layer == "batchnorm":
            cont_norm = nn.BatchNorm1d(len(continuous_cols))
        elif cont_norm_layer == "layernorm":
            cont_norm = nn.LayerNorm(len(continuous_cols))
        else:
            raise ValueError(
                "cont_norm_layer must be one of 'layernorm', 'batchnorm' or None"
            )

    return cont_norm


def _set_continous_embeddings_layer(
    column_idx: Dict[str, int],
    continuous_cols: List[str],
    embed_continuous_method: Literal["standard", "piecewise", "periodic"],
    cont_embed_dim: int,
    cont_embed_dropout: Optional[float],
    full_embed_dropout: Optional[bool],
    cont_embed_activation: Optional[str],
    quantization_setup: Optional[Dict[str, List[float]]],
    n_frequencies: Optional[int],
    sigma: Optional[float],
    share_last_layer: Optional[bool],
) -> TContEmbeddings:
    if embed_continuous_method == "standard":
        cont_embed: TContEmbeddings = ContEmbeddings(
            n_cont_cols=len(continuous_cols),
            embed_dim=cont_embed_dim,
            embed_dropout=0.0 if cont_embed_dropout is None else cont_embed_dropout,
            full_embed_dropout=(
                False if full_embed_dropout is None else full_embed_dropout
            ),
            activation_fn=cont_embed_activation,
        )

    elif embed_continuous_method == "piecewise":
        assert (
            quantization_setup is not None
        ), "If 'embed_continuous_method' is 'piecewise', 'quantization_setup' must be provided"
        min_cont_col_index = min([column_idx[col] for col in continuous_cols])
        cont_embed = PiecewiseContEmbeddings(
            column_idx={
                col: column_idx[col] - min_cont_col_index for col in continuous_cols
            },
            quantization_setup=quantization_setup,
            embed_dim=cont_embed_dim,
            embed_dropout=0.0 if cont_embed_dropout is None else cont_embed_dropout,
            full_embed_dropout=(
                False if full_embed_dropout is None else full_embed_dropout
            ),
            activation_fn=cont_embed_activation,
        )

    elif embed_continuous_method == "periodic":
        assert (
            n_frequencies is not None
            and sigma is not None
            and share_last_layer is not None
        ), (
            "If 'embed_continuous_method' is 'periodic', 'n_frequencies', 'sigma' and "
            "'share_last_layer' must be provided"
        )
        cont_embed = PeriodicContEmbeddings(
            n_cont_cols=len(continuous_cols),
            embed_dim=cont_embed_dim,
            embed_dropout=0.0 if cont_embed_dropout is None else cont_embed_dropout,
            full_embed_dropout=(
                False if full_embed_dropout is None else full_embed_dropout
            ),
            n_frequencies=n_frequencies,
            sigma=sigma,
            share_last_layer=share_last_layer,
            activation_fn=cont_embed_activation,
        )
    else:
        raise ValueError(
            "embed_continuous_method must be one of 'standard', 'piecewise', 'periodic' or 'none'"
        )

    return cont_embed
