from torch import nn

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models._get_activation_fn import get_activation_fn
from pytorch_widedeep.models.tabular.embeddings_layers import (
    DiffSizeCatAndContEmbeddings,
    SameSizeCatAndContEmbeddings,
)


class BaseTabularModelWithoutAttention(nn.Module):
    def __init__(
        self,
        column_idx: Dict[str, int],
        cat_embed_input: Optional[List[Tuple[str, int, int]]],
        cat_embed_dropout: float,
        use_cat_bias: bool,
        cat_embed_activation: Optional[str],
        continuous_cols: Optional[List[str]],
        cont_norm_layer: str,
        embed_continuous: bool,
        cont_embed_dim: int,
        cont_embed_dropout: float,
        use_cont_bias: bool,
        cont_embed_activation: Optional[str],
    ):
        super().__init__()

        self.column_idx = column_idx
        self.cat_embed_input = cat_embed_input
        self.cat_embed_dropout = cat_embed_dropout
        self.use_cat_bias = use_cat_bias
        self.cat_embed_activation = cat_embed_activation

        self.continuous_cols = continuous_cols
        self.cont_norm_layer = cont_norm_layer
        self.embed_continuous = embed_continuous
        self.cont_embed_dim = cont_embed_dim
        self.cont_embed_dropout = cont_embed_dropout
        self.use_cont_bias = use_cont_bias
        self.cont_embed_activation = cont_embed_activation

        self.cat_and_cont_embed = DiffSizeCatAndContEmbeddings(
            column_idx,
            cat_embed_input,
            cat_embed_dropout,
            use_cat_bias,
            continuous_cols,
            cont_norm_layer,
            embed_continuous,
            cont_embed_dim,
            cont_embed_dropout,
            use_cont_bias,
        )
        self.cat_embed_act_fn = (
            get_activation_fn(cat_embed_activation)
            if cat_embed_activation is not None
            else None
        )
        self.cont_embed_act_fn = (
            get_activation_fn(cont_embed_activation)
            if cont_embed_activation is not None
            else None
        )

    def _get_embeddings(self, X: Tensor) -> Tensor:
        x_cat, x_cont = self.cat_and_cont_embed(X)
        if x_cat is not None:
            x = (
                self.cat_embed_act_fn(x_cat)
                if self.cat_embed_act_fn is not None
                else x_cat
            )
        if x_cont is not None:
            if self.cont_embed_act_fn is not None:
                x_cont = self.cont_embed_act_fn(x_cont)
            x = torch.cat([x, x_cont], 1) if x_cat is not None else x_cont
        return x


class BaseTabularModelWithAttention(nn.Module):
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
        embed_continuous: bool,
        cont_embed_dropout: float,
        use_cont_bias: bool,
        cont_embed_activation: Optional[str],
        input_dim: int,
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
        self.embed_continuous = embed_continuous
        self.cont_embed_dropout = cont_embed_dropout
        self.use_cont_bias = use_cont_bias
        self.cont_embed_activation = cont_embed_activation

        self.input_dim = input_dim

        self.cat_and_cont_embed = SameSizeCatAndContEmbeddings(
            input_dim,
            column_idx,
            cat_embed_input,
            cat_embed_dropout,
            use_cat_bias,
            full_embed_dropout,
            shared_embed,
            add_shared_embed,
            frac_shared_embed,
            continuous_cols,
            cont_norm_layer,
            embed_continuous,
            cont_embed_dropout,
            use_cont_bias,
        )
        self.cat_embed_act_fn = (
            get_activation_fn(cat_embed_activation)
            if cat_embed_activation is not None
            else None
        )
        self.cont_embed_act_fn = (
            get_activation_fn(cont_embed_activation)
            if cont_embed_activation is not None
            else None
        )

    def _get_embeddings(self, X: Tensor) -> Tensor:
        x_cat, x_cont = self.cat_and_cont_embed(X)
        if x_cat is not None:
            x = (
                self.cat_embed_act_fn(x_cat)
                if self.cat_embed_act_fn is not None
                else x_cat
            )
        if x_cont is not None:
            if self.cont_embed_act_fn is not None:
                x_cont = self.cont_embed_act_fn(x_cont)
            x = torch.cat([x, x_cont], 1) if x_cat is not None else x_cont
        return x

    @property
    def attention_weights(self):
        raise NotImplementedError
