import torch
from torch import nn

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models.tab_mlp import MLP
from pytorch_widedeep.models.transformers.layers import (
    SaintEncoder,
    SharedEmbeddings,
    TransformerEncoder,
    FullEmbeddingDropout,
)


class TabTransformer(nn.Module):
    def __init__(
        self,
        column_idx: Dict[str, int],
        embed_input: List[Tuple[str, int]],
        continuous_cols: Optional[List[str]] = None,
        embed_dropout: float = 0.1,
        full_embed_dropout: bool = False,
        shared_embed: bool = False,
        add_shared_embed: bool = False,
        frac_shared_embed: int = 8,
        input_dim: int = 32,
        n_heads: int = 8,
        n_blocks: int = 6,
        dropout: float = 0.1,
        keep_attn_weights: bool = False,
        ff_hidden_dim: int = 32 * 4,
        transformer_activation: str = "gelu",
        with_special_token: bool = False,
        embed_continuous: bool = False,
        mlp_hidden_dims: Optional[List[int]] = None,
        mlp_activation: str = "relu",
        mlp_batchnorm: bool = False,
        mlp_batchnorm_last: bool = False,
        mlp_linear_first: bool = True,
    ):
        super(TabTransformer, self).__init__()

        self.column_idx = column_idx
        self.embed_input = embed_input
        self.continuous_cols = continuous_cols
        self.embed_dropout = embed_dropout
        self.full_embed_dropout = full_embed_dropout
        self.shared_embed = shared_embed
        self.add_shared_embed = add_shared_embed
        self.frac_shared_embed = frac_shared_embed
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.keep_attn_weights = keep_attn_weights
        self.ff_hidden_dim = ff_hidden_dim
        self.transformer_activation = transformer_activation
        self.with_special_token = with_special_token
        self.embed_continuous = embed_continuous
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.mlp_linear_first = mlp_linear_first

        # Categorical: val + 1 because 0 is reserved for padding/unseen cateogories.
        if shared_embed:
            self.cat_embed_layers = nn.ModuleDict(
                {
                    "emb_layer_"
                    + col: SharedEmbeddings(
                        val + 1,
                        input_dim,
                        embed_dropout,
                        full_embed_dropout,
                        add_shared_embed,
                        frac_shared_embed,
                    )
                    for col, val in self.embed_input
                }
            )
        else:
            self.cat_embed_layers = nn.ModuleDict(
                {
                    "emb_layer_" + col: nn.Embedding(val + 1, input_dim, padding_idx=0)
                    for col, val in self.embed_input
                }
            )
            if full_embed_dropout:
                self.embedding_dropout = FullEmbeddingDropout(embed_dropout)
            else:
                self.embedding_dropout = nn.Dropout(embed_dropout)  # type: ignore[assignment]

        # Continuous
        if self.continuous_cols is not None:
            self.cont_norm = nn.LayerNorm(len(continuous_cols))
            if self.embed_continuous:
                self.cont_embed_layers = nn.ModuleDict(
                    {
                        "emb_layer_"
                        + col: nn.Sequential(nn.Linear(1, input_dim), nn.ReLU())
                        for col in self.continuous_cols
                    }
                )

        self.transformer_blks = nn.Sequential()
        for i in range(n_blocks):
            self.transformer_blks.add_module(
                "block" + str(i),
                TransformerEncoder(
                    input_dim,
                    n_heads,
                    keep_attn_weights,
                    ff_hidden_dim,
                    dropout,
                    transformer_activation,
                ),
            )

        if keep_attn_weights:
            self.attention_weights: List[Any] = [None] * n_blocks

        if not mlp_hidden_dims:
            mlp_hidden_dims = self._set_mlp_hidden_dims()

        self.transformer_mlp = MLP(
            mlp_hidden_dims,
            mlp_activation,
            dropout,
            mlp_batchnorm,
            mlp_batchnorm_last,
            mlp_linear_first,
        )

        # the output_dim attribute will be used as input_dim when "merging" the models
        self.output_dim = mlp_hidden_dims[-1]

    def forward(self, X: Tensor) -> Tensor:

        cat_embed = [
            self.cat_embed_layers["emb_layer_" + col](
                X[:, self.column_idx[col]].long()
            ).unsqueeze(1)
            for col, _ in self.embed_input
        ]
        x = torch.cat(cat_embed, 1)
        if not self.shared_embed and self.embedding_dropout is not None:
            x = self.embedding_dropout(x)

        if self.continuous_cols is not None and self.embed_continuous:
            cont_embed = [
                self.cont_embed_layers["emb_layer_" + col](
                    X[:, self.column_idx[col]].float().unsqueeze(1)
                ).unsqueeze(1)
                for col in self.continuous_cols
            ]
            x_cont = torch.cat(cont_embed, 1)
            x = torch.cat([x, x_cont], 1)

        for i, blk in enumerate(self.transformer_blks):
            x = blk(x)
            if self.keep_attn_weights:
                if hasattr(blk, "inter_sample_attn"):
                    self.attention_weights[i] = (
                        blk.self_attn.attn_weights,
                        blk.inter_sample_attn.attn_weights,
                    )
                else:
                    self.attention_weights[i] = blk.self_attn.attn_weight

        if self.with_special_token:
            x = x[:, 0, :]
        else:
            x = x.flatten(1)

        if self.continuous_cols is not None and not self.embed_continuous:
            cont_idx = [self.column_idx[col] for col in self.continuous_cols]
            x_cont = self.cont_norm((X[:, cont_idx].float()))
            x = torch.cat([x, x_cont], 1)

        return self.transformer_mlp(x)

    def _set_mlp_hidden_dims(self) -> List[int]:
        if self.with_special_token:
            if self.embed_continuous:
                mlp_hidden_dims = [
                    self.input_dim,
                    self.input_dim * 4,
                    self.input_dim * 2,
                ]
            else:
                mlp_inp_l = self.input_dim + len(self.continuous_cols)
                mlp_hidden_dims = [mlp_inp_l, mlp_inp_l * 4, mlp_inp_l * 2]
        elif self.embed_continuous:
            mlp_inp_l = (
                len(self.embed_input) + len(self.continuous_cols)
            ) * self.input_dim
            mlp_hidden_dims = [mlp_inp_l, mlp_inp_l * 4, mlp_inp_l * 2]
        else:
            mlp_inp_l = len(self.embed_input) * self.input_dim + len(
                self.continuous_cols
            )
            mlp_hidden_dims = [mlp_inp_l, mlp_inp_l * 4, mlp_inp_l * 2]
        return mlp_hidden_dims


class SAINT(TabTransformer):
    def __init__(
        self,
        column_idx: Dict[str, int],
        embed_input: List[Tuple[str, int]],
        continuous_cols: Optional[List[str]] = None,
        embed_dropout: float = 0.1,
        full_embed_dropout: bool = False,
        shared_embed: bool = False,
        add_shared_embed: bool = False,
        frac_shared_embed: int = 8,
        input_dim: int = 32,
        n_heads: int = 8,
        n_blocks: int = 6,
        dropout: float = 0.1,
        keep_attn_weights: bool = False,
        ff_hidden_dim: int = 32 * 4,
        transformer_activation: str = "gelu",
        with_special_token: bool = False,
        embed_continuous: bool = False,
        mlp_hidden_dims: Optional[List[int]] = None,
        mlp_activation: str = "relu",
        mlp_batchnorm: bool = False,
        mlp_batchnorm_last: bool = False,
        mlp_linear_first: bool = True,
    ):
        super().__init__(
            column_idx,
            embed_input,
            continuous_cols,
            embed_dropout,
            full_embed_dropout,
            shared_embed,
            add_shared_embed,
            frac_shared_embed,
            input_dim,
            n_heads,
            n_blocks,
            dropout,
            keep_attn_weights,
            ff_hidden_dim,
            transformer_activation,
            with_special_token,
            embed_continuous,
            mlp_hidden_dims,
            mlp_activation,
            mlp_batchnorm,
            mlp_batchnorm_last,
            mlp_linear_first,
        )

        if embed_continuous:
            n_feats = len(embed_input) + len(continuous_cols)
        else:
            n_feats = len(embed_input)
        self.transformer_blks = nn.Sequential()
        for i in range(n_blocks):
            self.transformer_blks.add_module(
                "block" + str(i),
                SaintEncoder(
                    input_dim,
                    n_heads,
                    keep_attn_weights,
                    ff_hidden_dim,
                    dropout,
                    transformer_activation,
                    n_feats,
                ),
            )
