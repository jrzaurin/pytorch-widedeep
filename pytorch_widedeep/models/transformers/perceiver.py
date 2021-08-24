import torch
import einops
from torch import nn

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models.tab_mlp import MLP
from pytorch_widedeep.models.transformers.layers import (
    TransformerEncoder,
    CatAndContEmbeddings,
)


class Perceiver(nn.Module):
    def __init__(
        self,
        column_idx: Dict[str, int],
        embed_input: Optional[List[Tuple[str, int]]] = None,
        embed_dropout: float = 0.1,
        full_embed_dropout: bool = False,
        shared_embed: bool = False,
        add_shared_embed: bool = False,
        frac_shared_embed: float = 0.25,
        continuous_cols: Optional[List[str]] = None,
        embed_continuous_activation: str = None,
        cont_norm_layer: str = None,
        input_dim: int = 32,
        n_cross_attns: int = 1,
        n_cross_attn_heads: int = 4,
        n_latents: int = 16,
        latent_dim: int = 128,
        n_latent_heads: int = 4,
        n_latent_blocks: int = 4,
        n_perceiver_blocks: int = 4,
        share_weights: bool = True,
        dropout: float = 0.1,
        transformer_activation: str = "geglu",
        mlp_hidden_dims: Optional[List[int]] = None,
        mlp_activation: str = "relu",
        mlp_batchnorm: bool = False,
        mlp_batchnorm_last: bool = False,
        mlp_linear_first: bool = True,
    ):
        super(Perceiver, self).__init__()

        self.column_idx = column_idx
        self.embed_input = embed_input
        self.embed_dropout = embed_dropout
        self.full_embed_dropout = full_embed_dropout
        self.shared_embed = shared_embed
        self.add_shared_embed = add_shared_embed
        self.frac_shared_embed = frac_shared_embed
        self.continuous_cols = continuous_cols
        self.embed_continuous_activation = embed_continuous_activation
        self.cont_norm_layer = cont_norm_layer
        self.input_dim = input_dim
        self.n_cross_attns = n_cross_attns
        self.n_cross_attn_heads = n_cross_attn_heads
        self.n_latents = n_latents
        self.latent_dim = latent_dim
        self.n_latent_heads = n_latent_heads
        self.n_latent_blocks = n_latent_blocks
        self.n_perceiver_blocks = n_perceiver_blocks
        self.share_weights = share_weights
        self.dropout = dropout
        self.transformer_activation = transformer_activation
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.mlp_linear_first = mlp_linear_first

        if mlp_hidden_dims is not None:
            assert (
                mlp_hidden_dims[0] == latent_dim
            ), "The first mlp input dim must be equal to 'latent_dim'"

        self.cat_embed_and_cont = CatAndContEmbeddings(
            input_dim,
            column_idx,
            embed_input,
            embed_dropout,
            full_embed_dropout,
            shared_embed,
            add_shared_embed,
            frac_shared_embed,
            continuous_cols,
            True,  # embed_continuous,
            embed_continuous_activation,
            cont_norm_layer,
        )

        self.latents = nn.init.trunc_normal_(
            nn.Parameter(torch.empty(n_latents, latent_dim))
        )

        self.perceiver_blks = nn.ModuleDict()
        first_perceiver_block = self._build_perceiver_block()
        self.perceiver_blks["perceiver_block0"] = first_perceiver_block

        if share_weights:
            for n in range(1, n_perceiver_blocks):
                self.perceiver_blks["perceiver_block" + str(n)] = first_perceiver_block
        else:
            for n in range(1, n_perceiver_blocks):
                self.perceiver_blks[
                    "perceiver_block" + str(n)
                ] = self._build_perceiver_block()

        if not mlp_hidden_dims:
            mlp_hidden_dims = [latent_dim, latent_dim * 4, latent_dim * 2]
        self.perceiver_mlp = MLP(
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

        x_cat, x_cont = self.cat_embed_and_cont(X)
        x_emb = torch.cat([x_cat, x_cont], 1)

        x = einops.repeat(self.latents, "n d -> b n d", b=X.shape[0])

        for n in range(self.n_perceiver_blocks):
            cross_attns = self.perceiver_blks["perceiver_block" + str(n)]["cross_attns"]
            latent_transformer = self.perceiver_blks["perceiver_block" + str(n)][
                "latent_transformer"
            ]
            for cross_attn in cross_attns:
                x = cross_attn(x, x_emb)
            x = latent_transformer(x)

        # average along the latent index axis
        x = x.mean(dim=1)

        return self.perceiver_mlp(x)

    @property
    def attention_weights(self):
        if self.share_weights:
            cross_attns = self.perceiver_blks["perceiver_block0"]["cross_attns"]
            latent_transformer = self.perceiver_blks["perceiver_block0"][
                "latent_transformer"
            ]
            attention_weights = self._extract_attn_weights(
                cross_attns, latent_transformer
            )
        else:
            attention_weights = []
            for n in range(self.n_perceiver_blocks):
                cross_attns = self.perceiver_blks["perceiver_block" + str(n)][
                    "cross_attns"
                ]
                latent_transformer = self.perceiver_blks["perceiver_block" + str(n)][
                    "latent_transformer"
                ]
                attention_weights.append(
                    self._extract_attn_weights(cross_attns, latent_transformer)
                )
        return attention_weights

    def _build_perceiver_block(self) -> nn.ModuleDict:

        perceiver_block = nn.ModuleDict()

        # Cross Attention
        cross_attns = nn.ModuleList()
        for _ in range(self.n_cross_attns):
            cross_attns.append(
                TransformerEncoder(
                    self.input_dim,
                    self.n_cross_attn_heads,
                    False,  # use_bias
                    self.dropout,
                    self.transformer_activation,
                    self.latent_dim,  # q_dim,
                ),
            )
        perceiver_block["cross_attns"] = cross_attns

        # Latent Transformer
        latent_transformer = nn.Sequential()
        for i in range(self.n_latent_blocks):
            latent_transformer.add_module(
                "latent_block" + str(i),
                TransformerEncoder(
                    self.latent_dim,  # input_dim
                    self.n_latent_heads,
                    False,  # use_bias
                    self.dropout,
                    self.transformer_activation,
                ),
            )
        perceiver_block["latent_transformer"] = latent_transformer

        return perceiver_block

    @staticmethod
    def _extract_attn_weights(cross_attns, latent_transformer):
        attention_weights = []
        for cross_attn in cross_attns:
            attention_weights.append(cross_attn.attn.attn_weights)
        for latent_block in latent_transformer:
            attention_weights.append(latent_block.attn.attn_weights)
        return attention_weights
