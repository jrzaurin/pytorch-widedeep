import torch
import einops
from torch import nn

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models.tab_mlp import MLP
from pytorch_widedeep.models.transformers._encoders import PerceiverEncoder
from pytorch_widedeep.models.transformers._embeddings_layers import (
    CatAndContEmbeddings,
)


class TabPerceiver(nn.Module):
    r"""Defines an adaptation of a ``Perceiver`` model
    (`arXiv:2103.03206 <https://arxiv.org/abs/2103.03206>`_) that can be used
    as the ``deeptabular`` component of a Wide & Deep model.

    Parameters
    ----------
    column_idx: Dict
        Dict containing the index of the columns that will be passed through
        the model. Required to slice the tensors. e.g.
        {'education': 0, 'relationship': 1, 'workclass': 2, ...}
    embed_input: List
        List of Tuples with the column name and number of unique values
        e.g. [('education', 11), ...]
    embed_dropout: float, default = 0.1
        Dropout to be applied to the embeddings matrix
    full_embed_dropout: bool, default = False
        Boolean indicating if an entire embedding (i.e. the representation of
        one column) will be dropped in the batch. See:
        :obj:`pytorch_widedeep.models.transformers._layers.FullEmbeddingDropout`.
        If ``full_embed_dropout = True``, ``embed_dropout`` is ignored.
    shared_embed: bool, default = False
        The idea behind ``shared_embed`` is described in the Appendix A in the
        `TabTransformer paper <https://arxiv.org/abs/2012.06678>`_: `'The
        goal of having column embedding is to enable the model to distinguish
        the classes in one column from those in the other columns'`. In other
        words, the idea is to let the model learn which column is embedded
        at the time.
    add_shared_embed: bool, default = False,
        The two embedding sharing strategies are: 1) add the shared embeddings to the column
        embeddings or 2) to replace the first ``frac_shared_embed`` with the shared
        embeddings. See :obj:`pytorch_widedeep.models.transformers._layers.SharedEmbeddings`
    frac_shared_embed: float, default = 0.25
        The fraction of embeddings that will be shared (if ``add_shared_embed
        = False``) by all the different categories for one particular
        column.
    continuous_cols: List, Optional, default = None
        List with the name of the numeric (aka continuous) columns
    embed_continuous_activation: str, default = None
        String indicating the activation function to be applied to the
        continuous embeddings, if any. ``tanh``, ``relu``, ``leaky_relu`` and
        ``gelu`` are supported.
    cont_norm_layer: str, default =  None,
        Type of normalization layer applied to the continuous features before
        they are embedded. Options are: ``layernorm``, ``batchnorm`` or
        ``None``.
    input_dim: int, default = 32
        The so-called *dimension of the model*. In general, is the number of
        embeddings used to encode the categorical and/or continuous columns.
    n_cross_attns: int, default = 1
        Number of times each perceiver block will cross attend to the input
        data (i.e. number of cross attention components per perceiver block).
        This should normally be 1. However, in the paper they describe some
        architectures (normally computer vision-related problems) where the
        Perceiver attends multiple times to the input array. Therefore, maybe
        multiple cross attention to the input array is also useful in some
        cases for tabular data
    n_cross_attn_heads: int, default = 4
        Number of attention heads for the cross attention component
    n_latents: int, default = 16
        Number of latents. This is the *N* parameter in the paper. As
        indicated in the paper, this number should be significantly lower
        than *M* (the number of columns in the dataset). Setting *N* closer
        to *M* defies the main purpose of the Perceiver, which is to overcome
        the transformer quadratic bottleneck
    latent_dim: int, default = 128
        Latent dimension.
    n_latent_heads: int, default = 4
        Number of attention heads per Latent Transformer
    n_latent_blocks: int, default = 4
        Number of transformer encoder blocks (normalised MHA + normalised FF)
        per Latent Transformer
    n_perceiver_blocks: int, default = 4
        Number of Perceiver blocks defined as [Cross Attention + Latent
        Transformer]
    share_weights: Boolean, default = False
        Boolean indicating if the weights will be shared between Perceiver
        blocks
    attn_dropout: float, default = 0.2
        Dropout that will be applied to the Multi-Head Attention layers
    ff_dropout: float, default = 0.1
        Dropout that will be applied to the FeedForward network
    transformer_activation: str, default = "gelu"
        Transformer Encoder activation function. ``tanh``, ``relu``,
        ``leaky_relu``, ``gelu``, ``geglu`` and ``reglu`` are supported
    mlp_hidden_dims: List, Optional, default = None
        MLP hidden dimensions. If not provided it will default to ``[l, 4*l,
        2*l]`` where ``l`` is the MLP input dimension
    mlp_activation: str, default = "relu"
        MLP activation function. ``tanh``, ``relu``, ``leaky_relu`` and
        ``gelu`` are supported
    mlp_dropout: float, default = 0.1
        Dropout that will be applied to the final MLP
    mlp_batchnorm: bool, default = False
        Boolean indicating whether or not to apply batch normalization to the
        dense layers
    mlp_batchnorm_last: bool, default = False
        Boolean indicating whether or not to apply batch normalization to the
        last of the dense layers
    mlp_linear_first: bool, default = False
        Boolean indicating whether the order of the operations in the dense
        layer. If ``True: [LIN -> ACT -> BN -> DP]``. If ``False: [BN -> DP ->
        LIN -> ACT]``

    Attributes
    ----------
    cat_and_cont_embed: ``nn.Module``
        This is the module that processes the categorical and continuous columns
    perceiver_blks: ``nn.ModuleDict``
        ModuleDict with the Perceiver blocks
    latents: ``nn.Parameter``
        Latents that will be used for prediction
    perceiver_mlp: ``nn.Module``
        MLP component in the model
    output_dim: int
        The output dimension of the model. This is a required attribute
        neccesary to build the WideDeep class

    Example
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import TabPerceiver
    >>> X_tab = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
    >>> colnames = ['a', 'b', 'c', 'd', 'e']
    >>> embed_input = [(u,i) for u,i in zip(colnames[:4], [4]*4)]
    >>> continuous_cols = ['e']
    >>> column_idx = {k:v for v,k in enumerate(colnames)}
    >>> model = TabPerceiver(column_idx=column_idx, embed_input=embed_input,
    ... continuous_cols=continuous_cols, n_latents=2, latent_dim=16,
    ... n_perceiver_blocks=2)
    >>> out = model(X_tab)
    """

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
        share_weights: bool = False,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        transformer_activation: str = "geglu",
        mlp_hidden_dims: Optional[List[int]] = None,
        mlp_activation: str = "relu",
        mlp_dropout: float = 0.1,
        mlp_batchnorm: bool = False,
        mlp_batchnorm_last: bool = False,
        mlp_linear_first: bool = True,
    ):
        super(TabPerceiver, self).__init__()

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
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
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

        # This should be named 'cat_and_cont_embed' since the continuous cols
        # will always be embedded for the TabPerceiver. However is very
        # convenient for other funcionalities to name
        # it 'cat_and_cont_embed'
        self.cat_and_cont_embed = CatAndContEmbeddings(
            input_dim,
            column_idx,
            embed_input,
            embed_dropout,
            full_embed_dropout,
            shared_embed,
            add_shared_embed,
            frac_shared_embed,
            False,  # use_embed_bias
            continuous_cols,
            True,  # embed_continuous,
            embed_continuous_activation,
            True,  # use_cont_bias
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
            self.mlp_hidden_dims = [latent_dim, latent_dim * 4, latent_dim * 2]
        else:
            assert mlp_hidden_dims[0] == latent_dim, (
                f"The input dim of the MLP must be {latent_dim}. "
                f"Got {mlp_hidden_dims[0]} instead"
            )
        self.perceiver_mlp = MLP(
            self.mlp_hidden_dims,
            mlp_activation,
            mlp_dropout,
            mlp_batchnorm,
            mlp_batchnorm_last,
            mlp_linear_first,
        )

        # the output_dim attribute will be used as input_dim when "merging" the models
        self.output_dim = self.mlp_hidden_dims[-1]

    def forward(self, X: Tensor) -> Tensor:

        x_cat, x_cont = self.cat_and_cont_embed(X)
        if x_cat is not None:
            x_emb = x_cat
        if x_cont is not None:
            x_emb = torch.cat([x_emb, x_cont], 1) if x_cat is not None else x_cont

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
    def attention_weights(self) -> List:
        r"""List with the attention weights. If the weights are not shared
        between perceiver blocks each element of the list will be a list
        itself containing the Cross Attention and Latent Transformer
        attention weights respectively

        The shape of the attention weights is:

            - Cross Attention: :math:`(N, C, L, F)`
            - Latent Attention: :math:`(N, T, L, L)`

        WHere *N* is the batch size, *C* is the number of Cross Attention
        heads, *L* is the number of Latents, *F* is the number of
        features/columns in the dataset and *T* is the number of Latent
        Attention heads
        """
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
                PerceiverEncoder(
                    self.input_dim,
                    self.n_cross_attn_heads,
                    False,  # use_bias
                    self.attn_dropout,
                    self.ff_dropout,
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
                PerceiverEncoder(
                    self.latent_dim,  # input_dim
                    self.n_latent_heads,
                    False,  # use_bias
                    self.attn_dropout,
                    self.ff_dropout,
                    self.transformer_activation,
                ),
            )
        perceiver_block["latent_transformer"] = latent_transformer

        return perceiver_block

    @staticmethod
    def _extract_attn_weights(cross_attns, latent_transformer) -> List:
        attention_weights = []
        for cross_attn in cross_attns:
            attention_weights.append(cross_attn.attn.attn_weights)
        for latent_block in latent_transformer:
            attention_weights.append(latent_block.attn.attn_weights)
        return attention_weights
