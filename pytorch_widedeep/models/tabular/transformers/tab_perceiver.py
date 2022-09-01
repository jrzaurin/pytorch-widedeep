import torch
import einops
from torch import nn

from pytorch_widedeep.wdtypes import Dict, List, Tuple, Tensor, Optional
from pytorch_widedeep.models.tabular.mlp._layers import MLP
from pytorch_widedeep.models.tabular._base_tabular_model import (
    BaseTabularModelWithAttention,
)
from pytorch_widedeep.models.tabular.transformers._encoders import (
    PerceiverEncoder,
)


class TabPerceiver(BaseTabularModelWithAttention):
    r"""Defines an adaptation of a [Perceiver](https://arxiv.org/abs/2103.03206)
     that can be used as the `deeptabular` component of a Wide & Deep model
     or independently by itself.

    :information_source: **NOTE**: while there are scientific publications for
     the `TabTransformer`, `SAINT` and `FTTransformer`, the `TabPerceiver`
     and the `TabFastFormer` are our own adaptations of the
     [Perceiver](https://arxiv.org/abs/2103.03206) and the
     [FastFormer](https://arxiv.org/abs/2108.09084) for tabular data.

    Parameters
    ----------
    column_idx: Dict
        Dict containing the index of the columns that will be passed through
        the model. Required to slice the tensors. e.g.
        _{'education': 0, 'relationship': 1, 'workclass': 2, ...}_
    cat_embed_input: List, Optional, default = None
        List of Tuples with the column name and number of unique values for
        each categorical component e.g. _[(education, 11), ...]_
    cat_embed_dropout: float, default = 0.1
        Categorical embeddings dropout
    use_cat_bias: bool, default = False,
        Boolean indicating if bias will be used for the categorical embeddings
    cat_embed_activation: Optional, str, default = None,
        Activation function for the categorical embeddings, if any. _'tanh'_,
        _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported.
    full_embed_dropout: bool, default = False
        Boolean indicating if an entire embedding (i.e. the representation of
        one column) will be dropped in the batch. See:
        `pytorch_widedeep.models.transformers._layers.FullEmbeddingDropout`.
        If `full_embed_dropout = True`, `cat_embed_dropout` is ignored.
    shared_embed: bool, default = False
        The idea behind `shared_embed` is described in the Appendix A in the
        [TabTransformer paper](https://arxiv.org/abs/2012.06678): the
        goal of having column embedding is to enable the model to distinguish
        the classes in one column from those in the other columns. In other
        words, the idea is to let the model learn which column is embedded
        at the time.
    add_shared_embed: bool, default = False,
        The two embedding sharing strategies are: 1) add the shared embeddings
        to the column embeddings or 2) to replace the first
        `frac_shared_embed` with the shared embeddings.
        See `pytorch_widedeep.models.transformers._layers.SharedEmbeddings`
    frac_shared_embed: float, default = 0.25
        The fraction of embeddings that will be shared (if `add_shared_embed
        = False`) by all the different categories for one particular
        column.
    continuous_cols: List, Optional, default = None
        List with the name of the numeric (aka continuous) columns
    cont_norm_layer: str, default =  "batchnorm"
        Type of normalization layer applied to the continuous features. Options
        are: _'layernorm'_, _'batchnorm'_ or None.
    cont_embed_dropout: float, default = 0.1,
        Continuous embeddings dropout
    use_cont_bias: bool, default = True,
        Boolean indicating if bias will be used for the continuous embeddings
    cont_embed_activation: str, default = None
        Activation function to be applied to the continuous embeddings, if
        any. _'tanh'_, _'relu'_, _'leaky_relu'_ and _'gelu'_ are supported.
    input_dim: int, default = 32
        The so-called *dimension of the model*. Is the number of embeddings
        used to encode the categorical and/or continuous columns.
    n_cross_attns: int, default = 1
        Number of times each perceiver block will cross attend to the input
        data (i.e. number of cross attention components per perceiver block).
        This should normally be 1. However, in the paper they describe some
        architectures (normally computer vision-related problems) where the
        Perceiver attends multiple times to the input array. Therefore, maybe
        multiple cross attention to the input array is also useful in some
        cases for tabular data :shrug: .
    n_cross_attn_heads: int, default = 4
        Number of attention heads for the cross attention component
    n_latents: int, default = 16
        Number of latents. This is the $N$ parameter in the paper. As
        indicated in the paper, this number should be significantly lower
        than $M$ (the number of columns in the dataset). Setting $N$ closer
        to $M$ defies the main purpose of the Perceiver, which is to overcome
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
        Transformer Encoder activation function. _'tanh'_, _'relu'_,
        _'leaky_relu'_, _'gelu'_, _'geglu'_ and _'reglu'_ are supported
    mlp_hidden_dims: List, Optional, default = None
        MLP hidden dimensions. If not provided it will default to $[l, 4
        \times l, 2 \times l]$ where $l$ is the MLP's input dimension
    mlp_activation: str, default = "relu"
        MLP activation function. _'tanh'_, _'relu'_, _'leaky_relu'_ and
        _'gelu'_ are supported
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
        layer. If `True: [LIN -> ACT -> BN -> DP]`. If `False: [BN -> DP ->
        LIN -> ACT]`

    Attributes
    ----------
    cat_and_cont_embed: nn.Module
        This is the module that processes the categorical and continuous columns
    encoder: nn.ModuleDict
        ModuleDict with the Perceiver blocks
    latents: nn.Parameter
        Latents that will be used for prediction
    mlp: nn.Module
        MLP component in the model

    Examples
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import TabPerceiver
    >>> X_tab = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
    >>> colnames = ['a', 'b', 'c', 'd', 'e']
    >>> cat_embed_input = [(u,i) for u,i in zip(colnames[:4], [4]*4)]
    >>> continuous_cols = ['e']
    >>> column_idx = {k:v for v,k in enumerate(colnames)}
    >>> model = TabPerceiver(column_idx=column_idx, cat_embed_input=cat_embed_input,
    ... continuous_cols=continuous_cols, n_latents=2, latent_dim=16,
    ... n_perceiver_blocks=2)
    >>> out = model(X_tab)
    """

    def __init__(
        self,
        column_idx: Dict[str, int],
        cat_embed_input: Optional[List[Tuple[str, int]]] = None,
        cat_embed_dropout: float = 0.1,
        use_cat_bias: bool = False,
        cat_embed_activation: Optional[str] = None,
        full_embed_dropout: bool = False,
        shared_embed: bool = False,
        add_shared_embed: bool = False,
        frac_shared_embed: float = 0.25,
        continuous_cols: Optional[List[str]] = None,
        cont_norm_layer: str = None,
        cont_embed_dropout: float = 0.1,
        use_cont_bias: bool = True,
        cont_embed_activation: Optional[str] = None,
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
        super(TabPerceiver, self).__init__(
            column_idx=column_idx,
            cat_embed_input=cat_embed_input,
            cat_embed_dropout=cat_embed_dropout,
            use_cat_bias=use_cat_bias,
            cat_embed_activation=cat_embed_activation,
            full_embed_dropout=full_embed_dropout,
            shared_embed=shared_embed,
            add_shared_embed=add_shared_embed,
            frac_shared_embed=frac_shared_embed,
            continuous_cols=continuous_cols,
            cont_norm_layer=cont_norm_layer,
            embed_continuous=True,
            cont_embed_dropout=cont_embed_dropout,
            use_cont_bias=use_cont_bias,
            cont_embed_activation=cont_embed_activation,
            input_dim=input_dim,
        )

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
        self.mlp_dropout = mlp_dropout
        self.mlp_batchnorm = mlp_batchnorm
        self.mlp_batchnorm_last = mlp_batchnorm_last
        self.mlp_linear_first = mlp_linear_first

        # Embeddings are instantiated at the base model
        # Transformer blocks
        self.latents = nn.init.trunc_normal_(
            nn.Parameter(torch.empty(n_latents, latent_dim))
        )

        self.encoder = nn.ModuleDict()
        first_perceiver_block = self._build_perceiver_block()
        self.encoder["perceiver_block0"] = first_perceiver_block

        if share_weights:
            for n in range(1, n_perceiver_blocks):
                self.encoder["perceiver_block" + str(n)] = first_perceiver_block
        else:
            for n in range(1, n_perceiver_blocks):
                self.encoder["perceiver_block" + str(n)] = self._build_perceiver_block()

        self.mlp_first_hidden_dim = self.latent_dim

        # Mlp
        if mlp_hidden_dims is not None:
            self.mlp = MLP(
                [self.mlp_first_hidden_dim] + mlp_hidden_dims,
                mlp_activation,
                mlp_dropout,
                mlp_batchnorm,
                mlp_batchnorm_last,
                mlp_linear_first,
            )
        else:
            self.mlp = None

    def forward(self, X: Tensor) -> Tensor:

        x_emb = self._get_embeddings(X)

        x = einops.repeat(self.latents, "n d -> b n d", b=X.shape[0])

        for n in range(self.n_perceiver_blocks):
            cross_attns = self.encoder["perceiver_block" + str(n)]["cross_attns"]
            latent_transformer = self.encoder["perceiver_block" + str(n)][
                "latent_transformer"
            ]
            for cross_attn in cross_attns:
                x = cross_attn(x, x_emb)
            x = latent_transformer(x)

        # average along the latent index axis
        x = x.mean(dim=1)

        if self.mlp is not None:
            x = self.mlp(x)

        return x

    @property
    def output_dim(self) -> int:
        r"""The output dimension of the model. This is a required property
        neccesary to build the `WideDeep` class
        """
        return (
            self.mlp_hidden_dims[-1]
            if self.mlp_hidden_dims is not None
            else self.mlp_first_hidden_dim
        )

    @property
    def attention_weights(self) -> List:
        r"""List with the attention weights. If the weights are not shared
        between perceiver blocks each element of the list will be a list
        itself containing the Cross Attention and Latent Transformer
        attention weights respectively

        The shape of the attention weights is:

        - Cross Attention: $(N, C, L, F)$

        - Latent Attention: $(N, T, L, L)$

        WHere $N$ is the batch size, $C$ is the number of Cross Attention
        heads, $L$ is the number of Latents, $F$ is the number of
        features/columns in the dataset and $T$ is the number of Latent
        Attention heads
        """
        if self.share_weights:
            cross_attns = self.encoder["perceiver_block0"]["cross_attns"]
            latent_transformer = self.encoder["perceiver_block0"]["latent_transformer"]
            attention_weights = self._extract_attn_weights(
                cross_attns, latent_transformer
            )
        else:
            attention_weights = []
            for n in range(self.n_perceiver_blocks):
                cross_attns = self.encoder["perceiver_block" + str(n)]["cross_attns"]
                latent_transformer = self.encoder["perceiver_block" + str(n)][
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
