from torch import nn

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models.transformers.layers import SaintEncoder
from pytorch_widedeep.models.transformers.tab_transformer import TabTransformer


class SAINT(TabTransformer):
    r"""Adaptation of SAINT model
    (https://arxiv.org/abs/2106.01342) model that can be used as the
    deeptabular component of a Wide & Deep model.

    Parameters for this model are identical to those of the ``TabTransformer``

    Parameters
    ----------
    column_idx: Dict
        Dict containing the index of the columns that will be passed through
        the DeepDense model. Required to slice the tensors. e.g. {'education':
        0, 'relationship': 1, 'workclass': 2, ...}
    embed_input: List
        List of Tuples with the column name and number of unique values
        e.g. [(education, 11), ...]
    embed_dropout: float, default = 0.1
        Dropout to be applied to the embeddings matrix
    full_embed_dropout: bool, default = False
        Boolean indicating if an entire embedding (i.e. the representation of
        one column) will be dropped in the batch. See:
        :obj:`pytorch_widedeep.models.transformers.layers.FullEmbeddingDropout`.
        If ``full_embed_dropout = True``, ``embed_dropout`` is ignored.
    shared_embed: bool, default = False
        The idea behind ``shared_embed`` is described in the Appendix A in the paper:
        `'The goal of having column embedding is to enable the model to distinguish the
        classes in one column from those in the other columns'`. In other words, the idea
        is to let the model learn which column is embedding at the time.
    add_shared_embed: bool, default = False,
        The two embedding sharing strategies are: 1) add the shared embeddings to the column
        embeddings or 2) to replace the first ``frac_shared_embed`` with the shared
        embeddings. See :obj:`pytorch_widedeep.models.transformers.layers.SharedEmbeddings`
    frac_shared_embed: float, default = 0.25
        The fraction of embeddings that will be shared by all the different categories for
        one particular column.
    continuous_cols: List, Optional, default = None
        List with the name of the numeric (aka continuous) columns
    embed_continuous: bool, default = False,
        Boolean indicating if the continuous features will be "embedded". See
        ``pytorch_widedeep.models.transformers.layers.ContinuousEmbeddings``
    embed_continuous_activation: str, default = "relu"
        String indicating the activation function to be applied to the
        continuous embeddings, if any.
        'relu', 'leaky_relu' and 'gelu' are supported.
    cont_norm_layer: str, default =  "layernorm",
        Type of normalization layer applied to the continuous features if they
        are not embedded. Options are: 'layernorm' or 'batchnorm'.
    input_dim: int, default = 32
        The so-called *dimension of the model*. Is the number of embeddings used to encode
        the categorical columns
    n_heads: int, default = 8
        Number of attention heads per Transformer block
    n_blocks: int, default = 6
        Number of Transformer blocks
    dropout: float, default = 0.1
        Dropout that will be applied internally to the ``TransformerEncoder``
        (see :obj:`pytorch_widedeep.models.transformers.layers.TransformerEncoder`)
        and the output MLP
    keep_attn_weights: bool, default = False
        If set to ``True`` the model will store the attention weights in the ``attention_weights``
        attribute.
    ff_hidden_dim: int, default = 128
        Hidden dimension of the ``FeedForward`` Layer. See
        :obj:`pytorch_widedeep.models.transformers.layers.FeedForward`.
    transformer_activation: str, default = "gelu"
        Transformer Encoder activation function. 'relu', 'leaky_relu', 'gelu'
        and 'geglu' are supported
    mlp_hidden_dims: List, Optional, default = None
        MLP hidden dimensions. If not provided it will default to ``[4*l,
        2*l]`` where ``l`` is the mlp input dimension
    mlp_activation: str, default = "relu"
        MLP activation function. 'relu', 'leaky_relu' and 'gelu' are supported
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
    cat_embed_layers: ``nn.ModuleDict``
        Dict with the embeddings per column
    cont_embed: ``nn.ModuleDict``
        Dict with the embeddings per column if ``embed_continuous=True``
    cont_norm_layer: NormLayers
        Continuous normalization layer if ``continuous_cols`` is not None
    transformer_blks: ``nn.Sequential``
        Sequence of Transformer blocks
    attention_weights: List
        List of tuples with the attention weights per block if
        ``keep_attn_weights = True``. The first element in each tuples is the
        attention weights corresponding to the self attention mechanism
        (i.e. column attention) and the second element is the inter-sample
        attention weights (i.e. row attention)
    transformer_mlp: ``nn.Module``
        MLP component in the model
    output_dim: int
        The output dimension of the model. This is a required attribute
        neccesary to build the WideDeep class

    Example
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import SAINT
    >>> X_tab = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
    >>> colnames = ['a', 'b', 'c', 'd', 'e']
    >>> embed_input = [(u,i) for u,i in zip(colnames[:4], [4]*4)]
    >>> continuous_cols = ['e']
    >>> column_idx = {k:v for v,k in enumerate(colnames)}
    >>> model = SAINT(column_idx=column_idx, embed_input=embed_input, continuous_cols=continuous_cols)
    >>> out = model(X_tab)
    """

    def __init__(
        self,
        column_idx: Dict[str, int],
        embed_input: List[Tuple[str, int]],
        embed_dropout: float = 0.1,
        full_embed_dropout: bool = False,
        shared_embed: bool = False,
        add_shared_embed: bool = False,
        frac_shared_embed: float = 0.25,
        continuous_cols: Optional[List[str]] = None,
        embed_continuous: bool = False,
        embed_continuous_activation: str = None,
        cont_norm_layer: str = "layernorm",
        input_dim: int = 32,
        n_heads: int = 8,
        n_blocks: int = 6,
        dropout: float = 0.1,
        keep_attn_weights: bool = False,
        ff_hidden_dim: int = 32 * 4,
        transformer_activation: str = "gelu",
        mlp_hidden_dims: Optional[List[int]] = None,
        mlp_activation: str = "relu",
        mlp_batchnorm: bool = False,
        mlp_batchnorm_last: bool = False,
        mlp_linear_first: bool = True,
    ):
        super().__init__(
            column_idx,
            embed_input,
            embed_dropout,
            full_embed_dropout,
            shared_embed,
            add_shared_embed,
            frac_shared_embed,
            continuous_cols,
            embed_continuous,
            embed_continuous_activation,
            cont_norm_layer,
            input_dim,
            n_heads,
            n_blocks,
            dropout,
            keep_attn_weights,
            ff_hidden_dim,
            transformer_activation,
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
