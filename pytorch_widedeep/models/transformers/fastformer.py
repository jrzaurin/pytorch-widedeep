from torch import nn

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models.transformers.layers import FastFormerEncoder
from pytorch_widedeep.models.transformers.tab_transformer import TabTransformer


class FastFormer(TabTransformer):
    r"""Adaptation of FastFormer model
    (https://arxiv.org/abs/2108.09084) model that can be used as the
    deeptabular component of a Wide & Deep model.

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
        Note that setting this to true is equivalent to the so called
        `FT-Transformer <https://arxiv.org/abs/2106.11959>`_
        (Feature Tokenizer + Transformer). The only difference is that our
        implementation does not consider using bias for the categorical
        embeddings.
    embed_continuous_activation: str, default = None
        String indicating the activation function to be applied to the
        continuous embeddings, if any.
        'tanh', 'relu', 'leaky_relu' and 'gelu' are supported.
    cont_norm_layer: str, default =  None,
        Type of normalization layer applied to the continuous features. Options
        are: 'layernorm', 'batchnorm' or None.
    input_dim: int, default = 32
        The so-called *dimension of the model*. Is the number of embeddings used to encode
        the categorical and/or continuous columns
    n_heads: int, default = 8
        Number of attention heads per Transformer block
    use_bias: bool, default = False
        Boolean indicating whether or not to use bias in the Q, K, and V
        projection layers
    n_blocks: int, default = 6
        Number of Transformer blocks
    dropout: float, default = 0.1
        Dropout that will be applied internally to the ``TransformerEncoder``
        (see :obj:`pytorch_widedeep.models.transformers.layers.TransformerEncoder`)
        and the output MLP
    share_qv_weights: bool, default = True
        Following the original publication, this is a boolean indicating if
        the the value and query transformation parameters will be shared
    share_weights: bool, default = True
        In addition to sharing the value and query transformation parameters,
        the parameters across different Fastformer layers are also shared in
        the paper.
    transformer_activation: str, default = "gelu"
        Transformer Encoder activation
        function. 'tanh', 'relu', 'leaky_relu', 'gelu' and 'geglu' are
        supported
    mlp_hidden_dims: List, Optional, default = None
        MLP hidden dimensions. If not provided it will default to ``[4*l,
        2*l]`` where ``l`` is the mlp input dimension
    mlp_activation: str, default = "relu"
        MLP activation function. 'tanh', 'relu', 'leaky_relu' and 'gelu' are
        supported
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
    cat_embed_and_cont: ``nn.Module``
        Module that processese the categorical and continuous columns
    transformer_blks: ``nn.Sequential``
        Sequence of Transformer blocks
    transformer_mlp: ``nn.Module``
        MLP component in the model
    output_dim: int
        The output dimension of the model. This is a required attribute
        neccesary to build the WideDeep class

    Properties
    -----------
    attention_weights: List
        List with the attention weights

    Example
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import TabTransformer
    >>> X_tab = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
    >>> colnames = ['a', 'b', 'c', 'd', 'e']
    >>> embed_input = [(u,i) for u,i in zip(colnames[:4], [4]*4)]
    >>> continuous_cols = ['e']
    >>> column_idx = {k:v for v,k in enumerate(colnames)}
    >>> model = TabTransformer(column_idx=column_idx, embed_input=embed_input, continuous_cols=continuous_cols)
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
        embed_continuous: bool = True,
        embed_continuous_activation: str = None,
        cont_norm_layer: str = None,
        input_dim: int = 32,
        n_heads: int = 8,
        use_bias: bool = False,
        n_blocks: int = 6,
        dropout: float = 0.1,
        share_qv_weights: bool = True,
        share_weights: bool = True,
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
            use_bias,
            n_blocks,
            dropout,
            transformer_activation,
            mlp_hidden_dims,
            mlp_activation,
            mlp_batchnorm,
            mlp_batchnorm_last,
            mlp_linear_first,
        )

        self.share_qv_weights = share_qv_weights
        self.share_weights = share_weights

        self.transformer_blks = nn.Sequential()
        first_fastformer_block = FastFormerEncoder(
            input_dim,
            n_heads,
            use_bias,
            dropout,
            share_qv_weights,
            transformer_activation,
        )
        self.transformer_blks.add_module("fastformer_block0", first_fastformer_block)
        for i in range(1, n_blocks):
            if share_weights:
                self.transformer_blks.add_module(
                    "fastformer_block" + str(i), first_fastformer_block
                )
            else:
                self.transformer_blks.add_module(
                    "fastformer_block" + str(i),
                    FastFormerEncoder(
                        input_dim,
                        n_heads,
                        use_bias,
                        dropout,
                        share_qv_weights,
                        transformer_activation,
                    ),
                )
