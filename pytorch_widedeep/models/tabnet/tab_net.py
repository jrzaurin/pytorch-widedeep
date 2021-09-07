import torch
from torch import nn

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models.tab_mlp import CatEmbeddingsAndCont
from pytorch_widedeep.models.tabnet._layers import (
    TabNetEncoder,
    initialize_non_glu,
)


class TabNet(nn.Module):
    r"""Defines a ``TabNet`` model (https://arxiv.org/abs/1908.07442) model
    that can be used as the ``deeptabular`` component of a Wide & Deep
    model.

    The implementation in this library is fully based on that here:
    https://github.com/dreamquark-ai/tabnet, simply adapted so that it can
    work within the ``WideDeep`` frame. Therefore, **all credit to the
    dreamquark-ai team**

    Parameters
    ----------
    column_idx: Dict
        Dict containing the index of the columns that will be passed through
        the model. Required to slice the tensors. e.g. {'education':
        0, 'relationship': 1, 'workclass': 2, ...}
    embed_input: List
        List of Tuples with the column name, number of unique values and
        embedding dimension. e.g. [(education, 11, 32), ...]
    embed_dropout: float, default = 0.
        embeddings dropout
    continuous_cols: List, Optional, default = None
        List with the name of the numeric (aka continuous) columns
    cont_norm_layer: str, default =  "batchnorm"
        Type of normalization layer applied to the continuous features. Options
        are: 'layernorm', 'batchnorm' or None.
    n_steps: int, default = 3
        number of decision steps
    step_dim: int, default = 8
        Step's output dimension. This is the output dimension that
        ``WideDeep`` will collect and connect to the output neuron(s). For
        a better understanding of the function of this and the upcoming
        parameters, please see the `paper
        <https://arxiv.org/abs/1908.07442>`_.
    attn_dim: int, default = 8
        Attention dimension
    dropout: float, default = 0.0
        GLU block's internal dropout
    n_glu_step_dependent: int, default = 2
        number of GLU Blocks [FC -> BN -> GLU] that are step dependent
    n_glu_shared: int, default = 2
        number of GLU Blocks [FC -> BN -> GLU] that will be shared
        across decision steps
    ghost_bn: bool, default=True
        Boolean indicating if `Ghost Batch Normalization
        <https://arxiv.org/abs/1705.08741>`_ will be used.
    virtual_batch_size: int, default = 128
        Batch size when using Ghost Batch Normalization
    momentum: float, default = 0.02
        Ghost Batch Normalization's momentum. The dreamquark-ai advises for
        very low values. However high values are used in the original
        publication. During our tests higher values lead to better results
    gamma: float, default = 1.3
        Relaxation parameter in the paper. When gamma = 1, a feature is
        enforced to be used only at one decision step. As gamma increases,
        more flexibility is provided to use a feature at multiple decision
        steps
    epsilon: float, default = 1e-15
        Float to avoid log(0). Always keep low
    mask_type: str, default = "sparsemax"
        Mask function to use. Either "sparsemax" or "entmax"

    Attributes
    ----------
    cat_embed_and_cont: ``nn.Module``
        This is the module that processes the categorical and continuous columns
    tabnet_encoder: ``nn.Module``
        ``Module`` containing the TabNet encoder. See the `paper
        <https://arxiv.org/abs/1908.07442>`_.
    output_dim: int
        The output dimension of the model. This is a required attribute
        neccesary to build the WideDeep class

    Example
    --------
    >>> import torch
    >>> from pytorch_widedeep.models import TabNet
    >>> X_tab = torch.cat((torch.empty(5, 4).random_(4), torch.rand(5, 1)), axis=1)
    >>> colnames = ['a', 'b', 'c', 'd', 'e']
    >>> embed_input = [(u,i,j) for u,i,j in zip(colnames[:4], [4]*4, [8]*4)]
    >>> column_idx = {k:v for v,k in enumerate(colnames)}
    >>> model = TabNet(column_idx=column_idx, embed_input=embed_input, continuous_cols = ['e'])
    """

    def __init__(
        self,
        column_idx: Dict[str, int],
        embed_input: List[Tuple[str, int, int]],
        embed_dropout: float = 0.0,
        continuous_cols: Optional[List[str]] = None,
        cont_norm_layer: str = None,
        n_steps: int = 3,
        step_dim: int = 8,
        attn_dim: int = 8,
        dropout: float = 0.0,
        n_glu_step_dependent: int = 2,
        n_glu_shared: int = 2,
        ghost_bn: bool = True,
        virtual_batch_size: int = 128,
        momentum: float = 0.02,
        gamma: float = 1.3,
        epsilon: float = 1e-15,
        mask_type: str = "sparsemax",
    ):
        super(TabNet, self).__init__()

        self.column_idx = column_idx
        self.embed_input = embed_input
        self.embed_dropout = embed_dropout
        self.continuous_cols = continuous_cols
        self.cont_norm_layer = cont_norm_layer
        self.n_steps = n_steps
        self.step_dim = step_dim
        self.attn_dim = attn_dim
        self.dropout = dropout
        self.n_glu_step_dependent = n_glu_step_dependent
        self.n_glu_shared = n_glu_shared
        self.ghost_bn = ghost_bn
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.gamma = gamma
        self.epsilon = epsilon
        self.mask_type = mask_type

        self.cat_embed_and_cont = CatEmbeddingsAndCont(
            column_idx,
            embed_input,
            embed_dropout,
            continuous_cols,
            cont_norm_layer,
        )

        self.embed_and_cont_dim = self.cat_embed_and_cont.output_dim
        self.tabnet_encoder = TabNetEncoder(
            self.embed_and_cont_dim,
            n_steps,
            step_dim,
            attn_dim,
            dropout,
            n_glu_step_dependent,
            n_glu_shared,
            ghost_bn,
            virtual_batch_size,
            momentum,
            gamma,
            epsilon,
            mask_type,
        )
        self.output_dim = step_dim

    def forward(self, X: Tensor) -> Tuple[Tensor, Tensor]:

        x_emb, x_cont = self.cat_embed_and_cont(X)
        if x_emb is not None:
            x = x_emb
        if x_cont is not None:
            x = torch.cat([x, x_cont], 1) if x_emb is not None else x_cont

        steps_output, M_loss = self.tabnet_encoder(x)
        res = torch.sum(torch.stack(steps_output, dim=0), dim=0)

        return (res, M_loss)

    def forward_masks(self, X: Tensor) -> Tuple[Tensor, Dict[int, Tensor]]:

        x_emb, x_cont = self.cat_embed_and_cont(X)
        if x_emb is not None:
            x = x_emb
        if x_cont is not None:
            x = torch.cat([x, x_cont], 1) if x_emb is not None else x_cont

        return self.tabnet_encoder.forward_masks(x)


class TabNetPredLayer(nn.Module):
    def __init__(self, inp, out):
        r"""This class is a 'hack' required because TabNet is a very particular
        model within ``WideDeep``.

        TabNet's forward method within ``WideDeep`` outputs two tensors, one
        with the last layer's activations and the sparse regularization
        factor. Since the output needs to be collected by ``WideDeep`` to then
        Sequentially build the output layer (connection to the output
        neuron(s)) I need to code a custom TabNetPredLayer that accepts two
        inputs. This will be used by the ``WideDeep`` class.
        """
        super(TabNetPredLayer, self).__init__()
        self.pred_layer = nn.Linear(inp, out, bias=False)
        initialize_non_glu(self.pred_layer, inp, out)

    def forward(self, tabnet_tuple: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        res, M_loss = tabnet_tuple[0], tabnet_tuple[1]
        return self.pred_layer(res), M_loss
