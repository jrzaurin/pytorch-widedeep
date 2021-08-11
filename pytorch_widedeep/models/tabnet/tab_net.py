"""
Most of the code here is a direct copy and paste from the fantastic tabnet
implementation here: https://github.com/dreamquark-ai/tabnet

Therefore, ALL CREDIT TO THE DREAMQUARK-AI TEAM
-----------------------------------------------

Here I simply adapted what I needed the TabNet to work within pytorch-widedeep
"""

import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from pytorch_widedeep.wdtypes import *  # noqa: F403
from pytorch_widedeep.models.tabnet import sparsemax


def initialize_non_glu(module, input_dim: int, output_dim: int):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(4 * input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    return


def initialize_glu(module, input_dim: int, output_dim: int):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    return


class GBN(torch.nn.Module):
    """
    Ghost Batch Normalization
    https://arxiv.org/abs/1705.08741
    """

    def __init__(
        self, input_dim: int, virtual_batch_size: int = 128, momentum: float = 0.01
    ):
        super(GBN, self).__init__()
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(input_dim, momentum=momentum)

    def forward(self, X: Tensor) -> Tensor:
        chunks = X.chunk(int(np.ceil(X.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]
        return torch.cat(res, dim=0)


class GLU_Layer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float,
        fc: nn.Module = None,
        ghost_bn: bool = True,
        virtual_batch_size: int = 128,
        momentum: float = 0.02,
    ):
        super(GLU_Layer, self).__init__()

        if fc:
            self.fc = fc
        else:
            self.fc = nn.Linear(input_dim, 2 * output_dim, bias=False)
        initialize_glu(self.fc, input_dim, 2 * output_dim)

        if ghost_bn:
            self.bn: Union[GBN, nn.BatchNorm1d] = GBN(
                2 * output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum
            )
        else:
            self.bn = nn.BatchNorm1d(2 * output_dim, momentum=momentum)

        self.dp = nn.Dropout(dropout)

    def forward(self, X: Tensor) -> Tensor:
        return self.dp(F.glu(self.bn(self.fc(X))))


class GLU_Block(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float,
        n_glu: int = 2,
        first: bool = False,
        shared_layers: nn.ModuleList = None,
        ghost_bn: bool = True,
        virtual_batch_size: int = 128,
        momentum: float = 0.02,
    ):
        super(GLU_Block, self).__init__()
        self.first = first

        if (shared_layers is not None) and (n_glu != len(shared_layers)):
            self.n_glu = len(shared_layers)
            warnings.warn(
                "If 'shared_layers' is nor None, 'n_glu' must be equal to the number of shared_layers."
                "Got n_glu = {} and n shared_layers = {}. 'n_glu' has been set to {}".format(
                    n_glu, len(shared_layers), len(shared_layers)
                ),
                UserWarning,
            )
        else:
            self.n_glu = n_glu

        glu_dim = [input_dim] + [output_dim] * self.n_glu
        self.glu_layers = nn.ModuleList()
        for i in range(self.n_glu):
            fc = shared_layers[i] if shared_layers else None
            self.glu_layers.append(
                GLU_Layer(
                    glu_dim[i],
                    glu_dim[i + 1],
                    dropout,
                    fc=fc,
                    ghost_bn=ghost_bn,
                    virtual_batch_size=virtual_batch_size,
                    momentum=momentum,
                )
            )

    def forward(self, X: Tensor) -> Tensor:
        scale = torch.sqrt(torch.FloatTensor([0.5]).to(X.device))

        if self.first:  # the first layer of the block has no scale multiplication
            x = self.glu_layers[0](X)
            layers_left = range(1, self.n_glu)
        else:
            x = nn.Identity()(X)
            layers_left = range(self.n_glu)

        for glu_id in layers_left:
            x = torch.add(x, self.glu_layers[glu_id](x)) * scale

        return x


class FeatTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float,
        shared_layers: nn.ModuleList,
        n_glu_step_dependent: int,
        ghost_bn=True,
        virtual_batch_size=128,
        momentum=0.02,
    ):
        super(FeatTransformer, self).__init__()

        params = {
            "ghost_bn": ghost_bn,
            "virtual_batch_size": virtual_batch_size,
            "momentum": momentum,
        }

        self.shared = GLU_Block(
            input_dim,
            output_dim,
            dropout,
            n_glu=len(shared_layers),
            first=True,
            shared_layers=shared_layers,
            **params
        )

        self.step_dependent = GLU_Block(
            output_dim,
            output_dim,
            dropout,
            n_glu=n_glu_step_dependent,
            first=False,
            **params
        )

    def forward(self, X: Tensor) -> Tensor:
        return self.step_dependent(self.shared(X))


class AttentiveTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mask_type: str = "sparsemax",
        ghost_bn=True,
        virtual_batch_size=128,
        momentum=0.02,
    ):

        super(AttentiveTransformer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        initialize_non_glu(self.fc, input_dim, output_dim)
        if ghost_bn:
            self.bn: Union[GBN, nn.BatchNorm1d] = GBN(
                output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum
            )
        else:
            self.bn = nn.BatchNorm1d(output_dim, momentum=momentum)

        if mask_type == "sparsemax":
            self.mask: Union[Sparsemax, Entmax15] = sparsemax.Sparsemax(dim=-1)
        elif mask_type == "entmax":
            self.mask = sparsemax.Entmax15(dim=-1)
        else:
            raise NotImplementedError(
                "Please choose either 'sparsemax' or 'entmax' as masktype"
            )

    def forward(self, priors: Tensor, processed_feat: Tensor) -> Tensor:
        x = self.bn(self.fc(processed_feat))
        x = torch.mul(x, priors)
        return self.mask(x)


class TabNetEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
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
        super(TabNetEncoder, self).__init__()

        self.input_dim = input_dim
        self.n_steps = n_steps
        self.step_dim = step_dim
        self.attn_dim = attn_dim
        self.gamma = gamma
        self.epsilon = epsilon

        self.initial_bn = nn.BatchNorm1d(input_dim, momentum=0.01)

        params = {
            "ghost_bn": ghost_bn,
            "virtual_batch_size": virtual_batch_size,
            "momentum": momentum,
        }

        shared_layers = nn.ModuleList()
        for i in range(n_glu_shared):
            if i == 0:
                shared_layers.append(
                    nn.Linear(input_dim, 2 * (step_dim + attn_dim), bias=False)
                )
            else:
                shared_layers.append(
                    nn.Linear(
                        step_dim + attn_dim, 2 * (step_dim + attn_dim), bias=False
                    )
                )

        self.initial_splitter = FeatTransformer(
            input_dim,
            step_dim + attn_dim,
            dropout,
            shared_layers,
            n_glu_step_dependent,
            **params
        )

        self.feat_transformers = nn.ModuleList()
        self.attn_transformers = nn.ModuleList()
        for step in range(n_steps):
            feat_transformer = FeatTransformer(
                input_dim,
                step_dim + attn_dim,
                dropout,
                shared_layers,
                n_glu_step_dependent,
                **params
            )
            attn_transformer = AttentiveTransformer(
                attn_dim, input_dim, mask_type, **params
            )
            self.feat_transformers.append(feat_transformer)
            self.attn_transformers.append(attn_transformer)

    def forward(self, X: Tensor) -> Tuple[List[Tensor], Tensor]:
        x = self.initial_bn(X)

        # P[n_step = 0] is initialized as all ones, 1^(B×D)
        prior = torch.ones(x.shape).to(x.device)

        # sparsity regularization
        M_loss = torch.FloatTensor([0.0]).to(x.device)

        # split block
        attn = self.initial_splitter(x)[:, self.step_dim :]

        steps_output = []
        for step in range(self.n_steps):
            # learnable mask: M[i] = sparsemax(prior[i − 1] · hi(a[i − 1]))
            # where hi = FC + BN
            M = self.attn_transformers[step](prior, attn)

            # update prior: P[i] = \prod_{i}^{j=1} (γ − M[j])
            prior = torch.mul(self.gamma - M, prior)

            # sparsity regularization
            M_loss += torch.mean(
                torch.sum(torch.mul(M, torch.log(M + self.epsilon)), dim=1)
            )

            # update attention and d_out
            masked_x = torch.mul(M, x)
            out = self.feat_transformers[step](masked_x)
            attn = out[:, self.step_dim :]
            d_out = nn.ReLU()(out[:, : self.step_dim])
            steps_output.append(d_out)

        M_loss /= self.n_steps  # type: ignore[has-type]

        return steps_output, M_loss

    def forward_masks(self, X: Tensor) -> Tuple[Tensor, Dict[int, Tensor]]:
        x = self.initial_bn(X)

        prior = torch.ones(x.shape).to(x.device)
        M_explain = torch.zeros(x.shape).to(x.device)
        attn = self.initial_splitter(x)[:, self.step_dim :]
        masks = {}

        for step in range(self.n_steps):
            M = self.attn_transformers[step](prior, attn)
            masks[step] = M

            prior = torch.mul(self.gamma - M, prior)

            masked_x = torch.mul(M, x)
            out = self.feat_transformers[step](masked_x)
            attn = out[:, self.step_dim :]
            # 'decision contribution' in the paper
            d_out = nn.ReLU()(out[:, : self.step_dim])

            # aggregate decision contribution
            agg_decision_contrib = torch.sum(d_out, dim=1)
            M_explain += torch.mul(M, agg_decision_contrib.unsqueeze(dim=1))

        return M_explain, masks


class EmbeddingsAndContinuous(nn.Module):
    def __init__(
        self,
        column_idx: Dict[str, int],
        embed_input: List[Tuple[str, int, int]],
        embed_dropout: float,
        continuous_cols: Optional[List[str]],
        cont_norm_layer: str,
    ):
        super(EmbeddingsAndContinuous, self).__init__()

        self.column_idx = column_idx
        self.embed_input = embed_input
        self.continuous_cols = continuous_cols
        self.cont_norm_layer = cont_norm_layer

        # Embeddings: val + 1 because 0 is reserved for padding/unseen cateogories.
        self.embed_layers = nn.ModuleDict(
            {
                "emb_layer_" + col: nn.Embedding(val + 1, dim, padding_idx=0)
                for col, val, dim in self.embed_input
            }
        )
        self.embedding_dropout = nn.Dropout(embed_dropout)
        emb_out_dim = np.sum([embed[2] for embed in self.embed_input])

        # Continuous
        if self.continuous_cols is not None:
            self.cont_idx = [self.column_idx[col] for col in self.continuous_cols]
            cont_out_dim = len(self.continuous_cols)
            if self.cont_norm_layer == "batchnorm":
                self.cont_norm: NormLayers = nn.BatchNorm1d(cont_out_dim)
            if self.cont_norm_layer == "layernorm":
                self.cont_norm = nn.LayerNorm(cont_out_dim)
            else:
                self.cont_norm = nn.Identity()
        else:
            cont_out_dim = 0

        self.output_dim: int = emb_out_dim + cont_out_dim  # type: ignore[assignment]

    def forward(self, X: Tensor) -> Tensor:
        embed = [
            self.embed_layers["emb_layer_" + col](X[:, self.column_idx[col]].long())
            for col, _, _ in self.embed_input
        ]
        x = torch.cat(embed, 1)
        x = self.embedding_dropout(x)
        if self.continuous_cols is not None:
            x_cont = self.cont_norm((X[:, self.cont_idx].float()))
            x = torch.cat([x, x_cont], 1) if self.embed_input is not None else x_cont
        return x


class TabNet(nn.Module):
    r"""TabNet model (https://arxiv.org/abs/1908.07442) model that can be used
    as the deeptabular component of a Wide & Deep model.

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
        GLU block 'internal' dropout
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
        very low values. The results in the paper use significantly higher
        values. Higher values lead to better results in my experimentations
    gamma: float, default = 1.3
        Relaxation parameter in the paper. When gamma = 1, a feature is
        enforced to be used only at one decision step and as gamma
        increases, more flexibility is provided to use a feature at
        multiple decision steps
    epsilon: float, default = 1e-15
        Float to avoid log(0). Always keep low
    mask_type: str, default = "sparsemax"
        Mask function to use. Either "sparsemax" or "entmax"

    Attributes
    ----------
    embed_and_cont: ``nn.ModuleDict``
        ``ModuleDict`` with the embeddings and continuous setup
    embed_and_cont_dim: int
        embeddings plus continuous dimension
    output_dim: int
        The output dimension of the model. This is a required attribute
        neccesary to build the WideDeep class
    tabnet_encoder: ``nn.Module``
        ``Module`` containing the TabNet encoder. See the `paper
        <https://arxiv.org/abs/1908.07442>`_.

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

        self.embed_and_cont = EmbeddingsAndContinuous(
            column_idx, embed_input, embed_dropout, continuous_cols, cont_norm_layer
        )
        self.embed_and_cont_dim = self.embed_and_cont.output_dim
        self.tabnet_encoder = TabNetEncoder(
            self.embed_and_cont.output_dim,
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
        x = self.embed_and_cont(X)
        steps_output, M_loss = self.tabnet_encoder(x)
        res = torch.sum(torch.stack(steps_output, dim=0), dim=0)
        return (res, M_loss)

    def forward_masks(self, X: Tensor) -> Tuple[Tensor, Dict[int, Tensor]]:
        x = self.embed_and_cont(X)
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
