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
import sparsemax

from pytorch_widedeep.wdtypes import *  # noqa: F403


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

        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]
        return torch.cat(res, dim=0)


class GLU_Layer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        fc: nn.Module = None,
        ghost_bn: bool = True,
        virtual_batch_size: int = 128,
        momentum: float = 0.02,
    ):
        super(GLU_Layer, self).__init__()

        self.output_dim = output_dim
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

    def forward(self, x):
        return F.glu(self.bn(self.fc(x)))


class GLU_Block(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_glu: int = 2,
        first: bool = False,
        shared_layers: List = None,
        ghost_bn: bool = True,
        virtual_batch_size: int = 128,
        momentum: float = 0.02,
    ):
        super(GLU_Block, self).__init__()
        self.first = first
        self.shared_layers = shared_layers
        if (shared_layers is not None) and (n_glu != len(shared_layers)):
            self.n_glu = len(shared_layers)
            warnings.warn(
                "If 'shared_layers' is nor None, 'n_glu' must be equal to the number of shared_layers."
                "Got n_glu = {} and n shared_layers = {}. 'n_glu' has be set to {}".format(
                    n_glu, len(shared_layers), len(shared_layers)
                ),
                UserWarning,
            )
        else:
            self.n_glu = n_glu
        self.glu_layers = nn.ModuleList()

        glu_dim = [input_dim] + [output_dim] * self.n_glu
        for i in range(self.n_glu):
            fc = shared_layers[i] if shared_layers else None
            self.glu_layers.append(
                GLU_Layer(
                    glu_dim[i],
                    glu_dim[i + 1],
                    fc=fc,
                    ghost_bn=ghost_bn,
                    virtual_batch_size=virtual_batch_size,
                    momentum=momentum,
                )
            )

    def forward(self, x):
        scale = torch.sqrt(torch.FloatTensor([0.5]).to(x.device))

        if self.first:  # the first layer of the block has no scale multiplication
            x = self.glu_layers[0](x)
            layers_left = range(1, self.n_glu)
        else:
            layers_left = range(self.n_glu)

        for glu_id in layers_left:
            x = torch.add(x, self.glu_layers[glu_id](x)) * scale

        return x


class FeatTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        shared_layers: List,
        n_glu_step_dependent: int,
        ghost_bn: bool = True,
        virtual_batch_size=128,
        momentum=0.02,
    ):
        super(FeatTransformer, self).__init__()

        params = {
            "ghost_bn": ghost_bn,
            "virtual_batch_size": virtual_batch_size,
            "momentum": momentum,
        }

        if shared_layers is None:
            self.shared: Union[nn.Identity, GLU_Block] = nn.Identity()
            is_first = True
        else:
            self.shared = GLU_Block(
                input_dim,
                output_dim,
                n_glu=len(shared_layers),
                first=True,
                shared_layers=shared_layers,
                **params
            )
            is_first = False

        if n_glu_step_dependent == 0:
            self.step_dependent: Union[nn.Identity, GLU_Block] = nn.Identity()
        else:
            self.step_dependent = GLU_Block(
                input_dim if is_first else output_dim,
                output_dim,
                n_glu=n_glu_step_dependent,
                first=is_first,
                **params
            )

    def forward(self, x):
        return self.step_dependent(self.shared(x))


class AttentiveTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        mask_type="sparsemax",
        ghost_bn: bool = True,
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
                "Please choose either sparsemax" + "or entmax as masktype"
            )

    def forward(self, priors, processed_feat):
        x = self.bn(self.fc(processed_feat))
        x = torch.mul(x, priors)
        return self.mask(x)


class TabNetEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        step_dim=8,
        attn_dim=8,
        n_steps=3,
        n_glu_step_dependent=2,
        n_glu_shared=2,
        ghost_bn=True,
        virtual_batch_size=128,
        momentum=0.02,
        gamma=1.3,
        epsilon=1e-15,
        mask_type="sparsemax",
    ):
        super(TabNetEncoder, self).__init__()
        self.input_dim = input_dim
        self.step_dim = step_dim
        self.attn_dim = attn_dim
        self.n_steps = n_steps
        self.n_glu_step_dependent = n_glu_step_dependent
        self.n_glu_shared = n_glu_shared
        self.ghost_bn = ghost_bn
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.gamma = gamma
        self.epsilon = epsilon
        self.mask_type = mask_type
        self.initial_bn = BatchNorm1d(self.input_dim, momentum=0.01)

        params = {
            "ghost_bn": ghost_bn,
            "virtual_batch_size": virtual_batch_size,
            "momentum": momentum,
        }

        if self.n_glu_shared > 0:
            shared_layers_dims = [self.input_dim] + [
                2 * (step_dim + attn_dim)
            ] * self.n_glu_shared
            shared_layers = nn.ModuleList()
            for i in range(self.n_glu_shared):
                shared_layers.append(
                    nn.Linear(
                        shared_layers_dims[i], shared_layers_dims[i + 1], bias=False
                    )
                )
        else:
            shared_layers = None

        self.initial_splitter = FeatTransformer(
            self.input_dim,
            step_dim + attn_dim,
            shared_layers,
            n_glu_step_dependent=self.n_glu_step_dependent,
            **params
        )

        self.feat_transformers = nn.ModuleList()
        self.attn_transformers = nn.ModuleList()

        for step in range(n_steps):
            feat_transformer = FeatTransformer(
                self.input_dim,
                step_dim + attn_dim,
                shared_layers,
                n_glu_step_dependent=self.n_glu_step_dependent,
                **params
            )
            attn_transformer = AttentiveTransformer(
                attn_dim, self.input_dim, mask_type=self.mask_type, **params
            )
            self.feat_transformers.append(feat_transformer)
            self.attn_transformers.append(attn_transformer)

    def forward(self, x, prior=None):
        x = self.initial_bn(x)

        if prior is None:
            # P[0] is initialized as all ones, 1^(B×D)
            prior = torch.ones(x.shape).to(x.device)

        # sparsity regularization
        M_loss = 0

        # split block
        attn = self.initial_splitter(x)[:, self.step_dim :]

        steps_output = []
        for step in range(self.n_steps):
            # learnable mask: M[i] = sparsemax(prior[i − 1] · hi(a[i − 1]))
            # where hi = FC + BN
            M = self.attn_transformers[step](prior, attn)

            # sparsity regularization
            M_loss += torch.mean(
                torch.sum(torch.mul(M, torch.log(M + self.epsilon)), dim=1)
            )

            # update prior: P[i] = \prod_{i}^{j=1} (γ − M[j])
            prior = torch.mul(self.gamma - M, prior)

            # update attention and d_out
            masked_x = torch.mul(M, x)
            out = self.feat_transformers[step](masked_x)
            attn = out[:, self.step_dim :]
            d_out = nn.ReLU()(out[:, : self.step_dim])
            steps_output.append(d_out)

        M_loss /= self.n_steps

        return steps_output, M_loss

    def forward_masks(self, x):
        x = self.initial_bn(x)

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
            # decision contribution
            d_out = ReLU()(out[:, : self.step_dim])

            # aggregate decision contribution
            agg_decision_contrib = torch.sum(d_out, dim=1)
            M_explain += torch.mul(M, agg_decision_contrib.unsqueeze(dim=1))

        return M_explain, masks


class EmbeddingsAndContinuous(object):
    def __init__(
        self,
        column_idx: Dict[str, int],
        embed_input: List[Tuple[str, int, int]],
        embed_dropout: float,
        continuous_cols: Optional[List[str]],
        batchnorm_cont: bool,
    ):
        super(EmbeddingsAndContinuous, self).__init__()

        self.column_idx = column_idx
        self.embed_input = embed_input
        self.continuous_cols = continuous_cols
        self.batchnorm_cont = batchnorm_cont

        # Embeddings: val + 1 because 0 is reserved for padding/unseen cateogories.
        self.embed_layers = nn.ModuleDict(
            {
                "emb_layer_" + col: nn.Embedding(val + 1, dim, padding_idx=0)
                for col, val, dim in self.embed_input
            }
        )
        self.embedding_dropout = nn.Dropout(embed_dropout)
        emb_inp_dim = np.sum([embed[2] for embed in self.embed_input])

        # Continuous
        if self.continuous_cols is not None:
            cont_inp_dim = len(self.continuous_cols)
            if self.batchnorm_cont:
                self.norm = nn.BatchNorm1d(cont_inp_dim)
        else:
            cont_inp_dim = 0

        self.output_dim = emb_inp_dim + cont_inp_dim

    def forward(self, x):
        embed = [
            self.embed_layers["emb_layer_" + col](X[:, self.column_idx[col]].long())
            for col, _, _ in self.embed_input
        ]
        x = torch.cat(embed, 1)
        x = self.embedding_dropout(x)
        if self.continuous_cols is not None:
            cont_idx = [self.column_idx[col] for col in self.continuous_cols]
            x_cont = X[:, cont_idx].float()
            if self.batchnorm_cont:
                x_cont = self.norm(x_cont)
            x = torch.cat([x, x_cont], 1) if self.embed_input is not None else x_cont
        return x


class TabNet(nn.Module):
    def __init__(
        self,
        column_idx: Dict[str, int],
        embed_input: List[Tuple[str, int, int]],
        embed_dropout: float = 0.0,
        continuous_cols: Optional[List[str]] = None,
        batchnorm_cont: bool = False,
        step_dim=8,
        attn_dim=8,
        n_steps=3,
        n_glu_step_dependent=2,
        n_glu_shared=2,
        ghost_bn=True,
        virtual_batch_size=128,
        momentum=0.02,
        gamma=1.3,
        epsilon=1e-15,
        mask_type="sparsemax",
    ):
        super(TabNet, self).__init__()

        if self.n_glu_step_dependent == 0 and self.n_glu_shared == 0:
            raise ValueError(
                "'n_glu_shared' and 'n_glu_step_dependent' can't be both zero."
            )

        self.embed_and_cont = EmbeddingsAndContinuous(
            column_idx, embed_input, embed_dropout, continuous_cols, batchnorm_cont
        )
        self.tabnet_encoder = TabNetEncoder(
            self.embed_and_cont.output_dim,
            step_dim,
            attn_dim,
            n_steps,
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

    def forward(self, x):
        self.embed_and_cont(x)
        steps_output, M_loss = self.tabnet_encoder(x)
        res = torch.sum(torch.stack(steps_output, dim=0), dim=0)
        return (res, M_loss)

    def forward_masks(self, x):
        self.embed_and_cont(x)
        return self.tabnet_encoder.forward_masks(x)
