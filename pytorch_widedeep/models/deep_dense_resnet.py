import numpy as np
import torch
from torch import nn

from ..wdtypes import *


class BasicBlock(nn.Module):
    def __init__(self, inp: int, out: int, p: float = 0.0):
        super(BasicBlock, self).__init__()

        self.lin1 = nn.Linear(inp, out)
        self.bn1 = nn.BatchNorm1d(out)
        self.lin2 = nn.Linear(out, out)
        self.bn2 = nn.BatchNorm1d(out)

        self.leaky_relu = nn.LeakyReLU(inplace=True)
        if p > 0.:
            self.dropout = True
            self.dp1 = nn.Dropout(p)
            self.dp2 = nn.Dropout(p)

    def forward(self, x):

        identity = x

        out = self.lin1(x)
        out = self.bn1(out)
        out = self.leaky_relu(out)
        if self.dropout:
            out = self.dp1(out)

        out = self.lin2(out)
        out = self.bn2(out)

        out += identity
        out = self.leaky_relu(out)
        if self.dropout:
            out = self.dp2(out)

        return out


class DeepDenseResnet(nn.Module):
    def __init__(
        self,
        deep_column_idx: Dict[str, int],
        blocks: List[int],
        p: float = 0.0,
        embed_p: float = 0.0,
        embed_input: Optional[List[Tuple[str, int, int]]] = None,
        continuous_cols: Optional[List[str]] = None,
    ):
        super(DeepDenseResnet, self).__init__()

        self.embed_input = embed_input
        self.continuous_cols = continuous_cols
        self.deep_column_idx = deep_column_idx

        # Embeddings
        if self.embed_input is not None:
            self.embed_layers = nn.ModuleDict(
                {
                    "emb_layer_" + col: nn.Embedding(val, dim)
                    for col, val, dim in self.embed_input
                }
            )
            self.embed_dropout = nn.Dropout(embed_p)
            emb_inp_dim = np.sum([embed[2] for embed in self.embed_input])
        else:
            emb_inp_dim = 0

        # Continuous
        if self.continuous_cols is not None:
            cont_inp_dim = len(self.continuous_cols)
        else:
            cont_inp_dim = 0

        # ResNet comprised of dense layers
        input_dim = emb_inp_dim + cont_inp_dim

        if input_dim != blocks[0]:
            self.lin1 = nn.Linear(input_dim, blocks[0])
            self.bn1 = nn.BatchNorm1d(blocks[0])

        self.dense_resnet = nn.Sequential()
        for i in range(1, len(blocks)):
            self.dense_resnet.add_module(
                "block_{}".format(i - 1), BasicBlock(blocks[i - 1], blocks[i], p)
            )

        # the output_dim attribute will be used as input_dim when "merging" the models
        self.output_dim = blocks[-1]

    def forward(self, X: Tensor) -> Tensor:  # type: ignore
        r"""Forward pass that concatenates the continuous features with the
        embeddings. The result is then passed through a series of dense layers
        """
        if self.embed_input is not None:
            x = [
                self.embed_layers["emb_layer_" + col](
                    X[:, self.deep_column_idx[col]].long()
                )
                for col, _, _ in self.embed_input
            ]
            x = torch.cat(x, 1)  # type: ignore
            x = self.embed_dropout(x)  # type: ignore
        if self.continuous_cols is not None:
            cont_idx = [self.deep_column_idx[col] for col in self.continuous_cols]
            x_cont = X[:, cont_idx].float()
            x = torch.cat([x, x_cont], 1) if self.embed_input is not None else x_cont  # type: ignore
        return self.dense_resnet(x)  # type: ignore
