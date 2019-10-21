import numpy as np
import torch

from torch import nn
from ..wdtypes import *


def dense_layer(inp:int, out:int, dropout:float, batchnorm=False):
    if batchnorm:
        return nn.Sequential(
            nn.Linear(inp, out),
            nn.BatchNorm1d(out),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout)
            )
    else:
        return nn.Sequential(
            nn.Linear(inp, out),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout)
            )


class DeepDense(nn.Module):
    def __init__(self,
        deep_column_idx:Dict[str,int],
        hidden_layers:List[int],
        dropout:List[float]=0.,
        embed_input:Optional[List[Tuple[str,int,int]]]=None,
        continuous_cols:Optional[List[str]]=None,
        batchnorm:bool=False,
        output_dim:int=1):

        super(DeepDense, self).__init__()

        self.embed_input = embed_input
        self.continuous_cols = continuous_cols
        self.deep_column_idx = deep_column_idx

        # Embeddings
        if self.embed_input is not None:
            self.embed_layers = nn.ModuleDict({'emb_layer_'+col: nn.Embedding(val, dim)
                for col, val, dim in self.embed_input})
            emb_inp_dim = np.sum([embed[2] for embed in self.embed_input])
        else:
            emb_inp_dim = 0

        # Continuous
        if self.continuous_cols is not None: cont_inp_dim = len(self.continuous_cols)
        else: cont_inp_dim = 0

        # Dense Layers
        input_dim = emb_inp_dim + cont_inp_dim
        hidden_layers = [input_dim] + hidden_layers
        dropout = [0.0] + dropout
        self.dense = nn.Sequential()
        for i in range(1, len(hidden_layers)):
            self.dense.add_module(
                'dense_layer_{}'.format(i-1),
                dense_layer( hidden_layers[i-1], hidden_layers[i], dropout[i-1], batchnorm))

        # Last Linear (Deep Dense Linear ddlinear)
        self.dense.add_module('ddlinear', nn.Linear(hidden_layers[-1], output_dim))

    def forward(self, X:Tensor)->Tensor:
        if self.embed_input is not None:
            embed = [self.embed_layers['emb_layer_'+col](X[:,self.deep_column_idx[col]].long())
                for col,_,_ in self.embed_input]
        if self.continuous_cols is not None:
            cont_idx = [self.deep_column_idx[col] for col in self.continuous_cols]
            cont = X[:, cont_idx].float()
        try:
            out = self.dense(torch.cat(embed+[cont], 1))
        except:
            try:
                out = self.dense(torch.cat(embed, 1))
            except:
                out = self.dense(cont)
        return out
