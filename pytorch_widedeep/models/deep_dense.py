import torch
import numpy as np
import torch.nn as nn
from torch import Tensor

from ..wdtypes import *


def dense_layer(inp:int, out:int, dropout:float):
    return nn.Sequential(
        nn.Linear(inp, out),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(dropout)
        )

class DeepDense(nn.Module):
    def __init__(self, embed_input:List[Tuple[str,int,int]],
        embed_encoding_dict:Dict[str,Any], continuous_cols:List[str],
        deep_column_idx:Dict[str,int], hidden_layers:List[int],
        dropout:List[float], output_dim:int):

        super(DeepDense, self).__init__()

        self.embed_input = embed_input
        self.embed_encoding_dict = embed_encoding_dict
        self.continuous_cols = continuous_cols
        self.deep_column_idx = deep_column_idx

        self.embed_layers = nn.ModuleDict({'emb_layer_'+col: nn.Embedding(val, dim)
            for col, val, dim in embed_input})
        input_embed_dim = np.sum([emb[2] for emb in embed_input])+len(continuous_cols)
        hidden_layers = [input_embed_dim] + hidden_layers
        dropout = [0.0] + dropout
        self.dense = nn.Sequential()
        for i in range(1, len(hidden_layers)):
            self.dense.add_module(
                'dense_layer_{}'.format(i-1),
                dense_layer( hidden_layers[i-1], hidden_layers[i], dropout[i-1])
                )
        self.dense.add_module('last_linear', nn.Linear(hidden_layers[-1], output_dim))

    def forward(self, X:Tensor)->Tensor:
        emb = [self.embed_layers['emb_layer_'+col](X[:,self.deep_column_idx[col]].long())
            for col,_,_ in self.embed_input]
        if self.continuous_cols:
            cont_idx = [self.deep_column_idx[col] for col in self.continuous_cols]
            cont = [X[:, cont_idx].float()]
            inp = torch.cat(emb+cont, 1)
        else:
            inp = torch.cat(emb, 1)
        out = self.dense(inp)
        return out

