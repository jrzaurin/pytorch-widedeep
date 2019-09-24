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
    def __init__(self, embeddings_input:List[Tuple[str,int,int]],
        embeddings_encoding_dict:Dict[str,Any], continuous_cols:List[str],
        deep_column_idx:Dict[str,int], hidden_layers:List[int], dropout:List[float],
        output_dim:int):
        """
        Model consisting in a series of Dense Layers that receive continous
        features concatenated with categorical features represented with
        embeddings

        Parameters:
        embeddings_input: List
            List of Tuples with the column name, number of unique values and
            embedding dimension. e.g. [(education, 11, 32), ...]
        embeddings_encoding_dict: Dict
            Dict containing the encoding mappings
        continuous_cols: List
            List with the name of the so called continuous cols
        deep_column_idx: Dict
            Dict containing the index of the embedding columns. Required to
            slice the tensors.
        hidden_layers: List
            List with the number of neurons per dense layer. e.g: [64,32]
        dropout: List
            List with the dropout between the dense layers. We do not apply dropout
            between Embeddings and first dense or last dense and output. Therefore
            this list must contain len(hidden_layers)-1 elements. e.g: [0.5]
        output_dim: int
            1 for logistic regression or regression, N-classes for multiclass
        """
        super(DeepDense, self).__init__()

        self.embeddings_input = embeddings_input
        self.embeddings_encoding_dict = embeddings_encoding_dict
        self.continuous_cols = continuous_cols
        self.deep_column_idx = deep_column_idx

        for col,val,dim in embeddings_input:
            setattr(self, 'emb_layer_'+col, nn.Embedding(val, dim))
        input_emb_dim = np.sum([emb[2] for emb in embeddings_input])+len(continuous_cols)
        hidden_layers = [input_emb_dim] + hidden_layers
        dropout = [0.0] + dropout
        self.dense = nn.Sequential()
        for i in range(1, len(hidden_layers)):
            self.dense.add_module(
                'dense_layer_{}'.format(i-1),
                dense_layer( hidden_layers[i-1], hidden_layers[i], dropout[i-1])
                )
        self.dense.add_module('last_linear', nn.Linear(hidden_layers[-1], output_dim))

    def forward(self, X:Tensor)->Tensor:
        emb = [getattr(self, 'emb_layer_'+col)(X[:,self.deep_column_idx[col]].long())
               for col,_,_ in self.embeddings_input]
        if self.continuous_cols:
            cont_idx = [self.deep_column_idx[col] for col in self.continuous_cols]
            cont = [X[:, cont_idx].float()]
            inp = torch.cat(emb+cont, 1)
        else:
            inp = torch.cat(emb, 1)
        out = self.dense(inp)
        return out

