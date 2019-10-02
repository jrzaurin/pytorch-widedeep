import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..wdtypes import *


class DeepText(nn.Module):
    def __init__(self, vocab_size:int, embed_dim:int, hidden_dim:int, n_layers:int,
        rnn_dropout:float, spatial_dropout:float, padding_idx:int, output_dim:int,
        bidirectional:bool=False, embedding_matrix:Optional[np.ndarray]=None,
        pretrained:bool=True):
        super(DeepText, self).__init__()
        """
        Standard Text Classifier/Regressor with a stack of RNNs.
        """
        self.bidirectional = bidirectional
        self.spatial_dropout = spatial_dropout
        self.word_embed_dropout = nn.Dropout2d(spatial_dropout)
        if isinstance(embedding_matrix, np.ndarray):
            self.word_embed = nn.Embedding(vocab_size, embedding_matrix.shape[1], padding_idx = padding_idx)
            self.word_embed.weight = nn.Parameter(torch.Tensor(embedding_matrix))
            embed_dim = embedding_matrix.shape[1]
        else:
            self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx = padding_idx)
        self.pretrained = pretrained
        self.rnn = nn.GRU(embed_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=rnn_dropout,
            batch_first=True)
        input_dim = hidden_dim*2 if bidirectional else hidden_dim
        self.dtlinear = nn.Linear(input_dim, output_dim)

    def forward(self, X:Tensor)->Tensor:

        embed = self.word_embed(X.long())
        # Spatial dropout: dropping an entire channel (word-vector dimension)
        if self.spatial_dropout > 0.:
            sd_embed = embed.unsqueeze(2)
            sd_embed = sd_embed.permute(0, 3, 2, 1)
            sd_embed = self.word_embed_dropout(sd_embed)
            sd_embed = sd_embed.permute(0, 3, 2, 1)
            embed = sd_embed.squeeze(2)
        o, h = self.rnn(embed)
        if self.bidirectional:
            last_h = torch.cat((h[-2], h[-1]), dim = 1)
        else:
            last_h = h[-1]
        out = self.dtlinear(last_h)
        return out
