import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..wdtypes import *


class DeepText(nn.Module):
    def __init__(self,
        vocab_size:int,
        embed_dim:Optional[int]=None,
        hidden_dim:int=64,
        n_layers:int=3,
        rnn_dropout:float=0.,
        spatial_dropout:float=0.,
        padding_idx:int=1,
        output_dim:int=1,
        bidirectional:bool=False,
        embedding_matrix:Optional[np.ndarray]=None):
        super(DeepText, self).__init__()
        """
        Standard Text Classifier/Regressor with a stack of RNNs.
        """

        if embed_dim is not None and embedding_matrix is not None and not embed_dim==embedding_matrix.shape[1]:
            warnings.warn(
                'the input embedding dimension {} and the dimension of the '
                'pretrained embeddings {} do not match. The pretrained embeddings '
                'dimension ({}) will be used'.format(embed_dim, embedding_matrix.shape[1],
                    embedding_matrix.shape[1]), UserWarning)

        self.bidirectional = bidirectional
        self.spatial_dropout = spatial_dropout
        self.word_embed_dropout = nn.Dropout2d(spatial_dropout)

        # Pre-trained Embeddings
        if isinstance(embedding_matrix, np.ndarray):
            self.word_embed = nn.Embedding(vocab_size, embedding_matrix.shape[1], padding_idx = padding_idx)
            self.word_embed.weight = nn.Parameter(torch.Tensor(embedding_matrix))
            embed_dim = embedding_matrix.shape[1]
        else:
            self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx = padding_idx)

        # stack of GRUs
        self.rnn = nn.GRU(embed_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=rnn_dropout,
            batch_first=True)
        input_dim = hidden_dim*2 if bidirectional else hidden_dim

        # Deep Text Linear (dtlinear)
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