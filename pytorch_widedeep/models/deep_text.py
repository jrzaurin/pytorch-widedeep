import numpy as np
import torch
import warnings

from torch import nn
from ..wdtypes import *


class DeepText(nn.Module):
    def __init__(self,
        vocab_size:int,
        hidden_dim:int=64,
        n_layers:int=3,
        rnn_dropout:float=0.,
        padding_idx:int=1,
        bidirectional:bool=False,
        embed_dim:Optional[int]=None,
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
        self.word_embed_dropout = nn.Dropout2d(spatial_dropout)

        # Pre-trained Embeddings
        if isinstance(embedding_matrix, np.ndarray):
            self.word_embed = nn.Embedding(vocab_size, embedding_matrix.shape[1], padding_idx = padding_idx)
            self.word_embed.weight = nn.Parameter(torch.Tensor(embedding_matrix))
            embed_dim = embedding_matrix.shape[1]
        else:
            self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx = padding_idx)

        # stack of RNNs (LSTMs)
        self.rnn = nn.LSTM(embed_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=rnn_dropout,
            batch_first=True)

        # the output_dim attribute will be used as input_dim when "merging" the models
        self.output_dim = hidden_dim*2 if bidirectional else hidden_dim

    def forward(self, X:Tensor)->Tensor:

        embed = self.word_embed(X.long())
        o, (h, c) = self.rnn(embed)
        if self.bidirectional:
            last_h = torch.cat((h[-2], h[-1]), dim = 1)
        else:
            last_h = h[-1]
        return last_h
