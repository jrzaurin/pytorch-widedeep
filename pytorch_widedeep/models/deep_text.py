import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..wdtypes import *


class DeepText(nn.Module):
    def __init__(self, vocab_size:int, embedding_dim:int, hidden_dim:int, n_layers:int,
        rnn_dropout:float, spatial_dropout:float, padding_idx:int, output_dim:int,
        attention:bool=False, bidirectional:bool=False,
        embedding_matrix:Optional[np.ndarray]=None):
        super(DeepText, self).__init__()
        """
        Standard Text Classifier/Regressor with a stack of RNNs.
        """

        self.bidirectional = bidirectional
        self.attention = attention
        self.spatial_dropout = spatial_dropout
        self.embedding_dropout = nn.Dropout2d(spatial_dropout)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = padding_idx)
        if isinstance(embedding_matrix, np.ndarray):
            self.embedding.weight = nn.Parameter(torch.Tensor(embedding_matrix))
        self.rnn = nn.GRU(embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=rnn_dropout,
            batch_first=True)
        input_dim = hidden_dim*2 if bidirectional else hidden_dim
        self.dtlinear = nn.Linear(input_dim, output_dim)

    def attention_net(self, output:Tensor, hidden:Tensor)->Tensor:
        """
        Attention through Soft alignment Score between output and last hidden.
        Read here (and references therein) for more details:
        https://machinelearningmastery.com/how-does-attention-work-in-encoder-decoder-recurrent-neural-networks/

        code from here (there are more sophisticated approaches but these will do):
        https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/models/LSTM_Attn.py
        """
        attn_weights = torch.bmm(output, hidden.unsqueeze(2)).squeeze(2)
        attn_weights = F.softmax(attn_weights, 1)
        new_hidden = torch.bmm(output.transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden

    def forward(self, X:Tensor)->Tensor:

        embedded = self.embedding(X)
        # Spatial dropout: dropping an entire channel (word-vector dimension)
        if self.spatial_dropout > 0.:
            sd_embedded = embedded.unsqueeze(2)
            sd_embedded = sd_embedded.permute(0, 3, 2, 1)
            sd_embedded = self.embedding_dropout(sd_embedded)
            sd_embedded = sd_embedded.permute(0, 3, 2, 1)
            embedded = sd_embedded.squeeze(2)
        o, h = self.rnn(embedded)
        if self.bidirectional:
            last_h = torch.cat((h[-2], h[-1]), dim = 1)
        else:
            last_h = h[-1]
        if self.attention:
            last_h = self.attention_net(o, last_h)
        out = self.dtlinear(last_h)
        return out
