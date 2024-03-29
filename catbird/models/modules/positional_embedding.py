import math

import torch
from torch import Tensor, nn


class PositionalEmbedding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEmbedding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1) # [[0], [1], [2], ..., [maxlen - 1]]
        pos_embedding = torch.zeros((maxlen, emb_size))
        # PE(pos, 2i) = sin(pos/1000^(2i/emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/emb_size))
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )
