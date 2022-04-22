import torch
import torch.nn as nn
import math
from .modules.state import State
from .modules.transformer_blocks import EncoderBlock, EncoderBlockPreNorm, positional_embedding



class TransformerAttentionEncoder(nn.Module):

    def __init__(self, vocab_size, pad_token_id=None, hidden_size=512, embedding_size=None,
                 num_layers=6, num_heads=8, inner_linear=2048, inner_groups=1, prenormalized=False,
                 batch_first=True, layer_norm=True, weight_norm=False, dropout=0, embedder=None):

        super(TransformerAttentionEncoder, self).__init__()
        self.pad_token_id = pad_token_id
        embedding_size = embedding_size or hidden_size
        if embedding_size != hidden_size:
            self.input_projection = nn.Parameter(
                torch.empty(embedding_size, hidden_size))
            nn.init.kaiming_uniform_(self.input_projection, a=math.sqrt(5))
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.mask_symbol = pad_token_id
        self.embedder = embedder or nn.Embedding(
            vocab_size, embedding_size, padding_idx=PAD)
        self.scale_embedding = hidden_size ** 0.5
        self.dropout = nn.Dropout(dropout, inplace=True)
        if prenormalized:
            block = EncoderBlockPreNorm
        else:
            block = EncoderBlock
        self.blocks = nn.ModuleList([block(hidden_size,
                                           num_heads=num_heads,
                                           inner_linear=inner_linear,
                                           inner_groups=inner_groups,
                                           layer_norm=layer_norm,
                                           weight_norm=weight_norm,
                                           batch_first=batch_first,
                                           dropout=dropout)
                                     for _ in range(num_layers)
                                     ])
        if layer_norm and prenormalized:
            self.lnorm = nn.LayerNorm(hidden_size)

    def forward(self, inputs, hidden=None):
        batch_dim, time_dim = (0, 1) if self.batch_first else (1, 0)
        if self.mask_symbol is not None:
            padding_mask = inputs.eq(self.mask_symbol)
        else:
            padding_mask = None
        x = self.embedder(inputs).mul_(self.scale_embedding)
        if hasattr(self, 'input_projection'):
            x = x @ self.input_projection
        pos_embedding = positional_embedding(x.size(time_dim), x.size(-1),
                                             device=x.device)
        x.add_(pos_embedding.unsqueeze(batch_dim))
        x = self.dropout(x)

        for block in self.blocks:
            block.set_mask(padding_mask)
            x = block(x)

        if hasattr(self, 'lnorm'):
            x = self.lnorm(x)

        return State(outputs=x, mask=padding_mask, batch_first=self.batch_first)

