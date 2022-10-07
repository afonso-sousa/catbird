import torch
import torch.nn.functional as F
from torch import nn

from ..modules import RecurrentLayer
from ..registry import DISCRIMINATORS


@DISCRIMINATORS.register_module
class RecurrentDiscriminator(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_token_id=None,
        hidden_size=128,
        embedding_size=None,
        out_size=None,
        num_layers=1,
        bias=True,
        dropout=0,
        bidirectional=False,
        mode="LSTM",
        residual=False,
    ):
        super(RecurrentDiscriminator, self).__init__()
        self.vocab_size = vocab_size
        self.layers = num_layers
        self.bidirectional = bidirectional
        self.pad_token_id = pad_token_id
        embedding_size = embedding_size or hidden_size
        out_size = out_size or hidden_size

        # self.embed_tokens = nn.Embedding(
        #     num_embeddings=vocab_size,
        #     embedding_dim=embedding_size,
        #     padding_idx=self.pad_token_id,
        # )
        self.embed_tokens = nn.Sequential(
            nn.Linear(self.vocab_size, hidden_size),
            nn.Threshold(1e-6, 0),
            nn.Linear(hidden_size, embedding_size),
            nn.Threshold(1e-6, 0),
        )
        self.rnn = RecurrentLayer(
            mode,
            embedding_size,
            hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=False,
            residual=residual,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(p=dropout)

        # Define the output projection.
        self.fc_out = nn.Linear(hidden_size, out_size)

    def forward(self, decoder_out, tgt):
        x = F.softmax(decoder_out, dim=-1)
        x = self.embed_tokens(x)
        # B x T x H -> T x B x H
        x = x.transpose(0, 1)
        _, hidden_t = self.rnn(x)
        encoded_out = self.fc_out(hidden_t)  # 1, B, H

        x = self.embed_tokens(F.one_hot(tgt, self.vocab_size).to(torch.float32))

        # B x T x H -> T x B x H
        x = x.transpose(0, 1)
        _, hidden_t = self.rnn(x)
        encoded_tgt = self.fc_out(hidden_t)

        return encoded_out.squeeze_(0), encoded_tgt.squeeze_(0)
