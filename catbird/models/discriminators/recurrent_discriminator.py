import torch
from torch import nn

from ..registry import DISCRIMINATORS
from ..utils import Recurrent


def one_hot(t, c):
    return torch.zeros(*t.size(), c, device=t.device).scatter_(
        -1, t.unsqueeze(-1).data.long(), 1
    )


@DISCRIMINATORS.register_module
class RecurrentDiscriminator(nn.Module):
    def __init__(
        self,
        vocabulary_size,
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
        self.vocabulary_size = vocabulary_size
        self.layers = num_layers
        self.bidirectional = bidirectional
        self.pad_token_id = pad_token_id
        embedding_size = embedding_size or hidden_size
        out_size = out_size or hidden_size
        
        # self.embed_tokens = nn.Embedding(
        #     num_embeddings=vocabulary_size,
        #     embedding_dim=embedding_size,
        #     padding_idx=self.pad_token_id,
        # )
        self.embeddings = nn.Sequential(
            nn.Linear(self.vocabulary_size, hidden_size),
            nn.Threshold(0.000001, 0),
            nn.Linear(hidden_size, embedding_size),
            nn.Threshold(0.000001, 0),
        )
        self.rnn = Recurrent(
            mode,
            embedding_size,
            hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            residual=residual,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        
        self.dropout = nn.Dropout(p=dropout)

        # Define the output projection.
        self.fc_out = nn.Linear(hidden_size, out_size)

    def forward(self, decoder_out, tgt):
        x = self.embeddings(torch.exp(decoder_out))
        _, hidden_t = self.rnn(x)
        encoded_out = self.fc_out(hidden_t)
        
        x = self.embeddings(one_hot(tgt, self.vocabulary_size))
        _, hidden_t = self.rnn(x)
        encoded_tgt = self.fc_out(hidden_t)
        
        return encoded_out, encoded_tgt
        
        
