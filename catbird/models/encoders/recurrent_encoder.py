from torch import nn

from ..registry import ENCODERS
from ..utils import Recurrent
from ..state import State


@ENCODERS.register_module
class RecurrentEncoder(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        pad_token_id=None,
        hidden_size=128,
        embedding_size=None,
        num_layers=1,
        bias=True,
        dropout=0,
        bidirectional=False,
        mode="LSTM",
        residual=False,
    ):
        super(RecurrentEncoder, self).__init__()
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pad_token_id = pad_token_id
        embedding_size = embedding_size or hidden_size
        self.hidden_size = hidden_size

        self.embed_tokens = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_size,
            padding_idx=self.pad_token_id,
        )
        self.dropout = nn.Dropout(p=dropout)

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
        

    def forward(self, input_ids, **kwargs):
        batch_size = input_ids.size(0)
        
        x = self.embed_tokens(input_ids)
        x = self.dropout(x)

        if self.bidirectional:
            state_size = 2 * self.num_layers, batch_size, self.hidden_size
        else:
            state_size = self.num_layers, batch_size, self.hidden_size
        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)
        outs, hidden = self.rnn(x, (h0, c0))

        state = State(
            outputs=outs,  # batch x seq_len x hidden
            hidden=hidden,  # num_layers x batch x num_directions*hidden
        )

        return state
