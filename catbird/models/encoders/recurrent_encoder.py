from torch import nn

from ..registry import ENCODERS
from ..utils import RecurrentLayer
import torch.nn.functional as F


@ENCODERS.register_module
class RecurrentEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
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
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pad_token_id = pad_token_id
        embedding_size = embedding_size or hidden_size
        self.hidden_size = hidden_size

        self.embed_tokens = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size,
            padding_idx=self.pad_token_id,
        )
        
        # self.embed_tokens = nn.Sequential(
        #     nn.Linear(vocab_size, embedding_size // 2),
        #     nn.Linear(embedding_size // 2, embedding_size)
        # )
        
        self.dropout = nn.Dropout(p=dropout)

        self.rnn = RecurrentLayer(
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
        # x = self.embed_tokens(F.one_hot(input_ids, self.vocab_size).float())
        
        x = self.dropout(x)
        
        # B x T x C -> T x B x C
        # x = x.transpose(0, 1)

        if self.bidirectional:
            state_size = 2 * self.num_layers, batch_size, self.hidden_size
        else:
            state_size = self.num_layers, batch_size, self.hidden_size
            
        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)
        outs, hidden = self.rnn(x, (h0, c0))
        # outs, hidden = self.rnn(x)

        return (
            outs,  # batch x seq_len x hidden
            hidden,  # num_layers x batch x num_directions*hidden
        )
