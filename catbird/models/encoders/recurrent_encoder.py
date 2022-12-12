import torch
from torch import nn

from ..modules import RecurrentCell
from ..registry import ENCODERS


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
        dropout_in=0,
        dropout_out=0,
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
        self.residual = residual

        self.embed_tokens = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size,
            padding_idx=pad_token_id,
        )

        self.dropout_in_module = nn.Dropout(p=dropout_in)
        self.dropout_out_module = nn.Dropout(p=dropout_out)

        self.layers = nn.ModuleList(
            [
                RecurrentCell(
                    input_size=embedding_size if layer == 0 else hidden_size,
                    hidden_size=hidden_size,
                    mode=mode,
                )
                for layer in range(num_layers)
            ]
        )

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def combine_bidir(self, outs, bsz: int):
        out = outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous()
        return out.view(self.num_layers, bsz, -1)

    def reorder_encoder_out(self, encoder_out, new_order):
        return tuple(
            (
                encoder_out[0].index_select(1, new_order),
                encoder_out[1].index_select(1, new_order),
                encoder_out[2].index_select(1, new_order),
                encoder_out[3].index_select(1, new_order),
            )
        )

    def forward(self, input_ids, **kwargs):

        batch_size, seqlen = input_ids.shape

        x = self.embed_tokens(input_ids)
        x = self.dropout_in_module(x)  # batch x seqlen x hidden

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        zero_state = x.new_zeros(batch_size, self.hidden_size)
        prev_hiddens = [
            zero_state
            for _ in range(self.num_layers * (2 if self.bidirectional else 1))
        ]
        prev_cells = [
            zero_state
            for _ in range(self.num_layers * (2 if self.bidirectional else 1))
        ]

        outs = []
        for j in range(seqlen):
            input = x[j]
            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = self.dropout_out_module(hidden)
                if self.residual:
                    input = input + prev_hiddens[i]
                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            out = self.dropout_out_module(hidden)
            outs.append(out)

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, batch_size, self.output_units)
        assert x.shape == (seqlen, batch_size, self.output_units)

        final_hiddens = torch.stack(prev_hiddens)
        final_cells = torch.stack(prev_cells)

        if self.bidirectional:
            final_hiddens = self.combine_bidir(final_hiddens, batch_size)
            final_cells = self.combine_bidir(final_cells, batch_size)

        encoder_padding_mask = input_ids.eq(self.pad_token_id).t()

        return tuple(
            (
                x,  # seqlen x batch x hidden
                final_hiddens,  # num_layers x batch x num_directions*hidden
                final_cells,  # num_layers x batch x num_directions*hidden
                encoder_padding_mask,  # seqlen x batch
            )
        )
