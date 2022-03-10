import torch
from torch import nn

from ..registry import DECODERS
from ..state import State
from ..utils import Recurrent, RecurrentCell
from .base_decoder import BaseDecoder


@DECODERS.register_module
class RecurrentDecoder2(BaseDecoder):
    def __init__(
        self,
        vocabulary_size,
        pad_token_id=None,
        encoder_output_units=128,
        embed_dim=128,
        hidden_dim=128,
        dropout_in=0.1,
        dropout_out=0.1,
        num_layers=1,
        mode="LSTM",
        residuals=False,
    ):
        super(RecurrentDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.mode = mode
        self.residuals = residuals
        self.embed_tokens = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embed_dim,
            padding_idx=pad_token_id,
        )
        self.dropout_in_module = nn.Dropout(p=dropout_in)
        self.dropout_out_module = nn.Dropout(p=dropout_out)

        # input feeding is described in arxiv.org/abs/1508.04025
        input_feed_size = 0 if encoder_output_units == 0 else hidden_dim
        self.layers = nn.ModuleList(
            [
                RecurrentCell(
                    input_size=input_feed_size + embed_dim
                    if layer == 0
                    else hidden_dim,
                    hidden_size=hidden_dim,
                    mode=mode,
                )
                for layer in range(num_layers)
            ]
        )

        # Define the output projection.
        self.fc_out = nn.Linear(hidden_dim, vocabulary_size)

    # During training Decoders are expected to take the entire target sequence
    # (shifted right by one position) and produce logits over the vocabulary.
    # The *prev_output_tokens* tensor begins with the end-of-sentence symbol,
    # ``dictionary.eos()``, followed by the target sequence.
    def forward(self, prev_output_tokens, state, **kwargs):
        """
        cfg:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention

        Returns:
            tuple:
                - the last decoder layer's output of shape
                  `(batch, tgt_len, vocab)`
                - the last decoder layer's attention weights of shape
                  `(batch, tgt_len, src_len)`
        """
        x, new_state = self.extract_features(prev_output_tokens, state)
        return self.fc_out(x), new_state

    def extract_features(self, prev_output_tokens, state):
        encoder_hiddens = state.hidden
        # encoder_hiddens = encoder_hiddens[:self.num_layers]

        batch_size, seqlen = prev_output_tokens.size()

        # Embed the target sequence, which has been shifted right by one
        # position and now starts with the end-of-sentence symbol.
        x = self.embed_tokens(prev_output_tokens)
        x = self.dropout_in_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_is_lstm = isinstance(encoder_hiddens, tuple)

        if encoder_is_lstm:
            hiddens, cells = encoder_hiddens
            prev_hiddens = [hiddens[i] for i in range(self.num_layers)]
            prev_cells = [cells[i] for i in range(self.num_layers)]
        else:
            if self.mode == "LSTM":
                prev_hiddens = [encoder_hiddens[i] for i in range(self.num_layers)]
                prev_cells = [
                    torch.zeros_like(prev_hiddens[0]) for _ in range(self.num_layers)
                ]
            else:
                prev_hiddens = [encoder_hiddens[i] for i in range(self.num_layers)]

        input_feed = x.new_zeros(batch_size, self.hidden_dim)

        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            input = torch.cat((x[j, :, :], input_feed), dim=1)

            for i, rnn in enumerate(self.layers):
                # different cell type between encoder and decoder
                # if not isinstance(encoder_hiddens, tuple) and self.mode == "LSTM":
                if self.mode == "LSTM":
                    hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))
                else:
                    hidden = rnn(input, prev_hiddens[i])

                # hidden state becomes the input to the next layer
                input = self.dropout_out_module(hidden)
                if self.residuals:
                    input = input + prev_hiddens[i]

                # save state for next time step
                prev_hiddens[i] = hidden
                if self.mode == "LSTM":
                    prev_cells[i] = cell

            out = hidden
            out = self.dropout_out_module(out)

            input_feed = out

            # save final output
            outs.append(out)

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, batch_size, self.hidden_dim)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        if self.mode == "LSTM":
            hidden_t = torch.stack(prev_hiddens), torch.stack(prev_cells)
        else:
            hidden_t = torch.stack(prev_hiddens)

        return x, State(hidden=hidden_t)


@DECODERS.register_module
class RecurrentDecoder(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        pad_token_id=None,
        hidden_size=128,
        embedding_size=None,
        num_layers=1,
        bias=True,
        dropout_in=0,
        dropout_out=0,
        mode="LSTM",
        residual=False,
    ):
        super(RecurrentDecoder, self).__init__()
        embedding_size = embedding_size or hidden_size
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        self.hidden_size = hidden_size
        self.embed_tokens = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_size,
            padding_idx=self.pad_token_id,
        )
        self.dropout_in_module = nn.Dropout(p=dropout_in)
        self.dropout_out_module = nn.Dropout(p=dropout_out)
        self.rnn = Recurrent(
            mode,
            embedding_size,
            self.hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            residual=residual,
            dropout=dropout_out,
            bidirectional=False,
        )

        self.dropout = nn.Dropout(dropout_out)
        self.fc_out = nn.Linear(hidden_size, vocabulary_size)


    def forward(self, prev_output_tokens, state, **kwargs):
        """
        cfg:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention

        Returns:
            tuple:
                - the last decoder layer's output of shape
                  `(batch, tgt_len, vocab)`
                - the last decoder layer's attention weights of shape
                  `(batch, tgt_len, src_len)`
        """
        x, new_state = self.extract_features(prev_output_tokens, state)
        return self.fc_out(x), new_state


    def extract_features(self, prev_output_tokens, state):
        hidden = state.hidden
        # hiddens, cells = state.hidden
        # hiddens = hiddens[:self.num_layers]
        # cells = cells[:self.num_layers]
        x = self.embed_tokens(prev_output_tokens)
        emb = self.dropout_in_module(x)
        x, hidden_t = self.rnn(emb, hidden)
        x = self.dropout_out_module(x)
        return x, State(hidden=hidden_t)
