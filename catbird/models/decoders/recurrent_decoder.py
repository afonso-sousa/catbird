import torch
from torch import nn

from ..registry import DECODERS
from ..state import State
from ..utils import RecurrentLayer, RecurrentCell
from .base_decoder import BaseDecoder
import torch.nn.functional as F


@DECODERS.register_module
class RecurrentDecoder3(BaseDecoder):
    def __init__(
        self,
        vocab_size,
        pad_token_id=None,
        encoder_output_units=128,
        embedding_size=128,
        hidden_size=128,
        dropout_in=0.1,
        dropout_out=0.1,
        num_layers=1,
        mode="LSTM",
        residuals=False,
    ):
        super(RecurrentDecoder3, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.mode = mode
        self.residuals = residuals
        self.embed_tokens = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size,
            padding_idx=pad_token_id,
        )
        self.dropout_in_module = nn.Dropout(p=dropout_in)
        self.dropout_out_module = nn.Dropout(p=dropout_out)

        # input feeding is described in arxiv.org/abs/1508.04025
        input_feed_size = 0 if encoder_output_units == 0 else hidden_size
        self.layers = nn.ModuleList(
            [
                RecurrentCell(
                    input_size=input_feed_size + embedding_size
                    if layer == 0
                    else hidden_size,
                    hidden_size=hidden_size,
                    mode=mode,
                )
                for layer in range(num_layers)
            ]
        )

        # Define the output projection.
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    # During training Decoders are expected to take the entire target sequence
    # (shifted right by one position) and produce logits over the vocabulary.
    # The *prev_output_tokens* tensor begins with the end-of-sentence symbol,
    # ``dictionary.eos()``, followed by the target sequence.
    def forward(self, input_ids, encoder_hidden_states, **kwargs):
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
        x, new_state = self.extract_features(input_ids, encoder_hidden_states)
        return self.fc_out(x), new_state

    def extract_features(self, input_ids, encoder_hidden_states):
        encoder_hiddens = encoder_hidden_states
        # encoder_hiddens = encoder_hiddens[:self.num_layers]

        batch_size, seqlen = input_ids.size()

        # Embed the target sequence, which has been shifted right by one
        # position and now starts with the end-of-sentence symbol.
        x = self.embed_tokens(input_ids)
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

        input_feed = x.new_zeros(batch_size, self.hidden_size)

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
        x = torch.cat(outs, dim=0).view(seqlen, batch_size, self.hidden_size)

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
        super(RecurrentDecoder, self).__init__()
        self.vocab_size = vocab_size
        embedding_size = embedding_size or hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pad_token_id = pad_token_id
        self.hidden_size = hidden_size
        self.embed_tokens = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size,
            padding_idx=self.pad_token_id,
        )
        
        self.dropout_in_module = nn.Dropout(p=dropout_in)
        self.dropout_out_module = nn.Dropout(p=dropout_out)
                
        self.rnn = RecurrentLayer(
            mode,
            embedding_size,
            self.hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            residual=residual,
            dropout=dropout_out,
            bidirectional=bidirectional,
        )

        self.fc_out = nn.Linear(hidden_size, vocab_size)


    def forward(self, input_ids, encoder_hidden_states, **kwargs):
        """
        cfg:
            input_ids (LongTensor): previous decoder outputs of shape
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
        x, new_state = self.extract_features(input_ids, encoder_hidden_states)
        return self.fc_out(x), new_state


    def extract_features(self, input_ids, encoder_hidden_states):
        # input_ids = input_ids.unsqueeze(0)
        hidden = encoder_hidden_states
        # hiddens, cells = state.hidden
        # hiddens = hiddens[:self.num_layers]
        # cells = cells[:self.num_layers]
        
        x = self.embed_tokens(input_ids)
        x = self.dropout_in_module(x)
        # x = self.embed_tokens(F.one_hot(input_ids, self.vocab_size).float())

        # B x T x C -> T x B x C
        #x = x.transpose(0, 1)
        
        x, hidden_t = self.rnn(x, hidden)
        x = self.dropout_out_module(x)
        return x, hidden_t
        # return x.squeeze(0), State(hidden=hidden_t)
    

    # def extract_features(self, prev_output_tokens, state):
    #     hidden = state.hidden
    #     # hiddens, cells = state.hidden
    #     # hiddens = hiddens[:self.num_layers]
    #     # cells = cells[:self.num_layers]
    #     x = self.embed_tokens(prev_output_tokens).unsqueeze(1)
    #     emb = self.dropout_in_module(x)
    #     x, hidden_t = self.rnn(emb, hidden) # [batch_size, 1, hidden_size], ([num_layers, batch_size, hidden_size])
    #     x = self.dropout_out_module(x)
    #     return x.squeeze(1), State(hidden=hidden_t)
