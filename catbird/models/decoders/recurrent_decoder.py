import torch
from torch import nn

from ..registry import DECODERS
from .base_decoder import BaseDecoder
from ..state import State


@DECODERS.register_module
class RecurrentDecoder(BaseDecoder):
    def __init__(
        self,
        vocabulary_size,
        pad_token_id=None,
        encoder_hidden_dim=128,
        embed_dim=128,
        hidden_dim=128,
        dropout_in=0.1,
        dropout_out=0.1,
        num_layers=1,
        residuals=False,
    ):
        super(RecurrentDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.residuals = residuals
        self.embed_tokens = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embed_dim,
            padding_idx=pad_token_id,
        )
        self.dropout_in_module = nn.Dropout(p=dropout_in)
        self.dropout_out_module = nn.Dropout(p=dropout_out)

        self.layers = nn.ModuleList(
            [
                nn.LSTMCell(
                    input_size=encoder_hidden_dim + embed_dim
                    if layer == 0
                    else hidden_dim,
                    hidden_size=hidden_dim,
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
    def forward(self, prev_output_tokens, state):
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

        bsz, seqlen = prev_output_tokens.size()

        # Embed the target sequence, which has been shifted right by one
        # position and now starts with the end-of-sentence symbol.
        x = self.embed_tokens(prev_output_tokens)
        x = self.dropout_in_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        prev_hiddens_t = [
            (encoder_hiddens[0][i], encoder_hiddens[1][i])
            for i in range(self.num_layers)
        ]
        # prev_cells = [torch.zeros_like(prev_hiddens[0]) for i in range(self.num_layers)]
        input_feed = x.new_zeros(bsz, self.hidden_dim)

        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            input = torch.cat((x[j, :, :], input_feed), dim=1)

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hiddens = rnn(input, prev_hiddens_t[i])

                # hidden state becomes the input to the next layer
                input = self.dropout_out_module(hiddens[0])
                if self.residuals:
                    input = input + prev_hiddens_t[i][0]

                # save state for next time step
                prev_hiddens_t[i] = hiddens

            out = hiddens[0]
            out = self.dropout_out_module(out)

            input_feed = out

            # save final output
            outs.append(out)

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_dim)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        hidden_t = tuple(torch.stack(i) for i in zip(*prev_hiddens_t))

        return x, State(hidden=hidden_t)
