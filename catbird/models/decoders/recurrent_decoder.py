import torch
from torch import nn

from ..incremental_decoding import with_incremental_state
from ..modules import RecurrentCell
from ..registry import DECODERS


@with_incremental_state
@DECODERS.register_module
class RecurrentDecoder(nn.Module):
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
        residual=False,
    ):
        super(RecurrentDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.mode = mode
        self.residual = residual

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
    def forward(self, input_ids, encoder_out, incremental_state=None, **kwargs):
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
        x, attn_scores = self.extract_features(
            input_ids, encoder_out, incremental_state
        )
        return self.fc_out(x), attn_scores

    def extract_features(self, input_ids, encoder_out, incremental_state=None):
        # print(incremental_state)

        # encoder_outs = encoder_out[0]
        encoder_hiddens = encoder_out[1]
        encoder_cells = encoder_out[2]
        # encoder_padding_mask = encoder_out[3]

        if incremental_state is not None and len(incremental_state) > 0:
            input_ids = input_ids[:, -1:]

        batch_size, seqlen = input_ids.shape

        x = self.embed_tokens(input_ids)
        x = self.dropout_in_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        if incremental_state is not None and len(incremental_state) > 0:
            prev_hiddens, prev_cells, input_feed = self.get_cached_state(
                incremental_state
            )
        else:
            # setup recurrent cells
            prev_hiddens = [encoder_hiddens[i] for i in range(self.num_layers)]
            prev_cells = [encoder_cells[i] for i in range(self.num_layers)]
            input_feed = x.new_zeros(batch_size, self.hidden_size)

        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            if input_feed is not None:
                input = torch.cat((x[j, :, :], input_feed), dim=1)
            else:
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

            out = hidden
            out = self.dropout_out_module(out)

            # input feeding
            if input_feed is not None:
                input_feed = out

            # save final output
            outs.append(out)

        # Stack all the necessary tensors together and store
        prev_hiddens_tensor = torch.stack(prev_hiddens)
        prev_cells_tensor = torch.stack(prev_cells)
        cache_state = {
            "prev_hiddens": prev_hiddens_tensor,
            "prev_cells": prev_cells_tensor,
            "input_feed": input_feed,
        }
        self.set_incremental_state(incremental_state, "cached_state", cache_state)

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, batch_size, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        attn_scores = None
        return x, attn_scores

    def get_cached_state(
        self,
        incremental_state,
    ):
        cached_state = self.get_incremental_state(incremental_state, "cached_state")
        assert cached_state is not None
        prev_hiddens_ = cached_state["prev_hiddens"]
        assert prev_hiddens_ is not None
        prev_cells_ = cached_state["prev_cells"]
        assert prev_cells_ is not None
        prev_hiddens = [prev_hiddens_[i] for i in range(self.num_layers)]
        prev_cells = [prev_cells_[j] for j in range(self.num_layers)]
        input_feed = cached_state[
            "input_feed"
        ]  # can be None for decoder-only language models
        return prev_hiddens, prev_cells, input_feed

    def reorder_incremental_state(
        self,
        incremental_state,
        new_order,
    ):
        if incremental_state is None or len(incremental_state) == 0:
            return
        prev_hiddens, prev_cells, input_feed = self.get_cached_state(incremental_state)
        prev_hiddens = [p.index_select(0, new_order) for p in prev_hiddens]
        prev_cells = [p.index_select(0, new_order) for p in prev_cells]
        if input_feed is not None:
            input_feed = input_feed.index_select(0, new_order)
        cached_state_new = {
            "prev_hiddens": torch.stack(prev_hiddens),
            "prev_cells": torch.stack(prev_cells),
            "input_feed": input_feed,
        }
        self.set_incremental_state(incremental_state, "cached_state", cached_state_new),
        return
