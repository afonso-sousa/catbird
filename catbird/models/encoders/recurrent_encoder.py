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
        self.layers = num_layers
        self.bidirectional = bidirectional
        self.pad_token_id = pad_token_id
        embedding_size = embedding_size or hidden_size

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

    def forward(self, src_tokens, hidden=None):
        """[summary]

        Args:
            src_tokens ([type]): hape `(batch, src_len)`

        Returns:
            [type]: [description]
        """

        # Note that the source is typically padded on the left. This can be
        # configured by adding the `--left-pad-source "False"` command-line
        # argument, but here we'll make the Encoder handle either kind of
        # padding by converting everything to be right-padded.
        # if self.cfg.left_pad_source:
        #     # Convert left-padding to right-padding.
        #     src_tokens = convert_padding_direction(
        #         src_tokens, padding_idx=self.pad_token_id, left_to_right=True
        #     )

        x = self.embed_tokens(src_tokens)
        x = self.dropout(x)

        # Pack the sequence into a PackedSequence object to feed to the LSTM.
        # x = nn.utils.rnn.pack_padded_sequence(
        #     x, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        # )

        x, hidden_t = self.rnn(x, hidden)

        state = State(
            outputs=x,  # batch x seq_len x hidden
            hidden=hidden_t,  # num_layers x batch x num_directions*hidden
        )

        return state
        # return tuple(
        #     (
        #         x,  # batch x seq_len x hidden
        #         final_hiddens,  # num_layers x batch x num_directions*hidden
        #         final_cells,  # num_layers x batch x num_directions*hidden
        #     )
        # )

    # Encoders are required to implement this method so that we can rearrange
    # the order of the batch elements during inference (e.g., beam search).
    # def reorder_encoder_out(self, encoder_out, new_order):
    #     """
    #     Reorder encoder output according to `new_order`.

    #     cfg:
    #         encoder_out: output from the ``forward()`` method
    #         new_order (LongTensor): desired order

    #     Returns:
    #         `encoder_out` rearranged according to `new_order`
    #     """
    #     final_hidden = encoder_out["final_hidden"]
    #     return {
    #         "final_hidden": final_hidden.index_select(0, new_order),
    #     }
