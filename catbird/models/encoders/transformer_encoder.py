import torch
from torch import nn

from ..modules import PositionalEmbedding, TokenEmbedding
from ..registry import ENCODERS


@ENCODERS.register_module
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_token_id=None,
        embedding_size=256,
        num_heads=8,
        num_layers=3,
        ffnn_size=512,
        dropout=0.1,
    ):
        super(TransformerEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ffnn_size = ffnn_size

        self.embed_tokens = TokenEmbedding(vocab_size, embedding_size)
        self.embed_positions = PositionalEmbedding(embedding_size, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            embedding_size,
            num_heads,
            ffnn_size,
            dropout,
        )
        encoder_norm = nn.LayerNorm(embedding_size)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

    def forward(
        self,
        input_ids,
    ):

        # seq_len = input_ids.shape[1]
        # device = next(self.parameters()).device

        embedded_tokens = self.embed_positions(self.embed_tokens(input_ids))
        # B x T x C -> T x B x C
        embedded_tokens = embedded_tokens.transpose(0, 1)

        # mask = torch.zeros((seq_len, seq_len),device=device).type(torch.bool)

        # padding_mask = None
        # if self.training:
        padding_mask = (input_ids == self.pad_token_id)
        # memory = self.encoder(embedded_tokens, mask=mask, src_key_padding_mask=padding_mask)
        memory = self.encoder(embedded_tokens, src_key_padding_mask=padding_mask)

        return (memory,)

    def reorder_encoder_out(self, encoder_out, new_order):
        return (encoder_out[0].index_select(1, new_order),)
