import torch
from torch import Tensor, nn

from ..modules import PositionalEmbedding, TokenEmbedding
from ..registry import DECODERS


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


@DECODERS.register_module
class TransformerDecoder(nn.Module):
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

        super(TransformerDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ffnn_size = ffnn_size

        self.dropout_module = nn.Dropout(p=dropout)

        self.embed_tokens = TokenEmbedding(vocab_size, embedding_size)
        self.embed_positions = PositionalEmbedding(embedding_size, dropout=dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            embedding_size, num_heads, ffnn_size, dropout
        )
        decoder_norm = nn.LayerNorm(embedding_size)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers, decoder_norm)
        self.fc_out = nn.Linear(embedding_size, vocab_size)

    def forward(
        self,
        input_ids,
        encoder_out,
        incremental_state=None,
        graph_embeddings=None,
    ):
        seq_len = input_ids.shape[1]

        device = next(self.parameters()).device
        mask = generate_square_subsequent_mask(seq_len).to(device)

        embedded_tokens = self.embed_positions(self.embed_tokens(input_ids))

        # B x T x C -> T x B x C
        embedded_tokens = embedded_tokens.transpose(0, 1)

        # padding_mask = None
        # if self.training:
        #     padding_mask = (input_ids == self.pad_token_id).transpose(0, 1)
        # output = self.decoder(embedded_tokens, encoder_out[0], tgt_mask=mask, tgt_key_padding_mask=padding_mask)
        output = self.decoder(embedded_tokens, encoder_out[0], tgt_mask=mask)

        # T x B x C -> B x T x C
        output = output.transpose(1, 0)

        return (self.fc_out(output),)
