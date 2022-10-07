import torch
from torch import Tensor, nn

from ..modules import PositionalEmbedding, TokenEmbedding
from ..registry import ENCODERS


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


@ENCODERS.register_module
class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *cfg.encoder.layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

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
        # sequence_length = input_ids.shape[0]
        # padding_mask = (input_ids == self.pad_token_id).transpose(0, 1)
        # device = next(self.parameters()).device
        # mask = generate_square_subsequent_mask(sequence_length).to(device)

        embedded_tokens = self.embed_positions(self.embed_tokens(input_ids))
        # memory = self.encoder(embedded_tokens, mask=mask, src_key_padding_mask=padding_mask)
        memory = self.encoder(embedded_tokens)

        return memory
