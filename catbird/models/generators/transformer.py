import torch

from catbird.models.generators.base import EncoderDecoderBase

from ..registry import GENERATORS


@GENERATORS.register_module
class VanillaTransformer(EncoderDecoderBase):
    """This is an implementation of the Transformer model from `"Attention Is All You Need"
    <https://arxiv.org/abs/1706.03762>`.
    """

    def __init__(
        self, pad_token_id, eos_token_id, decoder_start_token_id, encoder, decoder
    ):
        super(VanillaTransformer, self).__init__(
            pad_token_id, eos_token_id, decoder_start_token_id, encoder, decoder
        )
