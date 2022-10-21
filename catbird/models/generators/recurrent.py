from ..registry import GENERATORS
from .base import EncoderDecoderBase


@GENERATORS.register_module
class RecurrentModel(EncoderDecoderBase):
    def __init__(
        self, pad_token_id, eos_token_id, decoder_start_token_id, encoder, decoder
    ):
        super(RecurrentModel, self).__init__(
            pad_token_id, eos_token_id, decoder_start_token_id, encoder, decoder
        )
