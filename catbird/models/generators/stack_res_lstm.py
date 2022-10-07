from ..registry import GENERATORS
from .base import EncoderDecoderBase


@GENERATORS.register_module
class StackedResidualLSTM(EncoderDecoderBase):
    """This is an implementation of paper `Neural Paraphrase Generation
    with Stacked Residual LSTM Networks <https://aclanthology.org/C16-1275/>`.
    """

    def __init__(
        self, pad_token_id, eos_token_id, decoder_start_token_id, encoder, decoder
    ):
        super(StackedResidualLSTM, self).__init__(
            pad_token_id, eos_token_id, decoder_start_token_id, encoder, decoder
        )
