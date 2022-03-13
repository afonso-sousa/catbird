from typing import Optional

from catbird.models.losses import pair_wise_loss

from ..builder import build_discriminator
from ..postprocessing.search import SequenceGenerator
from ..registry import GENERATORS
from .base import Seq2Seq


@GENERATORS.register_module
class EDD(Seq2Seq):
    """
    LSTM-based Encoder-Decoder-Discriminator (EDD) architecture.

    This is an implementation of paper `Learning Semantic Sentence Embeddings
    using Sequential Pair-wise Discriminator <https://aclanthology.org/C18-1230/>`.
    """

    def __init__(self, decoder_start_token_id, encoder, decoder, discriminator):
        super(EDD, self).__init__(decoder_start_token_id, encoder, decoder)
        self.discriminator = build_discriminator(discriminator)

    def forward(self, input_ids, prev_output_tokens, tgt, return_loss=True, **kwargs):
        state = self.encoder(input_ids)
        decoder_out, _ = self.decoder(prev_output_tokens, state=state)
        discriminated_out, discriminated_tgt = self.discriminator(decoder_out, tgt)
        if return_loss:
            return self.loss(decoder_out, tgt, discriminated_out, discriminated_tgt)
        else:
            return decoder_out

    def loss(self, out, tgt, discriminated_out, discriminated_tgt):
        cel_loss = super().loss(out, tgt)
        pwd_loss = pair_wise_loss(discriminated_out, discriminated_tgt)
        return cel_loss + pwd_loss

