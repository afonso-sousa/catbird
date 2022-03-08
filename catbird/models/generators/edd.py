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

    def __init__(self, encoder, decoder, discriminator):
        super(EDD, self).__init__(encoder, decoder)
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

    def generate(
        self,
        input_encoder,
        beam_size=3,
        max_sequence_length=50,
        length_normalization_factor=0,
        autoregressive=True,
        eos_token_id: Optional[int] = 2,
    ):
        batch_size = input_encoder.size(0) if input_encoder.dim() > 1 else 1
        input_decoder = [[eos_token_id]] * batch_size
        state = self.forward_encoder(input_encoder,)
        state_list = state.as_list()
        params = dict(
            decode_step=self._decode_step,
            beam_size=beam_size,
            max_sequence_length=max_sequence_length,
            length_normalization_factor=length_normalization_factor,
        )
        if autoregressive:
            generator = SequenceGenerator(**params)
        seqs = generator.greedy_search(input_decoder, state_list)

        return seqs
